"""Upload finished Shorts to YouTube via the Data API v3.

The whole flow is OAuth 2.0 for an installed ("Desktop") app:

1. ``client_secret.json`` identifies this application to Google. It is not a
   user credential and grants nothing on its own.
2. The first authorisation opens a browser, the operator consents, and Google
   returns a refresh token. That token IS a credential — it grants upload
   access to the channel until revoked — and it is cached in
   ``storage/youtube_token.json`` with owner-only permissions.
3. Every later run refreshes silently from that cache. No browser, no prompt.

Consent is deliberately a separate, explicit step (``main.py --youtube-auth``).
An unattended daemon must never be the thing that pops a browser window, and a
first upload should not be the moment anyone discovers the account is wrong.

Nothing here uploads under DRY_RUN, and nothing uploads unless
``YOUTUBE_UPLOAD_ENABLED`` is on: publishing to a real channel is not a
side effect anyone should get by accident.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from config.settings import Settings, get_settings

#: Upload scope only. Deliberately not ``youtube.force-ssl`` or full ``youtube``:
#: this code publishes videos and has no business reading or deleting anything
#: else on the account.
SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/youtube.upload",)

#: 24 is "Entertainment" in the YouTube category list, which is where short
#: documentary/mystery content belongs. Category IDs are region-dependent for
#: *availability* but the numbering itself is global.
CATEGORY_ENTERTAINMENT = "24"

#: YouTube decides what is a Short from the file itself — vertical and under 60
#: seconds — but the tag and the title marker are what the Shorts shelf and the
#: mobile app key off, and cost nothing.
SHORTS_TAG = "Shorts"

#: Hard API limits. Exceeding any of them is a 400, so they are enforced here
#: rather than discovered on upload.
MAX_TITLE = 100
MAX_DESCRIPTION = 5000
MAX_TAGS_TOTAL = 500

#: Resumable upload chunk. Large enough that a 20MB Short is one or two round
#: trips, small enough that a dropped connection does not lose everything.
CHUNK_BYTES = 4 * 1024 * 1024

#: Transient HTTP statuses worth retrying, per Google's own guidance.
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})

#: Attempts per upload before giving up.
MAX_ATTEMPTS = 4


class YouTubeError(RuntimeError):
    """Raised when an upload cannot be completed."""


@dataclass
class UploadResult:
    """The outcome of one upload attempt."""

    ok: bool
    video_id: str = ""
    url: str = ""
    privacy_status: str = ""
    simulated: bool = False
    error: str = ""
    title: str = ""

    @property
    def watch_url(self) -> str:
        """Canonical Shorts link, which is what a phone should open."""
        return f"https://www.youtube.com/shorts/{self.video_id}" if self.video_id else ""

    def summary(self) -> str:
        if not self.ok:
            return f"upload failed: {self.error}"
        if self.simulated:
            return "upload simulated (dry run)"
        return f"published {self.video_id} [{self.privacy_status}]"


def _clean_tags(tags: Sequence[str]) -> list[str]:
    """Normalise hashtags into YouTube's tag list.

    The API counts the TOTAL characters across all tags, not each one, and
    rejects the whole request when the sum passes 500. Tags are added until the
    budget runs out rather than truncating mid-word.
    """
    seen: set[str] = set()
    chosen: list[str] = [SHORTS_TAG]
    budget = MAX_TAGS_TOTAL - len(SHORTS_TAG)

    for raw in tags:
        tag = str(raw).strip().lstrip("#")
        if not tag or tag.lower() in {t.lower() for t in chosen} or tag.lower() in seen:
            continue
        seen.add(tag.lower())
        if len(tag) + 1 > budget:
            break
        chosen.append(tag)
        budget -= len(tag) + 1

    return chosen


def build_description(description: str, tags: Sequence[str]) -> str:
    """Compose the description, ending with the hashtags YouTube indexes.

    ``#Shorts`` goes in the description as well as the tag list because the
    mobile app reads hashtags out of the description text.
    """
    body = (description or "").strip()
    hashes = " ".join(f"#{tag.lstrip('#')}" for tag in tags if str(tag).strip())
    marker = f"#{SHORTS_TAG}"
    if marker.lower() not in hashes.lower():
        hashes = f"{marker} {hashes}".strip()

    combined = f"{body}\n\n{hashes}".strip() if body else hashes
    return combined[:MAX_DESCRIPTION]


def build_title(title: str) -> str:
    """Trim the title to the API limit, keeping the Shorts marker if it fits."""
    text = (title or "Untitled Short").strip()
    marker = f" #{SHORTS_TAG}"

    if f"#{SHORTS_TAG}".lower() in text.lower():
        return text[:MAX_TITLE]
    if len(text) + len(marker) <= MAX_TITLE:
        return text + marker
    return text[:MAX_TITLE]


class YouTubePublisher:
    """Authorises once, then uploads Shorts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ---- configuration ------------------------------------------------------
    @property
    def configured(self) -> bool:
        """Whether a client secret exists to authorise against."""
        return self.settings.youtube_configured

    @property
    def authorized(self) -> bool:
        """Whether a cached token exists."""
        return self.settings.youtube_authorized

    def status(self) -> tuple[bool, str]:
        """Whether an upload could run right now, and why not if it could not."""
        if not self.settings.youtube_upload_enabled:
            return False, "YOUTUBE_UPLOAD_ENABLED is off"
        if not self.configured:
            return False, (
                f"no client secret at {self.settings.youtube_client_secret_path}"
            )
        if not self.authorized:
            return False, "not authorised yet — run: python main.py --youtube-auth"
        return True, ""

    # ---- credentials --------------------------------------------------------
    def load_credentials(self, *, interactive: bool = False) -> Any:
        """Return usable OAuth credentials, refreshing or minting as needed.

        :param interactive: allow opening a browser for first consent. False
            everywhere except the explicit ``--youtube-auth`` command, so no
            background job can ever block waiting on a human.
        :raises YouTubeError: if credentials are unavailable.
        """
        from google.auth.transport.requests import Request  # noqa: PLC0415
        from google.oauth2.credentials import Credentials  # noqa: PLC0415

        token_path = self.settings.youtube_token_path
        credentials = None

        if token_path.is_file():
            try:
                credentials = Credentials.from_authorized_user_file(
                    str(token_path), list(SCOPES)
                )
            except Exception as exc:  # noqa: BLE001 - a corrupt cache is recoverable
                logger.warning("Ignoring unreadable token cache {}: {}", token_path, exc)
                credentials = None

        if credentials and credentials.valid:
            return credentials

        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                self._save_credentials(credentials)
                logger.debug("Refreshed the cached YouTube token")
                return credentials
            except Exception as exc:  # noqa: BLE001
                # A revoked or expired refresh token lands here. Consent has to
                # be granted again; nothing else will fix it.
                logger.warning("Could not refresh the YouTube token: {}", exc)
                credentials = None

        if not interactive:
            raise YouTubeError(
                "No usable YouTube credentials. Run: python main.py --youtube-auth"
            )

        return self._run_consent_flow()

    def _run_consent_flow(self) -> Any:
        """Open a browser, take consent, cache the result."""
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: PLC0415

        secret = self.settings.youtube_client_secret_path
        if not secret.is_file():
            raise YouTubeError(f"No OAuth client secret at {secret}")

        logger.info("Opening a browser for YouTube consent…")
        flow = InstalledAppFlow.from_client_secrets_file(str(secret), list(SCOPES))
        # Port 0 lets the OS pick, which avoids colliding with anything already
        # bound; the loopback redirect is registered as a prefix, not a port.
        credentials = flow.run_local_server(port=0, prompt="consent")
        self._save_credentials(credentials)
        logger.success("YouTube authorised. Token cached at {}", self.settings.youtube_token_path)
        return credentials

    def _save_credentials(self, credentials: Any) -> None:
        """Cache the token with owner-only permissions.

        The refresh token inside grants upload access to the channel until it
        is revoked, so it is written 0600 and the mode is set BEFORE the
        contents land — otherwise there is a window where it is world-readable.
        """
        path = self.settings.youtube_token_path
        path.parent.mkdir(parents=True, exist_ok=True)

        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(credentials.to_json())

        # Re-assert the mode: O_CREAT only applies it when the file is new, so
        # an existing group-readable cache would otherwise keep its old mode.
        os.chmod(path, 0o600)

    def authorize(self) -> bool:
        """Run the interactive consent flow. Returns True on success."""
        try:
            self.load_credentials(interactive=True)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("YouTube authorisation failed: {}", exc)
            return False

    # ---- upload -------------------------------------------------------------
    async def upload_short(
        self,
        video_path: Path | str,
        title: str,
        description: str = "",
        tags: Sequence[str] = (),
        privacy_status: str | None = None,
    ) -> UploadResult:
        """Upload one vertical video as a YouTube Short.

        Runs the blocking Google client on a worker thread so an async caller —
        a Telegram callback, the daemon — is never stalled by a 20MB upload.

        :param video_path: the finished .mp4. Must be vertical and under 60s
            for YouTube to shelve it as a Short; the Bard already guarantees the
            aspect and warns when a read runs long.
        :param privacy_status: overrides ``YOUTUBE_PRIVACY_STATUS`` for this
            upload. One of public / private / unlisted.
        :returns: an :class:`UploadResult`; never raises for an expected
            failure, so a caller can report it without a try block.
        """
        target = Path(video_path)
        privacy = (privacy_status or self.settings.youtube_privacy_status).strip().lower()
        if privacy not in {"public", "private", "unlisted"}:
            return UploadResult(
                ok=False, title=title, error=f"invalid privacy status {privacy!r}"
            )

        if not target.is_file():
            return UploadResult(ok=False, title=title, error=f"no video at {target}")

        if self.settings.dry_run:
            # A simulated id that could never be mistaken for a real one.
            fake = f"dry-{random.randint(10**9, 10**10 - 1)}"
            logger.info("DRY RUN — not uploading {} to YouTube", target.name)
            return UploadResult(
                ok=True, video_id=fake, url=f"https://www.youtube.com/shorts/{fake}",
                privacy_status=privacy, simulated=True, title=title,
            )

        usable, reason = self.status()
        if not usable:
            return UploadResult(ok=False, title=title, error=reason)

        try:
            return await asyncio.to_thread(
                self._upload_blocking, target, title, description, tags, privacy
            )
        except Exception as exc:  # noqa: BLE001 - reported, never raised at callers
            logger.error("YouTube upload failed: {}", exc)
            return UploadResult(ok=False, title=title, error=str(exc))

    def _upload_blocking(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: Sequence[str],
        privacy: str,
    ) -> UploadResult:
        """The synchronous upload. Called on a worker thread."""
        from googleapiclient.discovery import build  # noqa: PLC0415
        from googleapiclient.errors import HttpError  # noqa: PLC0415
        from googleapiclient.http import MediaFileUpload  # noqa: PLC0415

        credentials = self.load_credentials(interactive=False)
        youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)

        clean_tags = _clean_tags(tags)
        body = {
            "snippet": {
                "title": build_title(title),
                "description": build_description(description, clean_tags),
                "tags": clean_tags,
                "categoryId": CATEGORY_ENTERTAINMENT,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(video_path), chunksize=CHUNK_BYTES, resumable=True, mimetype="video/mp4"
        )
        request = youtube.videos().insert(
            part="snippet,status", body=body, media_body=media
        )

        size_mb = video_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Uploading {} ({:.1f} MB) as {!r} [{}]",
            video_path.name, size_mb, body["snippet"]["title"], privacy,
        )

        response = None
        attempt = 0
        while response is None:
            try:
                progress, response = request.next_chunk()
                if progress:
                    logger.debug("Upload {:.0f}%", progress.progress() * 100)
            except HttpError as exc:
                status = getattr(exc.resp, "status", None)
                if status in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                    attempt += 1
                    delay = min(2 ** attempt + random.random(), 30)
                    logger.warning(
                        "YouTube returned {}; retry {}/{} in {:.1f}s",
                        status, attempt, MAX_ATTEMPTS, delay,
                    )
                    time.sleep(delay)
                    continue
                raise YouTubeError(self._explain(exc)) from exc

        video_id = str(response.get("id") or "")
        if not video_id:
            raise YouTubeError("YouTube accepted the upload but returned no video id")

        result = UploadResult(
            ok=True,
            video_id=video_id,
            url=f"https://www.youtube.com/shorts/{video_id}",
            privacy_status=privacy,
            title=body["snippet"]["title"],
        )
        logger.success("Published to YouTube: {}", result.url)
        return result

    @staticmethod
    def _explain(exc: Any) -> str:
        """Turn a Google HttpError into something worth putting in Telegram."""
        status = getattr(getattr(exc, "resp", None), "status", None)
        detail = ""
        try:
            import json  # noqa: PLC0415

            payload = json.loads(exc.content.decode("utf-8"))
            detail = payload.get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001 - the raw error still gets reported
            detail = str(exc)

        if status == 403 and "quota" in detail.lower():
            # The default quota allows roughly six uploads a day. Worth saying
            # plainly, because it reads like a permissions error otherwise.
            return (
                "YouTube upload quota exceeded. The default API allowance is "
                "about 6 uploads per day and resets at midnight Pacific. "
                f"({detail})"
            )
        if status == 401:
            return (
                "YouTube rejected the credentials. Re-authorise with: "
                f"python main.py --youtube-auth ({detail})"
            )
        if status == 403:
            return (
                "YouTube refused the upload. Check the channel exists and the "
                f"authorised account can upload to it. ({detail})"
            )
        return f"HTTP {status}: {detail}" if status else detail or str(exc)
