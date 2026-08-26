"""Optional stock B-roll from the Pexels Video API.

Generated stills carry a Short perfectly well, but every scene looking like a
still is what makes a faceless channel feel like a slideshow. A little real
footage — a storm, a corridor, fog moving through trees — changes the texture
without changing the story.

This is opt-in and degrades to nothing. With no ``PEXELS_API_KEY`` set, nothing
here runs and no scene text leaves the machine; every failure path returns
``None`` so the caller falls back to the still it already has.

Pexels footage is free for commercial use and needs no attribution, but the
licence does forbid redistributing the clips themselves as stock — using them
inside a finished video is exactly what it is for.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from config.settings import Settings, get_settings

#: The search endpoint. Videos only; the photo API is a different host path.
SEARCH_URL = "https://api.pexels.com/videos/search"

#: Stills, for Pinterest cards. Same key, same account, different path.
PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"

#: Refuse anything wider than it is tall. A landscape clip cropped to 9:16
#: loses most of its subject, which is worse than the still it replaced.
MIN_ASPECT = 1.2

#: Never download more than this. A 4K stock clip can run to hundreds of MB,
#: and it is going to be scaled to 1080x1920 regardless.
MAX_BYTES = 60 * 1024 * 1024

#: Words that make a stock search return nothing. Proper nouns and dates are
#: the story, not the setting, and no library has footage of a specific event.
_NOISE = re.compile(r"\b(?:\d{3,4}s?|the|a|an|of|in|on|at|and)\b", re.IGNORECASE)


@dataclass
class StockClip:
    """One downloaded piece of B-roll."""

    path: Path
    width: int
    height: int
    duration: float
    source_url: str
    photographer: str

    @property
    def credit(self) -> str:
        """Attribution line. Not required by the licence, but good manners."""
        return f"{self.photographer} (Pexels)"


def clean_query(raw: str) -> str:
    """Reduce a scene's query to terms a stock library can actually match."""
    text = _NOISE.sub(" ", raw or "")
    words = [w for w in re.findall(r"[a-zA-Z]+", text) if len(w) > 2]
    return " ".join(words[:4]).strip()


class StockVideoLibrary:
    """Finds and downloads vertical B-roll, or quietly declines to."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def configured(self) -> bool:
        return self.settings.stock_video_configured

    async def fetch(self, query: str, destination: Path) -> StockClip | None:
        """Search for ``query`` and download the best vertical match.

        :param query: scene keywords; cleaned before use.
        :param destination: where to write the .mp4.
        :returns: the clip, or ``None`` if unconfigured, unmatched or failed.
        """
        if not self.configured:
            return None

        terms = clean_query(query)
        if not terms:
            logger.debug("Stock query {!r} had nothing searchable in it", query)
            return None

        try:
            payload = await self._search(terms)
        except Exception as exc:  # noqa: BLE001 - B-roll is never worth failing over
            logger.warning("Pexels search for {!r} failed: {}", terms, exc)
            return None

        candidate = self._best(payload)
        if not candidate:
            logger.info("No vertical Pexels footage for {!r}", terms)
            return None

        video_file, meta = candidate
        try:
            await self._download(video_file["link"], destination)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download Pexels clip for {!r}: {}", terms, exc)
            destination.unlink(missing_ok=True)
            return None

        clip = StockClip(
            path=destination,
            width=int(video_file.get("width") or 0),
            height=int(video_file.get("height") or 0),
            duration=float(meta.get("duration") or 0.0),
            source_url=str(meta.get("url") or ""),
            photographer=str(meta.get("user", {}).get("name") or "unknown"),
        )
        logger.success(
            "Stock B-roll for {!r}: {}x{}, {:.0f}s, by {}",
            terms, clip.width, clip.height, clip.duration, clip.photographer,
        )
        return clip

    async def fetch_many(
        self, queries: Sequence[str], directory: Path
    ) -> dict[int, StockClip]:
        """Fetch clips for several scenes at once.

        Capped by ``stock_video_max_clips``: B-roll is a change of texture, and
        a Short cut entirely from stock looks like every other stock Short.
        Queries are tried in order and the cap counts successes, so a scene that
        finds nothing does not use up the budget.

        :returns: ``{scene_index: clip}`` for the scenes that got one.
        """
        if not self.configured:
            return {}

        directory.mkdir(parents=True, exist_ok=True)
        budget = self.settings.stock_video_max_clips
        found: dict[int, StockClip] = {}

        for index, query in enumerate(queries):
            if len(found) >= budget:
                break
            if not query:
                continue
            clip = await self.fetch(query, directory / f"broll{index:02d}.mp4")
            if clip:
                found[index] = clip

        return found

    # ---- internals ----------------------------------------------------------
    async def _search(self, terms: str) -> dict[str, Any]:
        """One search request. The key travels in a header, never in the URL."""
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds
        ) as client:
            response = await client.get(
                SEARCH_URL,
                params={
                    "query": terms,
                    "orientation": "portrait",
                    "size": "medium",
                    "per_page": 10,
                },
                headers={"Authorization": self.settings.pexels_api_key.strip()},
            )
            response.raise_for_status()
            return response.json()

    def _best(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Pick the most usable vertical file out of a search response.

        Prefers the smallest rendition that still covers the output frame:
        anything larger is bandwidth spent on pixels that get scaled away, and
        Pexels lists a 4K version of almost everything.
        """
        target_w, target_h = self.settings.video_size
        candidates: list[tuple[tuple[int, int], dict[str, Any], dict[str, Any]]] = []

        for video in payload.get("videos") or []:
            if not isinstance(video, dict):
                continue
            for file in video.get("video_files") or []:
                if not isinstance(file, dict) or file.get("file_type") != "video/mp4":
                    continue
                if not file.get("link"):
                    continue
                width = int(file.get("width") or 0)
                height = int(file.get("height") or 0)
                if not width or not height:
                    continue
                if height / width < MIN_ASPECT:
                    continue  # landscape or square: cropping would gut it

                area = width * height
                if width >= target_w and height >= target_h:
                    # Covers the frame: smallest such file wins.
                    key = (0, area)
                else:
                    # Undersized: the largest of the undersized is the least bad.
                    key = (1, -area)
                candidates.append((key, file, video))

        if not candidates:
            return None

        candidates.sort(key=lambda entry: entry[0])
        _, file, video = candidates[0]
        return file, video

    async def _download(self, url: str, destination: Path) -> None:
        """Stream a clip to disk, refusing anything oversized."""
        import httpx  # noqa: PLC0415

        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0

        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds, follow_redirects=True
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes(64 * 1024):
                        written += len(chunk)
                        if written > MAX_BYTES:
                            raise ValueError(
                                f"clip exceeded {MAX_BYTES // (1024 * 1024)}MB"
                            )
                        handle.write(chunk)

        if written < 1024:
            raise ValueError("clip was empty")


if __name__ == "__main__":  # pragma: no cover - manual probe
    import sys

    async def _main() -> None:
        library = StockVideoLibrary()
        if not library.configured:
            print("PEXELS_API_KEY is not set; stock video is off.")
            return
        query = " ".join(sys.argv[1:]) or "stormy sea"
        clip = await library.fetch(query, Path("/tmp/broll-probe.mp4"))
        print(clip)

    asyncio.run(_main())


class StockPhotoLibrary:
    """Real licensed photography for Pin cards.

    Preferred over a generated image wherever a key exists: an actual
    photograph of a desk is a photograph of a desk, whereas a generated one is
    a plausible invention. Neither is the specific product being linked to —
    Amazon's product images need PA-API access and may not be scraped — so both
    are used as lifestyle context, never as a claim about the item.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        #: The URL the most recent fetch came from, for callers needing a link.
        self.last_source_url: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.settings.pexels_api_key.strip())

    async def fetch(self, query: str, destination: Path) -> Path | None:
        """Download the best vertical photo for ``query``, or None."""
        if not self.configured:
            return None

        terms = clean_query(query)
        if not terms:
            return None

        import httpx  # noqa: PLC0415

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout_seconds
            ) as client:
                response = await client.get(
                    PHOTO_SEARCH_URL,
                    params={
                        "query": terms,
                        "orientation": "portrait",
                        "per_page": 15,
                    },
                    headers={"Authorization": self.settings.pexels_api_key.strip()},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001 - never fail a card over B-roll
            logger.warning("Pexels photo search for {!r} failed: {}", terms, exc)
            return None

        best = self._best(payload)
        if not best:
            logger.info("No vertical Pexels photo for {!r}", terms)
            return None

        url, photographer = best
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout_seconds, follow_redirects=True
            ) as client:
                image = await client.get(url)
                image.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(image.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download the Pexels photo: {}", exc)
            destination.unlink(missing_ok=True)
            return None

        logger.success("Stock photo for {!r} by {}", terms, photographer)
        # Pexels serves this URL publicly and indefinitely, so it doubles as an
        # image_url for anything downstream that wants a link, not bytes.
        self.last_source_url = url
        return destination

    def _best(self, payload: dict[str, Any]) -> tuple[str, str] | None:
        """The largest portrait rendition in the response."""
        candidates: list[tuple[int, str, str]] = []
        for photo in payload.get("photos") or []:
            if not isinstance(photo, dict):
                continue
            width = int(photo.get("width") or 0)
            height = int(photo.get("height") or 0)
            if not width or not height or height / width < MIN_ASPECT:
                continue
            sources = photo.get("src") or {}
            url = sources.get("portrait") or sources.get("large") or sources.get("original")
            if not url:
                continue
            candidates.append(
                (width * height, str(url), str(photo.get("photographer") or "unknown"))
            )

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][2]
