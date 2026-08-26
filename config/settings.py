"""Typed configuration for the Autonomous Agent Village.

Every tunable lives here and is loaded from the environment (or a local ``.env``)
by pydantic-settings, so nothing in the village reads ``os.environ`` directly.

The one flag that matters most is :attr:`Settings.dry_run`. While it is true the
pipeline never touches Printify or Etsy: images are generated locally, uploads
are simulated, and product ids are fabricated. That makes the whole chain —
Scout, Crafter, Scribe, Guard, Telegram, database — runnable end to end with no
live store connected.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Project root: the directory that contains ``main.py``. Resolved from this
#: file rather than from the working directory, so it is correct no matter
#: where the process was launched.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

#: How far up the tree to look for a workspace-level ``.env``. Two levels
#: covers the common layouts — the project checked out on its own, and the
#: project nested inside a larger workspace that holds the shared secrets.
ENV_SEARCH_DEPTH = 2

#: Set this to point at one specific file and skip discovery entirely.
ENV_OVERRIDE_VAR = "AGENT_VILLAGE_ENV_FILE"


def discover_env_files() -> tuple[Path, ...]:
    """Every ``.env`` worth loading, in ASCENDING order of precedence.

    pydantic-settings applies a sequence of env files left to right, with later
    files overriding earlier ones, so the order here is deliberate:

    1. the workspace root and any intermediate ancestors (shared secrets),
    2. the current working directory, if it is somewhere else entirely,
    3. ``agent_village/.env`` — the most specific file, so it always wins,
    4. whatever ``AGENT_VILLAGE_ENV_FILE`` names, which beats all of them.

    Real environment variables still take priority over every file, which is
    what makes container and CI overrides work.

    :returns: existing paths only, de-duplicated, lowest priority first.
    """
    candidates: list[Path] = []

    # Ancestors, furthest first: /workspace/.env before /workspace/app/.env.
    for depth in range(ENV_SEARCH_DEPTH, 0, -1):
        try:
            candidates.append(BASE_DIR.parents[depth - 1] / ".env")
        except IndexError:
            # Fewer ancestors than the search depth; nothing to add.
            continue

    # The directory the process was launched from, when it is neither of those.
    try:
        candidates.append(Path.cwd() / ".env")
    except OSError:
        # A deleted working directory raises here; discovery must not fail.
        pass

    # The project's own file: most specific, so highest normal precedence.
    candidates.append(BASE_DIR / ".env")

    # An explicit override beats everything, and is reported as missing if the
    # operator names a file that is not there.
    override = os.environ.get(ENV_OVERRIDE_VAR, "").strip()
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file():
            candidates.append(override_path)
        else:
            sys.stderr.write(
                f"{ENV_OVERRIDE_VAR} points at {override_path}, which does not exist. "
                "Falling back to the discovered .env files.\n"
            )

    found: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        found.append(resolved)

    return tuple(found)


def load_env_files(files: tuple[Path, ...] | None = None) -> dict[str, str]:
    """Merge the discovered files into the process environment.

    Two rules, and the second is the one that matters:

    1. A real environment variable always wins. Nothing here overwrites a value
       the shell, the container or CI already set.
    2. **An empty value never shadows a non-empty one.** ``.env.example`` ships
       keys as ``OPENROUTER_API_KEY=`` so people know they exist, so copying it
       into the project would otherwise blank out a working key inherited from
       the workspace root — and the failure is silent, because the agents just
       quietly fall back to canned content. A blank is treated as "unspecified
       here", not as "deliberately empty".

    :returns: the merged values that were applied, for logging.
    """
    from dotenv import dotenv_values

    merged: dict[str, str] = {}
    for path in files if files is not None else ENV_FILES:
        try:
            values = dotenv_values(path, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - an unreadable file is skipped
            sys.stderr.write(f"Could not read {path}: {exc}\n")
            continue

        for key, value in values.items():
            if value is None:
                continue
            if not value.strip() and merged.get(key, "").strip():
                # Rule 2: a later blank does not erase an earlier real value.
                continue
            merged[key] = value

    applied: dict[str, str] = {}
    for key, value in merged.items():
        # Rule 1: setdefault, so the real environment keeps priority.
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    return applied


#: Resolved once at import. Exposed so the CLI banner can show what was loaded.
ENV_FILES: tuple[Path, ...] = discover_env_files()

#: Applied at import, before Settings is constructed, so the merge rules above
#: govern what the model sees rather than pydantic's raw last-file-wins.
ENV_APPLIED: dict[str, str] = load_env_files()


class Settings(BaseSettings):
    """Runtime configuration, validated once at import time."""

    # No ``env_file`` here on purpose: the files were already merged into the
    # environment by load_env_files(), which applies the "a blank never shadows
    # a real value" rule that pydantic's own last-file-wins cannot express.
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # ---- mode ---------------------------------------------------------------
    dry_run: bool = Field(
        default=True,
        description="Simulate every Printify/Etsy call instead of performing it.",
    )
    log_level: str = Field(default="INFO")

    # ---- OpenRouter ---------------------------------------------------------
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_referer: str = Field(
        default="https://github.com/agent-village",
        description="Sent as HTTP-Referer; OpenRouter uses it for app attribution.",
    )
    openrouter_title: str = Field(default="Autonomous Agent Village")

    #: Text model used by the Scout and the Scribe.
    text_model: str = Field(default="google/gemini-2.5-flash")
    #: Image-capable model, used when the OpenRouter rung is reached. Must be a
    #: model whose output_modalities include "image" — only a handful are, and
    #: OpenRouter answers a request for any other id with a 404 rather than a
    #: helpful error. As of writing that is the Gemini *-image family and the
    #: OpenAI gpt-*-image family; there are no Stability/SDXL models on
    #: OpenRouter at all, so point HF_API_TOKEN at Hugging Face for those.
    #:
    #: The "-preview" suffixed id this used to default to no longer exists.
    image_model: str = Field(default="google/gemini-2.5-flash-image")

    #: Which image backends to try, in order, and which to skip entirely.
    #:
    #: Pollinations leads by default because it needs no key, no account and no
    #: billing, and cannot 404 on a renamed model. It is slower (~45s a scene)
    #: and caps at 576x1024, so put "openrouter" first for better art once a
    #: key is set and the model id is known good.
    image_provider_order: str = Field(default="pollinations,openrouter,huggingface")

    request_timeout_seconds: float = Field(default=120.0, ge=5.0)
    max_retries: int = Field(default=3, ge=0, le=10)

    # ---- Telegram -----------------------------------------------------------
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")

    # ---- Printify -----------------------------------------------------------
    printify_api_key: str = Field(default="")
    printify_shop_id: str = Field(default="")
    printify_base_url: str = Field(default="https://api.printify.com/v1")
    #: 384 = Unisex Heavy Cotton Tee in Printify's catalogue. Override per niche.
    printify_blueprint_id: int = Field(default=384)
    printify_print_provider_id: int = Field(default=1)
    #: Retail price in minor units (cents). Never a float — money never is.
    listing_price_cents: int = Field(default=2499, ge=100)

    # ---- storage and database ----------------------------------------------
    storage_dir: Path = Field(default=BASE_DIR / "storage")
    database_url: str = Field(default=f"sqlite:///{BASE_DIR / 'agent_village.db'}")

    # ---- image pipeline -----------------------------------------------------
    target_dpi: int = Field(default=300, ge=72)
    #: Finished print area in inches; 300 DPI over 12x16in is a standard DTG area.
    print_width_inches: float = Field(default=12.0, gt=0)
    print_height_inches: float = Field(default=16.0, gt=0)
    remove_background: bool = Field(default=True)

    # ---- The Bard: faceless YouTube Shorts ---------------------------------
    #: edge = edge-tts (free, no key). openai / elevenlabs need their own keys.
    #: none forces the synthesised placeholder track.
    tts_provider: str = Field(default="edge")
    #: Edge voice. Christopher is the warm, measured storytelling read that
    #: suits dark history; Guy is drier and more newsreel; Jenny is the female
    #: equivalent of Christopher.
    #:
    #: Accepts BARD_VOICE as well as TTS_VOICE, because to anyone editing .env
    #: the voice is a property of the Bard, not of a subsystem called "tts".
    tts_voice: str = Field(
        default="en-US-ChristopherNeural",
        validation_alias=AliasChoices("BARD_VOICE", "TTS_VOICE", "tts_voice"),
    )

    #: A few voices worth knowing, for the CLI banner. Not a whitelist — any
    #: edge-tts voice name is accepted.
    SUGGESTED_VOICES: ClassVar[tuple[str, ...]] = (
        "en-US-ChristopherNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-GB-RyanNeural",
    )

    #: Slightly slower than default suits dark-history narration.
    tts_rate: str = Field(default="-8%")
    tts_pitch: str = Field(default="-2Hz")

    #: OpenAI is used ONLY for text-to-speech; the LLM calls all go through
    #: OpenRouter, which has no TTS endpoint.
    openai_api_key: str = Field(default="")
    openai_tts_model: str = Field(default="gpt-4o-mini-tts")
    openai_tts_voice: str = Field(default="onyx")

    elevenlabs_api_key: str = Field(default="")
    elevenlabs_voice_id: str = Field(default="")
    elevenlabs_model: str = Field(default="eleven_multilingual_v2")

    #: Hugging Face inference token, for SDXL scene art. Optional: without it
    #: the free Pollinations endpoint is used instead, which needs no account
    #: but caps the free tier at 576x1024.
    hf_api_token: str = Field(default="")

    #: Use the free Pollinations endpoint when no keyed provider is available.
    #: On by default: the alternative is the drawn concentric-ring placeholder,
    #: which is not a background anybody wants behind a video. Turn it off to
    #: keep scene prompts entirely on machines you control.
    pollinations_enabled: bool = Field(default=True)

    # ---- The Morning Ledger -------------------------------------------------
    #: RSS feeds the Night Scribe reads, comma separated.
    #:
    #: rekt.news is listed because it is the best exploit source there is, but
    #: every path on that host answers 500 as of this writing — a dead feed is
    #: skipped with a warning rather than failing the scan, and it will start
    #: working again the moment they fix it.
    crypto_feeds: str = Field(
        default=(
            "https://rekt.news/feed.xml,"
            "https://decrypt.co/feed,"
            "https://www.coindesk.com/arc/outboundfeeds/rss/,"
            "https://cointelegraph.com/rss,"
            "https://thedefiant.io/api/feed"
        )
    )

    #: Model that writes the edition. Routed through OpenRouter, which already
    #: has a key here and serves the Anthropic range; a direct ANTHROPIC_API_KEY
    #: is used instead when one is set.
    #:
    #: NOT claude-3-7-sonnet-20250219: that id is retired and 404s.
    newsletter_model: str = Field(default="anthropic/claude-sonnet-4.5")

    #: Optional direct Anthropic key. Unset means "go through OpenRouter".
    anthropic_api_key: str = Field(default="")

    #: Night scan window, local time, and how often to scan inside it.
    night_scan_start_hour: int = Field(default=0, ge=0, le=23)
    night_scan_end_hour: int = Field(default=6, ge=0, le=23)
    night_scan_interval_hours: float = Field(default=2.0, ge=0.25, le=12.0)

    #: When the edition is built, and when an approved one is sent.
    ledger_build_time: str = Field(default="06:30")
    ledger_publish_time: str = Field(default="08:00")

    #: Overseer thresholds. Sponsorship needs the subscriber count AND the open
    #: rate to hold for the full streak; the paid tier needs scale on top.
    overseer_sponsor_subscribers: int = Field(default=500, ge=1)
    overseer_sponsor_open_rate: float = Field(default=0.40, ge=0.0, le=1.0)
    overseer_streak_days: int = Field(default=14, ge=1, le=365)
    overseer_paid_subscribers: int = Field(default=1000, ge=1)

    #: Public base URL of this dashboard, if it is reachable from the internet
    #: (a tunnel, or a deployed host). e.g. "https://village.example.com".
    #:
    #: Set it and the webhook's ``image_url`` points at the FINISHED Pin card —
    #: hook, price and disclosure burned in. Leave it unset and image_url falls
    #: back to the raw background's own public URL, which is a real image but
    #: not the designed card.
    public_base_url: str = Field(default="")

    #: Make.com webhook that posts the Pin on this project's behalf.
    #:
    #: The practical route while the Pinterest developer app is stuck on Trial
    #: access: Make holds its own Pinterest connection, so the v5 app-approval
    #: problem does not apply. When this is set it is used INSTEAD of the direct
    #: API, because posting the same Pin down both paths would duplicate it.
    make_pinterest_webhook_url: str = Field(default="")

    #: Pinterest API v5 access token (starts "pina_"). Minted in the Pinterest
    #: developer console; the app behind it must have Standard access, not
    #: Trial — see PinterestPublisher._explain for why that matters.
    pinterest_access_token: str = Field(default="")

    #: Board to pin to, BY NAME. Resolved to a board id on first use, because a
    #: name is what someone can actually read off the Pinterest UI.
    pinterest_board_name: str = Field(default="")

    #: Post a Pin when a deal is approved. Off by default: publishing to a real
    #: account on a button press is not something to enable by accident.
    pinterest_enabled: bool = Field(default=False)

    #: OAuth client for the YouTube Data API, downloaded from Google Cloud as
    #: a "Desktop app" credential. A relative path resolves against the project
    #: root, so the default works wherever the process was launched from.
    youtube_client_secret_file: str = Field(default="client_secret.json")

    #: Where the minted refresh token is cached. Kept separate from the client
    #: secret because this one grants actual upload access to the channel.
    youtube_token_file: str = Field(default="storage/youtube_token.json")

    #: Visibility of an uploaded Short. "private" is the deliberate default:
    #: a first upload can be checked on the channel before anyone sees it.
    youtube_privacy_status: str = Field(default="private")

    #: Upload on approval instead of only marking the Short ready. Off by
    #: default — publishing to a real channel on a button press is not
    #: something to switch on without the operator saying so.
    youtube_upload_enabled: bool = Field(default=False)

    #: Pexels supplies free stock video under a licence permitting commercial
    #: use without attribution. Unset means the Bard illustrates with generated
    #: stills only — scene keywords are never sent to a third party unless the
    #: operator opts in by setting this.
    pexels_api_key: str = Field(default="")

    #: Cap on how many scenes may use stock footage. B-roll is a change of
    #: texture, not the whole video; an all-stock Short looks like every other
    #: all-stock Short. Remaining scenes stay generated stills.
    stock_video_max_clips: int = Field(default=2, ge=0, le=5)

    #: Vertical 9:16, the only aspect Shorts/Reels/TikTok accept full-bleed.
    video_width: int = Field(default=1080, ge=360)
    video_height: int = Field(default=1920, ge=640)
    video_fps: int = Field(default=30, ge=12, le=60)
    #: Ken Burns travel over a scene, as a fraction of the frame.
    video_zoom: float = Field(default=0.16, ge=0.0, le=0.6)

    #: Cross-dissolve between scenes, in seconds. 0 cuts hard instead.
    #: The dissolve begins exactly on the scene boundary, so the total runtime
    #: is unchanged and the captions stay where the audio put them.
    video_transition_seconds: float = Field(default=0.5, ge=0.0, le=2.0)

    #: Amazon Associates tracking id, appended to every affiliate link as
    #: ``?tag=``. Without a real one the links still work but earn nothing, so
    #: the default is a placeholder that is obvious in a log.
    amazon_tracking_id: str = Field(default="village-20")

    #: What the Scout researches when no niche is given.
    amazon_default_niche: str = Field(default="workspace gadgets")

    #: Amazon storefront to link into. The tracking id is region-specific — a
    #: US tag earns nothing on amazon.co.uk — so these two move together.
    amazon_marketplace: str = Field(default="www.amazon.com")

    #: Categories the Scout rotates through when picking a niche itself.
    amazon_categories: str = Field(default="tech,gadgets,workspace,lifestyle")

    #: Times of day the daemon produces a Short, comma separated, 24h LOCAL
    #: time: "12:00,16:00,20:00". Empty means the daemon makes no Shorts on a
    #: schedule, which is the default — producing and dispatching video costs
    #: model quota, so it is opt-in.
    shorts_schedule_hours: str = Field(default="")

    #: Times of day the daemon curates an affiliate deal, same format as
    #: SHORTS_SCHEDULE_HOURS. Empty means the Scout runs on an interval instead,
    #: or only on demand if that is unset too.
    #:
    #: SCOUT_SCHEDULE_TIMES is accepted as well, because "times" is what these
    #: are and it is the name people reach for.
    scout_schedule_hours: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SCOUT_SCHEDULE_TIMES", "SCOUT_SCHEDULE_HOURS", "scout_schedule_hours"
        ),
    )

    #: Curate a deal every N hours instead of at fixed times. Fractional values
    #: are allowed (0.5 = every 30 minutes). Ignored when fixed times are set:
    #: a schedule cannot be both.
    scout_interval_hours: float = Field(default=0.0, ge=0.0, le=168.0)

    #: The same option for Shorts, for symmetry — the daemon runs both through
    #: one loop and there is no reason one can do this and the other cannot.
    shorts_interval_hours: float = Field(default=0.0, ge=0.0, le=168.0)

    #: How late a missed slot may still run, in minutes. Covers a restart or a
    #: slow pass; beyond it the slot is skipped rather than posted at the wrong
    #: time of day.
    shorts_schedule_grace_minutes: int = Field(default=30, ge=0, le=720)

    #: How many distinct visual beats a Short is cut into. Four is the floor
    #: for the picture to feel like it is moving through a story; above six,
    #: each scene is on screen too briefly for a Ken Burns move to read as
    #: anything but a jerk.
    shorts_scene_count: int = Field(default=5, ge=4, le=6)
    shorts_target_seconds: int = Field(default=40, ge=15, le=59)
    subtitle_words_per_cue: int = Field(default=3, ge=1, le=8)

    #: Revenue per 1,000 Shorts views, in cents. Shorts RPM is genuinely low —
    #: commonly 5-15 cents — and varies by month, geography and niche. This is
    #: the default applied to a Short when no per-video RPM has been recorded.
    youtube_rpm_cents: int = Field(default=10, ge=0, le=10_000)

    #: Count simulated (dry-run) sales in the treasury.
    #:
    #: Off by default, and it should stay off: a dry-run publish writes a
    #: ledger line exactly like a live one, so counting it would show income
    #: nobody earned. Turn it on only to eyeball the HUD with test data.
    count_dry_run_revenue: bool = Field(default=False)

    # ---- scheduling ---------------------------------------------------------
    daemon_interval_seconds: int = Field(default=3600, ge=60)
    daemon_batch_size: int = Field(default=1, ge=1, le=25)

    # ---- validation ---------------------------------------------------------
    @field_validator("storage_dir", mode="after")
    @classmethod
    def _ensure_storage(cls, value: Path) -> Path:
        """Create the asset directory eagerly so no agent has to guard for it."""
        value.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("youtube_privacy_status", mode="after")
    @classmethod
    def _normalise_privacy(cls, value: str) -> str:
        """YouTube rejects anything outside these three with a 400."""
        status = value.strip().lower()
        allowed = {"public", "private", "unlisted"}
        if status not in allowed:
            raise ValueError(
                f"youtube_privacy_status must be one of {sorted(allowed)}, got {value!r}"
            )
        return status

    @field_validator("tts_provider", mode="after")
    @classmethod
    def _normalise_tts(cls, value: str) -> str:
        provider = value.strip().lower()
        allowed = {"edge", "openai", "elevenlabs", "none"}
        if provider not in allowed:
            raise ValueError(f"tts_provider must be one of {sorted(allowed)}, got {value!r}")
        return provider

    @field_validator("log_level", mode="after")
    @classmethod
    def _normalise_level(cls, value: str) -> str:
        level = value.strip().upper()
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return level

    @model_validator(mode="after")
    def _check_live_credentials(self) -> "Settings":
        """A live run must actually be able to reach the services it will call.

        Dry runs deliberately skip this: the point of the flag is that the whole
        pipeline works with an empty ``.env``.
        """
        if self.dry_run:
            return self

        # Only what NOTHING can work without. Printify used to be here too,
        # which meant a live YouTube upload was impossible unless the operator
        # also configured a print-on-demand shop they may not use — two
        # independent channels, one shared gate. The Merchant already degrades
        # to a simulated publish when Printify is unset, so its absence is a
        # warning at the point of use, not a reason to refuse to start.
        missing = [
            name
            for name, value in (("OPENROUTER_API_KEY", self.openrouter_api_key),)
            if not value.strip()
        ]
        if missing:
            searched = (
                ", ".join(str(path) for path in ENV_FILES)
                if ENV_FILES
                else f"none found (looked in {BASE_DIR} and its parents)"
            )
            raise ValueError(
                "DRY_RUN is false but these are unset: "
                + ", ".join(missing)
                + f". Loaded .env files: {searched}."
                + " Set them there, or set DRY_RUN=true to simulate."
            )
        return self

    # ---- derived helpers ----------------------------------------------------
    @property
    def shorts_dir(self) -> Path:
        """Where Shorts assets land: scenes, narration, and the finished mp4."""
        directory = self.storage_dir / "shorts"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def video_size(self) -> tuple[int, int]:
        return (self.video_width, self.video_height)

    @property
    def tts_configured(self) -> bool:
        """Can the configured provider actually speak?"""
        return self.tts_status()[0]

    def tts_status(self) -> tuple[bool, str]:
        """Whether the configured voice can speak, and why not when it cannot.

        edge-tts needs no API key, which used to make this return a flat True —
        but it does need the ``edge_tts`` package, and without it the Bard cuts
        a silent video and only says so afterwards. Checking importability here
        lets the CLI banner tell the truth before a render starts.

        :returns: ``(usable, reason)``; reason is empty when usable.
        """
        provider = self.tts_provider

        if provider == "edge":
            import importlib.util  # noqa: PLC0415

            if importlib.util.find_spec("edge_tts") is None:
                return False, (
                    "the edge_tts package is not installed in this interpreter "
                    "(pip install 'edge-tts>=6.1.0')"
                )
            return True, ""

        if provider == "openai":
            if not self.openai_api_key.strip():
                return False, "OPENAI_API_KEY is not set"
            return True, ""

        if provider == "elevenlabs":
            if not self.elevenlabs_api_key.strip():
                return False, "ELEVENLABS_API_KEY is not set"
            if not self.elevenlabs_voice_id.strip():
                return False, "ELEVENLABS_VOICE_ID is not set"
            return True, ""

        return False, f"unknown TTS provider {provider!r}"

    def _resolve(self, value: str) -> Path:
        """Interpret a configured path relative to the project root."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (BASE_DIR / path)

    @property
    def youtube_client_secret_path(self) -> Path:
        """Absolute path to the OAuth client secret."""
        return self._resolve(self.youtube_client_secret_file)

    @property
    def youtube_token_path(self) -> Path:
        """Absolute path to the cached OAuth token."""
        return self._resolve(self.youtube_token_file)

    @property
    def youtube_configured(self) -> bool:
        """Whether a client secret is present to authorise against."""
        return self.youtube_client_secret_path.is_file()

    @property
    def youtube_authorized(self) -> bool:
        """Whether consent has already been granted and cached."""
        return self.youtube_token_path.is_file()

    def public_url(self, path: str) -> str:
        """An absolute URL for ``path``, or "" when nothing public is configured."""
        base = self.public_base_url.strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            return ""
        return f"{base}/{path.lstrip('/')}"

    @property
    def crypto_feed_list(self) -> tuple[str, ...]:
        """The configured feeds, cleaned."""
        return tuple(
            part.strip() for part in self.crypto_feeds.split(",")
            if part.strip().startswith("http")
        )

    @property
    def newsletter_configured(self) -> bool:
        """Whether an edition can actually be written."""
        return bool(self.anthropic_api_key.strip() or self.openrouter_api_key.strip())

    @property
    def make_webhook_configured(self) -> bool:
        """Whether a Make.com webhook is set up to post Pins."""
        url = self.make_pinterest_webhook_url.strip()
        return url.startswith("https://") or url.startswith("http://")

    @property
    def pinterest_configured(self) -> bool:
        """Whether a token and a board name are both present."""
        return bool(
            self.pinterest_access_token.strip() and self.pinterest_board_name.strip()
        )

    @property
    def amazon_category_list(self) -> tuple[str, ...]:
        """The configured categories, cleaned."""
        return tuple(
            part.strip() for part in self.amazon_categories.split(",") if part.strip()
        )

    @property
    def amazon_configured(self) -> bool:
        """Whether a real tracking id is set.

        The placeholder default still produces working links; they simply
        credit nobody, which is worth saying out loud rather than discovering
        after a month of posting.
        """
        tag = self.amazon_tracking_id.strip()
        return bool(tag) and tag != "village-20"

    @property
    def shorts_schedule_state_path(self) -> Path:
        """Where the daemon records which slots have already produced a Short."""
        return self.storage_dir / "shorts_schedule.json"

    @property
    def scout_schedule_state_path(self) -> Path:
        """Where the daemon records which slots have already curated a deal."""
        return self.storage_dir / "scout_schedule.json"

    @property
    def scout_scheduled(self) -> bool:
        """Whether the Scout runs on a schedule at all."""
        from village.scheduler import parse_schedule  # noqa: PLC0415 - avoids a cycle

        return bool(parse_schedule(self.scout_schedule_hours)) or self.scout_interval_hours > 0

    @property
    def shorts_scheduled(self) -> bool:
        """Whether Shorts run on a schedule at all."""
        from village.scheduler import parse_schedule  # noqa: PLC0415 - avoids a cycle

        return bool(parse_schedule(self.shorts_schedule_hours)) or self.shorts_interval_hours > 0

    @property
    def image_providers(self) -> tuple[str, ...]:
        """The configured backend order, filtered to ones this build knows."""
        known = {"pollinations", "openrouter", "huggingface"}
        chosen: list[str] = []
        for raw in self.image_provider_order.split(","):
            name = raw.strip().lower()
            if name in known and name not in chosen:
                chosen.append(name)
        return tuple(chosen)

    @property
    def stock_video_configured(self) -> bool:
        """Whether B-roll can be fetched at all."""
        return bool(self.pexels_api_key.strip()) and self.stock_video_max_clips > 0

    @property
    def print_pixel_size(self) -> tuple[int, int]:
        """The finished artwork size in pixels at the configured DPI."""
        return (
            int(round(self.print_width_inches * self.target_dpi)),
            int(round(self.print_height_inches * self.target_dpi)),
        )

    @property
    def openrouter_headers(self) -> dict[str, str]:
        """Attribution headers OpenRouter uses for app ranking."""
        return {
            "HTTP-Referer": self.openrouter_referer,
            "X-Title": self.openrouter_title,
        }

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token.strip() and self.telegram_chat_id.strip())

    @property
    def env_files(self) -> tuple[Path, ...]:
        """The .env files this configuration was built from, lowest priority first."""
        return ENV_FILES

    @staticmethod
    def env_files_searched_hint() -> str:
        """The directories discovery looked in, for a "nothing found" message."""
        places = [str(BASE_DIR)]
        for depth in range(1, ENV_SEARCH_DEPTH + 1):
            try:
                places.append(str(BASE_DIR.parents[depth - 1]))
            except IndexError:
                break
        try:
            places.append(str(Path.cwd()))
        except OSError:
            pass
        return ", ".join(dict.fromkeys(places))

    def describe_env_sources(self) -> str:
        """One line naming where configuration came from, for the CLI banner."""
        if not ENV_FILES:
            return "no .env found (using real environment variables only)"

        def label(path: Path) -> str:
            try:
                return str(path.relative_to(BASE_DIR.parent))
            except ValueError:
                return str(path)

        # Highest priority last, so read it as "these, then these win".
        return " < ".join(label(path) for path in ENV_FILES)

    @property
    def openrouter_configured(self) -> bool:
        return bool(self.openrouter_api_key.strip())

    @property
    def printify_configured(self) -> bool:
        return bool(self.printify_api_key.strip() and self.printify_shop_id.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once.

    Exits with a readable message rather than a traceback when the environment
    is misconfigured — this is the first thing every entry point calls.
    """
    try:
        return Settings()
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        sys.stderr.write(f"Configuration error:\n  {exc}\n")
        raise SystemExit(2) from exc


#: Import-friendly singleton. Prefer ``get_settings()`` inside functions so tests
#: can clear the cache.
settings = get_settings()
