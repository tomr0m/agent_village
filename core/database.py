"""Persistence for the village: the SQLAlchemy engine, session factory, and the
:class:`Listing` record every agent reads and writes.

A listing moves through a small, explicit state machine::

    DRAFTED -> PENDING_APPROVAL -> APPROVED  -> PUBLISHED
                               \\-> REJECTED
                    (any state) -> FAILED

The Telegram callback is the only thing that moves a row out of
``PENDING_APPROVAL``, which is what makes the human the gate rather than a
formality.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterator, Sequence

from loguru import logger
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config.settings import get_settings


class ListingStatus(str, Enum):
    """Every state a listing can occupy. Stored as its string value."""

    DRAFTED = "DRAFTED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


#: Transitions the application is allowed to make. Anything else is a bug, and
#: :func:`update_status` refuses it rather than silently corrupting history.
_ALLOWED_TRANSITIONS: dict[ListingStatus, set[ListingStatus]] = {
    ListingStatus.DRAFTED: {
        ListingStatus.PENDING_APPROVAL,
        ListingStatus.FAILED,
    },
    ListingStatus.PENDING_APPROVAL: {
        ListingStatus.APPROVED,
        ListingStatus.REJECTED,
        ListingStatus.FAILED,
    },
    ListingStatus.APPROVED: {ListingStatus.PUBLISHED, ListingStatus.FAILED},
    ListingStatus.REJECTED: set(),
    ListingStatus.PUBLISHED: set(),
    ListingStatus.FAILED: {ListingStatus.PENDING_APPROVAL},
}


class ShortStatus(str, Enum):
    """Lifecycle of one faceless YouTube Short."""

    DRAFTED = "DRAFTED"
    RENDERING = "RENDERING"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class DealStatus(str, Enum):
    """Lifecycle of one curated affiliate recommendation."""

    DRAFTED = "DRAFTED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"


_DEAL_TRANSITIONS: dict[DealStatus, set[DealStatus]] = {
    DealStatus.DRAFTED: {DealStatus.PENDING_APPROVAL, DealStatus.REJECTED},
    DealStatus.PENDING_APPROVAL: {DealStatus.APPROVED, DealStatus.REJECTED},
    # APPROVED -> PUBLISHED is the operator confirming it went out. There is no
    # automated poster here: where a deal gets published is a channel decision,
    # not something this pipeline should guess at.
    DealStatus.APPROVED: {DealStatus.PUBLISHED, DealStatus.REJECTED},
    DealStatus.REJECTED: set(),
    DealStatus.PUBLISHED: set(),
}


_SHORT_TRANSITIONS: dict[ShortStatus, set[ShortStatus]] = {
    ShortStatus.DRAFTED: {ShortStatus.RENDERING, ShortStatus.FAILED},
    ShortStatus.RENDERING: {ShortStatus.PENDING_APPROVAL, ShortStatus.FAILED},
    ShortStatus.PENDING_APPROVAL: {
        ShortStatus.APPROVED,
        ShortStatus.REJECTED,
        ShortStatus.FAILED,
        # A reroll sends the same row back through the pipeline.
        ShortStatus.DRAFTED,
    },
    ShortStatus.APPROVED: {ShortStatus.PUBLISHED, ShortStatus.FAILED},
    ShortStatus.REJECTED: {ShortStatus.DRAFTED},
    ShortStatus.PUBLISHED: set(),
    ShortStatus.FAILED: {ShortStatus.DRAFTED, ShortStatus.RENDERING},
}


class Channel(str, Enum):
    """Where a pound of revenue came from."""

    ETSY = "etsy"
    YOUTUBE = "youtube"
    DIGITAL = "digital"


#: Display metadata for each channel, shared by the HUD and the CLI.
CHANNEL_META: dict[str, dict[str, str]] = {
    Channel.ETSY.value: {"icon": "🛍️", "label": "Etsy & POD Shop", "accent": "#e0a53f"},
    Channel.YOUTUBE.value: {"icon": "🎬", "label": "YouTube Shorts", "accent": "#c8434f"},
    Channel.DIGITAL.value: {"icon": "📦", "label": "Digital / Other Assets", "accent": "#7ea34a"},
}


def month_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    """First and last instant of the current UTC month.

    Lives here rather than in the web layer because the CLI's treasury report
    needs the same window, and importing it from ``web.app`` would make
    ``python main.py --treasury`` depend on FastAPI being installed.
    """
    import calendar

    moment = now or datetime.now(timezone.utc)
    start = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_day = calendar.monthrange(moment.year, moment.month)[1]
    end = moment.replace(
        day=last_day, hour=23, minute=59, second=59, microsecond=999_999
    )
    return start, end


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every model in the village."""


class Listing(Base):
    """One generated print-on-demand listing, from idea to published product."""

    __tablename__ = "listings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---- provenance ---------------------------------------------------------
    niche: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    audience: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    concept: Mapped[str] = mapped_column(Text, nullable=False, default="")
    art_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # ---- copy ---------------------------------------------------------------
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Etsy allows exactly 13 tags; stored as a JSON array of strings.
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # ---- assets -------------------------------------------------------------
    raw_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    processed_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # ---- lifecycle ----------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ListingStatus.DRAFTED.value, index=True
    )
    status_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    guard_report: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # ---- storefront ---------------------------------------------------------
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    printify_image_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    printify_product_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    external_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    dry_run: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # ---- messaging ----------------------------------------------------------
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # ---- convenience --------------------------------------------------------
    @property
    def tags(self) -> list[str]:
        """The tag list, decoded. Never raises on malformed stored JSON."""
        try:
            value = json.loads(self.tags_json or "[]")
        except json.JSONDecodeError:
            logger.warning("Listing {} has unreadable tags_json", self.id)
            return []
        return [str(tag) for tag in value] if isinstance(value, list) else []

    @tags.setter
    def tags(self, value: Sequence[str]) -> None:
        self.tags_json = json.dumps([str(tag) for tag in value], ensure_ascii=False)

    @property
    def status_enum(self) -> ListingStatus:
        return ListingStatus(self.status)

    @property
    def price_display(self) -> str:
        return f"${self.price_cents / 100:,.2f}"

    def summary(self) -> str:
        """A one-line description for logs and Telegram captions."""
        return f"#{self.id} [{self.status}] {self.title or self.niche!r}"

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, safe to serialise and to hand to another thread."""
        return {
            "id": self.id,
            "niche": self.niche,
            "audience": self.audience,
            "concept": self.concept,
            "art_prompt": self.art_prompt,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "raw_image_path": self.raw_image_path,
            "processed_image_path": self.processed_image_path,
            "status": self.status,
            "status_reason": self.status_reason,
            "price_cents": self.price_cents,
            "printify_image_id": self.printify_image_id,
            "printify_product_id": self.printify_product_id,
            "external_url": self.external_url,
            "dry_run": bool(self.dry_run),
            "telegram_message_id": self.telegram_message_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Listing {self.summary()}>"


class Short(Base):
    """One generated vertical short: script, narration, scenes and the mp4."""

    __tablename__ = "shorts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---- concept -------------------------------------------------------------
    topic: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    category: Mapped[str] = mapped_column(String(60), nullable=False, default="")
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    hook: Mapped[str] = mapped_column(Text, nullable=False, default="")
    conclusion: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Full narration, exactly as it was sent to the voice engine.
    narration: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: JSON array of {narration, image_prompt, seconds, image_path}.
    scenes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    hashtags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # ---- assets --------------------------------------------------------------
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    video_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    #: How the video was produced: moviepy | ffmpeg | storyboard.
    render_backend: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    voice_backend: Mapped[str] = mapped_column(String(40), nullable=False, default="")

    # ---- lifecycle -----------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ShortStatus.DRAFTED.value, index=True
    )
    status_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reroll_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dry_run: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    youtube_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    #: The API's own id for the upload. Kept apart from the URL because it is
    #: the handle every later API call needs, and a URL can be rewritten.
    youtube_video_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Visibility at upload time — public / private / unlisted. Worth storing:
    #: a Short uploaded privately is published as far as this pipeline is
    #: concerned but is not yet earning anything.
    youtube_privacy: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ---- performance ---------------------------------------------------------
    #: Views are only ever what someone recorded. Uploading uses the
    #: youtube.upload scope, which grants no read access to analytics, so a
    #: zero here means "unmeasured", not "nobody watched".
    views: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Revenue per 1,000 views, in cents. Shorts RPM is typically 5-15 cents.
    rpm_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # ---- convenience ---------------------------------------------------------
    @property
    def scenes(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.scenes_json or "[]")
        except json.JSONDecodeError:
            logger.warning("Short {} has unreadable scenes_json", self.id)
            return []
        return value if isinstance(value, list) else []

    @scenes.setter
    def scenes(self, value: Sequence[dict[str, Any]]) -> None:
        self.scenes_json = json.dumps(list(value), ensure_ascii=False)

    @property
    def hashtags(self) -> list[str]:
        try:
            value = json.loads(self.hashtags_json or "[]")
        except json.JSONDecodeError:
            return []
        return [str(tag) for tag in value] if isinstance(value, list) else []

    @hashtags.setter
    def hashtags(self, value: Sequence[str]) -> None:
        self.hashtags_json = json.dumps([str(tag) for tag in value], ensure_ascii=False)

    @property
    def status_enum(self) -> ShortStatus:
        return ShortStatus(self.status)

    @property
    def estimated_cents(self) -> int:
        """Estimated AdSense payout: views / 1000 * RPM.

        An estimate, and labelled as one everywhere it is shown. Real payouts
        depend on watch time, geography and the month's ad market.
        """
        return int(round(self.views / 1000 * self.rpm_cents))

    def summary(self) -> str:
        return f"#{self.id} [{self.status}] {self.title or self.topic!r}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "category": self.category,
            "title": self.title,
            "hook": self.hook,
            "conclusion": self.conclusion,
            "narration": self.narration,
            "scenes": self.scenes,
            "hashtags": self.hashtags,
            "description": self.description,
            "audio_path": self.audio_path,
            "video_path": self.video_path,
            "thumbnail_path": self.thumbnail_path,
            "duration_seconds": self.duration_seconds,
            "render_backend": self.render_backend,
            "voice_backend": self.voice_backend,
            "status": self.status,
            "status_reason": self.status_reason,
            "reroll_count": self.reroll_count,
            "dry_run": bool(self.dry_run),
            "youtube_url": self.youtube_url,
            "youtube_video_id": self.youtube_video_id,
            "youtube_privacy": self.youtube_privacy,
            "telegram_message_id": self.telegram_message_id,
            "views": self.views,
            "rpm_cents": self.rpm_cents,
            "estimated_cents": self.estimated_cents,
            "metrics_updated_at": (
                self.metrics_updated_at.isoformat() if self.metrics_updated_at else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Short {self.summary()}>"


class RevenueEntry(Base):
    """One line in the village ledger.

    Append-mostly: an Etsy sale or a digital payment is inserted once and never
    changed. YouTube is the exception — an estimate for a given video is
    *replaced* as its view count is re-recorded, which is what ``source_key``
    is for. Without that, re-entering yesterday's views would double-count.
    """

    __tablename__ = "revenue_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    note: Mapped[str] = mapped_column(String(300), nullable=False, default="")

    #: Stable key for the thing that earned it, e.g. "listing:4", "short:7".
    #: Unique per channel, so re-recording replaces rather than duplicates.
    source_key: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    #: True when this line is a projection rather than money actually received.
    estimated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dry_run: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "channel": self.channel,
            "amount_cents": self.amount_cents,
            "amount": round(self.amount_cents / 100, 2),
            "note": self.note,
            "source_key": self.source_key,
            "estimated": bool(self.estimated),
            "dry_run": bool(self.dry_run),
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Revenue {self.channel} {self.amount_cents}c {self.source_key}>"


# ---------------------------------------------------------------------------
# Engine and sessions
# ---------------------------------------------------------------------------

_settings = get_settings()

#: SQLite needs ``check_same_thread=False`` because the Telegram worker and the
#: generation loop touch the session from different threads.
_connect_args = (
    {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    _settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _add_missing_columns() -> int:
    """Bring an existing database up to the current models.

    ``create_all`` creates missing TABLES but never missing COLUMNS, so a
    database written by an earlier build keeps its old shape and every query
    touching a new field fails with "no such column". This compares each
    model against the live schema and issues the ALTERs.

    Deliberately small: additive columns only. It will not drop, rename or
    retype anything — if a change needs that, it needs a real migration tool,
    and silently rewriting a user's data is worse than an honest error.
    """
    from sqlalchemy import inspect, text
    from sqlalchemy.schema import CreateColumn

    inspector = inspect(engine)
    added = 0

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {column["name"] for column in inspector.get_columns(table.name)}

            for column in table.columns:
                if column.name in existing:
                    continue
                # SQLite cannot add a NOT NULL column without a default, so a
                # non-nullable column is added nullable and backfilled.
                spec = CreateColumn(column).compile(engine).string
                if column.default is not None and column.default.is_scalar:
                    spec += f" DEFAULT {column.default.arg!r}"
                spec = spec.replace(" NOT NULL", "")

                connection.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN {spec}'))
                if column.default is not None and column.default.is_scalar:
                    connection.execute(
                        text(f'UPDATE "{table.name}" SET "{column.name}" = :value '
                             f'WHERE "{column.name}" IS NULL'),
                        {"value": column.default.arg},
                    )
                logger.info("Schema: added {}.{}", table.name, column.name)
                added += 1

    return added


def init_db() -> None:
    """Create every table, migrate additive columns, reconcile the ledger.

    Safe to call repeatedly, and safe to call against a database written by an
    older build of this project.
    """
    Base.metadata.create_all(engine)
    logger.debug("Database ready at {}", _settings.database_url)

    try:
        _add_missing_columns()
    except Exception as exc:  # noqa: BLE001 - surfaced, but never fatal at import
        logger.error("Could not migrate the schema: {}", exc)

    try:
        backfill_revenue()
    except Exception as exc:  # noqa: BLE001 - a reconcile must not block startup
        logger.warning("Could not backfill the revenue ledger: {}", exc)


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional scope: commit on success, roll back on any exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Queries and mutations
# ---------------------------------------------------------------------------


def create_listing(**fields: Any) -> Listing:
    """Insert a listing and return the detached, fully-populated row."""
    tags = fields.pop("tags", None)
    with session_scope() as session:
        listing = Listing(**fields)
        if tags is not None:
            listing.tags = tags
        session.add(listing)
        session.flush()
        session.refresh(listing)
        logger.info("Created listing {}", listing.summary())
        return listing


def get_listing(listing_id: int) -> Listing | None:
    """Fetch one listing by id, or ``None`` when it does not exist."""
    with session_scope() as session:
        return session.get(Listing, listing_id)


def list_by_status(status: ListingStatus, limit: int = 50) -> list[Listing]:
    """Most recent listings in a given state, newest first."""
    with session_scope() as session:
        statement = (
            select(Listing)
            .where(Listing.status == status.value)
            .order_by(Listing.created_at.desc())
            .limit(limit)
        )
        return list(session.scalars(statement).all())


def recent_listings(limit: int = 20) -> list[Listing]:
    """The newest listings regardless of state."""
    with session_scope() as session:
        statement = select(Listing).order_by(Listing.created_at.desc()).limit(limit)
        return list(session.scalars(statement).all())


def update_listing(listing_id: int, **fields: Any) -> Listing | None:
    """Patch arbitrary columns on a listing.

    ``tags`` is accepted as a real list and encoded for you.
    """
    tags = fields.pop("tags", None)
    with session_scope() as session:
        listing = session.get(Listing, listing_id)
        if listing is None:
            logger.warning("update_listing: no listing {}", listing_id)
            return None
        for key, value in fields.items():
            if not hasattr(listing, key):
                raise AttributeError(f"Listing has no column {key!r}")
            setattr(listing, key, value)
        if tags is not None:
            listing.tags = tags
        session.flush()
        session.refresh(listing)
        return listing


def update_status(
    listing_id: int,
    status: ListingStatus,
    reason: str = "",
    **extra: Any,
) -> Listing | None:
    """Move a listing to a new state, refusing transitions the machine forbids.

    Returning ``None`` for an illegal move (rather than raising) keeps a stale
    Telegram button — a second tap on an already-answered card — from taking the
    worker down.
    """
    with session_scope() as session:
        listing = session.get(Listing, listing_id)
        if listing is None:
            logger.warning("update_status: no listing {}", listing_id)
            return None

        current = ListingStatus(listing.status)
        if status is not current and status not in _ALLOWED_TRANSITIONS[current]:
            logger.warning(
                "Refusing illegal transition {} -> {} on listing {}",
                current.value,
                status.value,
                listing_id,
            )
            return None

        listing.status = status.value
        if reason:
            listing.status_reason = reason
        for key, value in extra.items():
            if not hasattr(listing, key):
                raise AttributeError(f"Listing has no column {key!r}")
            setattr(listing, key, value)

        session.flush()
        session.refresh(listing)
        logger.info("Listing {} -> {}", listing_id, status.value)
        return listing


class Deal(Base):
    """One curated Amazon affiliate recommendation."""

    __tablename__ = "deals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    niche: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    category: Mapped[str] = mapped_column(String(60), nullable=False, default="")
    product: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    hook: Mapped[str] = mapped_column(Text, nullable=False, default="")
    search_terms: Mapped[str] = mapped_column(String(300), nullable=False, default="")

    #: The whole Deal dataclass, as JSON. Kept alongside the queried columns
    #: rather than instead of them: the columns are what the dashboard filters
    #: on, this is what rebuilds the exact object that was approved.
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    affiliate_url: Mapped[str] = mapped_column(String(800), nullable=False, default="")
    tracking_id: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    price_low: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    price_high: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DealStatus.DRAFTED.value, index=True
    )
    status_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    dry_run: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ---- performance ---------------------------------------------------------
    #: Affiliate performance is only ever what someone recorded here. Amazon
    #: reports clicks and conversions in the Associates dashboard, and there is
    #: no API in this project reading them back, so a zero means "unmeasured",
    #: not "nobody clicked".
    #: Where this deal was pinned, once it has been. Kept so a deal cannot be
    #: pinned twice and so the operator can open the live Pin from a listing.
    pinterest_pin_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pinterest_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    clicks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conversions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    earnings_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    @property
    def conversion_rate(self) -> float:
        """Conversions per click, or 0.0 when nothing has been recorded."""
        return round(self.conversions / self.clicks, 4) if self.clicks else 0.0

    @property
    def earnings_per_click(self) -> float:
        """EPC in dollars — the number affiliate programmes actually rank on."""
        return round(self.earnings_cents / 100 / self.clicks, 4) if self.clicks else 0.0

    @property
    def status_enum(self) -> DealStatus:
        return DealStatus(self.status)

    @property
    def payload(self) -> dict[str, Any]:
        """The stored Deal, as a dict."""
        try:
            data = json.loads(self.payload_json or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    @payload.setter
    def payload(self, value: dict[str, Any]) -> None:
        self.payload_json = json.dumps(value or {}, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "niche": self.niche,
            "category": self.category,
            "product": self.product,
            "hook": self.hook,
            "search_terms": self.search_terms,
            "affiliate_url": self.affiliate_url,
            "tracking_id": self.tracking_id,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "status": self.status,
            "status_reason": self.status_reason,
            "dry_run": bool(self.dry_run),
            "pinterest_pin_id": self.pinterest_pin_id,
            "pinterest_url": self.pinterest_url,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "earnings_cents": self.earnings_cents,
            "earnings": round(self.earnings_cents / 100, 2),
            "conversion_rate": self.conversion_rate,
            "earnings_per_click": self.earnings_per_click,
            "metrics_updated_at": (
                self.metrics_updated_at.isoformat() if self.metrics_updated_at else None
            ),
            "payload": self.payload,
        }


class EditionStatus(str, Enum):
    """Lifecycle of one newsletter edition."""

    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"


_EDITION_TRANSITIONS: dict[EditionStatus, set[EditionStatus]] = {
    # APPROVED means "scheduled"; PUBLISHED means the 08:00 send actually went.
    EditionStatus.DRAFT: {EditionStatus.APPROVED, EditionStatus.REJECTED},
    # APPROVED -> DRAFT is un-approving, and it has to be possible: when a
    # publish attempt fails the edition is still good and the channel is not,
    # so it goes back to the queue. Leaving it APPROVED would mean the 08:00
    # delivery loop picks up and sends an edition that just failed to send.
    EditionStatus.APPROVED: {
        EditionStatus.PUBLISHED,
        EditionStatus.REJECTED,
        EditionStatus.DRAFT,
    },
    EditionStatus.PUBLISHED: set(),
    EditionStatus.REJECTED: {EditionStatus.DRAFT},
}


class RawCryptoEvent(Base):
    """One story the Night Scribe found, before any editorial judgement."""

    __tablename__ = "raw_crypto_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(60), nullable=False, default="", index=True)
    headline: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Unique, and that is the whole de-duplication strategy: the same story
    #: reaching us from two feeds is the same URL, and a second insert is a
    #: no-op rather than a second paragraph in the newsletter.
    url: Mapped[str] = mapped_column(String(800), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(20), nullable=False, default="market", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "headline": self.headline,
            "summary": self.summary,
            "url": self.url,
            "category": self.category,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class NewsletterEdition(Base):
    """One day's Morning Ledger."""

    __tablename__ = "newsletter_editions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: One edition per day, enforced by the database rather than by whoever
    #: happens to call the builder twice.
    publish_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    market_pulse_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    heist_story_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    alpha_tip_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    full_markdown: Mapped[str] = mapped_column(Text, nullable=False, default="")
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=EditionStatus.DRAFT.value, index=True
    )
    status_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: When an approved edition should go out. Set on approval, read at 08:00.
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    @property
    def status_enum(self) -> EditionStatus:
        return EditionStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "title": self.title,
            "market_pulse_md": self.market_pulse_md,
            "heist_story_md": self.heist_story_md,
            "alpha_tip_md": self.alpha_tip_md,
            "full_markdown": self.full_markdown,
            "telegram_message_id": self.telegram_message_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class VillageMetric(Base):
    """One day's subscriber and revenue snapshot, for the Overseer."""

    __tablename__ = "village_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recorded_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)
    total_subscribers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    open_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    click_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_revenue_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    monetization_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="FREE"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recorded_date": self.recorded_date.isoformat() if self.recorded_date else None,
            "total_subscribers": self.total_subscribers,
            "open_rate": self.open_rate,
            "click_rate": self.click_rate,
            "total_revenue_usd": self.total_revenue_usd,
            "monetization_status": self.monetization_status,
        }


def insert_crypto_events(events: Sequence[dict[str, Any]]) -> int:
    """Store newly seen stories, ignoring ones already held.

    De-duplication is the UNIQUE constraint on ``url``, checked per row rather
    than with one bulk insert: a single duplicate in a batch of forty must not
    discard the other thirty-nine.

    :returns: how many rows were actually new.
    """
    added = 0
    with session_scope() as session:
        for entry in events:
            url = str(entry.get("url") or "").strip()
            if not url:
                continue
            if session.scalar(select(RawCryptoEvent.id).where(RawCryptoEvent.url == url)):
                continue
            session.add(
                RawCryptoEvent(
                    source=str(entry.get("source") or "")[:60],
                    headline=str(entry.get("headline") or "")[:600],
                    summary=str(entry.get("summary") or "")[:4000],
                    url=url[:800],
                    category=str(entry.get("category") or "market")[:20],
                )
            )
            added += 1
        session.flush()

    if added:
        logger.info("Night Scribe stored {} new event(s)", added)
    return added


def recent_crypto_events(hours: int = 12, limit: int = 60) -> list[RawCryptoEvent]:
    """Stories collected in the last ``hours``, newest first."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, hours))
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(RawCryptoEvent)
                .where(RawCryptoEvent.created_at >= cutoff)
                .order_by(RawCryptoEvent.created_at.desc())
                .limit(max(1, limit))
            )
        )
        for row in rows:
            session.expunge(row)
        return rows


def create_edition(**fields: Any) -> NewsletterEdition:
    """Insert one edition, or return the existing one for that date.

    The date is unique, so a second builder run on the same day updates the
    draft rather than raising — regenerating is a normal thing to want.
    """
    publish_date = fields.get("publish_date") or datetime.now(timezone.utc).date()
    with session_scope() as session:
        edition = session.scalar(
            select(NewsletterEdition).where(NewsletterEdition.publish_date == publish_date)
        )
        if edition is None:
            edition = NewsletterEdition(publish_date=publish_date)
            session.add(edition)

        for key, value in fields.items():
            if key != "publish_date" and hasattr(edition, key):
                setattr(edition, key, value)
        # A regenerated edition is a draft again, whatever it was before.
        edition.status = EditionStatus.DRAFT.value
        session.flush()
        session.refresh(edition)
        session.expunge(edition)

    logger.info("Edition #{} [{}] {!r}", edition.id, edition.status, edition.title)
    _publish_edition(edition)
    return edition


def get_edition(edition_id: int) -> NewsletterEdition | None:
    with session_scope() as session:
        edition = session.get(NewsletterEdition, edition_id)
        if edition is None:
            return None
        session.expunge(edition)
        return edition


def edition_for_date(publish_date: date | None = None) -> NewsletterEdition | None:
    """The edition for a given day, defaulting to today."""
    wanted = publish_date or datetime.now(timezone.utc).date()
    with session_scope() as session:
        edition = session.scalar(
            select(NewsletterEdition).where(NewsletterEdition.publish_date == wanted)
        )
        if edition is None:
            return None
        session.expunge(edition)
        return edition


def recent_editions(limit: int = 10) -> list[NewsletterEdition]:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(NewsletterEdition).order_by(NewsletterEdition.id.desc()).limit(limit)
            )
        )
        for row in rows:
            session.expunge(row)
        return rows


def editions_by_status(status: EditionStatus) -> list[NewsletterEdition]:
    """Every edition currently in one state — the 08:00 sender reads this."""
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(NewsletterEdition)
                .where(NewsletterEdition.status == status.value)
                .order_by(NewsletterEdition.publish_date)
            )
        )
        for row in rows:
            session.expunge(row)
        return rows


def update_edition_status(
    edition_id: int,
    status: EditionStatus,
    *,
    reason: str = "",
    scheduled_for: datetime | None = None,
    telegram_message_id: int | None = None,
) -> NewsletterEdition | None:
    """Move an edition through its lifecycle, refusing impossible jumps."""
    with session_scope() as session:
        edition = session.get(NewsletterEdition, edition_id)
        if edition is None:
            logger.warning("No edition with id {}", edition_id)
            return None

        current = edition.status_enum
        if status is not current and status not in _EDITION_TRANSITIONS.get(current, set()):
            logger.warning(
                "Refusing edition {} transition {} -> {}",
                edition_id, current.value, status.value,
            )
            return None

        edition.status = status.value
        if reason:
            edition.status_reason = reason
        if scheduled_for is not None:
            edition.scheduled_for = scheduled_for
        if telegram_message_id is not None:
            edition.telegram_message_id = telegram_message_id
        if status is EditionStatus.PUBLISHED:
            edition.published_at = datetime.now(timezone.utc)

        session.flush()
        session.refresh(edition)
        session.expunge(edition)

    logger.info("Edition {} -> {}", edition_id, status.value)
    _publish_edition(edition)
    return edition


def record_village_metrics(
    recorded_date: date | None = None, **fields: Any
) -> VillageMetric:
    """Upsert one day's subscriber snapshot."""
    day = recorded_date or datetime.now(timezone.utc).date()
    with session_scope() as session:
        metric = session.scalar(
            select(VillageMetric).where(VillageMetric.recorded_date == day)
        )
        if metric is None:
            metric = VillageMetric(recorded_date=day)
            session.add(metric)

        for key, value in fields.items():
            if hasattr(metric, key) and key not in ("id", "recorded_date"):
                setattr(metric, key, value)

        session.flush()
        session.refresh(metric)
        session.expunge(metric)

    logger.info(
        "Metrics {}: {} subscribers, {:.0%} open, ${:.2f}",
        day, metric.total_subscribers, metric.open_rate, metric.total_revenue_usd,
    )
    return metric


def recent_metrics(days: int = 14) -> list[VillageMetric]:
    """The last ``days`` snapshots, oldest first so streaks read naturally."""
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(VillageMetric)
                .order_by(VillageMetric.recorded_date.desc())
                .limit(max(1, days))
            )
        )
        for row in rows:
            session.expunge(row)
        return list(reversed(rows))


def _publish_edition(edition: NewsletterEdition) -> None:
    """Push an edition onto the event bus, guarded like the deal publisher."""
    try:
        from core import events  # noqa: PLC0415 - avoids an import cycle

        events.edition_event(edition.to_dict())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not publish edition {}: {}", getattr(edition, "id", "?"), exc)


def _publish_deal(deal: Deal) -> None:
    """Push a deal onto the event bus so the dashboard updates live.

    Imported lazily and guarded: the database layer must keep working in a CLI
    with no dashboard attached, and a broken bus is never worth losing a write
    that has already committed.
    """
    try:
        from core import events  # noqa: PLC0415 - avoids an import cycle

        events.deal_event(deal.to_dict())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not publish deal {}: {}", getattr(deal, "id", "?"), exc)


def create_deal(payload: dict[str, Any], *, dry_run: bool = True) -> Deal:
    """Store a curated deal and return the detached row."""
    with session_scope() as session:
        deal = Deal(
            niche=str(payload.get("niche", ""))[:200],
            category=str(payload.get("category", ""))[:60],
            product=str(payload.get("product", ""))[:300],
            hook=str(payload.get("hook", "")),
            search_terms=str(payload.get("search_terms", ""))[:300],
            affiliate_url=str(payload.get("affiliate_url", ""))[:800],
            tracking_id=str(payload.get("tracking_id", ""))[:80],
            price_low=float(payload.get("price_low") or 0.0),
            price_high=float(payload.get("price_high") or 0.0),
            dry_run=1 if dry_run else 0,
            status=DealStatus.DRAFTED.value,
        )
        deal.payload = payload
        session.add(deal)
        session.flush()
        session.refresh(deal)
        session.expunge(deal)

    logger.info("Created deal #{} [{}] {!r}", deal.id, deal.status, deal.product)
    _publish_deal(deal)
    return deal


def get_deal(deal_id: int) -> Deal | None:
    """One deal by id, detached."""
    with session_scope() as session:
        deal = session.get(Deal, deal_id)
        if deal is None:
            return None
        session.expunge(deal)
        return deal


def recent_deals(limit: int = 10, status: DealStatus | None = None) -> list[Deal]:
    """The most recent deals, newest first."""
    with session_scope() as session:
        statement = select(Deal).order_by(Deal.id.desc()).limit(max(1, limit))
        if status is not None:
            statement = statement.where(Deal.status == status.value)
        rows = list(session.scalars(statement))
        for row in rows:
            session.expunge(row)
        return rows


def record_deal_pin(deal_id: int, *, pin_id: str, url: str) -> Deal | None:
    """Record that a deal was pinned, atomically with its identifiers."""
    with session_scope() as session:
        deal = session.get(Deal, deal_id)
        if deal is None:
            return None
        deal.pinterest_pin_id = pin_id
        deal.pinterest_url = url
        session.flush()
        session.refresh(deal)
        session.expunge(deal)

    logger.info("Deal {} pinned as {} ({})", deal_id, pin_id, url)
    _publish_deal(deal)
    return deal


def record_deal_metrics(
    deal_id: int,
    *,
    clicks: int | None = None,
    conversions: int | None = None,
    earnings_cents: int | None = None,
) -> Deal | None:
    """Record affiliate performance reported from the Associates dashboard.

    Each field is optional and REPLACES rather than adds: the Associates report
    gives running totals for a period, so adding them would double-count every
    time the same figures were entered twice.
    """
    with session_scope() as session:
        deal = session.get(Deal, deal_id)
        if deal is None:
            logger.warning("No deal with id {}", deal_id)
            return None

        if clicks is not None:
            deal.clicks = max(0, int(clicks))
        if conversions is not None:
            deal.conversions = max(0, int(conversions))
        if earnings_cents is not None:
            deal.earnings_cents = max(0, int(earnings_cents))
        deal.metrics_updated_at = datetime.now(timezone.utc)

        session.flush()
        session.refresh(deal)
        session.expunge(deal)

    logger.info(
        "Deal {} metrics: {} clicks, {} conversions, ${:.2f}",
        deal_id, deal.clicks, deal.conversions, deal.earnings_cents / 100,
    )
    _publish_deal(deal)
    return deal


def deal_counts_by_status() -> dict[str, int]:
    """How many deals sit in each state."""
    with session_scope() as session:
        rows = session.execute(
            select(Deal.status, func.count(Deal.id)).group_by(Deal.status)
        ).all()
    return {status: int(count) for status, count in rows}


def deal_metrics(*, include_dry_run: bool = False) -> dict[str, Any]:
    """Aggregate affiliate performance across published deals.

    Simulated deals are excluded by the same rule the revenue ledger uses, so
    the dashboard cannot report earnings for a link that was never posted.
    """
    with session_scope() as session:
        statement = select(
            func.count(Deal.id),
            func.coalesce(func.sum(Deal.clicks), 0),
            func.coalesce(func.sum(Deal.conversions), 0),
            func.coalesce(func.sum(Deal.earnings_cents), 0),
        ).where(Deal.status == DealStatus.PUBLISHED.value)
        if not include_dry_run:
            statement = statement.where(Deal.dry_run == 0)
        published, clicks, conversions, cents = session.execute(statement).one()

        curated = session.scalar(select(func.count(Deal.id))) or 0

    clicks, conversions, cents = int(clicks), int(conversions), int(cents)
    return {
        "curated": int(curated),
        "published": int(published or 0),
        "clicks": clicks,
        "conversions": conversions,
        "earningsCents": cents,
        "earnings": round(cents / 100, 2),
        "conversionRate": round(conversions / clicks, 4) if clicks else 0.0,
        "earningsPerClick": round(cents / 100 / clicks, 4) if clicks else 0.0,
    }


def update_deal_status(
    deal_id: int, status: DealStatus, *, reason: str = ""
) -> Deal | None:
    """Move a deal through its lifecycle, refusing impossible jumps.

    Returns None when the deal is missing or the transition is not allowed, so
    a double-tapped Telegram button cannot approve something already rejected.
    """
    with session_scope() as session:
        deal = session.get(Deal, deal_id)
        if deal is None:
            logger.warning("No deal with id {}", deal_id)
            return None

        current = deal.status_enum
        if status is not current and status not in _DEAL_TRANSITIONS.get(current, set()):
            logger.warning(
                "Refusing deal {} transition {} -> {}", deal_id, current.value, status.value
            )
            return None

        deal.status = status.value
        if reason:
            deal.status_reason = reason
        session.flush()
        session.refresh(deal)
        session.expunge(deal)

    logger.info("Deal {} -> {}", deal_id, status.value)
    _publish_deal(deal)
    return deal

def create_short(**fields: Any) -> Short:
    """Insert a short and return the detached row."""
    scenes = fields.pop("scenes", None)
    hashtags = fields.pop("hashtags", None)
    with session_scope() as session:
        short = Short(**fields)
        if scenes is not None:
            short.scenes = scenes
        if hashtags is not None:
            short.hashtags = hashtags
        session.add(short)
        session.flush()
        session.refresh(short)
        logger.info("Created short {}", short.summary())
        return short


def get_short(short_id: int) -> Short | None:
    with session_scope() as session:
        return session.get(Short, short_id)


def recent_shorts(limit: int = 20) -> list[Short]:
    with session_scope() as session:
        statement = select(Short).order_by(Short.created_at.desc()).limit(limit)
        return list(session.scalars(statement).all())


def shorts_by_status(status: ShortStatus, limit: int = 50) -> list[Short]:
    with session_scope() as session:
        statement = (
            select(Short)
            .where(Short.status == status.value)
            .order_by(Short.created_at.desc())
            .limit(limit)
        )
        return list(session.scalars(statement).all())


def update_short(short_id: int, **fields: Any) -> Short | None:
    """Patch columns on a short. ``scenes`` and ``hashtags`` take real lists."""
    scenes = fields.pop("scenes", None)
    hashtags = fields.pop("hashtags", None)
    with session_scope() as session:
        short = session.get(Short, short_id)
        if short is None:
            logger.warning("update_short: no short {}", short_id)
            return None
        for key, value in fields.items():
            if not hasattr(short, key):
                raise AttributeError(f"Short has no column {key!r}")
            setattr(short, key, value)
        if scenes is not None:
            short.scenes = scenes
        if hashtags is not None:
            short.hashtags = hashtags
        session.flush()
        session.refresh(short)
        return short


def update_short_status(
    short_id: int, status: ShortStatus, reason: str = "", **extra: Any
) -> Short | None:
    """Move a short through its state machine, refusing illegal transitions."""
    with session_scope() as session:
        short = session.get(Short, short_id)
        if short is None:
            logger.warning("update_short_status: no short {}", short_id)
            return None

        current = ShortStatus(short.status)
        if status is not current and status not in _SHORT_TRANSITIONS[current]:
            logger.warning(
                "Refusing illegal short transition {} -> {} on {}",
                current.value,
                status.value,
                short_id,
            )
            return None

        short.status = status.value
        if reason:
            short.status_reason = reason
        for key, value in extra.items():
            if not hasattr(short, key):
                raise AttributeError(f"Short has no column {key!r}")
            setattr(short, key, value)

        session.flush()
        session.refresh(short)
        logger.info("Short {} -> {}", short_id, status.value)
        return short


def short_counts_by_status() -> dict[str, int]:
    with session_scope() as session:
        rows = session.execute(
            select(Short.status, func.count(Short.id)).group_by(Short.status)
        ).all()
    return {status: count for status, count in rows}


def record_revenue(
    channel: Channel | str,
    amount_cents: int,
    *,
    note: str = "",
    source_key: str | None = None,
    estimated: bool = False,
    dry_run: bool = True,
    occurred_at: datetime | None = None,
) -> RevenueEntry:
    """Post a line to the ledger.

    When ``source_key`` is given, an existing line for that key on that channel
    is UPDATED rather than duplicated — which is what makes re-recording a
    video's view count idempotent.
    """
    # Channel subclasses str, so `isinstance(channel, str)` is True for a
    # member too — and `str(Channel.ETSY)` yields "Channel.ETSY", not "etsy".
    # Going through the constructor normalises both forms and rejects junk.
    value = channel.value if isinstance(channel, Channel) else Channel(str(channel)).value

    with session_scope() as session:
        existing = None
        if source_key:
            existing = session.scalars(
                select(RevenueEntry)
                .where(RevenueEntry.channel == value, RevenueEntry.source_key == source_key)
                .limit(1)
            ).first()

        if existing is not None:
            existing.amount_cents = int(amount_cents)
            existing.note = note or existing.note
            existing.estimated = 1 if estimated else 0
            existing.dry_run = 1 if dry_run else 0
            if occurred_at:
                existing.occurred_at = occurred_at
            session.flush()
            session.refresh(existing)
            logger.debug("Updated ledger line {} -> {}c", source_key, amount_cents)
            return existing

        entry = RevenueEntry(
            channel=value,
            amount_cents=int(amount_cents),
            note=note,
            source_key=source_key,
            estimated=1 if estimated else 0,
            dry_run=1 if dry_run else 0,
            occurred_at=occurred_at or _utcnow(),
        )
        session.add(entry)
        session.flush()
        session.refresh(entry)
        logger.info("Ledger: {} +{}c ({})", value, amount_cents, note or source_key or "")
        return entry


def revenue_by_channel(
    since: datetime | None = None,
    until: datetime | None = None,
    *,
    include_dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Totals per channel over an optional window.

    ``include_dry_run`` is False by default, and that default is the point: a
    simulated publish writes a ledger line just like a live one, so counting it
    would show a treasury full of money nobody earned. Simulated amounts are
    still returned, under ``simulatedCents``, so the operator can see the
    pipeline worked without mistaking it for income.

    Every channel appears in the result even with no entries, so the HUD's
    ledger never gains or loses rows as money arrives.
    """
    with session_scope() as session:
        statement = select(
            RevenueEntry.channel,
            RevenueEntry.dry_run,
            func.coalesce(func.sum(RevenueEntry.amount_cents), 0),
            func.count(RevenueEntry.id),
            func.max(RevenueEntry.estimated),
        ).group_by(RevenueEntry.channel, RevenueEntry.dry_run)

        if since is not None:
            statement = statement.where(RevenueEntry.occurred_at >= since)
        if until is not None:
            statement = statement.where(RevenueEntry.occurred_at <= until)

        rows = session.execute(statement).all()

    def blank(channel_id: str) -> dict[str, Any]:
        meta = CHANNEL_META.get(
            channel_id, {"icon": "❓", "label": channel_id.title(), "accent": "#8d7f6e"}
        )
        return {
            "channel": channel_id,
            "cents": 0,
            "amount": 0.0,
            "simulatedCents": 0,
            "simulated": 0.0,
            "entries": 0,
            "estimated": False,
            **meta,
        }

    totals = {channel.value: blank(channel.value) for channel in Channel}

    for channel_id, dry_run, cents, count, estimated in rows:
        # A channel written by something newer than this build: surface it
        # rather than silently dropping the money.
        row = totals.setdefault(channel_id, blank(channel_id))
        amount = int(cents or 0)

        if dry_run:
            row["simulatedCents"] += amount
            row["simulated"] = round(row["simulatedCents"] / 100, 2)
            if include_dry_run:
                row["cents"] += amount
                row["entries"] += int(count or 0)
        else:
            row["cents"] += amount
            row["entries"] += int(count or 0)

        row["amount"] = round(row["cents"] / 100, 2)
        if estimated and (include_dry_run or not dry_run):
            row["estimated"] = True

    return totals


def recent_revenue(limit: int = 20) -> list[RevenueEntry]:
    with session_scope() as session:
        statement = select(RevenueEntry).order_by(RevenueEntry.occurred_at.desc()).limit(limit)
        return list(session.scalars(statement).all())


def record_youtube_upload(
    short_id: int,
    *,
    video_id: str,
    url: str,
    privacy: str,
    simulated: bool = False,
) -> Short | None:
    """Store where a Short ended up on YouTube.

    Separate from ``update_short`` so the upload is recorded atomically with the
    status change: a row that says PUBLISHED but carries no link is the state
    that makes an operator go looking through the channel by hand.
    """
    with session_scope() as session:
        short = session.get(Short, short_id)
        if short is None:
            return None

        short.youtube_video_id = video_id
        short.youtube_url = url
        short.youtube_privacy = privacy
        session.flush()
        session.refresh(short)
        session.expunge(short)

    logger.info(
        "Short {} {} at {}", short_id,
        "simulated upload" if simulated else "uploaded", url,
    )
    return short


def record_short_metrics(short_id: int, views: int, rpm_cents: int | None = None) -> Short | None:
    """Record a Short's view count and re-post its estimated payout.

    Idempotent: the ledger line is keyed to the short, so entering views twice
    corrects the estimate instead of doubling it.
    """
    with session_scope() as session:
        short = session.get(Short, short_id)
        if short is None:
            logger.warning("record_short_metrics: no short {}", short_id)
            return None
        short.views = max(0, int(views))
        if rpm_cents is not None:
            short.rpm_cents = max(0, int(rpm_cents))
        short.metrics_updated_at = _utcnow()
        session.flush()
        session.refresh(short)
        snapshot = short

    record_revenue(
        Channel.YOUTUBE,
        snapshot.estimated_cents,
        note=f"{snapshot.views:,} views @ {snapshot.rpm_cents}c RPM",
        source_key=f"short:{snapshot.id}",
        estimated=True,
        dry_run=bool(snapshot.dry_run),
    )
    return snapshot


def youtube_metrics(*, include_dry_run: bool = False) -> dict[str, Any]:
    """Aggregate view/RPM figures across published Shorts.

    Simulated Shorts are excluded by the same rule the ledger uses, so the
    dashboard cannot report views for a video that was never uploaded.
    """
    with session_scope() as session:
        statement = select(
            func.count(Short.id),
            func.coalesce(func.sum(Short.views), 0),
            func.coalesce(func.avg(Short.rpm_cents), 0),
        ).where(Short.status == ShortStatus.PUBLISHED.value)
        if not include_dry_run:
            statement = statement.where(Short.dry_run == 0)
        rows = session.execute(statement).one()

        simulated = (
            session.scalar(
                select(func.count(Short.id)).where(
                    Short.status == ShortStatus.PUBLISHED.value, Short.dry_run == 1
                )
            )
            or 0
        )

    published, views, rpm = int(rows[0] or 0), int(rows[1] or 0), float(rows[2] or 0)
    return {
        "published": published,
        "simulatedPublished": int(simulated),
        "views": views,
        "rpmCents": round(rpm, 1),
        "rpm": round(rpm / 100, 4),
    }


def backfill_revenue() -> int:
    """Post ledger lines for published listings that predate the ledger.

    Idempotent by ``source_key``. Without this, upgrading an existing database
    would show a treasury of zero next to a shelf of published listings.
    """
    created = 0
    with session_scope() as session:
        listings = session.scalars(
            select(Listing).where(Listing.status == ListingStatus.PUBLISHED.value)
        ).all()
        existing = {
            key
            for (key,) in session.execute(
                select(RevenueEntry.source_key).where(RevenueEntry.channel == Channel.ETSY.value)
            ).all()
        }
        pending = [
            (row.id, row.price_cents, row.title, row.dry_run, row.created_at)
            for row in listings
            if f"listing:{row.id}" not in existing
        ]

    for listing_id, price, title, dry, created_at in pending:
        record_revenue(
            Channel.ETSY,
            price,
            note=(title or "")[:120],
            source_key=f"listing:{listing_id}",
            dry_run=bool(dry),
            occurred_at=created_at,
        )
        created += 1

    if created:
        logger.info("Backfilled {} ledger line(s) from published listings", created)
    return created


def counts_by_status() -> dict[str, int]:
    """How many listings sit in each state — used by the CLI status view."""
    with session_scope() as session:
        rows = session.execute(
            select(Listing.status, func.count(Listing.id)).group_by(Listing.status)
        ).all()
    return {status: count for status, count in rows}
