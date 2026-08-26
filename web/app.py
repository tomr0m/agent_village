"""FastAPI backend for the living village dashboard.

Serves the canvas, streams pipeline state over a WebSocket, and exposes the
"poke a villager" actions the dialogue boxes call.

Design notes:

* The pipeline is unchanged and unaware of the web. It publishes to
  :mod:`core.events`; this module subscribes and fans out to browsers. Run the
  CLI with no dashboard and nothing here costs anything.
* Triggered actions run as **background tasks** with a single global lock. A
  pipeline pass takes minutes; the HTTP call returns immediately and the browser
  watches the villagers move instead of waiting on a response.
* Asset serving is path-guarded: only files inside ``storage/`` are readable,
  by basename, with no traversal.
"""

from __future__ import annotations

import asyncio
import calendar
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from config.settings import get_settings
from core import events, roster
from core.database import (
    CHANNEL_META,
    Channel,
    Deal,
    DealStatus,
    deal_counts_by_status,
    deal_metrics,
    get_deal,
    recent_deals,
    record_deal_metrics,
    update_deal_status,
    month_bounds,
    Listing,
    ListingStatus,
    Short,
    ShortStatus,
    init_db,
    recent_revenue,
    record_revenue,
    record_short_metrics,
    revenue_by_channel,
    session_scope,
    youtube_metrics,
)
from core.events import bus, state

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Only one long job runs at a time — the pipeline shares a SQLite file and
#: rate-limited APIs, and two concurrent passes buy nothing.
_job_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats() -> dict[str, Any]:
    """Treasury figures, per channel, plus the listing counts the HUD chips use.

    Money comes from the ledger rather than from summing Listing rows: with
    three channels, a derived total per channel would mean three different
    definitions of revenue and no way to reconcile them.
    """
    settings = get_settings()
    start, end = month_bounds()

    # Simulated publishes are excluded unless the operator opts in: a dry-run
    # sale writes the same ledger line as a real one, and counting it would put
    # money in the treasury that nobody earned.
    counted = settings.count_dry_run_revenue
    month_channels = revenue_by_channel(since=start, until=end, include_dry_run=counted)
    lifetime_channels = revenue_by_channel(include_dry_run=counted)
    month_cents = sum(entry["cents"] for entry in month_channels.values())
    lifetime_cents = sum(entry["cents"] for entry in lifetime_channels.values())
    simulated_cents = sum(entry["simulatedCents"] for entry in lifetime_channels.values())

    with session_scope() as session:
        published_total = (
            session.scalar(
                select(func.count(Listing.id)).where(
                    Listing.status == ListingStatus.PUBLISHED.value
                )
            )
            or 0
        )
        month_published = (
            session.scalar(
                select(func.count(Listing.id)).where(
                    Listing.status == ListingStatus.PUBLISHED.value,
                    Listing.created_at >= start,
                    Listing.created_at <= end,
                )
            )
            or 0
        )
        by_status = {
            status: count
            for status, count in session.execute(
                select(Listing.status, func.count(Listing.id)).group_by(Listing.status)
            ).all()
        }
        shorts_by_state = {
            status: count
            for status, count in session.execute(
                select(Short.status, func.count(Short.id)).group_by(Short.status)
            ).all()
        }

    youtube = youtube_metrics(include_dry_run=counted)

    # The ledger rows the HUD renders, in a fixed order so the panel never
    # reflows as money arrives on a channel that previously had none.
    channels = []
    for channel in Channel:
        month = month_channels[channel.value]
        lifetime = lifetime_channels[channel.value]
        row = {
            "id": channel.value,
            "icon": month["icon"],
            "label": month["label"],
            "accent": month["accent"],
            "monthCents": month["cents"],
            "month": month["amount"],
            "lifetimeCents": lifetime["cents"],
            "lifetime": lifetime["amount"],
            "entries": lifetime["entries"],
            "estimated": lifetime["estimated"],
            "simulatedCents": lifetime["simulatedCents"],
            "simulated": lifetime["simulated"],
        }
        if channel is Channel.ETSY:
            row["detail"] = (
                f"{published_total} listing(s) published"
                if not row["simulatedCents"]
                else f"{published_total} published · all simulated"
            )
        elif channel is Channel.YOUTUBE:
            row["views"] = youtube["views"]
            row["rpmCents"] = youtube["rpmCents"]
            row["published"] = youtube["published"]
            if youtube["views"]:
                row["detail"] = f"{youtube['views']:,} views · {youtube['rpmCents']:.0f}c RPM"
            elif youtube["published"]:
                row["detail"] = f"{youtube['published']} live · no views recorded"
            elif youtube["simulatedPublished"]:
                row["detail"] = f"{youtube['simulatedPublished']} simulated · not uploaded"
            else:
                row["detail"] = "nothing published yet"
        else:
            row["detail"] = f"{lifetime['entries']} ledger line(s)"
        channels.append(row)

    return {
        "monthRevenueCents": month_cents,
        "monthRevenue": round(month_cents / 100, 2),
        "monthPublished": int(month_published),
        "lifetimeRevenueCents": lifetime_cents,
        "lifetimeRevenue": round(lifetime_cents / 100, 2),
        "publishedTotal": int(published_total),
        "simulatedRevenueCents": simulated_cents,
        "simulatedRevenue": round(simulated_cents / 100, 2),
        "countsDryRun": counted,
        "channels": channels,
        "pending": int(by_status.get(ListingStatus.PENDING_APPROVAL.value, 0)),
        "approved": int(by_status.get(ListingStatus.APPROVED.value, 0)),
        "rejected": int(by_status.get(ListingStatus.REJECTED.value, 0)),
        "failed": int(by_status.get(ListingStatus.FAILED.value, 0)),
        "byStatus": by_status,
        "shorts": {
            "total": sum(shorts_by_state.values()),
            "pending": int(shorts_by_state.get(ShortStatus.PENDING_APPROVAL.value, 0)),
            "published": int(shorts_by_state.get(ShortStatus.PUBLISHED.value, 0)),
            "byStatus": shorts_by_state,
            **youtube,
        },
        "dryRun": settings.dry_run,
        "month": start.strftime("%B %Y"),
        "currency": "USD",
    }


def recent_short_dicts(limit: int = 12) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(
            select(Short).order_by(Short.created_at.desc()).limit(limit)
        ).all()
        return [row.to_dict() for row in rows]


def recent_listing_dicts(limit: int = 12) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(
            select(Listing).order_by(Listing.created_at.desc()).limit(limit)
        ).all()
        return [row.to_dict() for row in rows]


def snapshot() -> dict[str, Any]:
    """Everything a freshly-loaded dashboard needs in one payload."""
    settings = get_settings()
    return {
        "villagers": roster.roster(),
        "buildings": roster.BUILDINGS,
        "socialSpots": list(roster.SOCIAL_SPOTS),
        "agents": state.all(),
        "stats": compute_stats(),
        "listings": recent_listing_dicts(),
        "shorts": recent_short_dicts(),
        "config": {
            "dryRun": settings.dry_run,
            "textModel": settings.text_model,
            "imageModel": settings.image_model,
            "telegram": settings.telegram_configured,
            "printify": settings.printify_configured,
            "openrouter": settings.openrouter_configured,
            "ttsProvider": settings.tts_provider,
            "videoBackend": _video_backend(),
        },
    }


def _video_backend() -> str:
    """Which renderer the Bard would use, for the dashboard banner."""
    try:
        from core.video_engine import available_backends

        return available_backends()[0]
    except Exception:  # noqa: BLE001 - never let a probe break the snapshot
        return "unknown"


# ---------------------------------------------------------------------------
# WebSocket fan-out
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Tracks live sockets so a stats refresh can reach all of them."""

    def __init__(self) -> None:
        self._sockets: set[WebSocket] = set()

    async def connect(self, socket: WebSocket) -> None:
        await socket.accept()
        self._sockets.add(socket)
        logger.info("Dashboard connected ({} open)", len(self._sockets))

    def disconnect(self, socket: WebSocket) -> None:
        self._sockets.discard(socket)
        logger.info("Dashboard disconnected ({} open)", len(self._sockets))

    @property
    def count(self) -> int:
        return len(self._sockets)


manager = ConnectionManager()


async def _stats_ticker(interval: float = 20.0) -> None:
    """Republish treasury figures periodically.

    The pipeline announces its own changes; this catches the ones it cannot —
    a Telegram approval that published a listing while the browser watched.
    """
    while True:
        await asyncio.sleep(interval)
        if manager.count == 0:
            continue
        try:
            events.stats_event(compute_stats())
        except Exception as exc:  # noqa: BLE001 - a stats read must not kill the loop
            logger.warning("Stats refresh failed: {}", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    logger.info("Village dashboard ready")
    ticker = asyncio.create_task(_stats_ticker())
    try:
        yield
    finally:
        ticker.cancel()
        with suppress(asyncio.CancelledError):
            await ticker


app = FastAPI(
    title="Agent Village",
    description="A living 16-bit village dashboard for the autonomous POD pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pages and assets
# ---------------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """The dashboard itself."""
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail="index.html is missing from web/static")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/assets/{filename}")
async def asset(filename: str) -> FileResponse:
    """Serve one generated asset out of ``storage/``.

    Only a basename is accepted and the resolved path is re-checked against the
    storage root, so neither ``..`` nor a symlink can escape it.
    """
    settings = get_settings()
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name:
        raise HTTPException(status_code=400, detail="Invalid asset name")

    root = settings.storage_dir.resolve()
    target = (root / safe_name).resolve()
    if root not in target.parents and target.parent != root:
        raise HTTPException(status_code=400, detail="Asset path escapes storage")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="No such asset")
    return FileResponse(target)


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------


@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(snapshot())


@app.get("/api/stats")
async def api_stats() -> JSONResponse:
    return JSONResponse(compute_stats())


@app.get("/api/listings")
async def api_listings(limit: int = 20) -> JSONResponse:
    return JSONResponse({"listings": recent_listing_dicts(max(1, min(limit, 200)))})


@app.get("/api/agents")
async def api_agents() -> JSONResponse:
    return JSONResponse({"villagers": roster.roster(), "agents": state.all()})


@app.get("/api/agents/{agent_id}")
async def api_agent(agent_id: str) -> JSONResponse:
    villager = roster.get(agent_id)
    agent = state.get(agent_id)
    if villager is None or agent is None:
        raise HTTPException(status_code=404, detail=f"No villager {agent_id!r}")
    return JSONResponse({"villager": villager.to_dict(), "agent": agent.to_dict()})


#: One publish at a time. A cron service that retries on timeout, or two
#: overlapping schedules, would otherwise run the pipeline twice and post the
#: same edition to the channel twice.
_cron_lock = asyncio.Lock()


def _cron_authorised(token: str | None, header: str | None) -> bool:
    """Whether a cron request may proceed.

    With no CRON_SECRET configured the route is open, which is what makes it
    usable from a bare cron URL — and is also why the check exists at all. This
    route publishes to a public channel with no human in the loop and answers
    GET, so a crawler or a link preview can fire it.
    """
    secret = get_settings().cron_secret.strip()
    if not secret:
        return True

    import hmac  # noqa: PLC0415

    supplied = (token or header or "").strip()
    # Constant time: this is a bearer secret, and a timing oracle on it is a
    # slow but real way to guess it.
    return bool(supplied) and hmac.compare_digest(supplied, secret)


@app.get("/api/cron/publish")
async def api_cron_publish(
    token: str | None = None,
    hours: int = 12,
    x_cron_token: str | None = Header(default=None),
) -> JSONResponse:
    """Run the full pipeline and publish straight to the channel.

    Scout gathers, Scribe writes, and the edition goes out — no approval card,
    no button. Intended for a cron service hitting one URL each morning.

    Synchronous on purpose: the response claims the newsletter was published,
    so it must not return until that is true. A full run takes roughly 30-60
    seconds, so give the caller a timeout above that.
    """
    if not _cron_authorised(token, x_cron_token):
        raise HTTPException(status_code=401, detail="Invalid or missing cron token")

    if _cron_lock.locked():
        # 409 rather than queueing: a cron retry should be told it was already
        # running, not silently produce a second edition.
        raise HTTPException(
            status_code=409, detail="A publish run is already in progress"
        )

    async with _cron_lock:
        settings = get_settings()

        if not settings.channel_configured:
            raise HTTPException(
                status_code=503,
                detail=(
                    "TELEGRAM_CHANNEL_ID is not set. Add it to .env "
                    "(e.g. TELEGRAM_CHANNEL_ID=@TheMorningLedger) and make the "
                    "bot an administrator of that channel."
                ),
            )

        from core.database import EditionStatus, update_edition_status  # noqa: PLC0415
        from village.morning_ledger import MorningLedger  # noqa: PLC0415
        from village.night_scribe import NightScribe  # noqa: PLC0415
        from village.town_crier import TownCrier  # noqa: PLC0415

        events.log("Cron: the Morning Ledger run has started.", "info")

        # ---- 1. the Scout gathers -------------------------------------------
        try:
            scan = await NightScribe(settings).scan()
        except Exception as exc:  # noqa: BLE001 - a bad scan need not stop the run
            logger.warning("Cron scan failed ({}); building from what is held", exc)
            scan = None

        # ---- 2. the Scribe writes -------------------------------------------
        ledger = MorningLedger(settings)
        try:
            row = await ledger.publish_draft(hours=hours, dispatch=False)
        except Exception as exc:
            logger.exception("Cron build failed")
            raise HTTPException(
                status_code=500, detail=f"Could not build the edition: {exc}"
            ) from exc
        finally:
            await ledger.aclose()

        # ---- 3. straight past the approval gate ------------------------------
        # The row still moves through APPROVED rather than jumping to PUBLISHED:
        # the transition table is what keeps every other path honest, and a row
        # that skipped a state is one no other code can reason about.
        update_edition_status(
            row.id, EditionStatus.APPROVED, reason="Auto-approved by the cron route."
        )

        sent, detail = await TownCrier(settings).broadcast_edition(row.id)
        if not sent:
            # Back to DRAFT, not PUBLISHED: nothing was published, and the row
            # must not claim otherwise.
            update_edition_status(
                row.id, EditionStatus.DRAFT, reason=f"Channel publish failed: {detail}"
            )
            raise HTTPException(
                status_code=502, detail=f"Telegram refused the post: {detail}"
            )

        update_edition_status(
            row.id, EditionStatus.PUBLISHED, reason=f"Published by cron — {detail}"
        )
        events.log(f"Cron published: {row.title}", "success")

        logger.success("Cron publish complete: edition {} — {}", row.id, detail)
        return JSONResponse(
            {
                "status": "success",
                "message": "Newsletter generated and published successfully",
                # Everything below is additive. The two fields above are the
                # contract; these are what make a failed morning debuggable.
                "edition_id": row.id,
                "title": row.title,
                "publish_date": str(row.publish_date),
                "stories_scanned": getattr(scan, "stored", 0) if scan else 0,
                "channel": settings.telegram_channel_id.strip(),
                "detail": detail,
            }
        )


@app.get("/api/deals")
async def api_deals(limit: int = 12, status: str | None = None) -> JSONResponse:
    """Recent curated affiliate deals, newest first."""
    wanted: DealStatus | None = None
    if status:
        try:
            wanted = DealStatus(status.upper())
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unknown deal status {status!r}"
            ) from None

    rows = recent_deals(max(1, min(limit, 100)), wanted)
    return JSONResponse(
        {
            "deals": [row.to_dict() for row in rows],
            "counts": deal_counts_by_status(),
            "metrics": deal_metrics(
                include_dry_run=get_settings().count_dry_run_revenue
            ),
        }
    )


@app.get("/api/deals/{deal_id}/pin.jpg")
async def api_deal_pin_image(deal_id: int) -> FileResponse:
    """Serve the composed Pin card.

    Exists so ``image_url`` in the Make payload can point at the finished card
    rather than at the raw background — but only helps when this dashboard is
    actually reachable from the internet, which is what PUBLIC_BASE_URL asserts.
    """
    settings = get_settings()
    card = (settings.storage_dir / "pins" / f"deal-{deal_id}.jpg").resolve()

    # Same containment check the other media routes use: a deal id is a path
    # component, and a path component is never trusted.
    root = settings.storage_dir.resolve()
    if root not in card.parents or not card.is_file():
        raise HTTPException(status_code=404, detail=f"No Pin card for deal {deal_id}")

    return FileResponse(card, media_type="image/jpeg")


@app.get("/api/deals/{deal_id}")
async def api_deal(deal_id: int) -> JSONResponse:
    row = get_deal(deal_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return JSONResponse(row.to_dict())


@app.post("/api/deals/{deal_id}/approve")
async def api_approve_deal(deal_id: int) -> JSONResponse:
    """Approve a deal from the dashboard, then mark it ready to post."""
    row = get_deal(deal_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")

    if row.status_enum is DealStatus.DRAFTED:
        # Straight from the dashboard skips the Telegram card, so walk the row
        # through the intermediate state rather than allowing the jump.
        update_deal_status(deal_id, DealStatus.PENDING_APPROVAL, reason="Dashboard review.")

    approved = update_deal_status(
        deal_id, DealStatus.APPROVED, reason="Approved in the dashboard."
    )
    if approved is None:
        raise HTTPException(
            status_code=409, detail=f"Deal {deal_id} is {row.status}, not approvable"
        )

    published = update_deal_status(
        deal_id, DealStatus.PUBLISHED, reason="Approved for posting."
    )
    return JSONResponse((published or approved).to_dict())


@app.post("/api/deals/{deal_id}/reject")
async def api_reject_deal(deal_id: int) -> JSONResponse:
    updated = update_deal_status(
        deal_id, DealStatus.REJECTED, reason="Rejected in the dashboard."
    )
    if updated is None:
        raise HTTPException(
            status_code=409, detail=f"Deal {deal_id} could not be rejected"
        )
    return JSONResponse(updated.to_dict())


@app.post("/api/deals/{deal_id}/metrics")
async def api_deal_metrics(deal_id: int, payload: dict[str, Any]) -> JSONResponse:
    """Record clicks / conversions / earnings reported by Amazon Associates."""
    def number(key: str) -> int | None:
        value = payload.get(key)
        if value in (None, ""):
            return None
        try:
            return max(0, int(float(value)))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400, detail=f"{key} must be a number"
            ) from None

    updated = record_deal_metrics(
        deal_id,
        clicks=number("clicks"),
        conversions=number("conversions"),
        earnings_cents=number("earnings_cents"),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return JSONResponse(updated.to_dict())


@app.get("/api/shorts")
async def api_shorts(limit: int = 12) -> JSONResponse:
    return JSONResponse({"shorts": recent_short_dicts(max(1, min(limit, 100)))})


@app.get("/api/shorts/{short_id}")
async def api_short(short_id: int) -> JSONResponse:
    with session_scope() as session:
        row = session.get(Short, short_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No short {short_id}")
        return JSONResponse(row.to_dict())


def _short_media(short_id: int, field: str) -> Path:
    """Resolve a Short's media file, refusing anything outside storage/."""
    settings = get_settings()
    with session_scope() as session:
        row = session.get(Short, short_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No short {short_id}")
        raw = getattr(row, field, None)

    if not raw:
        raise HTTPException(status_code=404, detail=f"Short {short_id} has no {field}")

    root = settings.storage_dir.resolve()
    target = Path(raw).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=400, detail="Media path escapes storage")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Media file is missing on disk")
    return target


@app.get("/api/shorts/{short_id}/video")
async def api_short_video(short_id: int) -> FileResponse:
    """Serve the finished video (or the storyboard PNG, when that is what exists)."""
    target = _short_media(short_id, "video_path")
    media_type = "video/mp4" if target.suffix.lower() == ".mp4" else "image/png"
    # accept-ranges lets the browser scrub without downloading the whole file.
    return FileResponse(target, media_type=media_type, filename=target.name)


@app.get("/api/shorts/{short_id}/thumbnail")
async def api_short_thumbnail(short_id: int) -> FileResponse:
    return FileResponse(_short_media(short_id, "thumbnail_path"), media_type="image/jpeg")


@app.get("/api/shorts/{short_id}/audio")
async def api_short_audio(short_id: int) -> FileResponse:
    target = _short_media(short_id, "audio_path")
    media_type = "audio/mpeg" if target.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(target, media_type=media_type)


class ShortRequest(BaseModel):
    topic: str | None = Field(default=None, max_length=200)


@app.post("/api/shorts/generate")
async def api_generate_short(body: ShortRequest, background: BackgroundTasks) -> JSONResponse:
    """Produce a new Short. Long-running, so it returns immediately."""
    if _job_lock.locked():
        return JSONResponse(
            {"accepted": False, "reason": "The village is already busy."}, status_code=409
        )

    from web.jobs import generate_short

    background.add_task(
        _run_guarded, "The Bard's Short", lambda: generate_short(topic=body.topic)
    )
    return JSONResponse({"accepted": True})


@app.post("/api/shorts/{short_id}/approve")
async def api_approve_short(short_id: int, background: BackgroundTasks) -> JSONResponse:
    from web.jobs import approve_short

    background.add_task(_run_guarded, f"Approve short #{short_id}", lambda: approve_short(short_id))
    return JSONResponse({"accepted": True})


@app.post("/api/shorts/{short_id}/reject")
async def api_reject_short(short_id: int) -> JSONResponse:
    from village.town_crier import reject_short_from_cli

    ok = reject_short_from_cli(short_id, "Rejected from the village dashboard.")
    if ok:
        events.toast(f"Short #{short_id} rejected.", "warn")
    return JSONResponse({"ok": ok})


@app.post("/api/shorts/{short_id}/reroll")
async def api_reroll_short(short_id: int, background: BackgroundTasks) -> JSONResponse:
    if _job_lock.locked():
        return JSONResponse(
            {"accepted": False, "reason": "The village is already busy."}, status_code=409
        )

    from web.jobs import reroll_script

    background.add_task(
        _run_guarded, f"Reroll short #{short_id}", lambda: reroll_script(short_id=short_id)
    )
    return JSONResponse({"accepted": True})


@app.get("/api/revenue")
async def api_revenue(limit: int = 25) -> JSONResponse:
    """The ledger: per-channel totals plus the most recent lines."""
    return JSONResponse(
        {
            "channels": compute_stats()["channels"],
            "entries": [entry.to_dict() for entry in recent_revenue(max(1, min(limit, 200)))],
        }
    )


class MetricsRequest(BaseModel):
    """Record what a Short actually did on the platform."""

    views: int = Field(..., ge=0, le=10_000_000_000)
    rpm_cents: int | None = Field(default=None, ge=0, le=10_000)


@app.post("/api/shorts/{short_id}/metrics")
async def api_short_metrics(short_id: int, body: MetricsRequest) -> JSONResponse:
    """Set a Short's view count and re-estimate its payout.

    Idempotent: the ledger line is keyed to the Short, so re-recording corrects
    the estimate rather than adding to it.
    """
    settings = get_settings()
    short = record_short_metrics(
        short_id, body.views, body.rpm_cents if body.rpm_cents is not None else settings.youtube_rpm_cents
    )
    if short is None:
        raise HTTPException(status_code=404, detail=f"No short {short_id}")

    events.stats_event(compute_stats())
    events.toast(
        f"Short #{short_id}: {short.views:,} views → "
        f"${short.estimated_cents / 100:.2f} estimated.",
        "info",
    )
    return JSONResponse(short.to_dict())


class RevenueRequest(BaseModel):
    """Post a line to the ledger by hand — the digital/other channel."""

    channel: str = Field(default=Channel.DIGITAL.value)
    amount_cents: int = Field(..., ge=0, le=100_000_000)
    note: str = Field(default="", max_length=300)
    source_key: str | None = Field(default=None, max_length=120)
    estimated: bool = False


@app.post("/api/revenue")
async def api_record_revenue(body: RevenueRequest) -> JSONResponse:
    settings = get_settings()
    try:
        channel = Channel(body.channel)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"channel must be one of {[c.value for c in Channel]}",
        ) from None

    entry = record_revenue(
        channel,
        body.amount_cents,
        note=body.note,
        source_key=body.source_key,
        estimated=body.estimated,
        dry_run=settings.dry_run,
    )
    events.stats_event(compute_stats())
    events.toast(
        f"{CHANNEL_META[channel.value]['icon']} {CHANNEL_META[channel.value]['label']}: "
        f"+${body.amount_cents / 100:.2f}",
        "success",
    )
    return JSONResponse(entry.to_dict())


@app.get("/api/health")
async def api_health() -> JSONResponse:
    settings = get_settings()
    return JSONResponse(
        {
            "status": "ok",
            "dryRun": settings.dry_run,
            "sockets": manager.count,
            "busSubscribers": bus.subscriber_count,
            "jobRunning": _job_lock.locked(),
            "videoBackend": _video_backend(),
            "ttsProvider": settings.tts_provider,
        }
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class TriggerRequest(BaseModel):
    """Body for poking a villager."""

    action: str = Field(..., description="Action id offered by that villager")
    hint: str | None = Field(default=None, max_length=200)
    listing_id: int | None = None


async def _run_guarded(name: str, coro_factory: Any) -> None:
    """Run one background job under the global lock, reporting either way."""
    if _job_lock.locked():
        events.toast(f"{name} skipped — the village is already busy.", "warn")
        return

    async with _job_lock:
        events.toast(f"{name} started.", "info")
        try:
            await coro_factory()
            events.toast(f"{name} finished.", "success")
        except Exception as exc:  # noqa: BLE001 - a failed job must not kill the server
            logger.exception("{} failed", name)
            events.toast(f"{name} failed: {exc}", "error")
        finally:
            state.reset_idle()
            events.stats_event(compute_stats())
            for agent in state.all():
                bus.publish("agent_state", agent=agent)


@app.post("/api/agents/{agent_id}/trigger")
async def api_trigger(
    agent_id: str, body: TriggerRequest, background: BackgroundTasks
) -> JSONResponse:
    """Poke a villager: the dialogue box's action buttons land here."""
    if roster.get(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"No villager {agent_id!r}")
    if not roster.has_action(agent_id, body.action):
        raise HTTPException(
            status_code=400, detail=f"{agent_id!r} does not offer {body.action!r}"
        )

    # Imported lazily: pulling the pipeline in at module load would make the
    # web server pay for onnxruntime before it serves a single page.
    from web.jobs import ACTIONS

    handler = ACTIONS.get(body.action)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"Unimplemented action {body.action!r}")

    if _job_lock.locked() and body.action != "review_logs":
        return JSONResponse(
            {"accepted": False, "reason": "The village is already busy."}, status_code=409
        )

    if body.action == "review_logs":
        agent = state.get(agent_id)
        return JSONResponse({"accepted": True, "logs": agent.to_dict()["history"] if agent else []})

    label = f"{roster.get(agent_id).name}: {body.action}"
    background.add_task(
        _run_guarded, label, lambda: handler(agent_id=agent_id, hint=body.hint, listing_id=body.listing_id)
    )
    return JSONResponse({"accepted": True, "action": body.action, "agent": agent_id})


class GenerateRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=10)
    hint: str | None = Field(default=None, max_length=200)


@app.post("/api/generate")
async def api_generate(body: GenerateRequest, background: BackgroundTasks) -> JSONResponse:
    """Run the full pipeline, the same work ``--generate`` does."""
    if _job_lock.locked():
        return JSONResponse(
            {"accepted": False, "reason": "The village is already busy."}, status_code=409
        )

    from web.jobs import run_pipeline

    background.add_task(
        _run_guarded,
        f"Village run x{body.count}",
        lambda: run_pipeline(count=body.count, hint=body.hint),
    )
    return JSONResponse({"accepted": True, "count": body.count})


class DecisionRequest(BaseModel):
    listing_id: int


@app.post("/api/listings/{listing_id}/approve")
async def api_approve(listing_id: int, background: BackgroundTasks) -> JSONResponse:
    from web.jobs import approve_listing

    background.add_task(_run_guarded, f"Approve #{listing_id}", lambda: approve_listing(listing_id))
    return JSONResponse({"accepted": True})


@app.post("/api/listings/{listing_id}/reject")
async def api_reject(listing_id: int) -> JSONResponse:
    from village.town_crier import reject_from_cli

    ok = reject_from_cli(listing_id, "Rejected from the village dashboard.")
    if ok:
        events.toast(f"Listing #{listing_id} rejected.", "warn")
        events.stats_event(compute_stats())
    return JSONResponse({"ok": ok})


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket) -> None:
    """Stream village events to one browser.

    The client receives a full snapshot first, then a replay of recent events,
    then the live stream — so a page opened mid-run is immediately correct.
    """
    await manager.connect(socket)
    queue = bus.subscribe()

    try:
        await socket.send_json({"type": "snapshot", "data": snapshot()})
        for event in bus.replay(limit=30):
            await socket.send_json(event)

        while True:
            # Wait on either an outbound event or an inbound client message,
            # whichever lands first, so a client ping cannot stall the stream.
            receive = asyncio.create_task(socket.receive_text())
            send = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {receive, send}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

            if send in done:
                await socket.send_json(send.result())
            if receive in done:
                message = receive.result()
                if message == "ping":
                    await socket.send_json({"type": "pong", "ts": asyncio.get_event_loop().time()})
                elif message == "refresh":
                    await socket.send_json({"type": "snapshot", "data": snapshot()})

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - a broken socket is routine
        logger.debug("Dashboard socket closed: {}", exc)
    finally:
        bus.unsubscribe(queue)
        manager.disconnect(socket)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    logger.exception("Unhandled error on {}", request.url.path)
    return JSONResponse({"error": str(exc)}, status_code=500)


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the dashboard. Called by ``python main.py --web``."""
    import uvicorn

    logger.info("Village dashboard on http://{}:{}", host, port)
    uvicorn.run(
        "web.app:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )
