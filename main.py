#!/usr/bin/env python3
"""Autonomous Agent Village — unified CLI.

    python main.py --generate 1     Run the pipeline and dispatch to Telegram
    python main.py --bot            Poll Telegram for approve/reject taps
    python main.py --daemon         Both: scheduled generation plus the bot
    python main.py --web            Serve the living village dashboard
    python main.py --shorts         Produce one faceless YouTube Short

Supporting commands::

    python main.py --status                 Counts by state and the recent tail
    python main.py --list                   The most recent listings
    python main.py --show 7                 Everything about listing 7
    python main.py --approve 7              Approve and publish without Telegram
    python main.py --reject 7               Reject without Telegram
    python main.py --check                  Verify the configuration and exit
    python main.py --init-db                Create the tables and exit
    python main.py --shorts --topic "..."   Steer the Bard at a subject
    python main.py --list-shorts            The Bard's recent videos
    python main.py --treasury               Revenue by channel, and the ledger
    python main.py --record-views 3 12400   Log a Short's views and re-estimate
    python main.py --record-revenue 1250 --channel digital --note "...""
    python main.py --approve-short 3        Mark a Short ready to upload
    python main.py --reject-short 3         Reject a Short

Every command honours ``DRY_RUN``. With it enabled — the default — nothing is
sent to Printify or Etsy, and the whole chain still runs.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, NoReturn

from loguru import logger

from config.settings import Settings, get_settings
from core.database import (
    CHANNEL_META,
    Channel,
    month_bounds,
    ListingStatus,
    counts_by_status,
    get_listing,
    get_short,
    init_db,
    recent_listings,
    recent_revenue,
    record_revenue,
    record_short_metrics,
    recent_shorts,
    revenue_by_channel,
    short_counts_by_status,
    youtube_metrics,
)
from village.mayor import Mayor
from village.town_crier import (
    TownCrier,
    approve_from_cli,
    approve_short_from_cli,
    reject_from_cli,
    reject_short_from_cli,
)


def configure_logging(level: str) -> None:
    """One readable sink on stderr, plus a rolling file in ``storage/``."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
    )
    settings = get_settings()
    logger.add(
        settings.storage_dir / "village.log",
        level="DEBUG",
        rotation="5 MB",
        retention=5,
        encoding="utf-8",
    )


def print_banner(mayor: Mayor) -> None:
    """Show how this run is wired before anything happens."""
    config = mayor.describe_configuration()
    width = max(len(key) for key in config)
    logger.info("Agent Village starting")
    for key, value in config.items():
        logger.info("  {}  {}", key.ljust(width), value)
    if config["mode"] == "DRY RUN":
        logger.warning("DRY RUN: Printify and Etsy calls are simulated.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_generate(count: int, hint: str | None) -> int:
    """Run the village pipeline ``count`` times."""
    mayor = Mayor()
    print_banner(mayor)
    try:
        results = await mayor.run_batch(count, hint)
    finally:
        await mayor.aclose()

    print()
    for result in results:
        marker = "✓" if result.ok else "✗"
        print(f"{marker} {result.summary()}")
        for stage in result.stages:
            print(f"    · {stage}")
        for warning in result.warnings:
            print(f"    ! {warning}")
        for error in result.errors:
            print(f"    ✗ {error}")

    failed = [item for item in results if not item.ok]
    if failed:
        print(f"\n{len(failed)}/{len(results)} pass(es) failed.")
        return 1

    pending = [item for item in results if not item.dispatched]
    if pending:
        ids = ", ".join(str(item.listing_id) for item in pending)
        print(
            f"\nTelegram is not configured, so nothing was dispatched.\n"
            f"Approve locally with:  python main.py --approve {ids.split(',')[0].strip()}"
        )
    return 0


def cmd_bot() -> int:
    """Poll Telegram until interrupted."""
    settings = get_settings()
    if not settings.telegram_configured:
        logger.error(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set to run the bot."
        )
        return 2

    init_db()
    crier = TownCrier(settings)
    try:
        crier.run_polling()
    except KeyboardInterrupt:  # pragma: no cover - operator interrupt
        logger.info("Bot stopped.")
    return 0


async def _scheduled_loop(
    label: str,
    schedule: str,
    state_path: Path,
    grace_minutes: int,
    job: Callable[[], Awaitable[str]],
    stopping: asyncio.Event,
    interval_hours: float = 0.0,
) -> None:
    """Run ``job`` at each configured time of day.

    Generic because the Bard and the Scout want exactly the same behaviour on
    different clocks: fire once per slot, survive a restart without repeating,
    and skip a slot missed by more than the grace window rather than posting it
    at the wrong time of day.

    Each runs as its own task. A Short takes minutes to render, and sharing a
    loop would drag deal curation around behind it.

    :param job: produces the work and returns a one-line summary for the log.
    """
    from village.scheduler import build_scheduler  # noqa: PLC0415

    scheduler = build_scheduler(
        schedule, interval_hours, state_path, grace_minutes=grace_minutes
    )
    if scheduler is None or not scheduler.enabled:
        return

    # "local time" only means something for fixed times of day.
    from village.scheduler import ShortsScheduler  # noqa: PLC0415

    qualifier = " (local time)" if isinstance(scheduler, ShortsScheduler) else ""
    logger.info("{} schedule: {}{}", label, scheduler.describe(), qualifier)

    def announce() -> None:
        upcoming = scheduler.next_run()
        if upcoming is None:
            return
        # Say which day when it is not today, or "09:00" right after the 18:00
        # slot reads like the clock has gone backwards.
        when = upcoming.strftime("%H:%M")
        if upcoming.date() == date.today() + timedelta(days=1):
            when += " tomorrow"
        elif upcoming.date() != date.today():
            when += upcoming.strftime(" on %a %d %b")
        logger.info("Next {} scheduled for: {}", label, when)

    announce()

    while not stopping.is_set():
        slot = scheduler.due()

        if slot is not None:
            logger.info("Scheduled slot {} reached — running {}", slot.label, label)
            try:
                summary = await job()
                logger.success("{} complete: {}", label, summary)
            except Exception:  # noqa: BLE001 - a bad slot must not kill the daemon
                logger.exception("Scheduled {} raised; continuing", label)

            # Marked whatever happened. Retrying a failed slot in a loop would
            # burn quota against the same broken thing until the next slot.
            scheduler.mark_fired(slot)
            announce()

        # Wake before the next slot so a long sleep cannot overshoot it, and cap
        # the wait so a stop request is still noticed promptly.
        delay = min(scheduler.seconds_until_next(), 300.0)
        try:
            await asyncio.wait_for(stopping.wait(), timeout=max(1.0, delay))
        except asyncio.TimeoutError:
            continue


async def _produce_short() -> str:
    """One scheduled Short, dispatched to Telegram for approval."""
    from village.bard import BardAgent  # noqa: PLC0415

    bard = BardAgent(get_settings())
    try:
        result = await bard.produce()
    finally:
        await bard.aclose()

    if not result.ok:
        raise RuntimeError("; ".join(result.errors) or "unknown error")
    return result.summary()


async def _curate_deal() -> str:
    """One scheduled affiliate deal, dispatched to Telegram for approval."""
    from core.database import create_deal  # noqa: PLC0415
    from village.dealscout import DealScout  # noqa: PLC0415

    settings = get_settings()
    scout = DealScout(settings)
    try:
        deal = await scout.find_deal(None)
    finally:
        await scout.aclose()

    row = create_deal(deal.to_dict(), dry_run=settings.dry_run)

    if settings.telegram_configured:
        from village.town_crier import TownCrier  # noqa: PLC0415

        await TownCrier(settings).dispatch_deal(row.id, deal)
        return f"deal #{row.id} '{deal.product}' — sent to Telegram"

    return f"deal #{row.id} '{deal.product}' — awaiting CLI approval"


async def _night_scan_loop(settings: Settings, stopping: asyncio.Event) -> None:
    """Read the wires every couple of hours through the small hours.

    Scans only inside the configured window (00:00-06:00 by default). Outside
    it the loop sleeps until the window opens rather than spinning: there is no
    point reading feeds at noon for a newsletter written at dawn.
    """
    from village.night_scribe import NightScribe  # noqa: PLC0415

    start = settings.night_scan_start_hour
    end = settings.night_scan_end_hour
    every = timedelta(hours=settings.night_scan_interval_hours)

    def in_window(now: datetime) -> bool:
        # A window that wraps midnight (22 -> 04) is the normal case for a
        # night routine, so it has to be handled, not assumed away.
        return start <= now.hour < end if start < end else (now.hour >= start or now.hour < end)

    logger.info(
        "Night Scribe: scanning {:02d}:00-{:02d}:00 every {:g}h",
        start, end, settings.night_scan_interval_hours,
    )

    last: datetime | None = None
    while not stopping.is_set():
        now = datetime.now()

        if in_window(now) and (last is None or now - last >= every):
            try:
                await NightScribe(settings).scan()
                last = now
            except Exception:  # noqa: BLE001 - a bad scan must not kill the daemon
                logger.exception("Night scan failed; continuing")
                last = now

        try:
            await asyncio.wait_for(stopping.wait(), timeout=300)
        except asyncio.TimeoutError:
            continue


async def _ledger_loop(settings: Settings, stopping: asyncio.Event) -> None:
    """Build the edition at 06:30 and deliver approved ones at 08:00.

    One loop rather than two: they are two moments on the same clock, and
    splitting them would mean two schedulers agreeing about the same day.
    """
    from core.database import EditionStatus, edition_for_date, editions_by_status  # noqa: PLC0415

    def at(clock: str) -> tuple[int, int]:
        hour, _, minute = clock.strip().partition(":")
        try:
            return int(hour), int(minute or 0)
        except ValueError:
            return (6, 30) if clock is settings.ledger_build_time else (8, 0)

    build_h, build_m = at(settings.ledger_build_time)
    send_h, send_m = at(settings.ledger_publish_time)
    logger.info(
        "Morning Ledger: build at {:02d}:{:02d}, deliver at {:02d}:{:02d}",
        build_h, build_m, send_h, send_m,
    )

    built_on: date | None = None

    while not stopping.is_set():
        now = datetime.now()
        today = now.date()

        # ---- 06:30: write the edition -------------------------------------
        past_build = (now.hour, now.minute) >= (build_h, build_m)
        if past_build and built_on != today:
            existing = edition_for_date(today)
            if existing is None:
                try:
                    from village.morning_ledger import MorningLedger  # noqa: PLC0415

                    ledger = MorningLedger(settings)
                    try:
                        # The card is posted below, so not here as well.
                        row = await ledger.publish_draft(dispatch=False)
                    finally:
                        await ledger.aclose()

                    logger.success("Edition #{} drafted: {}", row.id, row.title)
                    if settings.telegram_configured:
                        from village.town_crier import TownCrier  # noqa: PLC0415

                        await TownCrier(settings).dispatch_edition(row.id)
                except Exception:  # noqa: BLE001
                    logger.exception("Ledger build failed; will retry tomorrow")
            built_on = today

        # ---- 08:00: deliver whatever was approved --------------------------
        if (now.hour, now.minute) >= (send_h, send_m):
            for edition in editions_by_status(EditionStatus.APPROVED):
                due = edition.scheduled_for
                if due is not None and due > now:
                    continue
                try:
                    from village.town_crier import TownCrier  # noqa: PLC0415

                    await TownCrier(settings).publish_edition(edition.id)
                except Exception:  # noqa: BLE001
                    logger.exception("Could not deliver edition {}", edition.id)

        try:
            await asyncio.wait_for(stopping.wait(), timeout=60)
        except asyncio.TimeoutError:
            continue


async def _overseer_loop(settings: Settings, stopping: asyncio.Event) -> None:
    """Review the newsletter economy once a day."""
    from village.village_overseer import VillageOverseer  # noqa: PLC0415

    reviewed_on: date | None = None
    while not stopping.is_set():
        today = datetime.now().date()
        if reviewed_on != today:
            try:
                await VillageOverseer(settings).run_daily()
            except Exception:  # noqa: BLE001
                logger.exception("Overseer review failed; continuing")
            reviewed_on = today

        try:
            await asyncio.wait_for(stopping.wait(), timeout=900)
        except asyncio.TimeoutError:
            continue


async def cmd_daemon(interval: int, batch: int, hint: str | None) -> int:
    """Run the Telegram bot and a generation schedule in one process."""
    settings = get_settings()
    init_db()

    mayor = Mayor(settings)
    print_banner(mayor)
    logger.info("Daemon: {} listing(s) every {}s", batch, interval)

    application = None
    if settings.telegram_configured:
        crier = TownCrier(settings)
        application = await crier.run_polling_async()
    else:
        logger.warning("Telegram not configured - the daemon will generate but not dispatch.")

    stopping = asyncio.Event()

    def request_stop() -> None:
        logger.info("Shutdown requested…")
        stopping.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
            pass

    scheduled: list[asyncio.Task[None]] = []

    if settings.shorts_scheduled:
        scheduled.append(asyncio.create_task(_scheduled_loop(
            "Short",
            settings.shorts_schedule_hours,
            settings.shorts_schedule_state_path,
            settings.shorts_schedule_grace_minutes,
            _produce_short,
            stopping,
            settings.shorts_interval_hours,
        )))
    else:
        logger.info(
            "No Shorts schedule set. Add SHORTS_SCHEDULE_HOURS=12:00,16:00,20:00 "
            "to .env for daily Shorts."
        )

    if settings.scout_scheduled:
        scheduled.append(asyncio.create_task(_scheduled_loop(
            "Deal",
            settings.scout_schedule_hours,
            settings.scout_schedule_state_path,
            settings.shorts_schedule_grace_minutes,
            _curate_deal,
            stopping,
            settings.scout_interval_hours,
        )))
    else:
        logger.info(
            "No Scout schedule set. Add SCOUT_SCHEDULE_TIMES=10:00,14:00,18:00 "
            "or SCOUT_INTERVAL_HOURS=3 to .env for automatic deals."
        )

    # The Ledger's three routines. Each is its own task: a feed scan, a model
    # call and a delivery run have nothing to say to each other, and a slow one
    # must not hold up the rest.
    scheduled.append(asyncio.create_task(_night_scan_loop(settings, stopping)))
    scheduled.append(asyncio.create_task(_ledger_loop(settings, stopping)))
    scheduled.append(asyncio.create_task(_overseer_loop(settings, stopping)))

    try:
        while not stopping.is_set():
            try:
                await mayor.run_batch(batch, hint)
            except Exception:  # noqa: BLE001 - a bad pass must not kill the daemon
                logger.exception("Generation pass failed; continuing")

            logger.info("Sleeping {}s until the next pass", interval)
            try:
                await asyncio.wait_for(stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
    finally:
        for task in scheduled:
            # `stopping` is already set, so each loop exits on its own; the wait
            # bounds how long work in flight may hold up shutdown.
            try:
                await asyncio.wait_for(task, timeout=30)
            except asyncio.TimeoutError:
                logger.warning("Scheduled work still running; cancelling it")
                task.cancel()
            except asyncio.CancelledError:  # pragma: no cover - teardown
                pass
        await mayor.aclose()
        if application is not None:
            await TownCrier.stop_application(application)
        logger.info("Daemon stopped.")
    return 0


async def cmd_approve(listing_id: int) -> int:
    """Approve and publish a listing without Telegram."""
    init_db()
    try:
        result = await approve_from_cli(listing_id)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    print(result.summary())
    for step in result.steps:
        print(f"  · {step}")
    return 0 if result.ok else 1


def cmd_reject(listing_id: int) -> int:
    init_db()
    if reject_from_cli(listing_id):
        print(f"Listing #{listing_id} rejected.")
        return 0
    logger.error("Could not reject listing {} — check its current state.", listing_id)
    return 1


async def cmd_shorts(topic: str | None, count: int) -> int:
    """Produce one or more faceless Shorts."""
    from core.video_engine import describe_toolchain  # noqa: PLC0415
    from village.bard import BardAgent  # noqa: PLC0415

    settings = get_settings()
    init_db()

    chain = describe_toolchain()
    logger.info("The Bard's Theater")
    logger.info("  render backend   {}", chain["preferred"])
    logger.info("  ffmpeg           {}", chain["ffmpeg"] or "not found")
    logger.info("  moviepy          {}", chain["moviepy"])
    voice_ok, voice_reason = settings.tts_status()
    logger.info(
        "  voice            {} ({})",
        settings.tts_provider,
        f"{settings.tts_voice}, ready" if voice_ok else f"UNAVAILABLE — {voice_reason}",
    )
    logger.info("  frame            {}x{} @ {}fps",
                settings.video_width, settings.video_height, settings.video_fps)

    if not voice_ok:
        # Rendering 1080x1920 takes real minutes; finding out afterwards that
        # the result is silent wastes all of them.
        logger.warning(
            "The Short will have NO NARRATION: {}. Fix that first, or accept a "
            "silent placeholder track.", voice_reason,
        )

    if chain["preferred"] == "storyboard":
        logger.warning(
            "No video toolchain found — the Bard will emit a storyboard PNG. "
            "Install ffmpeg (brew install ffmpeg / apt install ffmpeg) for real video."
        )

    bard = BardAgent(settings)
    results = []
    try:
        for index in range(1, count + 1):
            if count > 1:
                logger.info("--- Short {}/{} ---", index, count)
            results.append(await bard.produce(topic))
    finally:
        await bard.aclose()

    print()
    for result in results:
        marker = "\u2713" if result.ok else "\u2717"
        print(f"{marker} {result.summary()}")
        for stage in result.stages:
            print(f"    - {stage}")
        for error in result.errors:
            print(f"    x {error}")
        if result.video_path:
            print(f"    file: {result.video_path}")

    failed = [item for item in results if not item.ok]
    if failed:
        print(f"\n{len(failed)}/{len(results)} short(s) failed.")
        return 1

    undispatched = [item for item in results if not item.dispatched]
    if undispatched:
        print(
            f"\nTelegram is not configured, so nothing was sent.\n"
            f"Approve locally with:  python main.py --approve-short "
            f"{undispatched[0].short_id}"
        )
    return 0


def cmd_list_shorts(limit: int) -> int:
    init_db()
    rows = recent_shorts(limit=limit)
    if not rows:
        print("The Bard has not written anything yet. Run:  python main.py --shorts")
        return 0

    counts = short_counts_by_status()
    print("Shorts by state")
    for status, count in sorted(counts.items()):
        print(f"  {status:<18} {count}")

    print("\nRecent")
    for short in rows:
        flag = " [dry-run]" if short.dry_run else ""
        print(
            f"  {short.summary()}{flag}\n"
            f"      {short.duration_seconds:.0f}s | voice={short.voice_backend or 'n/a'}"
            f" | cut={short.render_backend or 'n/a'}\n"
            f"      {short.video_path or '(no media)'}"
        )
    return 0


async def cmd_approve_short(short_id: int) -> int:
    init_db()
    try:
        ok = await approve_short_from_cli(short_id)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    if not ok:
        logger.error("Could not approve short {} - check its current state.", short_id)
        return 1

    short = get_short(short_id)
    print(f"Short #{short_id} approved and ready to upload.")
    if short:
        print(f"  file: {short.video_path}")
        print(f"  title: {short.title}")
        print(f"  tags: {' '.join('#' + t for t in short.hashtags)}")
    return 0


async def cmd_scout(niche: str | None, count: int, dispatch: bool) -> int:
    """Curate Amazon affiliate recommendations and store them."""
    from core.database import create_deal  # noqa: PLC0415
    from village.dealscout import DealScout  # noqa: PLC0415

    init_db()
    settings = get_settings()

    if not settings.amazon_configured:
        # The links work either way; they simply credit nobody. Worth one line
        # now rather than a month of untracked clicks.
        logger.warning(
            "AMAZON_TRACKING_ID is still the {!r} placeholder — links will work "
            "but earn nothing. Set your Associates tag in .env.",
            settings.amazon_tracking_id,
        )

    scout = DealScout(settings)
    try:
        deals = await scout.find_deals(niche, count)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    finally:
        await scout.aclose()

    crier = None
    if dispatch and settings.telegram_configured:
        from village.town_crier import TownCrier  # noqa: PLC0415

        crier = TownCrier(settings)

    for deal in deals:
        row = create_deal(deal.to_dict(), dry_run=settings.dry_run)
        print()
        print(f"deal #{row.id}  [{deal.category}]  {deal.niche}")
        print(deal.as_markdown())

        if crier is not None:
            sent = await crier.dispatch_deal(row.id, deal)
            print(f"\n  telegram: {'sent for approval' if sent else 'not sent'}")
        elif dispatch:
            print("\n  telegram: not configured — approve with "
                  f"--approve-deal {row.id}")

    return 0


def cmd_deals(limit: int) -> int:
    """List recent curated deals."""
    from core.database import recent_deals  # noqa: PLC0415

    init_db()
    rows = recent_deals(limit)
    if not rows:
        print("No deals yet. Run: python main.py --scout")
        return 0

    print(f"{'ID':>4}  {'STATUS':<17} {'CAT':<10} {'PRICE':<14} PRODUCT")
    for row in rows:
        low, high = row.price_low, row.price_high
        price = (
            f"~${low:,.0f}-${high:,.0f}" if low and high and low != high
            else (f"~${high or low:,.0f}" if (high or low) else "varies")
        )
        flag = " (dry)" if row.dry_run else ""
        print(f"{row.id:>4}  {row.status:<17} {row.category:<10} {price:<14} "
              f"{row.product[:44]}{flag}")
    return 0


def cmd_approve_deal(deal_id: int, reject: bool = False) -> int:
    """Approve or reject a curated deal from the CLI."""
    from core.database import DealStatus, get_deal, update_deal_status  # noqa: PLC0415

    init_db()
    row = get_deal(deal_id)
    if row is None:
        logger.error("No deal with id {}", deal_id)
        return 1

    target = DealStatus.REJECTED if reject else DealStatus.APPROVED
    if row.status_enum is target:
        # Idempotent, but say so rather than implying something changed.
        print(f"Deal #{deal_id} is already {row.status}.")
        return 0

    if not reject and row.status_enum is DealStatus.DRAFTED:
        # Approving straight from the CLI skips the Telegram card, so walk the
        # row through the intermediate state rather than allowing the jump.
        update_deal_status(deal_id, DealStatus.PENDING_APPROVAL, reason="CLI review.")

    updated = update_deal_status(
        deal_id, target, reason="Rejected from the CLI." if reject else "Approved from the CLI."
    )
    if updated is None:
        logger.error("Could not {} deal {} (it is {})",
                     "reject" if reject else "approve", deal_id, row.status)
        return 1

    print(f"Deal #{deal_id} {updated.status}.")
    if not reject:
        print(f"  {updated.affiliate_url}")
    return 0


async def cmd_pin_deal(deal_id: int) -> int:
    """Post one approved deal to Pinterest."""
    from core.database import get_deal  # noqa: PLC0415
    from village.town_crier import TownCrier  # noqa: PLC0415

    init_db()
    settings = get_settings()

    row = get_deal(deal_id)
    if row is None:
        logger.error("No deal with id {}", deal_id)
        return 1
    if row.pinterest_pin_id:
        print(f"Deal #{deal_id} is already pinned: {row.pinterest_url}")
        return 0

    result = await TownCrier(settings).pin_deal(deal_id)
    if result is None:
        from village.pinterest_publisher import PinterestPublisher  # noqa: PLC0415

        _, reason = PinterestPublisher(settings).status()
        logger.error("Pinterest is not available: {}", reason)
        return 1

    if not result.ok:
        logger.error("Pin failed: {}", result.error)
        return 1

    print(f"Deal #{deal_id} {'simulated' if result.simulated else 'pinned'}.")
    if result.url:
        print(f"  {result.url}")
    return 0


def cmd_preview_pin(deal_id: int) -> int:
    """Render a deal's Pin card and print the exact payload, without posting.

    The rehearsal step: everything a real post does except the network call, so
    the card and the copy can be checked before anything is public.
    """
    from core.database import get_deal  # noqa: PLC0415
    from village.dealscout import Deal as DealPayload  # noqa: PLC0415
    from village.pinterest_publisher import (  # noqa: PLC0415
        PIN_SIZE, build_pin_text, render_pin_image, source_pin_background,
    )

    init_db()
    settings = get_settings()

    row = get_deal(deal_id)
    if row is None:
        logger.error("No deal with id {}", deal_id)
        return 1

    deal = DealPayload.from_dict(row.payload)
    destination = settings.storage_dir / "pins" / f"preview-{deal_id}.jpg"

    # Source the same background a real post would, so the preview is a
    # preview and not a different picture.
    art, art_url = asyncio.run(
        source_pin_background(
            deal, destination.with_name(f"preview-{deal_id}-bg.jpg"), settings
        )
    )
    render_pin_image(deal, destination, PIN_SIZE, art)
    title, description = build_pin_text(deal)

    print(f"Pin preview for deal #{deal_id}")
    print(f"  board        {settings.pinterest_board_name or '(unset)'}")
    print(f"  title        {title}")
    print(f"  link         {deal.affiliate_url}")
    print(f"  background   {art.name if art else 'none — plain typographic card'}")
    print(f"  image        {destination}  "
          f"({destination.stat().st_size // 1024} KB)")
    print("  description  |")
    for line in description.splitlines():
        print(f"    {line}")
    if settings.make_webhook_configured:
        # Make cannot map fields it has never seen, so print the exact shape it
        # will receive. The five the Pinterest module needs go first.
        from core.webhooks import build_deal_payload  # noqa: PLC0415

        payload = asyncio.run(
            build_deal_payload(
                deal, deal_id, destination,
                settings.public_url(f"/api/deals/{deal_id}/pin.jpg"),
            )
        )
        primary = ("deal_id", "title", "caption", "image_url", "affiliate_url")
        print("\n  Make webhook payload |")
        for key in primary:
            value = str(payload.get(key, "")).replace("\n", " ")
            print(f"    {key:<16} {value[:76]}")
        print("    " + "-" * 20)
        for key in sorted(set(payload) - set(primary)):
            value = str(payload[key]).replace("\n", " ")
            print(f"    {key:<16} {value[:76]}")

    print("\nNothing was posted. Open the image above, then post it with:")
    print(f"  python main.py --pin-deal {deal_id}")
    return 0


def cmd_pinterest_check() -> int:
    """Verify the Pinterest token and board without posting anything."""
    from village.pinterest_publisher import PinterestPublisher  # noqa: PLC0415

    settings = get_settings()
    publisher = PinterestPublisher(settings)

    print("Pinterest")
    print(f"  token                {'set' if settings.pinterest_access_token.strip() else 'not set'}")
    print(f"  board                {settings.pinterest_board_name or '(unset)'}")
    print(f"  post on approval     {'on' if settings.pinterest_enabled else 'off'}")

    ok, detail = asyncio.run(publisher.check())
    print(f"  status               {'ready' if ok else 'UNAVAILABLE'}")
    print(f"  {detail}")
    return 0 if ok else 1


async def cmd_night_scan() -> int:
    """Read the feeds now, outside the night window."""
    from village.night_scribe import NightScribe  # noqa: PLC0415

    init_db()
    result = await NightScribe(get_settings()).scan()
    print(f"Night Scribe: {result.summary()}")
    for name, count in sorted(result.sources.items()):
        print(f"  {name:<16} {count}")
    for failure in result.errors:
        print(f"  ! {failure[:110]}")
    return 0


async def cmd_ledger(hours: int, dispatch: bool) -> int:
    """Build today's Morning Ledger."""
    from village.morning_ledger import MorningLedger  # noqa: PLC0415

    init_db()
    settings = get_settings()

    if not settings.newsletter_configured:
        logger.warning(
            "No OPENROUTER_API_KEY or ANTHROPIC_API_KEY — the edition will be "
            "assembled from the wire without an editorial pass."
        )

    ledger = MorningLedger(settings)
    try:
        row = await ledger.publish_draft(hours=hours, dispatch=dispatch)
    finally:
        await ledger.aclose()

    print(f"\nEdition #{row.id} [{row.status}] — {row.title}")
    print(f"  {row.publish_date} · {len(row.full_markdown):,} characters")
    print()
    print(row.full_markdown)
    if not dispatch:
        print(f"\nApprove it with: python main.py --approve-edition {row.id}")
    return 0


def cmd_editions(limit: int) -> int:
    """List recent editions."""
    from core.database import recent_editions  # noqa: PLC0415

    init_db()
    rows = recent_editions(limit)
    if not rows:
        print("No editions yet. Run: python main.py --ledger")
        return 0

    print(f"{'ID':>4}  {'DATE':<12} {'STATUS':<10} TITLE")
    for row in rows:
        print(f"{row.id:>4}  {row.publish_date!s:<12} {row.status:<10} {row.title[:52]}")
    return 0


def cmd_show_edition(edition_id: int) -> int:
    """Print one edition in full."""
    from core.database import get_edition  # noqa: PLC0415

    init_db()
    row = get_edition(edition_id)
    if row is None:
        logger.error("No edition with id {}", edition_id)
        return 1
    print(row.full_markdown)
    return 0


async def cmd_approve_edition(edition_id: int, reject: bool = False) -> int:
    """Approve (and schedule) or reject an edition from the CLI."""
    from core.database import EditionStatus, get_edition, update_edition_status  # noqa: PLC0415
    from village.town_crier import _next_publish_time  # noqa: PLC0415

    init_db()
    settings = get_settings()

    row = get_edition(edition_id)
    if row is None:
        logger.error("No edition with id {}", edition_id)
        return 1

    if reject:
        updated = update_edition_status(
            edition_id, EditionStatus.REJECTED, reason="Rejected from the CLI."
        )
        if updated is None:
            logger.error("Could not reject edition {} (it is {})", edition_id, row.status)
            return 1
        print(f"Edition #{edition_id} REJECTED.")
        return 0

    when = _next_publish_time(settings.ledger_publish_time)
    updated = update_edition_status(
        edition_id, EditionStatus.APPROVED,
        reason=f"Approved from the CLI; scheduled for {when:%Y-%m-%d %H:%M}.",
        scheduled_for=when,
    )
    if updated is None:
        logger.error("Could not approve edition {} (it is {})", edition_id, row.status)
        return 1

    print(f"Edition #{edition_id} APPROVED, scheduled for {when:%H:%M on %d %B}.")
    print("  The daemon delivers it; leave it running, or send it now with:")
    print(f"    python main.py --send-edition {edition_id}")
    return 0


async def cmd_send_edition(edition_id: int) -> int:
    """Deliver an approved edition immediately."""
    from village.town_crier import TownCrier  # noqa: PLC0415

    init_db()
    if await TownCrier(get_settings()).publish_edition(edition_id):
        print(f"Edition #{edition_id} delivered.")
        return 0
    logger.error("Edition {} was not delivered — is it APPROVED?", edition_id)
    return 1


async def cmd_overseer() -> int:
    """Run the Overseer's economic review now."""
    from village.village_overseer import VillageOverseer  # noqa: PLC0415

    init_db()
    review = await VillageOverseer(get_settings()).run_daily()

    print("Village Overseer")
    print(f"  subscribers      {review.subscribers:,}")
    print(f"  open rate        {review.open_rate:.1%}")
    print(f"  click rate       {review.click_rate:.1%}")
    print(f"  revenue          ${review.revenue_usd:,.2f}")
    print(f"  qualifying run   {review.streak_days} day(s)")
    print(f"  sponsor ready    {'yes' if review.sponsor_ready else 'no'}")
    print(f"  paid tier ready  {'yes' if review.paid_ready else 'no'}")
    print(f"  recommendation   {review.recommendation}")
    for note in review.notes:
        print(f"    · {note}")
    return 0


def cmd_record_metrics(values: list[str]) -> int:
    """Record today's subscriber snapshot: SUBSCRIBERS OPEN_RATE [CLICK_RATE] [REVENUE]."""
    from core.database import record_village_metrics  # noqa: PLC0415

    init_db()
    try:
        subscribers = int(values[0])
        open_rate = float(values[1])
        click_rate = float(values[2]) if len(values) > 2 else 0.0
        revenue = float(values[3]) if len(values) > 3 else 0.0
    except (IndexError, ValueError):
        logger.error(
            "Usage: --record-metrics SUBSCRIBERS OPEN_RATE [CLICK_RATE] [REVENUE_USD]"
        )
        return 1

    # Percentages are entered either way round in practice; 45 means 45%.
    if open_rate > 1:
        open_rate /= 100
    if click_rate > 1:
        click_rate /= 100

    metric = record_village_metrics(
        total_subscribers=subscribers, open_rate=open_rate,
        click_rate=click_rate, total_revenue_usd=revenue,
    )
    print(f"Recorded {metric.recorded_date}: {metric.total_subscribers:,} subscribers, "
          f"{metric.open_rate:.1%} open, ${metric.total_revenue_usd:,.2f}")
    return 0


def cmd_youtube_auth() -> int:
    """Grant YouTube upload consent once, interactively.

    Kept separate from every other command on purpose: this is the only place
    allowed to open a browser. A daemon or a Telegram callback that could block
    on a consent screen would simply hang.
    """
    from village.youtube_publisher import YouTubePublisher  # noqa: PLC0415

    settings = get_settings()
    publisher = YouTubePublisher(settings)

    if not publisher.configured:
        logger.error(
            "No OAuth client secret at {}. Download one from Google Cloud "
            "(APIs & Services > Credentials > OAuth client ID > Desktop app) "
            "and save it there.",
            settings.youtube_client_secret_path,
        )
        return 1

    if publisher.authorized:
        print(f"Already authorised. Token: {settings.youtube_token_path}")
        print("Re-running will replace it with fresh consent.")

    print("A browser window will open for Google sign-in.")
    print("Sign in as the account that owns the YouTube channel you publish to.\n")

    if not publisher.authorize():
        return 1

    print(f"\nAuthorised. Token cached at {settings.youtube_token_path} (mode 600).")
    if not settings.youtube_upload_enabled:
        print("\nUploading is still OFF. To publish on approval, set:")
        print("  YOUTUBE_UPLOAD_ENABLED=true")
    print(f"Uploads will be {settings.youtube_privacy_status.upper()} "
          "(change with YOUTUBE_PRIVACY_STATUS).")
    return 0


async def cmd_upload_short(short_id: int) -> int:
    """Upload an already-approved Short, or retry one that failed."""
    from village.town_crier import TownCrier  # noqa: PLC0415

    init_db()
    settings = get_settings()

    short = get_short(short_id)
    if short is None:
        logger.error("No short with id {}", short_id)
        return 1
    if not short.video_path or not Path(short.video_path).is_file():
        logger.error("Short {} has no video file on disk", short_id)
        return 1

    from village.youtube_publisher import YouTubePublisher  # noqa: PLC0415

    usable, reason = YouTubePublisher(settings).status()
    if not usable and not settings.dry_run:
        logger.error("Cannot upload: {}", reason)
        return 1

    result = await TownCrier(settings).upload_to_youtube(short_id)
    if not result.ok:
        logger.error("Upload failed: {}", result.error)
        return 1

    print(f"Short #{short_id} {'simulated' if result.simulated else 'published'}.")
    print(f"  {result.watch_url or result.url}")
    if result.privacy_status != "public" and not result.simulated:
        print(f"  visibility: {result.privacy_status} — not publicly visible yet")
    return 0


def cmd_reject_short(short_id: int) -> int:
    init_db()
    if reject_short_from_cli(short_id):
        print(f"Short #{short_id} rejected.")
        return 0
    logger.error("Could not reject short {} - check its current state.", short_id)
    return 1


def cmd_treasury() -> int:
    """Revenue by channel, plus the most recent ledger lines."""
    init_db()
    settings = get_settings()
    counted = settings.count_dry_run_revenue

    start, end = month_bounds()
    month = revenue_by_channel(since=start, until=end, include_dry_run=counted)
    lifetime = revenue_by_channel(include_dry_run=counted)
    youtube = youtube_metrics(include_dry_run=counted)

    total_month = sum(row["cents"] for row in month.values())
    total_life = sum(row["cents"] for row in lifetime.values())
    simulated = sum(row["simulatedCents"] for row in lifetime.values())

    money = lambda cents: f"${cents / 100:,.2f}"  # noqa: E731 - display only

    print(f"TREASURY - {start.strftime('%B %Y')}\n")
    print(f"  {'Channel':<26}{'This month':>13}{'Lifetime':>13}")
    print(f"  {'-' * 52}")
    for channel in Channel:
        meta = CHANNEL_META[channel.value]
        print(
            f"  {meta['icon']} {meta['label']:<23}"
            f"{money(month[channel.value]['cents']):>13}"
            f"{money(lifetime[channel.value]['cents']):>13}"
        )
    print(f"  {'-' * 52}")
    print(f"  {'TOTAL':<26}{money(total_month):>13}{money(total_life):>13}")

    if simulated and not counted:
        print(
            f"\n  {money(simulated)} of dry-run activity is recorded but NOT counted."
            "\n  Nothing was actually sold. Set COUNT_DRY_RUN_REVENUE=true to include it."
        )

    if youtube["published"]:
        print(
            f"\n  YouTube: {youtube['published']} live, {youtube['views']:,} views recorded, "
            f"{youtube['rpmCents']:.0f}c average RPM (estimates, not confirmed payouts)"
        )
    elif youtube["simulatedPublished"]:
        print(
            f"\n  YouTube: {youtube['simulatedPublished']} simulated Short(s) — "
            "none uploaded, so no views and no revenue."
        )

    entries = recent_revenue(limit=10)
    if entries:
        print("\n  Recent ledger lines")
        for entry in entries:
            flag = " [est]" if entry.estimated else ""
            dry = " [dry-run]" if entry.dry_run else ""
            when = entry.occurred_at.strftime("%Y-%m-%d") if entry.occurred_at else "?"
            print(
                f"    {when}  {CHANNEL_META.get(entry.channel, {}).get('icon', '?')} "
                f"{money(entry.amount_cents):>10}{flag}{dry}  {entry.note[:44]}"
            )
    else:
        print("\n  The ledger is empty. Publish a listing or a Short to open it.")
    return 0


def cmd_record_views(short_id: int, views: int, rpm: int | None) -> int:
    """Log a Short's view count and re-estimate its payout."""
    init_db()
    settings = get_settings()
    short = record_short_metrics(short_id, views, rpm if rpm is not None else settings.youtube_rpm_cents)
    if short is None:
        logger.error("No short {}", short_id)
        return 1
    print(
        f"Short #{short.id}: {short.views:,} views @ {short.rpm_cents}c RPM "
        f"-> ${short.estimated_cents / 100:,.2f} estimated"
    )
    print("  This is an estimate. Real payouts depend on watch time and the ad market.")
    return 0


def cmd_record_revenue(amount_cents: int, channel: str, note: str) -> int:
    """Post a line to the ledger by hand."""
    init_db()
    try:
        target = Channel(channel)
    except ValueError:
        logger.error("channel must be one of %s", [c.value for c in Channel])
        return 2

    entry = record_revenue(
        target, amount_cents, note=note, dry_run=get_settings().dry_run
    )
    meta = CHANNEL_META[target.value]
    print(f"{meta['icon']} {meta['label']}: +${entry.amount_cents / 100:,.2f}  {entry.note}")
    return 0


def cmd_web(host: str, port: int, reload: bool) -> int:
    """Serve the living village dashboard."""
    init_db()
    try:
        from web.app import serve
    except ImportError as exc:
        logger.error(
            "The dashboard needs FastAPI and uvicorn: pip install -r requirements.txt ({})", exc
        )
        return 2

    print(f"\n  Oakhaven is open at http://{host}:{port}\n")
    try:
        serve(host=host, port=port, reload=reload)
    except KeyboardInterrupt:  # pragma: no cover - operator interrupt
        logger.info("Dashboard stopped.")
    return 0


def cmd_status() -> int:
    """The whole village at a glance: who lives here and what each has made.

    This used to return early when there were no listings, which meant a
    village busy producing Shorts and curating deals reported itself as empty.
    Every pipeline now gets a line whether or not it has produced anything.
    """
    from core.database import deal_counts_by_status, deal_metrics  # noqa: PLC0415
    from core.database import recent_deals, recent_shorts  # noqa: PLC0415
    from core.database import short_counts_by_status  # noqa: PLC0415
    from core.roster import VILLAGERS  # noqa: PLC0415

    init_db()
    settings = get_settings()

    print("Oakhaven — village overview")
    print(f"  mode  {'DRY RUN' if settings.dry_run else 'LIVE'}")

    # ---- citizens -----------------------------------------------------------
    print("\nCitizens")
    def cadence(times: str, interval: float, on: bool) -> str:
        if not on:
            return ""
        return times or (f"every {interval:g}h" if interval else "")

    schedules = {
        "bard": cadence(
            settings.shorts_schedule_hours,
            settings.shorts_interval_hours,
            settings.shorts_scheduled,
        ),
        "dealscout": cadence(
            settings.scout_schedule_hours,
            settings.scout_interval_hours,
            settings.scout_scheduled,
        ),
    }
    for villager in VILLAGERS.values():
        note = schedules.get(villager.id) or ""
        suffix = f"  ⏰ {note}" if note else ""
        print(f"  {villager.emoji}  {villager.name:<24} {villager.title}{suffix}")

    # ---- listings -----------------------------------------------------------
    listing_counts = counts_by_status()
    print("\nListings (Etsy / print-on-demand)")
    if listing_counts:
        for status in ListingStatus:
            count = listing_counts.get(status.value, 0)
            if count:
                print(f"  {status.value:<18} {count}")
        for listing in recent_listings(limit=3):
            print(f"    · {listing.summary()}")
    else:
        print("  none yet — python main.py --generate 1")

    # ---- shorts -------------------------------------------------------------
    short_counts = short_counts_by_status()
    print("\nShorts (YouTube)")
    if short_counts:
        for status, count in sorted(short_counts.items()):
            print(f"  {status:<18} {count}")
        for short in recent_shorts(limit=3):
            flag = " [dry]" if short.dry_run else ""
            print(f"    · #{short.id} {short.title[:48]}{flag}")
    else:
        print("  none yet — python main.py --shorts")

    # ---- deals --------------------------------------------------------------
    counts = deal_counts_by_status()
    print("\nDeals (Amazon affiliate)")
    if counts:
        for status, count in sorted(counts.items()):
            print(f"  {status:<18} {count}")

        metrics = deal_metrics(include_dry_run=settings.count_dry_run_revenue)
        if metrics["clicks"]:
            print(
                f"  {'reported':<18} {metrics['clicks']:,} clicks · "
                f"${metrics['earnings']:,.2f} · EPC ${metrics['earningsPerClick']:.4f}"
            )
        pinned = sum(1 for d in recent_deals(limit=100) if d.pinterest_pin_id)
        if pinned:
            print(f"  {'pinned':<18} {pinned}")
        for deal in recent_deals(limit=3):
            flag = " [dry]" if deal.dry_run else ""
            pin = " 📌" if deal.pinterest_pin_id else ""
            print(f"    · #{deal.id} {deal.product[:48]}{flag}{pin}")

        if not settings.amazon_configured:
            print(f"  ⚠ AMAZON_TRACKING_ID is still {settings.amazon_tracking_id!r} "
                  "— links earn nothing")
    else:
        print("  none yet — python main.py --scout")

    return 0


def cmd_list(limit: int) -> int:
    init_db()
    rows = recent_listings(limit=limit)
    if not rows:
        print("No listings yet.")
        return 0
    for listing in rows:
        flag = " [dry-run]" if listing.dry_run else ""
        print(f"{listing.summary()}{flag}")
    return 0


def cmd_show(listing_id: int) -> int:
    init_db()
    listing = get_listing(listing_id)
    if listing is None:
        logger.error("No listing {}", listing_id)
        return 1

    data: dict[str, Any] = listing.to_dict()
    print(f"Listing #{listing.id}")
    for key in (
        "status", "status_reason", "niche", "audience", "concept", "title",
        "price_cents", "raw_image_path", "processed_image_path",
        "printify_product_id", "external_url", "dry_run", "created_at",
    ):
        print(f"  {key:<22} {data.get(key)}")
    print(f"  {'tags':<22} {', '.join(listing.tags)}")
    print("\nGuard report")
    print("  " + (listing.guard_report or "n/a").replace("\n", "\n  "))
    print("\nDescription")
    print("  " + (listing.description or "n/a").replace("\n", "\n  "))
    return 0


def cmd_check() -> int:
    """Validate the configuration and report what is reachable."""
    settings = get_settings()
    init_db()

    print("Configuration")
    for key, value in Mayor(settings).describe_configuration().items():
        print(f"  {key:<14} {value}")

    print("\n.env discovery")
    if settings.env_files:
        # Highest priority last, matching how the files are actually applied.
        for index, path in enumerate(settings.env_files, start=1):
            rank = "lowest" if index == 1 and len(settings.env_files) > 1 else (
                "highest" if index == len(settings.env_files) and len(settings.env_files) > 1
                else "only"
            )
            print(f"  {index}. {path}  ({rank} priority)")
    else:
        print("  none found — using real environment variables only")
        print(f"  searched: {settings.env_files_searched_hint()}")

    # Credentials are reported as present or absent, never echoed.
    print("\nCredentials")
    for label, present in (
        ("OPENROUTER_API_KEY", settings.openrouter_configured),
        ("TELEGRAM_BOT_TOKEN", bool(settings.telegram_bot_token.strip())),
        ("TELEGRAM_CHAT_ID", bool(settings.telegram_chat_id.strip())),
        ("PRINTIFY", settings.printify_configured),
    ):
        print(f"  {label:<20} {'set' if present else 'not set'}")

    # The voice is a package, not a key, so it fails in a different way from
    # everything above: nothing is missing from .env, the video just comes out
    # silent. Report it next to the credentials it is easily mistaken for.
    voice_ok, voice_reason = settings.tts_status()
    print("\nVoice")
    print(f"  provider             {settings.tts_provider}")
    print(f"  voice                {settings.tts_voice}")
    print(f"  status               {'ready' if voice_ok else 'UNAVAILABLE'}")
    if not voice_ok:
        print(f"  reason               {voice_reason}")
    # The override is worth naming here: the voice is the single most audible
    # choice in the whole pipeline and there is no other place it is written down.
    others = [v for v in Settings.SUGGESTED_VOICES if v != settings.tts_voice]
    print(f"  override with        BARD_VOICE=... in .env")
    print(f"  others              {', '.join(others[:3])}")

    print("\nVisuals")
    print(f"  scenes per short     {settings.shorts_scene_count}")
    print(f"  frame                {settings.video_width}x{settings.video_height} @ {settings.video_fps}fps")
    print(f"  ken burns zoom       {settings.video_zoom:.2f}")
    from core.scene_art import SceneArtist  # noqa: PLC0415

    artist = SceneArtist(settings)
    chain = artist.describe_chain()
    print(f"  scene art            {' -> '.join(chain)}")
    if chain[0].startswith("pollinations"):
        # Worth saying before someone waits four minutes wondering.
        print("                       (free tier: ~45s per scene, 576x1024, upscaled)")

    # Verify the image model BEFORE a render rather than 404ing on every scene.
    if settings.openrouter_configured and any(c.startswith("openrouter") for c in chain):
        model_ok, model_why = asyncio.run(artist.validate_openrouter_model())
        print(f"  image model          {settings.image_model} — "
              f"{'ok' if model_ok else 'UNUSABLE'}")
        if not model_why.startswith("ok"):
            print(f"                       {model_why}")
    else:
        model_ok = True
    transition = settings.video_transition_seconds
    print(f"  scene transition     {f'{transition:.2f}s cross-dissolve' if transition else 'hard cut'}")
    if settings.stock_video_configured:
        print(f"  stock b-roll         Pexels, up to {settings.stock_video_max_clips} clip(s) per short")
    else:
        print("  stock b-roll         off (set PEXELS_API_KEY to enable)")

    from village.youtube_publisher import YouTubePublisher  # noqa: PLC0415

    publisher = YouTubePublisher(settings)
    upload_ok, upload_reason = publisher.status()
    print("\nYouTube")
    print(f"  client secret        {'found' if publisher.configured else 'missing'} "
          f"({settings.youtube_client_secret_path})")
    print(f"  authorised           {'yes' if publisher.authorized else 'no'}")
    print(f"  upload on approval   {'on' if settings.youtube_upload_enabled else 'off'}")
    print(f"  visibility           {settings.youtube_privacy_status}")
    print(f"  status               {'ready' if upload_ok else upload_reason}")

    print("\nPinterest")
    if settings.make_webhook_configured:
        # Say which route is live: two are configured and only one will fire.
        print("  route                Make.com webhook (used instead of the direct API)")
        host = settings.make_pinterest_webhook_url.split("/")[2:3]
        print(f"  webhook host         {host[0] if host else '(unparsed)'}")
    else:
        print("  route                direct Pinterest API v5")
    print(f"  token                {'set' if settings.pinterest_access_token.strip() else 'not set'}")
    print(f"  board                {settings.pinterest_board_name or '(unset)'}")
    print(f"  post on approval     {'on' if settings.pinterest_enabled or settings.make_webhook_configured else 'off'}")
    if settings.pinterest_enabled and not settings.make_webhook_configured:
        print("  verify with          python main.py --pinterest-check")

    problems: list[str] = []
    if settings.pinterest_enabled and not settings.pinterest_configured:
        problems.append(
            "Pinterest posting is on but the token or board name is missing."
        )
    if not model_ok:
        problems.append(f"IMAGE_MODEL is unusable — {model_why}")
    if settings.youtube_upload_enabled and not upload_ok:
        problems.append(f"YouTube uploading is on but not usable — {upload_reason}.")
    if not voice_ok:
        problems.append(f"Shorts will be silent — {voice_reason}.")
    if not settings.dry_run and not settings.printify_configured:
        problems.append("DRY_RUN is off but Printify is not configured (will simulate).")
    if not settings.openrouter_configured:
        problems.append("OPENROUTER_API_KEY is unset (Scout/Scribe/Crafter will use fallbacks).")
    if not settings.telegram_configured:
        problems.append("Telegram is unset (approvals must go through the CLI).")

    print("\nFindings")
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
    else:
        print("  none — every integration is configured.")
    print("\nThe pipeline is runnable either way; unset integrations degrade to simulation.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Autonomous Agent Village — Etsy POD listings with human approval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --generate 1\n"
            "  python main.py --generate 3 --hint 'gardening'\n"
            "  python main.py --bot\n"
            "  python main.py --daemon --interval 1800\n"
            "  python main.py --shorts\n"
            "  python main.py --shorts 3 --topic 'deep sea discoveries'\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", type=int, metavar="N", help="create N listings")
    mode.add_argument("--bot", action="store_true", help="run the Telegram approval worker")
    mode.add_argument("--daemon", action="store_true", help="generate on a schedule and serve the bot")
    mode.add_argument("--approve", type=int, metavar="ID", help="approve and publish a listing")
    mode.add_argument("--reject", type=int, metavar="ID", help="reject a listing")
    mode.add_argument("--status", action="store_true", help="counts by state")
    mode.add_argument("--list", type=int, nargs="?", const=20, metavar="N", help="recent listings")
    mode.add_argument("--show", type=int, metavar="ID", help="everything about one listing")
    mode.add_argument("--check", action="store_true", help="validate the configuration")
    mode.add_argument("--init-db", action="store_true", help="create the tables and exit")
    mode.add_argument("--web", action="store_true", help="serve the village dashboard")
    mode.add_argument(
        "--shorts", type=int, nargs="?", const=1, metavar="N",
        help="produce N faceless YouTube Shorts (default 1)",
    )
    mode.add_argument("--list-shorts", type=int, nargs="?", const=10, metavar="N",
                      help="the Bard's recent videos")
    mode.add_argument("--approve-short", type=int, metavar="ID", help="mark a Short ready to upload")
    mode.add_argument("--reject-short", type=int, metavar="ID", help="reject a Short")
    mode.add_argument(
        "--scout", action="store_true",
        help="curate an Amazon affiliate recommendation (see --niche)",
    )
    mode.add_argument(
        "--deals", action="store_true", help="list recent curated deals",
    )
    mode.add_argument(
        "--approve-deal", type=int, metavar="ID", help="approve a curated deal",
    )
    mode.add_argument(
        "--reject-deal", type=int, metavar="ID", help="reject a curated deal",
    )
    mode.add_argument(
        "--pin-deal", type=int, metavar="ID", help="post an approved deal to Pinterest",
    )
    mode.add_argument(
        "--preview-pin", type=int, metavar="ID",
        help="render a deal's Pin card and print the payload WITHOUT posting",
    )
    mode.add_argument("--ledger", action="store_true", help="build today's Morning Ledger")
    mode.add_argument("--night-scan", action="store_true", help="read the crypto feeds now")
    mode.add_argument("--editions", action="store_true", help="list recent editions")
    mode.add_argument("--show-edition", type=int, metavar="ID", help="print one edition")
    mode.add_argument("--approve-edition", type=int, metavar="ID", help="approve and schedule an edition")
    mode.add_argument("--reject-edition", type=int, metavar="ID", help="reject an edition")
    mode.add_argument("--send-edition", type=int, metavar="ID", help="deliver an approved edition now")
    mode.add_argument("--overseer", action="store_true", help="run the economic review")
    mode.add_argument(
        "--record-metrics", nargs="+", metavar="N",
        help="record today's metrics: SUBSCRIBERS OPEN_RATE [CLICK_RATE] [REVENUE]",
    )
    mode.add_argument(
        "--pinterest-check", action="store_true",
        help="verify the Pinterest token and board without posting",
    )
    mode.add_argument(
        "--youtube-auth", action="store_true",
        help="grant YouTube upload consent (opens a browser, once)",
    )
    mode.add_argument(
        "--upload-short", type=int, metavar="ID",
        help="upload an approved Short to YouTube, or retry a failed upload",
    )
    mode.add_argument("--treasury", action="store_true", help="revenue by channel")
    mode.add_argument(
        "--record-views", type=int, nargs=2, metavar=("ID", "VIEWS"),
        help="log a Short's views and re-estimate its payout",
    )
    mode.add_argument(
        "--record-revenue", type=int, metavar="CENTS",
        help="post a ledger line by hand (see --channel)",
    )

    parser.add_argument("--hint", type=str, default=None, help="steer the Scout toward a theme")
    parser.add_argument("--topic", type=str, default=None, help="steer the Bard at a subject")
    parser.add_argument("--channel", type=str, default="digital", help="ledger channel")
    parser.add_argument("--note", type=str, default="", help="ledger line note")
    parser.add_argument("--rpm", type=int, default=None, help="RPM in cents for --record-views")
    parser.add_argument("--interval", type=int, default=None, help="daemon seconds between passes")
    parser.add_argument("--batch", type=int, default=None, help="daemon listings per pass")
    parser.add_argument(
        "--niche", type=str, default=None,
        help="what --scout researches, e.g. --niche \"desk setup\"",
    )
    parser.add_argument("--count", type=int, default=1, help="how many deals --scout curates")
    parser.add_argument("--hours", type=int, default=12, help="lookback window for --ledger")
    parser.add_argument("--limit", type=int, default=10, help="rows shown by --deals")
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="print a curated deal instead of sending it for approval",
    )
    parser.add_argument("--log-level", type=str, default=None, help="TRACE..CRITICAL")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="dashboard bind address")
    parser.add_argument("--port", type=int, default=8000, help="dashboard port")
    parser.add_argument("--reload", action="store_true", help="dashboard autoreload (dev)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(args.log_level.upper() if args.log_level else settings.log_level)

    if args.init_db:
        init_db()
        print(f"Database ready: {settings.database_url}")
        return 0
    if args.check:
        return cmd_check()
    if args.status:
        return cmd_status()
    if args.list is not None:
        return cmd_list(args.list)
    if args.show is not None:
        return cmd_show(args.show)
    if args.reject is not None:
        return cmd_reject(args.reject)
    if args.list_shorts is not None:
        return cmd_list_shorts(args.list_shorts)
    if args.reject_short is not None:
        return cmd_reject_short(args.reject_short)
    if args.treasury:
        return cmd_treasury()
    if args.record_views is not None:
        return cmd_record_views(args.record_views[0], args.record_views[1], args.rpm)
    if args.record_revenue is not None:
        return cmd_record_revenue(args.record_revenue, args.channel, args.note)
    if args.bot:
        return cmd_bot()
    if args.web:
        return cmd_web(args.host, args.port, args.reload)

    if args.approve is not None:
        return asyncio.run(cmd_approve(args.approve))
    if args.approve_short is not None:
        return asyncio.run(cmd_approve_short(args.approve_short))

    if args.scout:
        return asyncio.run(cmd_scout(args.niche, args.count, not args.no_telegram))

    if args.deals:
        return cmd_deals(args.limit)

    if args.approve_deal is not None:
        return cmd_approve_deal(args.approve_deal)

    if args.reject_deal is not None:
        return cmd_approve_deal(args.reject_deal, reject=True)

    if args.pin_deal is not None:
        return asyncio.run(cmd_pin_deal(args.pin_deal))

    if args.preview_pin is not None:
        return cmd_preview_pin(args.preview_pin)

    if args.ledger:
        return asyncio.run(cmd_ledger(args.hours, not args.no_telegram))

    if args.night_scan:
        return asyncio.run(cmd_night_scan())

    if args.editions:
        return cmd_editions(args.limit)

    if args.show_edition is not None:
        return cmd_show_edition(args.show_edition)

    if args.approve_edition is not None:
        return asyncio.run(cmd_approve_edition(args.approve_edition))

    if args.reject_edition is not None:
        return asyncio.run(cmd_approve_edition(args.reject_edition, reject=True))

    if args.send_edition is not None:
        return asyncio.run(cmd_send_edition(args.send_edition))

    if args.overseer:
        return asyncio.run(cmd_overseer())

    if args.record_metrics:
        return cmd_record_metrics(args.record_metrics)

    if args.pinterest_check:
        return cmd_pinterest_check()

    if args.youtube_auth:
        return cmd_youtube_auth()

    if args.upload_short is not None:
        return asyncio.run(cmd_upload_short(args.upload_short))
    if args.shorts is not None:
        if args.shorts < 1:
            logger.error("--shorts needs a positive count")
            return 2
        return asyncio.run(cmd_shorts(args.topic, args.shorts))
    if args.generate is not None:
        if args.generate < 1:
            logger.error("--generate needs a positive count")
            return 2
        return asyncio.run(cmd_generate(args.generate, args.hint))
    if args.daemon:
        return asyncio.run(
            cmd_daemon(
                args.interval or settings.daemon_interval_seconds,
                args.batch or settings.daemon_batch_size,
                args.hint,
            )
        )

    build_parser().print_help()
    return 2


def _run() -> NoReturn:
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover - operator interrupt
        logger.info("Interrupted.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    _run()
