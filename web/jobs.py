"""Background jobs the dashboard's action buttons trigger.

Kept out of :mod:`web.app` so the HTTP layer never imports the pipeline (and its
heavy image dependencies) at module load. Every job publishes to the event bus
as it goes, which is what animates the villagers on the canvas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import get_settings
from core import events
from core.database import (
    ListingStatus,
    ShortStatus,
    get_listing,
    get_short,
    list_by_status,
    recent_listings,
    recent_shorts,
    update_listing,
    update_short_status,
    update_status,
)


async def run_pipeline(count: int = 1, hint: str | None = None) -> dict[str, Any]:
    """A full village pass — what ``--generate`` does, from the dashboard."""
    from village.mayor import Mayor

    mayor = Mayor(get_settings())
    try:
        results = await mayor.run_batch(count, hint)
    finally:
        await mayor.aclose()

    created = [item.listing_id for item in results if item.ok]
    return {"created": created, "failed": len(results) - len(created)}


async def force_scan(hint: str | None = None, **_: Any) -> dict[str, Any]:
    """Rowan hunts a niche on demand, without building a listing.

    Useful on its own: the operator sees what the Scout would pick before
    committing an image generation to it.
    """
    from village.scout import Scout

    scout = Scout(get_settings())
    events.agent_working("scout", "Climbing the tower for an unscheduled sweep…", progress=0.2)
    try:
        brief = await scout.find_niche(hint)
    finally:
        await scout.aclose()

    events.agent_done("scout", f"Spotted: {brief.niche}")
    events.agent_output(
        "scout",
        "brief",
        niche=brief.niche,
        audience=brief.audience,
        concept=brief.concept,
        prompt=brief.art_prompt,
        keywords=brief.keywords,
        source=brief.source,
    )
    return brief.to_dict()


async def reroll_design(listing_id: int | None = None, **_: Any) -> dict[str, Any]:
    """Bram re-renders the artwork for a listing and re-runs the print prep."""
    from core.image_processor import process_image
    from village.crafter import Crafter

    listing = _resolve_listing(listing_id)
    if listing is None:
        events.toast("No listing to reroll.", "warn")
        return {"ok": False, "reason": "no listing"}

    settings = get_settings()
    crafter = Crafter(settings)
    events.agent_working(
        "crafter",
        f"Reworking the design for #{listing.id}…",
        listing_id=listing.id,
        progress=0.4,
    )
    try:
        crafted = await crafter.craft(listing.art_prompt, label=f"{listing.niche}-reroll")
    finally:
        await crafter.aclose()

    import asyncio

    processed = await asyncio.to_thread(
        process_image, crafted.path, settings.storage_dir / f"{crafted.path.stem}-print.png"
    )
    update_listing(
        listing.id,
        raw_image_path=str(crafted.path),
        processed_image_path=str(processed.path),
    )

    events.agent_done("crafter", f"Reforged the artwork for #{listing.id}.")
    events.agent_output(
        "crafter",
        "artwork",
        path=str(processed.path),
        image_url=f"/api/assets/{processed.path.name}",
        simulated=crafted.simulated,
        model=crafted.model,
        prompt=crafted.prompt[:400],
    )
    _broadcast_listing(listing.id)
    return {"ok": True, "listing_id": listing.id, "path": str(processed.path)}


async def rewrite_copy(listing_id: int | None = None, **_: Any) -> dict[str, Any]:
    """Lyra rewrites the title, tags and description for a listing."""
    from village.scribe import Scribe

    listing = _resolve_listing(listing_id)
    if listing is None:
        events.toast("No listing to rewrite.", "warn")
        return {"ok": False, "reason": "no listing"}

    scribe = Scribe(get_settings())
    events.agent_working(
        "scribe", f"Re-inking the copy for #{listing.id}…", listing_id=listing.id, progress=0.5
    )
    try:
        copy = await scribe.write(
            niche=listing.niche, audience=listing.audience, concept=listing.concept
        )
    finally:
        await scribe.aclose()

    update_listing(
        listing.id, title=copy.title, description=copy.description, tags=copy.tags
    )
    events.agent_done("scribe", f"Rewrote #{listing.id}: {copy.title[:60]}…")
    events.agent_output(
        "scribe",
        "copy",
        title=copy.title,
        tags=copy.tags,
        description=copy.description[:600],
        source=copy.source,
        repairs=copy.repairs,
    )
    _broadcast_listing(listing.id)
    return {"ok": True, "listing_id": listing.id, "title": copy.title}


async def rescreen(listing_id: int | None = None, **_: Any) -> dict[str, Any]:
    """Garrison re-runs the full Guard validation on a listing."""
    from village.guard import Guard

    listing = _resolve_listing(listing_id)
    if listing is None:
        events.toast("No listing to screen.", "warn")
        return {"ok": False, "reason": "no listing"}

    events.agent_working(
        "guard", f"Re-checking #{listing.id} at the gate…", listing_id=listing.id, progress=0.6
    )
    report = Guard(get_settings()).validate(
        title=listing.title,
        tags=listing.tags,
        description=listing.description,
        image_path=listing.processed_image_path or listing.raw_image_path,
        art_prompt=listing.art_prompt,
        niche=listing.niche,
    )
    update_listing(listing.id, guard_report=report.render())

    if report.ok:
        events.agent_done("guard", f"#{listing.id} still passes.")
    else:
        events.agent_error("guard", f"#{listing.id} now fails.", report.summary)
    events.agent_output(
        "guard",
        "screen",
        ok=report.ok,
        summary=report.summary,
        errors=report.errors,
        warnings=report.warnings,
    )
    _broadcast_listing(listing.id)
    return {"ok": report.ok, "summary": report.summary}


async def dispatch_pending(**_: Any) -> dict[str, Any]:
    """Pippin sends every pending listing to Telegram."""
    from village.town_crier import TownCrier

    pending = list_by_status(ListingStatus.PENDING_APPROVAL, limit=10)
    if not pending:
        events.toast("Nothing is waiting to be announced.", "info")
        return {"dispatched": 0}

    crier = TownCrier(get_settings())
    delivered = 0
    for index, listing in enumerate(pending, start=1):
        events.agent_working(
            "crier",
            f"Announcing #{listing.id} ({index}/{len(pending)})…",
            listing_id=listing.id,
            progress=index / len(pending),
        )
        if await crier.dispatch(listing.id):
            delivered += 1

    events.agent_done("crier", f"Announced {delivered}/{len(pending)} listing(s).")
    return {"dispatched": delivered, "pending": len(pending)}


async def publish_pending(**_: Any) -> dict[str, Any]:
    """Oakhaven publishes every approved listing.

    Honours dry-run exactly as the CLI does — the Merchant simulates when the
    village is not live, so this button is safe to press with no store.
    """
    from village.merchant import Merchant

    approved = list_by_status(ListingStatus.APPROVED, limit=10)
    if not approved:
        events.toast("No approved listings to trade.", "info")
        return {"published": 0}

    merchant = Merchant(get_settings())
    published = 0
    for index, listing in enumerate(approved, start=1):
        events.agent_working(
            "merchant",
            f"Trading #{listing.id} ({index}/{len(approved)})…",
            listing_id=listing.id,
            progress=index / len(approved),
        )
        result = await merchant.publish(
            title=listing.title,
            description=listing.description,
            tags=listing.tags,
            image_path=listing.processed_image_path or listing.raw_image_path or "",
            price_cents=listing.price_cents,
        )
        if result.ok:
            published += 1
            update_status(
                listing.id,
                ListingStatus.PUBLISHED,
                reason=result.summary(),
                printify_image_id=result.image_id,
                printify_product_id=result.product_id,
                external_url=result.external_url,
                dry_run=1 if result.simulated else 0,
            )
        else:
            update_status(listing.id, ListingStatus.FAILED, reason=result.summary())
        events.agent_output(
            "merchant",
            "publish",
            listing_id=listing.id,
            ok=result.ok,
            simulated=result.simulated,
            product_id=result.product_id,
            url=result.external_url,
            error=result.error,
        )
        _broadcast_listing(listing.id)

    events.agent_done("merchant", f"Traded {published}/{len(approved)} listing(s).")
    return {"published": published, "approved": len(approved)}


async def approve_listing(listing_id: int, **_: Any) -> dict[str, Any]:
    """Approve and publish one listing from the dashboard."""
    from village.town_crier import approve_from_cli

    events.agent_working("merchant", f"Taking #{listing_id} to market…", listing_id=listing_id)
    try:
        result = await approve_from_cli(listing_id)
    except ValueError as exc:
        events.agent_error("merchant", f"Could not take #{listing_id} to market.", str(exc))
        return {"ok": False, "reason": str(exc)}

    if result.ok:
        events.agent_done("merchant", f"Sold #{listing_id} ({result.product_id}).")
    else:
        events.agent_error("merchant", f"#{listing_id} failed to publish.", result.error or "")
    events.agent_output(
        "merchant",
        "publish",
        listing_id=listing_id,
        ok=result.ok,
        simulated=result.simulated,
        product_id=result.product_id,
        url=result.external_url,
        error=result.error,
    )
    _broadcast_listing(listing_id)
    return {"ok": result.ok, "product_id": result.product_id}


async def generate_short(topic: str | None = None, **_: Any) -> dict[str, Any]:
    """The Bard writes, voices, illustrates and cuts one Short."""
    from village.bard import BardAgent

    bard = BardAgent(get_settings())
    try:
        result = await bard.produce(topic)
    finally:
        await bard.aclose()

    if not result.ok:
        events.toast(f"The Bard's short failed: {'; '.join(result.errors)}", "error")
        return {"ok": False, "errors": result.errors}

    _broadcast_shorts()
    return {
        "ok": True,
        "short_id": result.short_id,
        "title": result.title,
        "duration": result.duration,
        "backend": result.render_backend,
    }


async def reroll_script(short_id: int | None = None, topic: str | None = None, **_: Any) -> dict[str, Any]:
    """Rewrite and re-cut the newest Short, or a named one."""
    from village.bard import BardAgent

    target = short_id
    if target is None:
        rows = recent_shorts(limit=1)
        if not rows:
            events.toast("There is no Short to reroll yet.", "warn")
            return {"ok": False, "reason": "no short"}
        target = rows[0].id

    bard = BardAgent(get_settings())
    try:
        result = await bard.reroll(target, topic)
    finally:
        await bard.aclose()

    _broadcast_shorts()
    return {"ok": result.ok, "short_id": result.short_id, "title": result.title}


async def approve_short(short_id: int, **_: Any) -> dict[str, Any]:
    """Mark a Short approved and ready to upload."""
    from village.town_crier import approve_short_from_cli

    try:
        ok = await approve_short_from_cli(short_id)
    except ValueError as exc:
        events.toast(str(exc), "warn")
        return {"ok": False, "reason": str(exc)}

    events.agent_output(
        "bard", "approval", short_id=short_id, ok=ok, ready_to_upload=ok
    )
    _broadcast_shorts()
    return {"ok": ok}


def _broadcast_shorts() -> None:
    """Push the Bard's latest rows to the dashboard."""
    try:
        rows = [row.to_dict() for row in recent_shorts(limit=12)]
        events.bus.publish("shorts", shorts=rows)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not broadcast shorts: {}", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_listing(listing_id: int | None):
    """The named listing, or the most recent one when none is named."""
    if listing_id is not None:
        return get_listing(listing_id)
    rows = recent_listings(limit=1)
    return rows[0] if rows else None


def _broadcast_listing(listing_id: int) -> None:
    """Push a listing's current row to the dashboard."""
    try:
        row = get_listing(listing_id)
        if row is not None:
            events.listing_event(row.to_dict())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not broadcast listing {}: {}", listing_id, exc)


#: Action id -> coroutine. The endpoint checks the roster first, so only actions
#: a villager actually offers can reach this table.
ACTIONS: dict[str, Any] = {
    "run_pipeline": lambda **kw: run_pipeline(count=1, hint=kw.get("hint")),
    "force_scan": force_scan,
    "reroll_design": reroll_design,
    "rewrite_copy": rewrite_copy,
    "rescreen": rescreen,
    "publish_pending": publish_pending,
    "dispatch_pending": dispatch_pending,
    "generate_short": lambda **kw: generate_short(topic=kw.get("hint")),
    "reroll_script": lambda **kw: reroll_script(
        short_id=kw.get("listing_id"), topic=kw.get("hint")
    ),
    "review_logs": lambda **kw: None,
}
