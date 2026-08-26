"""The Mayor: orchestrates one pass through the village.

    Scout -> Crafter -> ImageProcessor -> Scribe -> Guard -> database -> Town Crier

Each stage's output is the next stage's only input, and the listing row is
written as soon as there is something worth keeping — so a failure at the Guard
still leaves the artwork and the copy on disk to inspect rather than discarding
the whole run.

Nothing here decides to publish. The Mayor's job ends at "a human has been
asked"; the Town Crier's callback is what moves a listing forward.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import Settings, get_settings
from core import events
from core.database import ListingStatus, create_listing, init_db, update_listing, update_status
from core.image_processor import process_image
from village.crafter import Crafter
from village.guard import Guard
from village.scout import Scout
from village.scribe import Scribe
from village.town_crier import TownCrier


def get_listing_safe(listing_id: int) -> dict[str, Any] | None:
    """Fetch a listing as a dict, swallowing any read error.

    Used only to enrich dashboard events: a broadcast must never be the thing
    that breaks a pipeline run.
    """
    try:
        from core.database import get_listing

        row = get_listing(listing_id)
        return row.to_dict() if row is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read listing {} for broadcast: {}", listing_id, exc)
        return None


@dataclass
class PipelineResult:
    """What one pass through the village produced."""

    ok: bool
    listing_id: int | None = None
    title: str = ""
    niche: str = ""
    image_path: Path | None = None
    dispatched: bool = False
    simulated_art: bool = False
    guard_ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            return f"FAILED: {'; '.join(self.errors) or 'unknown error'}"
        gate = "dispatched to Telegram" if self.dispatched else "awaiting CLI approval"
        return f"listing #{self.listing_id} '{self.title}' — {gate}"


class Mayor:
    """Runs the pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.scout = Scout(self.settings)
        self.crafter = Crafter(self.settings)
        self.scribe = Scribe(self.settings)
        self.guard = Guard(self.settings)
        self.town_crier = TownCrier(self.settings)

    async def run_once(self, hint: str | None = None) -> PipelineResult:
        """Generate one listing end to end.

        Never raises: every failure is captured on the result so a batch keeps
        going and the operator sees which stage broke.
        """
        result = PipelineResult(ok=False)
        init_db()

        # ---- 1. Scout -------------------------------------------------------
        events.pipeline_event("run_started", hint=hint)
        events.agent_working(
            "mayor", "Opening the ledger and dispatching the village.", progress=0.05
        )
        events.agent_working(
            "scout",
            "Scanning the horizon for a niche" + (f" near {hint!r}" if hint else "") + "…",
            progress=0.1,
        )
        try:
            brief = await self.scout.find_niche(hint)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"scout: {exc}")
            events.agent_error("scout", "The scan failed.", str(exc))
            logger.exception("Scout stage failed")
            return result

        events.agent_done("scout", f"Found a market: {brief.niche}")
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

        result.niche = brief.niche
        result.stages.append(f"scout: {brief.niche} ({brief.source})")
        logger.info("Niche: {} | audience: {}", brief.niche, brief.audience)

        # The row exists from here on, so nothing generated below is orphaned.
        listing = create_listing(
            niche=brief.niche,
            audience=brief.audience,
            concept=brief.concept,
            art_prompt=brief.art_prompt,
            price_cents=self.settings.listing_price_cents,
            dry_run=1 if self.settings.dry_run else 0,
            status=ListingStatus.DRAFTED.value,
            status_reason=f"Brief from {brief.source}.",
        )
        result.listing_id = listing.id
        events.listing_event(listing.to_dict())

        # ---- 2. Crafter -----------------------------------------------------
        events.agent_working(
            "crafter",
            f"Hammering out artwork for {brief.niche}…",
            listing_id=listing.id,
            progress=0.3,
        )
        try:
            crafted = await self.crafter.craft(brief.art_prompt, label=brief.niche)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"crafter: {exc}")
            events.agent_error("crafter", "The forge went cold.", str(exc))
            update_status(listing.id, ListingStatus.FAILED, reason=f"Crafter failed: {exc}")
            logger.exception("Crafter stage failed")
            return result

        events.agent_done(
            "crafter", "Struck the design" + (" (simulated)" if crafted.simulated else "") + "."
        )
        events.agent_output(
            "crafter",
            "artwork",
            path=str(crafted.path),
            image_url=f"/api/assets/{crafted.path.name}",
            simulated=crafted.simulated,
            model=crafted.model,
            prompt=crafted.prompt[:400],
        )

        result.simulated_art = crafted.simulated
        result.stages.append(
            f"crafter: {crafted.path.name}" + (" (simulated)" if crafted.simulated else "")
        )
        update_listing(listing.id, raw_image_path=str(crafted.path))

        # ---- 3. Image processing -------------------------------------------
        try:
            processed = await asyncio.to_thread(
                process_image,
                crafted.path,
                self.settings.storage_dir / f"{crafted.path.stem}-print.png",
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"image processing: {exc}")
            update_status(
                listing.id, ListingStatus.FAILED, reason=f"Image processing failed: {exc}"
            )
            logger.exception("Image processing failed")
            return result

        result.image_path = processed.path
        result.stages.append(
            f"processed: {processed.width}x{processed.height} @ {processed.dpi} DPI"
        )
        update_listing(listing.id, processed_image_path=str(processed.path))
        events.agent_output(
            "crafter",
            "print_file",
            path=str(processed.path),
            image_url=f"/api/assets/{processed.path.name}",
            width=processed.width,
            height=processed.height,
            dpi=processed.dpi,
            background_removed=processed.background_removed,
        )

        # ---- 4. Scribe ------------------------------------------------------
        events.agent_working(
            "scribe",
            f"Inking the listing for {brief.niche}…",
            listing_id=listing.id,
            progress=0.6,
        )
        try:
            copy = await self.scribe.write(
                niche=brief.niche,
                audience=brief.audience,
                concept=brief.concept,
                keywords=brief.keywords,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"scribe: {exc}")
            events.agent_error("scribe", "The quill snapped.", str(exc))
            update_status(listing.id, ListingStatus.FAILED, reason=f"Scribe failed: {exc}")
            logger.exception("Scribe stage failed")
            return result

        events.agent_done("scribe", f"Wrote the listing: {copy.title[:60]}…")
        events.agent_output(
            "scribe",
            "copy",
            title=copy.title,
            tags=copy.tags,
            description=copy.description[:600],
            source=copy.source,
            repairs=copy.repairs,
        )

        result.title = copy.title
        result.stages.append(f"scribe: {len(copy.tags)} tags ({copy.source})")
        update_listing(
            listing.id, title=copy.title, description=copy.description, tags=copy.tags
        )

        # ---- 5. Guard -------------------------------------------------------
        events.agent_working(
            "guard",
            "Checking the sigils against the banned register…",
            listing_id=listing.id,
            progress=0.8,
        )
        report = self.guard.validate(
            title=copy.title,
            tags=copy.tags,
            description=copy.description,
            image_path=processed.path,
            art_prompt=brief.art_prompt,
            niche=brief.niche,
        )
        result.guard_ok = report.ok
        result.warnings.extend(report.warnings)
        result.stages.append(f"guard: {report.summary}")
        update_listing(listing.id, guard_report=report.render())

        events.agent_output(
            "guard",
            "screen",
            ok=report.ok,
            summary=report.summary,
            errors=report.errors,
            warnings=report.warnings,
        )

        if not report.ok:
            result.errors.extend(report.errors)
            events.agent_error("guard", "Turned the listing back at the gate.", report.summary)
            update_status(
                listing.id,
                ListingStatus.FAILED,
                reason=f"Guard blocked the listing. {report.summary}",
            )
            events.listing_event(get_listing_safe(listing.id) or {})
            events.pipeline_event("run_finished", listing_id=listing.id, ok=False)
            logger.error("Listing {} blocked by the Guard", listing.id)
            return result

        events.agent_done("guard", "Cleared the listing for the herald.")

        # ---- 6. Hand to the human ------------------------------------------
        update_status(
            listing.id,
            ListingStatus.PENDING_APPROVAL,
            reason="Passed the Guard; awaiting operator approval.",
        )
        events.agent_working(
            "crier",
            "Climbing the bell tower to announce it…",
            listing_id=listing.id,
            progress=0.95,
        )
        result.dispatched = await self.town_crier.dispatch(listing.id)
        result.stages.append(
            "town crier: dispatched" if result.dispatched else "town crier: not configured"
        )
        if result.dispatched:
            events.agent_done("crier", "Rang the bell — awaiting the operator's verdict.")
        else:
            events.agent_done("crier", "No herald configured; approve from the CLI.")
        events.agent_output(
            "crier",
            "dispatch",
            listing_id=listing.id,
            delivered=result.dispatched,
            title=copy.title,
        )

        events.agent_done("mayor", f"Listing #{listing.id} is on the board.")
        events.listing_event(get_listing_safe(listing.id) or {})
        events.pipeline_event("run_finished", listing_id=listing.id, ok=True)

        result.ok = True
        logger.success("Pipeline complete — {}", result.summary())
        return result

    async def run_batch(self, count: int, hint: str | None = None) -> list[PipelineResult]:
        """Run the pipeline ``count`` times, sequentially.

        Sequential on purpose: the stages are rate-limited API calls and a
        shared SQLite file, and a burst buys nothing but throttling.
        """
        results: list[PipelineResult] = []
        for index in range(1, count + 1):
            logger.info("--- Village pass {}/{} ---", index, count)
            results.append(await self.run_once(hint))

        succeeded = sum(1 for item in results if item.ok)
        logger.info("Batch finished: {}/{} listings created", succeeded, count)
        return results

    async def aclose(self) -> None:
        """Release every HTTP client the agents opened."""
        await asyncio.gather(
            self.scout.aclose(), self.crafter.aclose(), self.scribe.aclose(),
            return_exceptions=True,
        )

    def describe_configuration(self) -> dict[str, Any]:
        """A snapshot of how this run is wired, for the CLI banner."""
        return {
            "mode": "DRY RUN" if self.settings.dry_run else "LIVE",
            "openrouter": "configured" if self.settings.openrouter_configured else "absent",
            "text_model": self.settings.text_model,
            "image_model": self.settings.image_model,
            "telegram": "configured" if self.settings.telegram_configured else "absent",
            "printify": "configured" if self.settings.printify_configured else "absent",
            "print_canvas": "x".join(str(v) for v in self.settings.print_pixel_size),
            "dpi": self.settings.target_dpi,
            "storage": str(self.settings.storage_dir),
            "database": self.settings.database_url,
            "env files": self.settings.describe_env_sources(),
        }
