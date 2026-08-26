"""The Village Overseer: says when the newsletter is worth money.

Reads ``village_metrics`` and answers one question — has the Ledger earned the
right to ask for money yet? Two gates, both from the spec:

* **Sponsorship** — more than 500 subscribers AND better than a 40% open rate,
  held for 14 consecutive days.
* **Paid tier** — the same engagement, plus more than 1,000 subscribers.

"Consecutive" is the load-bearing word. A single good day is noise; a fortnight
is a pattern. The streak is therefore measured over days that were actually
recorded, and a gap in the record breaks it rather than being skipped over —
otherwise two good days a month apart would read as a streak of two.

Advisories fire once per threshold. An agent that tells you the same thing
every morning is one you stop reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Sequence

from loguru import logger

from config.settings import Settings, get_settings
from core import events

#: Monetization states, in the order they are unlocked. These are the exact
#: values the ``village_metrics.monetization_status`` column stores.
FREE = "FREE"
#: Sponsorship threshold met: the audience is worth selling a slot against.
PAID_READY = "PAID_READY"
#: Paid-tier threshold met as well.
SPONSORED = "SPONSORED"

ADVISORY_SPONSOR = (
    "👑 Village Overseer Advisory: The Morning Ledger has hit product-market "
    "fit metrics! Recommended action: add a B2B sponsor slot."
)

ADVISORY_PAID = (
    "👑 Village Overseer Advisory: The Morning Ledger has hit product-market "
    "fit metrics! Recommended action: Enable $9/mo paid tier / add B2B sponsor slot."
)


@dataclass
class EconomicReview:
    """What the Overseer concluded today."""

    subscribers: int = 0
    open_rate: float = 0.0
    click_rate: float = 0.0
    revenue_usd: float = 0.0
    streak_days: int = 0
    sponsor_ready: bool = False
    paid_ready: bool = False
    recommendation: str = ""
    advisory: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.subscribers} subscribers · {self.open_rate:.0%} open · "
            f"{self.streak_days}d streak · {self.recommendation}"
        )


class VillageOverseer:
    """Assesses whether the newsletter is ready to be monetized."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def qualifying(self, metric: Any) -> bool:
        """Whether one day clears the engagement bar."""
        return (
            int(metric.total_subscribers or 0) >= self.settings.overseer_sponsor_subscribers
            and float(metric.open_rate or 0.0) >= self.settings.overseer_sponsor_open_rate
        )

    def streak(self, metrics: Sequence[Any]) -> int:
        """Consecutive qualifying days ending today.

        Counts backwards from the most recent record and stops at the first day
        that either fails the bar or is missing from the record entirely. A gap
        is a break: an unrecorded day is a day we cannot vouch for.
        """
        if not metrics:
            return 0

        ordered = sorted(metrics, key=lambda m: m.recorded_date, reverse=True)
        run = 0
        expected: date | None = None

        for metric in ordered:
            if expected is not None and metric.recorded_date != expected:
                break                       # a missing day breaks the run
            if not self.qualifying(metric):
                break
            run += 1
            expected = metric.recorded_date - timedelta(days=1)

        return run

    def review(self, metrics: Sequence[Any] | None = None) -> EconomicReview:
        """Assess the current state of the newsletter economy."""
        from core.database import recent_metrics  # noqa: PLC0415

        window = max(self.settings.overseer_streak_days * 2, 30)
        rows = list(metrics) if metrics is not None else recent_metrics(days=window)

        if not rows:
            return EconomicReview(
                recommendation="no metrics recorded yet",
                notes=["Record a day with: python main.py --record-metrics"],
            )

        latest = max(rows, key=lambda m: m.recorded_date)
        run = self.streak(rows)
        needed = self.settings.overseer_streak_days

        review = EconomicReview(
            subscribers=int(latest.total_subscribers or 0),
            open_rate=float(latest.open_rate or 0.0),
            click_rate=float(latest.click_rate or 0.0),
            revenue_usd=float(latest.total_revenue_usd or 0.0),
            streak_days=run,
        )

        review.sponsor_ready = run >= needed
        review.paid_ready = (
            review.sponsor_ready
            and review.subscribers >= self.settings.overseer_paid_subscribers
        )

        if review.paid_ready:
            review.recommendation = "enable the paid tier and sell a sponsor slot"
            review.advisory = ADVISORY_PAID
        elif review.sponsor_ready:
            review.recommendation = "sell a sponsor slot"
            review.advisory = ADVISORY_SPONSOR
        else:
            short_by = max(0, needed - run)
            review.recommendation = f"keep growing — {short_by} more qualifying day(s)"
            if review.subscribers < self.settings.overseer_sponsor_subscribers:
                review.notes.append(
                    f"{self.settings.overseer_sponsor_subscribers - review.subscribers} "
                    "more subscribers needed"
                )
            if review.open_rate < self.settings.overseer_sponsor_open_rate:
                review.notes.append(
                    f"open rate {review.open_rate:.0%}, needs "
                    f"{self.settings.overseer_sponsor_open_rate:.0%}"
                )

        return review

    def already_advised(self, level: str) -> bool:
        """Whether this threshold has already been announced.

        Recorded on the metric row rather than in memory, so a daemon restart
        does not re-announce a milestone the operator has already seen.
        """
        from core.database import recent_metrics  # noqa: PLC0415

        return any(
            str(m.monetization_status or FREE) == level for m in recent_metrics(days=365)
        )

    async def run_daily(self, *, notify: bool = True) -> EconomicReview:
        """Today's assessment, announcing a threshold the first time it is met."""
        from core.database import record_village_metrics  # noqa: PLC0415

        events.agent_working("overseer", "Going through the books…", progress=0.4)
        review = self.review()

        level = FREE
        if review.paid_ready:
            level = SPONSORED
        elif review.sponsor_ready:
            level = PAID_READY

        fresh = bool(review.advisory) and not self.already_advised(level)

        if level != FREE:
            record_village_metrics(monetization_status=level)

        logger.info("Overseer: {}", review.summary())
        events.agent_done("overseer", review.summary())
        events.agent_output(
            "overseer", "economy",
            subscribers=review.subscribers, open_rate=review.open_rate,
            streak=review.streak_days, recommendation=review.recommendation,
        )

        if fresh and notify:
            await self._announce(review)
        elif review.advisory and not fresh:
            logger.debug("Threshold {} already announced; staying quiet", level)

        return review

    async def _announce(self, review: EconomicReview) -> None:
        """Send the Council Advisory to Telegram."""
        if not self.settings.telegram_configured:
            logger.warning("Overseer advisory not sent: Telegram is not configured")
            logger.success(review.advisory)
            return

        from village.town_crier import TownCrier  # noqa: PLC0415

        detail = (
            f"{review.advisory}\n\n"
            f"Subscribers: {review.subscribers:,}\n"
            f"Open rate: {review.open_rate:.1%} · Clicks: {review.click_rate:.1%}\n"
            f"Qualifying streak: {review.streak_days} days\n"
            f"Revenue to date: ${review.revenue_usd:,.2f}"
        )
        await TownCrier(self.settings).announce(detail)
        events.toast("Overseer advisory sent", "success")
