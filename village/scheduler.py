"""Fixed daily schedule for Short production.

The daemon's listing loop runs on a plain interval, which is fine for something
that can happen at any hour. Shorts are different: they are published to a
channel where posting time matters, so they run at named times of day instead.

Two things make this more than a clock comparison:

* **A slot must fire once.** The loop wakes far more often than the schedule
  fires, and every wake-up sees the same "it is past 12:00" condition. What has
  already run is therefore recorded on disk, not in memory — otherwise
  restarting the daemon at 12:05 produces a second Short for the 12:00 slot,
  and Shorts cost API quota and upload allowance.
* **A missed slot should usually stay missed.** If the machine was asleep from
  11:00 to 21:00, firing all three backlogged slots at once dumps three videos
  into the chat at midnight. Only a slot missed within the grace window runs
  late; older ones are logged and skipped.

Times are local, matching how someone thinks about posting times. This means a
DST jump can skip or repeat a slot once a year; the once-per-slot record keeps
the repeat from producing two Shorts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

#: How late a missed slot may still run, in minutes. Long enough to survive a
#: restart or a slow generation pass, short enough that a machine woken in the
#: evening does not post a Short intended for lunchtime.
DEFAULT_GRACE_MINUTES = 30


def parse_schedule(raw: str) -> tuple[tuple[int, int], ...]:
    """Parse ``"12:00,16:00,20:00"`` into sorted (hour, minute) pairs.

    Accepts ``H``, ``HH``, ``HH:MM`` and tolerates whitespace, so ``"9, 16:30"``
    works. Invalid entries are reported and dropped rather than taking the
    daemon down over a typo in a schedule.
    """
    slots: set[tuple[int, int]] = set()

    for chunk in (raw or "").split(","):
        text = chunk.strip()
        if not text:
            continue

        hour_text, _, minute_text = text.partition(":")
        try:
            hour = int(hour_text)
            minute = int(minute_text) if minute_text else 0
        except ValueError:
            logger.warning("Ignoring unparsable schedule entry {!r}", text)
            continue

        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            logger.warning("Ignoring out-of-range schedule entry {!r}", text)
            continue

        slots.add((hour, minute))

    return tuple(sorted(slots))


@dataclass
class Slot:
    """One scheduled firing time on a specific day."""

    at: datetime

    @property
    def key(self) -> str:
        """Stable identity for "this slot on this day"."""
        return self.at.strftime("%Y-%m-%dT%H:%M")

    @property
    def label(self) -> str:
        return self.at.strftime("%H:%M")


class ShortsScheduler:
    """Decides when the next Short is due, and remembers what already ran."""

    def __init__(
        self,
        schedule: str,
        state_path: Path,
        *,
        grace_minutes: int = DEFAULT_GRACE_MINUTES,
    ) -> None:
        self.slots = parse_schedule(schedule)
        self.state_path = Path(state_path)
        self.grace = timedelta(minutes=max(0, grace_minutes))
        self._fired: set[str] = self._load()

    @property
    def enabled(self) -> bool:
        """Whether any valid time was configured."""
        return bool(self.slots)

    def describe(self) -> str:
        """The configured times, for the banner."""
        return ", ".join(f"{h:02d}:{m:02d}" for h, m in self.slots) or "not scheduled"

    # ---- what is due --------------------------------------------------------
    def _slots_on(self, day: datetime) -> list[datetime]:
        return [
            day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            for hour, minute in self.slots
        ]

    def next_run(self, now: datetime | None = None) -> datetime | None:
        """When the next Short is due, today or tomorrow."""
        if not self.enabled:
            return None

        moment = now or datetime.now()
        for candidate in self._slots_on(moment):
            if candidate > moment and candidate.strftime("%Y-%m-%dT%H:%M") not in self._fired:
                return candidate

        tomorrow = (moment + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        upcoming = self._slots_on(tomorrow)
        return upcoming[0] if upcoming else None

    def due(self, now: datetime | None = None) -> Slot | None:
        """The slot that should run right now, if any.

        Returns the most recent past-due slot that is still inside the grace
        window. Anything older is recorded as handled so it cannot fire later
        in a burst — three backlogged Shorts arriving at midnight is worse than
        three missed ones.

        Note this does NOT mark the returned slot. The caller marks it once the
        Short is produced, so a crash mid-generation retries rather than
        silently losing the slot.
        """
        if not self.enabled:
            return None

        moment = now or datetime.now()
        earliest = moment - self.grace

        # Yesterday matters ONLY just after midnight, when a 23:30 slot can
        # still be inside its grace window. Yesterday's earlier slots are not
        # candidates at all: on a first run they are history, not missed work,
        # and treating them as missed fills the state file with warnings about
        # a day the daemon was never running.
        candidates = [
            slot
            for slot in self._slots_on(moment - timedelta(days=1))
            if slot >= earliest
        ] + self._slots_on(moment)
        overdue = [
            candidate
            for candidate in candidates
            if candidate <= moment
            and candidate.strftime("%Y-%m-%dT%H:%M") not in self._fired
        ]
        if not overdue:
            return None

        runnable = [candidate for candidate in overdue if candidate >= earliest]

        # Everything past the grace window is consumed quietly if it predates
        # this process's first look, and warned about if it is from today —
        # the second case means the daemon was actually down when it mattered.
        for stale in overdue:
            if stale in runnable:
                continue
            if stale.date() == moment.date():
                logger.warning(
                    "Missed the {} Short slot by {:.0f} minutes; skipping rather "
                    "than posting late",
                    stale.strftime("%H:%M"),
                    (moment - stale).total_seconds() / 60,
                )
            self.mark_fired(Slot(stale))

        if not runnable:
            return None

        # The newest runnable slot. If two are inside the window, the older one
        # is consumed too: posting both back to back helps nobody.
        newest = runnable[-1]
        for superseded in runnable[:-1]:
            logger.info(
                "Superseding the {} slot with {}",
                superseded.strftime("%H:%M"), newest.strftime("%H:%M"),
            )
            self.mark_fired(Slot(superseded))

        return Slot(newest)

    def seconds_until_next(self, now: datetime | None = None) -> float:
        """How long to sleep before the next slot, floored at one second."""
        moment = now or datetime.now()
        upcoming = self.next_run(moment)
        if upcoming is None:
            return 3600.0
        return max(1.0, (upcoming - moment).total_seconds())

    # ---- persistence --------------------------------------------------------
    def mark_fired(self, slot: Slot) -> None:
        """Record that a slot ran, so a restart does not repeat it."""
        self._fired.add(slot.key)
        self._prune()
        self._save()

    def _prune(self) -> None:
        """Drop records older than two days; the file should not grow forever."""
        cutoff = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M")
        self._fired = {key for key in self._fired if key >= cutoff}

    def _load(self) -> set[str]:
        if not self.state_path.is_file():
            return set()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return set(data.get("fired", []))
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not block
            logger.warning("Ignoring unreadable schedule state {}: {}", self.state_path, exc)
            return set()

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"fired": sorted(self._fired)}, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # noqa: BLE001 - losing the record is not fatal
            logger.warning("Could not persist schedule state: {}", exc)


class IntervalScheduler:
    """Fire every N hours instead of at fixed times of day.

    Deliberately the same interface as :class:`ShortsScheduler` — ``enabled``,
    ``describe``, ``due``, ``next_run``, ``seconds_until_next``, ``mark_fired``
    — so the daemon's loop takes either without knowing which it has.

    The last firing is recorded on disk for the same reason: the loop wakes far
    more often than the interval, and a restart must not reset the clock and
    fire again immediately.
    """

    def __init__(
        self,
        interval_hours: float,
        state_path: Path,
        *,
        fire_on_start: bool = True,
    ) -> None:
        self.interval = timedelta(hours=max(0.0, float(interval_hours or 0.0)))
        self.state_path = Path(state_path)
        #: With no history, run once at startup so a new schedule visibly works
        #: rather than going quiet for hours.
        self.fire_on_start = fire_on_start
        self._last: datetime | None = self._load()

    @property
    def enabled(self) -> bool:
        return self.interval > timedelta(0)

    def describe(self) -> str:
        if not self.enabled:
            return "not scheduled"
        seconds = self.interval.total_seconds()
        if seconds >= 3600 and (seconds / 3600).is_integer():
            return f"every {int(seconds // 3600)}h"
        if seconds >= 3600:
            return f"every {seconds / 3600:.1f}h"
        if seconds >= 60:
            return f"every {seconds / 60:.0f}m"
        # Sub-minute intervals are only ever a test, but reporting them as
        # "every 0m" makes a test look like a bug.
        return f"every {seconds:.0f}s"

    def next_run(self, now: datetime | None = None) -> datetime | None:
        if not self.enabled:
            return None
        moment = now or datetime.now()
        if self._last is None:
            return moment if self.fire_on_start else moment + self.interval
        return self._last + self.interval

    def due(self, now: datetime | None = None) -> Slot | None:
        """The firing that is owed right now, if any."""
        if not self.enabled:
            return None

        moment = now or datetime.now()
        if self._last is None:
            return Slot(moment) if self.fire_on_start else None

        # Exactly one firing, however far behind the clock has fallen. A
        # machine asleep for a day owes one deal, not twenty-four.
        return Slot(moment) if moment - self._last >= self.interval else None

    def seconds_until_next(self, now: datetime | None = None) -> float:
        moment = now or datetime.now()
        upcoming = self.next_run(moment)
        if upcoming is None:
            return 3600.0
        return max(1.0, (upcoming - moment).total_seconds())

    def mark_fired(self, slot: Slot) -> None:
        self._last = slot.at
        self._save()

    # ---- persistence --------------------------------------------------------
    def _load(self) -> datetime | None:
        if not self.state_path.is_file():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            raw = data.get("last")
            return datetime.fromisoformat(raw) if raw else None
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not block
            logger.warning("Ignoring unreadable schedule state {}: {}", self.state_path, exc)
            return None

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"last": self._last.isoformat() if self._last else None}),
                encoding="utf-8",
            )
        except OSError as exc:  # noqa: BLE001 - losing the record is not fatal
            logger.warning("Could not persist schedule state: {}", exc)


def build_scheduler(
    times: str,
    interval_hours: float,
    state_path: Path,
    *,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
) -> ShortsScheduler | IntervalScheduler | None:
    """Whichever scheduler the configuration asks for, or None.

    Fixed times win when both are set: they are the more specific instruction,
    and silently averaging the two would be worse than picking one and saying
    so.
    """
    if parse_schedule(times):
        if interval_hours:
            logger.warning(
                "Both fixed times and an interval are configured; using the "
                "fixed times ({}) and ignoring the {}h interval.",
                times, interval_hours,
            )
        return ShortsScheduler(times, state_path, grace_minutes=grace_minutes)

    if interval_hours and interval_hours > 0:
        return IntervalScheduler(interval_hours, state_path)

    return None
