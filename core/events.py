"""In-process event bus plus the live state of every villager.

The pipeline agents publish here; the dashboard's WebSocket layer subscribes.
Nothing in ``village/`` or ``core/`` knows the web exists — an event published
with no subscribers is a cheap no-op, so the CLI pays nothing for the dashboard.

Two things live here:

* :class:`EventBus` — fan-out to any number of asyncio queues, with a bounded
  replay buffer so a browser that connects mid-run sees what it missed.
* :class:`VillageState` — the current task and last output per villager, which
  is what a click-to-talk dialogue reads.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from loguru import logger

from core.roster import VILLAGER_ORDER, VILLAGERS

#: How many recent events a newly connected client is replayed.
REPLAY_LIMIT = 60

#: A slow client is dropped rather than allowed to stall the pipeline.
QUEUE_MAXSIZE = 256


@dataclass
class AgentState:
    """What one villager is doing right now."""

    id: str
    status: str = "idle"  # idle | working | done | error
    task: str = ""
    detail: str = ""
    listing_id: int | None = None
    progress: float = 0.0
    last_output: dict[str, Any] | None = None
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=25))
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        villager = VILLAGERS.get(self.id)
        return {
            "id": self.id,
            "name": villager.name if villager else self.id,
            "status": self.status,
            "task": self.task or (villager.idle_line if villager else ""),
            "detail": self.detail,
            "listingId": self.listing_id,
            "progress": self.progress,
            "lastOutput": self.last_output,
            "history": list(self.history),
            "updatedAt": self.updated_at,
        }


class VillageState:
    """Live state for every villager, updated as the pipeline runs."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentState] = {
            key: AgentState(id=key) for key in VILLAGER_ORDER
        }

    def get(self, agent_id: str) -> AgentState | None:
        return self._agents.get(agent_id)

    def all(self) -> list[dict[str, Any]]:
        return [self._agents[key].to_dict() for key in VILLAGER_ORDER]

    def set_status(
        self,
        agent_id: str,
        status: str,
        task: str = "",
        *,
        detail: str = "",
        listing_id: int | None = None,
        progress: float | None = None,
    ) -> AgentState | None:
        """Update a villager and record the change in their history."""
        agent = self._agents.get(agent_id)
        if agent is None:
            logger.warning("Unknown villager {!r}", agent_id)
            return None

        agent.status = status
        if task:
            agent.task = task
        agent.detail = detail
        if listing_id is not None:
            agent.listing_id = listing_id
        if progress is not None:
            agent.progress = max(0.0, min(1.0, progress))
        agent.updated_at = time.time()

        agent.history.append(
            {
                "at": agent.updated_at,
                "status": status,
                "task": task or agent.task,
                "detail": detail,
                "listingId": agent.listing_id,
            }
        )
        return agent

    def set_output(self, agent_id: str, output: dict[str, Any]) -> AgentState | None:
        """Attach the villager's most recent produced asset."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return None
        agent.last_output = output
        agent.updated_at = time.time()
        return agent

    def reset_idle(self) -> None:
        """Send everyone back to wandering — used when a run finishes."""
        for agent in self._agents.values():
            if agent.status == "working":
                agent.status = "idle"
                agent.progress = 0.0
                agent.updated_at = time.time()


class EventBus:
    """Fan-out pub/sub over asyncio queues, with replay for late joiners."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._replay: deque[dict[str, Any]] = deque(maxlen=REPLAY_LIMIT)
        self._sequence = 0

    # ---- subscription -------------------------------------------------------
    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._subscribers.add(queue)
        logger.debug("Event bus: {} subscriber(s)", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)
        logger.debug("Event bus: {} subscriber(s)", len(self._subscribers))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def replay(self, limit: int = REPLAY_LIMIT) -> list[dict[str, Any]]:
        """The most recent events, oldest first."""
        events = list(self._replay)
        return events[-limit:]

    # ---- publishing ---------------------------------------------------------
    def publish(self, event_type: str, **payload: Any) -> dict[str, Any]:
        """Emit an event to every subscriber.

        Safe to call from synchronous code and from any thread that has no
        running loop: with no subscribers it only appends to the replay buffer.
        A subscriber whose queue is full is dropped, because a stalled browser
        must never block the pipeline.
        """
        self._sequence += 1
        event = {"seq": self._sequence, "type": event_type, "ts": time.time(), **payload}
        self._replay.append(event)

        if not self._subscribers:
            return event

        stale: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Dropping a slow dashboard subscriber")
                stale.append(queue)
        for queue in stale:
            self._subscribers.discard(queue)

        return event


#: Process-wide singletons. The CLI never touches them; the dashboard does.
bus = EventBus()
state = VillageState()


# ---------------------------------------------------------------------------
# Convenience publishers, used by the agents
# ---------------------------------------------------------------------------


def agent_working(
    agent_id: str,
    task: str,
    *,
    detail: str = "",
    listing_id: int | None = None,
    progress: float | None = None,
) -> None:
    """A villager has started (or is continuing) a piece of work."""
    agent = state.set_status(
        agent_id, "working", task, detail=detail, listing_id=listing_id, progress=progress
    )
    if agent is not None:
        bus.publish("agent_state", agent=agent.to_dict())


def agent_done(agent_id: str, task: str = "", *, detail: str = "") -> None:
    """A villager finished and may go back to wandering."""
    agent = state.set_status(agent_id, "done", task, detail=detail, progress=1.0)
    if agent is not None:
        bus.publish("agent_state", agent=agent.to_dict())


def agent_error(agent_id: str, task: str, detail: str = "") -> None:
    agent = state.set_status(agent_id, "error", task, detail=detail)
    if agent is not None:
        bus.publish("agent_state", agent=agent.to_dict())


def agent_idle(agent_id: str) -> None:
    agent = state.set_status(agent_id, "idle", "", progress=0.0)
    if agent is not None:
        bus.publish("agent_state", agent=agent.to_dict())


def agent_output(agent_id: str, kind: str, **payload: Any) -> None:
    """Record and announce a villager's newest produced asset."""
    output = {"kind": kind, "at": time.time(), **payload}
    agent = state.set_output(agent_id, output)
    if agent is not None:
        bus.publish("agent_output", agent_id=agent_id, output=output)


def pipeline_event(stage: str, **payload: Any) -> None:
    bus.publish("pipeline", stage=stage, **payload)


def listing_event(listing: dict[str, Any]) -> None:
    bus.publish("listing", listing=listing)


def edition_event(edition: dict[str, Any]) -> None:
    """One Morning Ledger edition changed. Mirrors ``deal_event``."""
    bus.publish("edition", edition=edition)


def deal_event(deal: dict[str, Any]) -> None:
    """One curated affiliate deal changed. Mirrors ``listing_event``."""
    bus.publish("deal", deal=deal)


def stats_event(stats: dict[str, Any]) -> None:
    bus.publish("stats", stats=stats)


def toast(message: str, tone: str = "info") -> None:
    """A one-line notice for the dashboard's log rail, WITH a popup.

    For things the operator should see even if they are looking elsewhere.
    Routine progress belongs in :func:`log`, which is quieter.
    """
    bus.publish("toast", message=message, tone=tone)


def log(message: str, tone: str = "info") -> None:
    """A line in the Village Chronicle, without a popup.

    The difference from :func:`toast` is deliberate: a night scan filing forty
    stories is worth recording and not worth interrupting anyone for. Popping a
    notification for every routine step trains people to dismiss them.
    """
    bus.publish("log", message=message, tone=tone)


def broadcast_all(events: Iterable[tuple[str, dict[str, Any]]]) -> None:
    for event_type, payload in events:
        bus.publish(event_type, **payload)
