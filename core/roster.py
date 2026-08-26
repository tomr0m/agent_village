"""The villagers: one record per agent, shared by the engine and the dashboard.

The pipeline's modules are named for what they do (``scout``, ``crafter``); the
village names them for who they are (Rowan, Bram). Both live here so the HUD,
the dialogue boxes and the backend can never disagree about who is who.

Map coordinates deliberately do NOT live here — the canvas owns its own layout.
This module owns identity: name, title, home building, and the actions an
operator may trigger by poking that villager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """A button the dashboard offers on a villager's dialogue card."""

    id: str
    label: str
    description: str
    #: Long-running actions warn the operator before they fire.
    slow: bool = False


@dataclass(frozen=True)
class Villager:
    """One agent, as the village knows them."""

    id: str
    name: str
    title: str
    emoji: str
    #: Building id the canvas paths them to when they start working.
    home: str
    #: Palette hint the canvas uses for their tunic, so HUD and sprite agree.
    color: str
    #: What they are doing when nothing is happening, shown in dialogue.
    idle_line: str
    #: Animation the canvas plays while they work at their building.
    work_animation: str
    actions: tuple[Action, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "emoji": self.emoji,
            "home": self.home,
            "color": self.color,
            "idleLine": self.idle_line,
            "workAnimation": self.work_animation,
            "actions": [
                {
                    "id": action.id,
                    "label": action.label,
                    "description": action.description,
                    "slow": action.slow,
                }
                for action in self.actions
            ],
        }


REVIEW_LOGS = Action("review_logs", "📜 Review Logs", "Show this villager's recent activity")


VILLAGERS: dict[str, Villager] = {
    "mayor": Villager(
        id="mayor",
        name="Mayor Eldon",
        title="Town Orchestrator & Strategy",
        emoji="👑",
        home="town_hall",
        color="#c9a227",
        idle_line="Reviewing the ledgers and deciding what the village builds next.",
        work_animation="ponder",
        actions=(
            Action("run_pipeline", "⚙️ Run the Village", "Run one full pipeline pass", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "scout": Villager(
        id="scout",
        name="Rowan the Scout",
        title="Niche Hunter & Trend Seeker",
        emoji="🔭",
        home="watchtower",
        color="#3f7a4a",
        idle_line="Scanning the horizon from the watchtower for a market worth chasing.",
        work_animation="spyglass",
        actions=(
            Action("force_scan", "🔭 Force Scout Scan", "Hunt for a fresh niche now", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "dealscout": Villager(
        id="dealscout",
        name="Marlow the Deal Scout",
        title="Affiliate Broker & Bargain Hunter",
        emoji="🛒",
        # The tavern was the one building on the map with nobody in it, and a
        # broker who trades on gossip belongs there rather than in a shop.
        home="tavern",
        color="#b8862f",
        idle_line="Nursing a drink in the Gilded Stag, listening for what people are buying.",
        work_animation="ledger",
        actions=(
            Action("scout_deal", "🛒 Scout a Deal", "Curate an affiliate pick now", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "night_scribe": Villager(
        id="night_scribe",
        name="Vesper the Night Scribe",
        title="Nightwatch Chronicler & Wire Reader",
        emoji="🌙",
        home="observatory",
        color="#4a5a8f",
        idle_line="Waiting for the small hours, when the wires carry the interesting news.",
        work_animation="starchart",
        actions=(
            Action("night_scan", "🌙 Scan the Wires", "Read the feeds now", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "overseer": Villager(
        id="overseer",
        name="Overseer Aldric",
        title="Council Economist & Monetization Advisor",
        emoji="👑",
        home="counting_house",
        color="#8f6f2a",
        idle_line="Running the ledger again, looking for the week the numbers turn.",
        work_animation="abacus",
        actions=(
            Action("review_economy", "👑 Review the Books", "Assess monetization readiness", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "crafter": Villager(
        id="crafter",
        name="Bram the Crafter",
        title="Master Artisan & Prompt Weaver",
        emoji="🎨",
        home="forge",
        color="#a8763f",
        idle_line="Tempering the forge and sharpening chisels between commissions.",
        work_animation="hammer",
        actions=(
            Action("reroll_design", "🎨 Reroll Design", "Re-render the latest artwork", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "scribe": Villager(
        id="scribe",
        name="Lyra the Scribe",
        title="Royal Copywriter & SEO Sage",
        emoji="📜",
        home="scribe_cottage",
        color="#5b4a7d",
        idle_line="Grinding fresh ink and ruling lines for the next listing.",
        work_animation="quill",
        actions=(
            Action("rewrite_copy", "📜 Rewrite Copy", "Rewrite the latest title and tags", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "guard": Villager(
        id="guard",
        name="Garrison Guard",
        title="IP & Trademark Sentinel",
        emoji="🛡️",
        home="gatehouse",
        color="#31527d",
        idle_line="Standing watch at the gate, checking sigils against the banned register.",
        work_animation="inspect",
        actions=(
            Action("rescreen", "🛡️ Re-screen Listing", "Re-run the IP screen on the latest listing"),
            REVIEW_LOGS,
        ),
    ),
    "merchant": Villager(
        id="merchant",
        name="Oakhaven Merchant",
        title="Printify & Fulfillment Trader",
        emoji="🏪",
        home="market",
        color="#1f6f63",
        idle_line="Counting crates at the stall and waiting on the next approved order.",
        work_animation="trade",
        actions=(
            Action("publish_pending", "🏪 Publish Approved", "Publish every approved listing", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "bard": Villager(
        id="bard",
        name="Bard Finneas",
        title="The Video Bard — Faceless Shorts",
        emoji="🎭",
        home="theater",
        color="#8e4a7d",
        idle_line="Tuning the lute and turning over a story worth telling in forty seconds.",
        work_animation="lute",
        actions=(
            Action("generate_short", "🎬 Generate New Short", "Write, voice and cut a new Short", slow=True),
            Action("reroll_script", "🔀 Reroll Script", "Rewrite and re-cut the latest Short", slow=True),
            REVIEW_LOGS,
        ),
    ),
    "crier": Villager(
        id="crier",
        name="Pippin the Crier",
        title="Telegram Dispatcher & Royal Herald",
        emoji="🔔",
        home="bell_tower",
        color="#d97706",
        idle_line="Polishing the bell and waiting for a proclamation worth ringing.",
        work_animation="bell",
        actions=(
            Action("dispatch_pending", "🔔 Dispatch Pending", "Send pending listings to Telegram"),
            REVIEW_LOGS,
        ),
    ),
}

#: Display order for the HUD roster, which is also pipeline order.
VILLAGER_ORDER: tuple[str, ...] = (
    "mayor",
    "scout",
    # Marlow. Added to VILLAGERS last but not here, which left him with no
    # AgentState: every event for him was dropped as an unknown villager.
    "dealscout",
    "night_scribe",
    "overseer",
    "crafter",
    "scribe",
    "guard",
    "merchant",
    "crier",
    "bard",
)

#: Buildings the canvas draws. The frontend owns their coordinates; this is the
#: shared vocabulary of building ids and their human names.
BUILDINGS: dict[str, str] = {
    "town_hall": "Town Hall",
    "watchtower": "Watchtower",
    "forge": "The Forge",
    "scribe_cottage": "Scribe's Cottage",
    "gatehouse": "Gatehouse",
    "market": "Oakhaven Market",
    "bell_tower": "Bell Tower",
    "tavern": "The Gilded Stag",
    "theater": "The Bard's Theater",
}

#: Where a villager wanders when they have nothing to do.
SOCIAL_SPOTS: tuple[str, ...] = ("tavern", "picnic_north", "picnic_south", "well", "orchard")


def roster() -> list[dict[str, Any]]:
    """Every villager, in HUD order, as plain dicts."""
    return [VILLAGERS[key].to_dict() for key in VILLAGER_ORDER]


def get(villager_id: str) -> Villager | None:
    return VILLAGERS.get(villager_id)


def has_action(villager_id: str, action_id: str) -> bool:
    """Is this action offered by this villager? Guards the trigger endpoint."""
    villager = VILLAGERS.get(villager_id)
    if villager is None:
        return False
    return any(action.id == action_id for action in villager.actions)
