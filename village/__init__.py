"""The village: one module per agent, plus the Mayor that runs them in order.

    Scout    -> finds the Etsy niche and writes the art brief
    DealScout-> curates Amazon affiliate recommendations (a different job
                from Scout above, hence a separate module)
    Crafter  -> renders the artwork through OpenRouter
    Scribe   -> writes the Etsy title, 13 tags and description
    Guard    -> validates everything objective before a human is asked
    TownCrier-> Telegram approval cards and their callbacks
    Merchant -> Printify publishing, with an automatic dry-run fallback
    Mayor    -> orchestrates the pass
"""

from village.crafter import Crafter
from village.dealscout import Deal, DealScout
from village.guard import Guard
from village.mayor import Mayor, PipelineResult
from village.merchant import Merchant, PublishResult
from village.scout import NicheBrief, Scout
from village.scribe import ListingCopy, Scribe
from village.town_crier import TownCrier, approve_from_cli, reject_from_cli

__all__ = [
    "Crafter",
    "Deal",
    "DealScout",
    "Guard",
    "Mayor",
    "PipelineResult",
    "Merchant",
    "PublishResult",
    "NicheBrief",
    "Scout",
    "ListingCopy",
    "Scribe",
    "TownCrier",
    "approve_from_cli",
    "reject_from_cli",
]
