"""The Deal Scout: curates Amazon affiliate product recommendations.

Named "Scout" everywhere the operator touches it — ``--scout``, ``/scout`` —
but it lives here rather than in :mod:`village.scout`, which is a different
agent entirely: that one researches Etsy print niches and feeds the Mayor's
print-on-demand pipeline. Two unrelated jobs, two modules.

**On the links.** There is no Amazon Product Advertising API access in this
project — PA-API needs approved Associate credentials and three qualifying
sales before it is granted. So this agent does NOT invent product IDs. A model
asked for an ASIN will happily produce a well-formed one that points at
something else entirely, and an affiliate link to the wrong product is worse
than no link: it misleads the reader and it is the kind of thing that gets an
Associates account closed.

Instead every link is a **tagged search URL** for the product's search terms.
It always resolves, it always carries the tracking id, and it lands the reader
on real current listings with real current prices. Prices in the summary are
labelled as estimates for the same reason: this agent cannot see live pricing,
and stating a price it cannot verify is how a post becomes wrong within a day.

**On disclosure.** The Amazon Associates operating agreement requires affiliate
content to disclose the relationship. The disclosure is part of the rendered
post, not an optional extra, because a post that ships without it puts the
account at risk.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence
from urllib.parse import quote_plus

from loguru import logger
from openai import AsyncOpenAI

from config.settings import Settings, get_settings
from core import events

#: Required by the Associates operating agreement on any affiliate placement.
AFFILIATE_DISCLOSURE = (
    "As an Amazon Associate I earn from qualifying purchases."
)

DEAL_PROMPT = """You curate product recommendations for a deals channel. You \
are writing for someone who is deciding whether to spend their own money, so \
you are specific, you are honest about drawbacks, and you never gush.

Return ONE JSON object and nothing else:

{
  "product": "the product type and what makes this one specific, 3-9 words",
  "category": "tech | gadgets | workspace | lifestyle",
  "hook": "one line that makes someone stop scrolling. Under 16 words. \
Concrete benefit or a surprising detail, never 'you won't believe'.",
  "search_terms": "2-6 words someone would actually type into Amazon to find \
this class of product. No brand names unless the brand IS the point.",
  "benefits": ["3 to 5 short bullets, each a concrete benefit, under 12 words"],
  "pros": ["2 to 4 genuine strengths"],
  "cons": ["2 to 3 REAL drawbacks. Never invent a fake weakness like 'so good \
you'll want two'. If a product's weakness is price, say so."],
  "price_low": 29,
  "price_high": 49,
  "verdict": "one sentence on who should buy it and who should not",
  "audience": "who this is for, 3-8 words"
}

RULES:
- Recommend a PRODUCT CLASS, not a specific listing. You cannot see live \
inventory, so "a low-profile mechanical keyboard with hot-swap switches" is \
honest and "the Keychron K3 Pro at $79" is a guess that will be wrong.
- Prices are ESTIMATES of the typical street range in USD. Give a range.
- The cons must be real. A recommendation with no honest drawback reads as an \
advertisement and is worth nothing to the reader.
- No health, medical, supplement or financial products. No weapons. No items \
aimed at children.
- No superlatives you cannot support. No "best ever", no fake urgency, no \
invented discount percentages, no "limited time".
"""

#: Used when the model is unavailable, so the whole path stays exercised.
FALLBACK_DEALS: tuple[dict[str, Any], ...] = (
    {
        "product": "Under-desk cable management tray",
        "category": "workspace",
        "hook": "The reason your desk photo never looks like the ones online.",
        "search_terms": "under desk cable management tray",
        "benefits": [
            "Hides the power strip and every trailing cable",
            "Clamps or screws on, no adhesive to fail",
            "Makes the desk liftable without unplugging",
            "Stops dust collecting in a floor nest",
        ],
        "pros": [
            "Cheapest change with the biggest visual payoff",
            "Fits most desks without tools beyond a screwdriver",
        ],
        "cons": [
            "Screw-in versions mark the underside of the desk",
            "Deep trays can foul a sit-stand desk's travel",
        ],
        "price_low": 25,
        "price_high": 45,
        "verdict": "Worth it for any permanent desk; skip if you move often.",
        "audience": "anyone with a visible cable nest",
    },
    {
        "product": "Compact 65W GaN charger",
        "category": "tech",
        "hook": "One brick replaces the three chargers in your bag.",
        "search_terms": "65w gan charger usb c",
        "benefits": [
            "Charges a laptop, phone and earbuds together",
            "Roughly half the size of the stock brick",
            "Runs cooler than older silicon chargers",
        ],
        "pros": [
            "Genuinely removes weight from a bag",
            "Works across nearly every modern device",
        ],
        "cons": [
            "Splitting the wattage slows laptop charging",
            "Cheap units skip the safety certifications worth having",
        ],
        "price_low": 30,
        "price_high": 60,
        "verdict": "Buy if you carry a laptop daily; overkill for a phone alone.",
        "audience": "people who commute with a laptop",
    },
    {
        "product": "Monitor light bar",
        "category": "workspace",
        "hook": "Lights the desk without putting a reflection on the screen.",
        "search_terms": "monitor light bar",
        "benefits": [
            "No glare on the panel, unlike a desk lamp",
            "Frees the desk space a lamp would take",
            "Adjustable warmth for evening work",
        ],
        "pros": [
            "Noticeably easier on the eyes at night",
            "Clips on with no mounting or drilling",
        ],
        "cons": [
            "Does not fit every monitor bezel or curve",
            "The good ones cost more than a decent desk lamp",
        ],
        "price_low": 35,
        "price_high": 90,
        "verdict": "Good buy for late-night desk work; unnecessary in a bright room.",
        "audience": "people working after dark",
    },
)

#: Categories this agent refuses regardless of what was asked for.
BANNED_TERMS = (
    "supplement", "vitamin", "medication", "medicine", "cbd", "weight loss",
    "diet pill", "gun", "ammo", "knife", "weapon", "crypto", "investment",
)


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull one JSON object out of a model response."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in response") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON was not an object")
    return parsed


@dataclass
class Deal:
    """One curated product recommendation, ready to post."""

    product: str
    category: str
    hook: str
    search_terms: str
    benefits: list[str] = field(default_factory=list)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    price_low: float = 0.0
    price_high: float = 0.0
    verdict: str = ""
    audience: str = ""
    niche: str = ""
    #: A public HTTPS image for this product, when one is known.
    #:
    #: Empty today: the real source would be Amazon's own product image, which
    #: needs PA-API access this project does not have, and scraping a listing
    #: image breaches Amazon's terms. The field exists so that the moment a
    #: legitimate URL is available it is used, and so downstream code has one
    #: place to look.
    image_url: str = ""
    affiliate_url: str = ""
    tracking_id: str = ""
    source: str = "openrouter"
    simulated: bool = False
    created_at: str = ""

    @property
    def price_range(self) -> str:
        """Human-readable estimate, always labelled as one."""
        if self.price_low and self.price_high and self.price_low != self.price_high:
            return f"~${self.price_low:,.0f}-${self.price_high:,.0f}"
        single = self.price_high or self.price_low
        return f"~${single:,.0f}" if single else "price varies"

    def to_dict(self) -> dict[str, Any]:
        """Plain types only, so this round-trips through JSON and a DB column."""
        return {
            "product": self.product,
            "category": self.category,
            "hook": self.hook,
            "search_terms": self.search_terms,
            "benefits": list(self.benefits),
            "pros": list(self.pros),
            "cons": list(self.cons),
            "price_low": float(self.price_low),
            "price_high": float(self.price_high),
            "verdict": self.verdict,
            "audience": self.audience,
            "niche": self.niche,
            "image_url": self.image_url,
            "affiliate_url": self.affiliate_url,
            "tracking_id": self.tracking_id,
            "source": self.source,
            "simulated": self.simulated,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Deal":
        """Rebuild from stored JSON, tolerating missing and extra keys."""
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - documented API
        clean = {key: value for key, value in (payload or {}).items() if key in known}
        for listy in ("benefits", "pros", "cons"):
            value = clean.get(listy)
            clean[listy] = list(value) if isinstance(value, (list, tuple)) else []
        for number in ("price_low", "price_high"):
            try:
                clean[number] = float(clean.get(number) or 0.0)
            except (TypeError, ValueError):
                clean[number] = 0.0
        clean.setdefault("product", "Unknown product")
        clean.setdefault("category", "")
        clean.setdefault("hook", "")
        clean.setdefault("search_terms", clean["product"])
        return cls(**clean)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ---- rendering ----------------------------------------------------------
    def as_markdown(self) -> str:
        """The post, as plain markdown."""
        lines = [f"**{self.hook}**", "", f"**{self.product}** — {self.price_range}"]
        if self.benefits:
            lines += [""] + [f"• {item}" for item in self.benefits]
        if self.pros:
            lines += ["", "**Good:**"] + [f"  + {item}" for item in self.pros]
        if self.cons:
            lines += ["", "**Not so good:**"] + [f"  - {item}" for item in self.cons]
        if self.verdict:
            lines += ["", self.verdict]
        lines += ["", f"👉 {self.affiliate_url}", "", f"_{AFFILIATE_DISCLOSURE}_"]
        if self.simulated:
            lines += ["", "_(simulated — dry run)_"]
        return "\n".join(lines)

    def as_telegram_html(self) -> str:
        """The post, as the HTML subset Telegram accepts."""
        def esc(text: str) -> str:
            return (
                str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )

        lines = [f"🛒 <b>{esc(self.hook)}</b>", ""]
        lines.append(f"<b>{esc(self.product)}</b> — <i>{esc(self.price_range)}</i>")
        if self.audience:
            lines.append(f"<i>For: {esc(self.audience)}</i>")
        if self.benefits:
            lines += [""] + [f"• {esc(item)}" for item in self.benefits]
        if self.pros:
            lines += ["", "<b>👍 Good</b>"] + [f"  + {esc(item)}" for item in self.pros]
        if self.cons:
            lines += ["", "<b>👎 Trade-offs</b>"] + [f"  − {esc(item)}" for item in self.cons]
        if self.verdict:
            lines += ["", f"<i>{esc(self.verdict)}</i>"]
        lines += ["", f'<a href="{esc(self.affiliate_url)}">🔗 See it on Amazon</a>']
        lines += ["", f"<i>{esc(AFFILIATE_DISCLOSURE)}</i>"]
        if self.simulated:
            lines += ["", "<i>(simulated — dry run, nothing was posted)</i>"]
        return "\n".join(lines)


class DealScout:
    """Researches a niche and returns one honest product recommendation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.settings.openrouter_base_url,
                api_key=self.settings.openrouter_api_key,
                timeout=self.settings.request_timeout_seconds,
                max_retries=self.settings.max_retries,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ---- links --------------------------------------------------------------
    def affiliate_link(self, search_terms: str) -> str:
        """A tagged Amazon search URL for these terms.

        A search link rather than a product link, deliberately — see the module
        docstring. The tag is what earns; the search is what stays correct.
        """
        query = quote_plus((search_terms or "").strip() or "gadgets")
        tag = quote_plus(self.settings.amazon_tracking_id.strip() or "village-20")
        return f"https://{self.settings.amazon_marketplace}/s?k={query}&tag={tag}"

    # ---- research -----------------------------------------------------------
    async def find_deal(self, niche: str | None = None) -> Deal:
        """Curate one recommendation for ``niche``.

        Never raises for a model problem: an unusable response falls back to a
        curated recommendation so the CLI, the bot and the storage path are all
        still exercised.
        """
        target = (niche or self.settings.amazon_default_niche).strip()
        if not target:
            target = random.choice(self.settings.amazon_category_list or ("gadgets",))

        blocked = self._blocked_term(target)
        if blocked:
            raise ValueError(
                f"Refusing to research {target!r}: this agent does not cover "
                f"{blocked} products."
            )

        # Announced before the early returns below: a dry run or a missing key
        # still takes a moment and still produces a deal, and a villager that
        # teleports from idle to done never reads as having done anything.
        events.agent_working(
            "dealscout", f"Scouting deals for {target}…", progress=0.25
        )

        if self.settings.dry_run:
            logger.info("DRY RUN — Scout is reciting a curated deal for {!r}", target)
            return self._fallback(target, "dry run")

        if not self.settings.openrouter_configured:
            logger.warning("OPENROUTER_API_KEY unset — the Scout is reciting from memory")
            return self._fallback(target, "no api key")

        logger.info("Scout researching {!r}", target)
        events.agent_working(
            "dealscout", f"Reading the market for {target}…", progress=0.5
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.text_model,
                messages=[
                    {"role": "system", "content": DEAL_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Recommend one product for: {target}. "
                            "Pick something specific and genuinely useful, not "
                            "the most obvious thing in the category."
                        ),
                    },
                ],
                temperature=0.8,
                max_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001 - degrade rather than abort
            logger.error("Deal research failed ({}); using a curated one", exc)
            events.agent_error("dealscout", "The trail went cold.", str(exc))
            return self._fallback(target, f"api error: {exc}")

        try:
            payload = _extract_json(response.choices[0].message.content or "")
            deal = self._build(payload, target, source="openrouter")
        except Exception as exc:  # noqa: BLE001
            logger.error("Unusable deal response ({}); using a curated one", exc)
            return self._fallback(target, "unparsable response")

        logger.success("Scout found: {} ({})", deal.product, deal.price_range)
        events.agent_done("dealscout", f"Found: {deal.product}")
        events.agent_output(
            "dealscout", "deal",
            product=deal.product, hook=deal.hook, price=deal.price_range,
        )
        return deal

    async def find_deals(self, niche: str | None = None, count: int = 1) -> list[Deal]:
        """Several recommendations, one after another."""
        deals: list[Deal] = []
        for index in range(max(1, count)):
            if count > 1:
                logger.info("Deal {}/{}", index + 1, count)
            deals.append(await self.find_deal(niche))
        return deals

    # ---- internals ----------------------------------------------------------
    def _blocked_term(self, text: str) -> str | None:
        """The banned category this text falls into, if any."""
        lowered = (text or "").lower()
        return next((term for term in BANNED_TERMS if term in lowered), None)

    def _clean_list(self, value: Any, limit: int, width: int = 120) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        cleaned = [str(item).strip()[:width] for item in value if str(item).strip()]
        return cleaned[:limit]

    def _price(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        # A four-figure "deal" is either a hallucination or not a deal.
        return number if 0 < number < 5000 else 0.0

    def _build(self, payload: dict[str, Any], niche: str, *, source: str) -> Deal:
        """Validate a payload into a Deal."""
        def text(key: str, limit: int) -> str:
            return str(payload.get(key, "") or "").strip()[:limit]

        product = text("product", 120)
        if not product:
            raise ValueError("no product in response")

        blocked = self._blocked_term(f"{product} {text('search_terms', 120)}")
        if blocked:
            raise ValueError(f"model returned a banned category: {blocked}")

        search_terms = text("search_terms", 120) or product
        low = self._price(payload.get("price_low"))
        high = self._price(payload.get("price_high"))
        if low and high and low > high:
            low, high = high, low

        category = text("category", 40).lower()
        if category not in self.settings.amazon_category_list:
            category = (self.settings.amazon_category_list or ("general",))[0]

        return Deal(
            product=product,
            category=category,
            hook=text("hook", 200) or product,
            search_terms=search_terms,
            benefits=self._clean_list(payload.get("benefits"), 5),
            pros=self._clean_list(payload.get("pros"), 4),
            cons=self._clean_list(payload.get("cons"), 3),
            price_low=low,
            price_high=high,
            verdict=text("verdict", 300),
            audience=text("audience", 120),
            niche=niche,
            affiliate_url=self.affiliate_link(search_terms),
            tracking_id=self.settings.amazon_tracking_id.strip(),
            source=source,
            simulated=self.settings.dry_run,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def _fallback(self, niche: str, reason: str) -> Deal:
        """A curated recommendation, when the model cannot be used."""
        payload = dict(random.choice(FALLBACK_DEALS))
        deal = self._build(payload, niche, source=f"fallback ({reason})")
        deal.simulated = True
        logger.info("Scout fallback: {} ({})", deal.product, reason)
        # Still an outcome worth animating: the villager should settle back to
        # idle rather than staying stuck mid-scan.
        events.agent_done("dealscout", f"Recalled: {deal.product}")
        return deal


def render_deal(deal: Deal) -> str:
    """Module-level convenience for the CLI."""
    return deal.as_markdown()
