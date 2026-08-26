"""The Scout: finds a niche and turns it into a concrete art brief.

Everything downstream is shaped by this one turn, so the Scout is asked for a
structured brief rather than prose: a niche, the buyer it targets, a one-line
concept, and an image prompt precise enough for the Crafter to render without
interpretation.

The brief is screened for trademark risk here, before an image is ever
generated — catching "cartoon mouse in red shorts" at this stage costs a text
call; catching it after the Crafter runs costs an image generation.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from config.settings import Settings, get_settings
from core.trademark_guard import sanitise_prompt, screen_text

SYSTEM_PROMPT = """You are a print-on-demand market scout with a decade of Etsy \
sales data behind you. You find niches that people actually buy from, not niches \
that sound clever.

Return ONE JSON object and nothing else, with exactly these keys:

{
  "niche": "3-6 words naming the specific market",
  "audience": "the exact buyer: their role, their situation, the occasion they buy for",
  "concept": "one sentence describing the design and why that buyer wants it on a shirt",
  "art_prompt": "a complete image-generation prompt: subject, composition, art style, line weight, colour palette, and the words 'isolated on a plain white background, centered, no mockup, no text watermark'",
  "style": "the visual style in 2-4 words",
  "palette": ["#hex", "#hex", "#hex"],
  "keywords": ["8 to 12 lowercase Etsy search phrases a buyer would actually type"]
}

HARD RULES:
- The niche must be specific. "cats" is not a niche; "grumpy cats for night-shift \
nurses" is.
- The design must be ORIGINAL. Never reference a real brand, franchise, character, \
team, band, film, game, or living person. No parody, no "inspired by", no lookalikes.
- The art prompt must describe a single self-contained graphic suitable for direct-to-\
garment printing: bold shapes, high contrast, no photographic backgrounds, no gradients \
that would band, no small text.
- Any text inside the artwork must be spelled out in the prompt and kept under six words.
- Output raw JSON. No markdown, no code fences, no commentary."""

#: Used when OpenRouter is unavailable or in dry-run mode. Real briefs, so the
#: rest of the pipeline is exercised with realistic shapes rather than "test".
FALLBACK_BRIEFS: tuple[dict[str, Any], ...] = (
    {
        "niche": "sourdough baking for beginners",
        "audience": "home bakers in their thirties who name their starter and post it",
        "concept": "A friendly anthropomorphic sourdough starter jar with bubbles, "
        "for bakers who treat their culture like a pet.",
        "art_prompt": "Bold flat-vector illustration of a smiling glass sourdough "
        "starter jar with rising bubbles and a cloth lid, thick black outlines, "
        "warm cream and rye-brown palette, small caption 'FED TODAY' beneath, "
        "isolated on a plain white background, centered, no mockup, no text watermark",
        "style": "flat bold vector",
        "palette": ["#e8d5b7", "#8b5a2b", "#2f2a26"],
        "keywords": [
            "sourdough shirt", "bread baker gift", "starter jar tee",
            "baking lover shirt", "sourdough starter", "home baker gift",
            "funny baking shirt", "bread lover tee", "fermentation gift",
        ],
    },
    {
        "niche": "night-shift emergency nurses",
        "audience": "ER nurses working 7p-7a who buy humour that only colleagues get",
        "concept": "A deadpan coffee cup with tired eyes and an IV drip line, "
        "for nurses whose caffeine is basically medication.",
        "art_prompt": "Bold flat-vector illustration of a takeaway coffee cup with "
        "half-closed tired eyes connected to an IV drip bag, thick uniform outlines, "
        "teal scrubs and warm amber palette, caption 'SHIFT FUEL' below, "
        "isolated on a plain white background, centered, no mockup, no text watermark",
        "style": "bold line-art vector",
        "palette": ["#1f6f63", "#f0a33a", "#22293a"],
        "keywords": [
            "night shift nurse", "er nurse gift", "nurse coffee shirt",
            "funny nurse tee", "nurse humor shirt", "emergency nurse",
            "night shift gift", "nurse appreciation", "scrubs shirt",
        ],
    },
    {
        "niche": "cold-water swimming clubs",
        "audience": "year-round sea swimmers who meet at dawn and treat it as identity",
        "concept": "A stylised wave breaking over a thermometer at 4 degrees, "
        "for swimmers who measure the water before they measure the weather.",
        "art_prompt": "Bold flat-vector illustration of a curling wave wrapping "
        "around an outdoor thermometer reading four degrees, thick outlines, "
        "deep navy and pale ice-blue palette, caption 'DAWN PATROL' beneath, "
        "isolated on a plain white background, centered, no mockup, no text watermark",
        "style": "graphic wave vector",
        "palette": ["#12283f", "#7fd6c2", "#eceff4"],
        "keywords": [
            "cold water swimming", "sea swimmer gift", "wild swimming shirt",
            "open water swim", "dawn patrol tee", "cold plunge gift",
            "winter swimming", "swim club shirt", "outdoor swimmer",
        ],
    },
    {
        "niche": "houseplant propagation hobbyists",
        "audience": "plant keepers with more cuttings in jars than shelves to put them on",
        "concept": "A row of glass propagation jars with roots, for people whose "
        "windowsill is a nursery.",
        "art_prompt": "Bold flat-vector illustration of three glass propagation "
        "jars in a row holding cuttings with visible root systems, thick black "
        "outlines, sage green and terracotta palette, caption 'ROOTING FOR YOU' "
        "beneath, isolated on a plain white background, centered, no mockup, "
        "no text watermark",
        "style": "botanical flat vector",
        "palette": ["#5f7d4f", "#c1663f", "#2f2a26"],
        "keywords": [
            "plant propagation", "houseplant shirt", "plant lover gift",
            "propagation station", "plant parent tee", "cutting jar",
            "gardening gift", "plant mom shirt", "botanical tee",
        ],
    },
)


@dataclass
class NicheBrief:
    """A validated creative brief, ready for the Crafter."""

    niche: str
    audience: str
    concept: str
    art_prompt: str
    style: str = ""
    palette: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source: str = "openrouter"

    def to_dict(self) -> dict[str, Any]:
        return {
            "niche": self.niche,
            "audience": self.audience,
            "concept": self.concept,
            "art_prompt": self.art_prompt,
            "style": self.style,
            "palette": list(self.palette),
            "keywords": list(self.keywords),
            "source": self.source,
        }


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull one JSON object out of a model response.

    Tolerates the two things models do anyway: wrapping the object in a fenced
    block, and prefixing it with a sentence.
    """
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


class Scout:
    """Finds the niche and writes the art brief."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazily constructed OpenRouter client, shared across calls."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.settings.openrouter_base_url,
                api_key=self.settings.openrouter_api_key or "not-set",
                default_headers=self.settings.openrouter_headers,
                timeout=self.settings.request_timeout_seconds,
                max_retries=self.settings.max_retries,
            )
        return self._client

    async def find_niche(self, hint: str | None = None) -> NicheBrief:
        """Produce one brief.

        Falls back to a curated brief when OpenRouter is not configured or the
        call fails, so the pipeline is always runnable.
        """
        if not self.settings.openrouter_configured:
            logger.warning("OPENROUTER_API_KEY unset - Scout using a curated brief")
            return self._fallback(reason="no api key")

        user_prompt = (
            "Find one print-on-demand niche and write its brief now."
            if not hint
            else f"Find one print-on-demand niche related to: {hint}. Write its brief now."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.text_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.9,
                max_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001 - any transport failure degrades
            logger.error("Scout call failed ({}); using a curated brief", exc)
            return self._fallback(reason=f"api error: {exc}")

        content = (response.choices[0].message.content or "") if response.choices else ""
        try:
            payload = _extract_json(content)
        except ValueError as exc:
            logger.error("Scout returned unparseable JSON ({}); using a curated brief", exc)
            return self._fallback(reason="unparseable response")

        brief = self._coerce(payload)
        if brief is None:
            logger.error("Scout brief was missing required fields; using a curated brief")
            return self._fallback(reason="incomplete brief")

        return self._screen(brief)

    # ---- internals ----------------------------------------------------------
    def _coerce(self, payload: dict[str, Any]) -> NicheBrief | None:
        """Validate and normalise a model payload into a brief."""

        def text(key: str) -> str:
            value = payload.get(key, "")
            return value.strip() if isinstance(value, str) else ""

        def string_list(key: str) -> list[str]:
            value = payload.get(key, [])
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()]

        niche, concept, art_prompt = text("niche"), text("concept"), text("art_prompt")
        if not niche or not art_prompt:
            return None

        return NicheBrief(
            niche=niche[:120],
            audience=text("audience")[:200] or "general gift buyers",
            concept=concept or niche,
            art_prompt=art_prompt,
            style=text("style")[:60],
            palette=string_list("palette")[:6],
            keywords=[kw.lower() for kw in string_list("keywords")][:12],
        )

    def _screen(self, brief: NicheBrief) -> NicheBrief:
        """Repair a brief that trips the IP screen, rather than discarding it."""
        combined = " | ".join([brief.niche, brief.concept, brief.art_prompt])
        result = screen_text(combined)
        if result.ok:
            return brief

        logger.warning("Scout brief tripped the IP screen: {}", result.reason())
        brief.art_prompt = sanitise_prompt(brief.art_prompt)
        brief.concept = sanitise_prompt(brief.concept)
        brief.niche = sanitise_prompt(brief.niche)

        recheck = screen_text(" | ".join([brief.niche, brief.concept, brief.art_prompt]))
        if not recheck.ok:
            logger.error("Brief still blocked after sanitising; falling back")
            return self._fallback(reason="ip screen")

        brief.source = "openrouter+sanitised"
        return brief

    def _fallback(self, reason: str) -> NicheBrief:
        """A curated brief, chosen at random so repeated runs differ."""
        chosen = random.choice(FALLBACK_BRIEFS)
        logger.info("Scout fallback brief: {} ({})", chosen["niche"], reason)
        return NicheBrief(source=f"fallback ({reason})", **chosen)

    async def aclose(self) -> None:
        """Release the underlying HTTP connections."""
        if self._client is not None:
            await self._client.close()
            self._client = None
