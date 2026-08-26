"""The Scribe: writes the Etsy listing copy.

Etsy's constraints are hard limits, not guidance — a 141-character title is
rejected, a 14th tag is dropped, a 21-character tag is refused. So the Scribe
does two things: it asks for copy that already fits, and then it *repairs*
whatever comes back so the listing is always valid. A model that returns 11 tags
is a normal Tuesday; a pipeline that fails because of it is a bad design.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from loguru import logger
from openai import AsyncOpenAI

from config.settings import Settings, get_settings
from core.trademark_guard import screen_many

#: Etsy's published limits.
MAX_TITLE_CHARS = 140
TAG_COUNT = 13
MAX_TAG_CHARS = 20
MIN_DESCRIPTION_CHARS = 200

SYSTEM_PROMPT = f"""You are an Etsy SEO copywriter who has ranked hundreds of \
print-on-demand listings. You write for the search algorithm AND for the buyer \
who reads the first line and decides.

Return ONE JSON object and nothing else:

{{
  "title": "string",
  "tags": ["exactly {TAG_COUNT} strings"],
  "description": "string"
}}

TITLE RULES:
- At most {MAX_TITLE_CHARS} characters. Count them.
- Front-load the two highest-intent keywords; Etsy weights the opening words most.
- Read as a phrase a human would say, not a keyword pile. Separate ideas with \
"|" or "-", never commas.
- Name the product type and the buyer ("Nurse Coffee Shirt", "Gift for Bakers").
- No ALL CAPS words, no emoji, no quotes.

TAG RULES:
- EXACTLY {TAG_COUNT} tags. Not 12, not 14.
- Each at most {MAX_TAG_CHARS} characters INCLUDING spaces. Count them.
- Lowercase. Letters, numbers and single spaces only. No punctuation, no hyphens.
- Multi-word phrases beat single words: buyers search phrases.
- No tag repeats another tag's exact wording, and no tag repeats the whole title.
- Mix intent types: product ("nurse tee"), occasion ("nurse grad gift"), \
recipient ("gift for nurses"), style ("retro nurse art"), niche ("er nurse").

DESCRIPTION RULES:
- At least {MIN_DESCRIPTION_CHARS} characters.
- First sentence sells the feeling, names the buyer, and repeats the main keyword.
- Then a short paragraph on the design and who it is for.
- Then a "Details" list: material weight, unisex fit, print method, care.
- Then a "Shipping & Returns" line and a "Gift note" line.
- Plain text with line breaks. No markdown, no HTML, no emoji spam (at most two).

HARD RULES:
- Never reference a real brand, franchise, character, team, band, film, game or \
living person, in any field.
- Never promise a delivery date, a discount, or a review.
- Output raw JSON. No markdown, no code fences, no commentary."""


@dataclass
class ListingCopy:
    """Validated, ready-to-publish Etsy copy."""

    title: str
    tags: list[str]
    description: str
    source: str = "openrouter"
    repairs: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Does this copy satisfy every hard Etsy constraint?"""
        return (
            bool(self.title)
            and len(self.title) <= MAX_TITLE_CHARS
            and len(self.tags) == TAG_COUNT
            and all(0 < len(tag) <= MAX_TAG_CHARS for tag in self.tags)
            and len(set(self.tags)) == TAG_COUNT
            and len(self.description) >= MIN_DESCRIPTION_CHARS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "tags": list(self.tags),
            "description": self.description,
            "source": self.source,
            "repairs": list(self.repairs),
        }


def clean_tag(raw: str) -> str:
    """Normalise one tag to Etsy's character rules.

    Lowercase, letters/digits/single-spaces only, trimmed to the length limit on
    a word boundary where possible — a tag cut mid-word reads as a typo.
    """
    tag = re.sub(r"[^a-z0-9 ]+", " ", str(raw).lower())
    tag = re.sub(r"\s+", " ", tag).strip()
    if len(tag) <= MAX_TAG_CHARS:
        return tag

    clipped = tag[:MAX_TAG_CHARS]
    if " " in clipped:
        trimmed = clipped.rsplit(" ", 1)[0].strip()
        if len(trimmed) >= 4:
            return trimmed
    return clipped.strip()


def normalise_tags(
    candidates: Iterable[str],
    *,
    filler: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Force any tag list to exactly :data:`TAG_COUNT` valid, unique tags.

    :param candidates: the model's tags, best first.
    :param filler: keywords to top up from when the model returns too few.
    :returns: the tags and a list of the repairs performed.
    """
    repairs: list[str] = []
    seen: set[str] = set()
    tags: list[str] = []

    for candidate in candidates:
        tag = clean_tag(candidate)
        if not tag or tag in seen:
            continue
        if tag != str(candidate).strip().lower():
            repairs.append(f"cleaned {candidate!r} -> {tag!r}")
        seen.add(tag)
        tags.append(tag)
        if len(tags) == TAG_COUNT:
            break

    if len(tags) > TAG_COUNT:  # pragma: no cover - guarded by the loop above
        repairs.append(f"dropped {len(tags) - TAG_COUNT} surplus tag(s)")
        tags = tags[:TAG_COUNT]

    if len(tags) < TAG_COUNT:
        for candidate in filler:
            if len(tags) == TAG_COUNT:
                break
            tag = clean_tag(candidate)
            if not tag or tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
            repairs.append(f"added filler tag {tag!r}")

    # Still short: pad from the tags we have, so the count is always exact.
    suffixes = ("gift", "shirt", "tee", "design", "idea", "present", "art")
    index = 0
    while len(tags) < TAG_COUNT and tags:
        base = tags[index % len(tags)]
        suffix = suffixes[index % len(suffixes)]
        candidate = clean_tag(f"{base} {suffix}")
        index += 1
        if candidate and candidate not in seen:
            seen.add(candidate)
            tags.append(candidate)
            repairs.append(f"derived tag {candidate!r}")
        if index > 120:  # pragma: no cover - loop guard
            break

    while len(tags) < TAG_COUNT:  # pragma: no cover - only when tags was empty
        placeholder = clean_tag(f"custom design {len(tags) + 1}")
        tags.append(placeholder)
        seen.add(placeholder)
        repairs.append(f"placeholder tag {placeholder!r}")

    return tags[:TAG_COUNT], repairs


def truncate_title(title: str) -> tuple[str, str | None]:
    """Trim a title to the limit on a separator or word boundary."""
    cleaned = re.sub(r"\s+", " ", str(title).replace('"', "").strip())
    if len(cleaned) <= MAX_TITLE_CHARS:
        return cleaned, None

    window = cleaned[:MAX_TITLE_CHARS]
    for separator in (" | ", " - ", " "):
        if separator in window:
            trimmed = window.rsplit(separator, 1)[0].strip(" |-")
            if len(trimmed) >= 40:
                return trimmed, f"title trimmed from {len(cleaned)} to {len(trimmed)} chars"
    trimmed = window.strip()
    return trimmed, f"title hard-cut to {len(trimmed)} chars"


def _extract_json(raw: str) -> dict[str, Any]:
    """Same tolerant JSON extraction the Scout uses."""
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


class Scribe:
    """Writes and repairs the listing copy."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.settings.openrouter_base_url,
                api_key=self.settings.openrouter_api_key or "not-set",
                default_headers=self.settings.openrouter_headers,
                timeout=self.settings.request_timeout_seconds,
                max_retries=self.settings.max_retries,
            )
        return self._client

    async def write(
        self,
        *,
        niche: str,
        audience: str,
        concept: str,
        keywords: Sequence[str] = (),
        product: str = "unisex heavy cotton t-shirt",
    ) -> ListingCopy:
        """Produce validated copy for one listing."""
        if not self.settings.openrouter_configured:
            logger.warning("OPENROUTER_API_KEY unset - Scribe writing template copy")
            return self._template(niche, audience, concept, keywords, product, "no api key")

        user_prompt = (
            f"Product: {product}\n"
            f"Niche: {niche}\n"
            f"Buyer: {audience}\n"
            f"Design concept: {concept}\n"
            f"Seed keywords: {', '.join(keywords) if keywords else '(none supplied)'}\n\n"
            "Write the listing JSON now."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.text_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1600,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Scribe call failed ({}); using template copy", exc)
            return self._template(niche, audience, concept, keywords, product, f"api error: {exc}")

        content = (response.choices[0].message.content or "") if response.choices else ""
        try:
            payload = _extract_json(content)
        except ValueError as exc:
            logger.error("Scribe returned unparseable JSON ({}); using template copy", exc)
            return self._template(niche, audience, concept, keywords, product, "unparseable")

        copy = self._repair(payload, niche, audience, concept, keywords, product)
        screened = screen_many(
            {"title": copy.title, "tags": " ".join(copy.tags), "description": copy.description}
        )
        if not screened.ok:
            logger.error("Scribe copy blocked by the IP screen; using template copy")
            return self._template(
                niche, audience, concept, keywords, product, f"ip screen: {screened.reason()}"
            )

        if copy.repairs:
            logger.info("Scribe repairs: {}", "; ".join(copy.repairs))
        return copy

    # ---- internals ----------------------------------------------------------
    def _repair(
        self,
        payload: dict[str, Any],
        niche: str,
        audience: str,
        concept: str,
        keywords: Sequence[str],
        product: str,
    ) -> ListingCopy:
        """Coerce a model payload into copy that always satisfies Etsy."""
        repairs: list[str] = []

        raw_title = payload.get("title") or f"{niche.title()} {product.title()}"
        title, note = truncate_title(str(raw_title))
        if note:
            repairs.append(note)
        if not title:
            title = f"{niche.title()} Shirt"
            repairs.append("title was empty; synthesised from the niche")

        raw_tags = payload.get("tags")
        candidates = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
        if len(candidates) != TAG_COUNT:
            repairs.append(f"model returned {len(candidates)} tags; normalised to {TAG_COUNT}")
        tags, tag_repairs = normalise_tags(
            candidates, filler=[*keywords, *self._derive_keywords(niche, audience)]
        )
        repairs.extend(tag_repairs)

        description = str(payload.get("description") or "").strip()
        if len(description) < MIN_DESCRIPTION_CHARS:
            repairs.append(
                f"description was {len(description)} chars; extended to the minimum"
            )
            description = self._compose_description(
                title, niche, audience, concept, product, existing=description
            )

        return ListingCopy(title=title, tags=tags, description=description, repairs=repairs)

    @staticmethod
    def _derive_keywords(niche: str, audience: str) -> list[str]:
        """Cheap keyword seeds from the brief, used only to top up tags."""
        words = [w for w in re.split(r"[^a-z0-9]+", f"{niche} {audience}".lower()) if len(w) > 2]
        head = words[:4]
        seeds = [" ".join(head[:2]), " ".join(head[1:3])] if len(head) >= 3 else []
        return [
            *(s for s in seeds if s.strip()),
            *(f"{w} shirt" for w in head),
            *(f"{w} gift" for w in head),
            "gift idea",
            "graphic tee",
            "unisex shirt",
        ]

    @staticmethod
    def _compose_description(
        title: str,
        niche: str,
        audience: str,
        concept: str,
        product: str,
        *,
        existing: str = "",
    ) -> str:
        """Build a complete, compliant description."""
        opening = existing.strip() or (
            f"{concept.strip().rstrip('.')}. Made for {audience.strip().rstrip('.')}."
        )
        return (
            f"{opening}\n\n"
            f"This {product} was designed around one idea: {niche}. The artwork is "
            f"printed large and centred, with bold shapes that stay readable after "
            f"a hundred washes — not a faint transfer that cracks after three.\n\n"
            "Details\n"
            "- Unisex fit, true to size. Size up for a relaxed drape.\n"
            "- Soft mid-weight cotton, side-seamed, with a taped neck and shoulders.\n"
            "- Direct-to-garment print using water-based inks.\n"
            "- Machine wash cold inside out, tumble dry low, do not iron the print.\n\n"
            "Shipping & Returns\n"
            "Each shirt is printed to order and dispatched with tracking. If the fit "
            "is wrong or the print arrives damaged, message us and we will put it right.\n\n"
            "Gift note\n"
            f"Add a note at checkout and we will include it — this one lands well for "
            f"{audience.strip().rstrip('.')}."
        )

    def _template(
        self,
        niche: str,
        audience: str,
        concept: str,
        keywords: Sequence[str],
        product: str,
        reason: str,
    ) -> ListingCopy:
        """Deterministic copy used when the model is unavailable."""
        pretty = niche.strip().title() or "Original Design"
        title, _ = truncate_title(
            f"{pretty} Shirt | Gift for {audience.split(',')[0].strip().title() or 'Fans'} "
            f"| Original {product.title()}"
        )
        tags, _ = normalise_tags(
            [*keywords, *self._derive_keywords(niche, audience)],
        )
        description = self._compose_description(title, niche, audience, concept, product)
        logger.info("Scribe template copy for {!r} ({})", niche, reason)
        return ListingCopy(
            title=title,
            tags=tags,
            description=description,
            source=f"template ({reason})",
            repairs=["generated from template"],
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
