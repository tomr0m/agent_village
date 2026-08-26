"""Intellectual-property screening for generated listing copy and art prompts.

Print-on-demand accounts are suspended for trademark infringement faster than
for anything else, so this runs before a listing is ever shown to a human and
again before it is published.

The screen is deliberately conservative and deliberately dumb: a curated list of
protected marks, franchise names and high-risk phrase patterns, matched on word
boundaries. It is not a legal opinion and does not pretend to be one — it exists
to catch the obvious infringement an image model will happily produce when asked
for "cartoon mouse t-shirt design".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from loguru import logger


class Severity(str, Enum):
    """How hard a hit should stop the pipeline."""

    BLOCK = "BLOCK"
    WARN = "WARN"


#: Marks that must never appear. Brands, franchises, characters and leagues that
#: are aggressively enforced on print-on-demand marketplaces.
BLOCKED_MARKS: dict[str, tuple[str, ...]] = {
    "entertainment": (
        "disney", "pixar", "marvel", "avengers", "spider-man", "spiderman",
        "star wars", "mandalorian", "baby yoda", "grogu", "mickey mouse",
        "minnie mouse", "winnie the pooh", "frozen elsa", "moana", "encanto",
        "harry potter", "hogwarts", "gryffindor", "lord of the rings",
        "game of thrones", "stranger things", "squid game", "bluey",
        "paw patrol", "peppa pig", "cocomelon", "sesame street", "muppets",
        "looney tunes", "bugs bunny", "scooby doo", "the simpsons",
        "rick and morty", "south park", "family guy", "spongebob",
        "hello kitty", "sanrio", "studio ghibli", "totoro",
    ),
    "gaming": (
        "nintendo", "pokemon", "pikachu", "mario", "super mario", "luigi",
        "zelda", "kirby", "sonic the hedgehog", "minecraft", "fortnite",
        "roblox", "call of duty", "playstation", "xbox", "among us",
        "animal crossing", "league of legends", "overwatch",
    ),
    "anime": (
        "naruto", "dragon ball", "goku", "one piece", "attack on titan",
        "demon slayer", "my hero academia", "jujutsu kaisen", "sailor moon",
        "death note", "hunter x hunter", "chainsaw man",
    ),
    "brands": (
        "nike", "just do it", "adidas", "puma", "reebok", "under armour",
        "supreme", "gucci", "louis vuitton", "chanel", "prada", "versace",
        "coca-cola", "coca cola", "pepsi", "starbucks", "mcdonald",
        "apple inc", "iphone", "google", "microsoft", "amazon prime",
        "netflix", "spotify", "tesla", "harley davidson", "jeep",
        "lego", "barbie", "hot wheels", "monster energy", "red bull",
    ),
    "sports": (
        "nfl", "nba", "mlb", "nhl", "fifa", "uefa", "premier league",
        "olympics", "olympic", "super bowl", "world cup", "wwe",
        "manchester united", "real madrid", "lakers", "yankees",
    ),
    "music": (
        "taylor swift", "eras tour", "beyonce", "drake", "bts", "blackpink",
        "grateful dead", "rolling stones", "pink floyd", "nirvana",
        "the beatles", "metallica", "ac/dc",
    ),
}

#: Phrases that are usually fine but often ride alongside infringement. These
#: warn rather than block, because the false-positive rate is real.
RISKY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bofficial(?:ly)?\s+licen[sc]ed\b", "claims an official licence"),
    (r"\bofficial\s+merchandise\b", "claims official merchandise"),
    (r"\bauthenti[c|city]\w*\s+brand\b", "claims brand authenticity"),
    (r"\binspired\s+by\s+[A-Z]", "'inspired by' a named work"),
    (r"\bfan\s*(?:art|made|merch)\b", "fan-work framing"),
    (r"\bparody\s+of\b", "parody framing"),
    (r"\bin\s+the\s+style\s+of\s+[A-Z]", "imitates a named artist"),
    (r"\bcopyright(?:ed)?\b", "mentions copyright"),
    (r"\btrademark(?:ed)?\b", "mentions trademark"),
    (r"\b(?:tm|®|©)\b", "carries a rights symbol"),
    (r"\breplica\b", "describes a replica"),
    (r"\bbootleg\b", "describes a bootleg"),
)

#: Characters people use to slip a mark past a naive substring check.
_HOMOGLYPHS = str.maketrans(
    {
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
        "$": "s", "@": "a", "!": "i", "|": "i",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-",
    }
)


@dataclass(frozen=True)
class Hit:
    """One screening match."""

    term: str
    category: str
    severity: Severity
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.severity.value}: {self.term} [{self.category}]{suffix}"


@dataclass(frozen=True)
class ScreenResult:
    """The verdict for one piece of text, or for a whole listing."""

    ok: bool
    hits: tuple[Hit, ...] = field(default=())

    @property
    def blocking(self) -> tuple[Hit, ...]:
        return tuple(hit for hit in self.hits if hit.severity is Severity.BLOCK)

    @property
    def warnings(self) -> tuple[Hit, ...]:
        return tuple(hit for hit in self.hits if hit.severity is Severity.WARN)

    def reason(self) -> str:
        """A single line explaining the verdict, for a status column."""
        if self.ok and not self.hits:
            return "No trademark or IP concerns detected."
        if self.ok:
            return "Cleared with warnings: " + "; ".join(h.term for h in self.warnings)
        return "Blocked on: " + "; ".join(h.term for h in self.blocking)

    def report(self) -> str:
        """A multi-line report suitable for storing on the listing."""
        if not self.hits:
            return "IP screen: clean."
        lines = ["IP screen:"]
        for hit in self.hits:
            lines.append(f"  - {hit}")
        return "\n".join(lines)


def normalise(text: str) -> str:
    """Fold text so that near-miss spellings still match.

    Strips accents, maps common homoglyph substitutions, collapses runs of
    punctuation and whitespace, and lowercases.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    folded = stripped.lower().translate(_HOMOGLYPHS)
    # Keep word characters and single spaces; a mark split by punctuation should
    # still be caught ("m.a.r.v.e.l" -> "marvel" is out of scope, but
    # "marvel-style" is not).
    collapsed = re.sub(r"[^a-z0-9'&/ -]+", " ", folded)
    return re.sub(r"\s+", " ", collapsed).strip()


def _term_pattern(term: str) -> re.Pattern[str]:
    """Word-boundary pattern for a mark, tolerant of internal spacing."""
    parts = [re.escape(part) for part in term.split()]
    body = r"[\s\-]*".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


#: Compiled once: the screen runs on every generated listing.
_COMPILED_MARKS: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(
    (_term_pattern(term), term, category)
    for category, terms in BLOCKED_MARKS.items()
    for term in terms
)

_COMPILED_RISKY: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), detail) for pattern, detail in RISKY_PATTERNS
)


def screen_text(text: str, *, extra_blocked: Sequence[str] = ()) -> ScreenResult:
    """Screen a single string.

    :param text: the copy or prompt to check.
    :param extra_blocked: additional marks to treat as blocking, for a caller
        that maintains its own list.
    """
    if not text or not text.strip():
        return ScreenResult(ok=True)

    haystack = normalise(text)
    hits: list[Hit] = []
    seen: set[str] = set()

    for pattern, term, category in _COMPILED_MARKS:
        if pattern.search(haystack) and term not in seen:
            seen.add(term)
            hits.append(Hit(term=term, category=category, severity=Severity.BLOCK))

    for term in extra_blocked:
        normalised = normalise(term)
        if normalised and _term_pattern(normalised).search(haystack) and normalised not in seen:
            seen.add(normalised)
            hits.append(Hit(term=normalised, category="custom", severity=Severity.BLOCK))

    # Risky patterns run against the ORIGINAL text: several depend on
    # capitalisation ("inspired by Batman") that normalisation would destroy.
    for pattern, detail in _COMPILED_RISKY:
        match = pattern.search(text)
        if match and match.group(0).lower() not in seen:
            seen.add(match.group(0).lower())
            hits.append(
                Hit(
                    term=match.group(0).strip(),
                    category="phrasing",
                    severity=Severity.WARN,
                    detail=detail,
                )
            )

    ordered = tuple(sorted(hits, key=lambda hit: (hit.severity is Severity.WARN, hit.term)))
    return ScreenResult(ok=not any(h.severity is Severity.BLOCK for h in ordered), hits=ordered)


def screen_many(
    fields: dict[str, str] | Iterable[tuple[str, str]],
    *,
    extra_blocked: Sequence[str] = (),
) -> ScreenResult:
    """Screen several named fields and merge the verdicts.

    Each hit is labelled with the field it came from, so a report can point at
    the title rather than at "the listing".
    """
    items = fields.items() if isinstance(fields, dict) else fields
    merged: list[Hit] = []

    for name, value in items:
        result = screen_text(value or "", extra_blocked=extra_blocked)
        for hit in result.hits:
            merged.append(
                Hit(
                    term=hit.term,
                    category=f"{name}:{hit.category}",
                    severity=hit.severity,
                    detail=hit.detail,
                )
            )

    ok = not any(hit.severity is Severity.BLOCK for hit in merged)
    if not ok:
        logger.warning(
            "IP screen blocked: {}",
            ", ".join(hit.term for hit in merged if hit.severity is Severity.BLOCK),
        )
    return ScreenResult(ok=ok, hits=tuple(merged))


def sanitise_prompt(prompt: str) -> str:
    """Strip blocked marks out of a generation prompt.

    Used by the Scout so a risky idea is repaired before it costs an image
    generation, rather than being caught after the money is spent.
    """
    cleaned = prompt
    for _pattern, term, _category in _COMPILED_MARKS:
        cleaned = re.sub(
            rf"(?i)(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", "original character", cleaned
        )
    return re.sub(r"\s+", " ", cleaned).strip()
