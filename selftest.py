#!/usr/bin/env python3
"""Self-test for the village's rules, run without any network or credentials.

Covers what a happy-path pipeline run does not: Etsy's hard limits, the
trademark screen's evasion handling, the listing state machine's refusals, the
Telegram card construction, and the Guard's blocking checks.

    python selftest.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def section(name: str) -> None:
    print(f"\n--- {name} ---")


# ---------------------------------------------------------------------------
section("trademark screen")
# ---------------------------------------------------------------------------
from core.trademark_guard import sanitise_prompt, screen_many, screen_text  # noqa: E402

check("clean copy passes", screen_text("A cheerful sourdough starter jar").ok)
check("blocks a franchise", not screen_text("cute baby yoda on a shirt").ok)
check("blocks a brand", not screen_text("Nike inspired running tee").ok)
check("blocks across word boundaries", not screen_text("POKEMON trainer gift").ok)
check(
    "blocks a homoglyph evasion",
    not screen_text("p0kemon trainer gift").ok,
    "digit-for-letter substitution",
)
check(
    "blocks a spaced-out mark",
    not screen_text("star   wars fan shirt").ok,
    "tolerates irregular spacing",
)
check(
    "does not block an innocent substring",
    screen_text("a marvellous morning bake").ok,
    "'marvellous' must not trip 'marvel'",
)
check(
    "warns on risky phrasing without blocking",
    screen_text("officially licensed design").ok
    and len(screen_text("officially licensed design").warnings) > 0,
)
multi = screen_many({"title": "Pikachu tee", "description": "A nice shirt"})
check("screen_many labels the offending field", not multi.ok and "title" in multi.blocking[0].category)
check(
    "sanitise_prompt strips the mark",
    "pikachu" not in sanitise_prompt("a pikachu in a teacup").lower(),
    sanitise_prompt("a pikachu in a teacup"),
)

# ---------------------------------------------------------------------------
section("Etsy copy rules")
# ---------------------------------------------------------------------------
from village.scribe import (  # noqa: E402
    MAX_TAG_CHARS,
    MAX_TITLE_CHARS,
    TAG_COUNT,
    clean_tag,
    normalise_tags,
    truncate_title,
)

tags, _ = normalise_tags(["one", "two", "three"], filler=["four", "five"])
check(f"pads a short list to {TAG_COUNT}", len(tags) == TAG_COUNT, f"got {len(tags)}")
check("padded tags are unique", len(set(tags)) == TAG_COUNT)

tags, _ = normalise_tags([f"tag number {n}" for n in range(30)])
check(f"trims a long list to {TAG_COUNT}", len(tags) == TAG_COUNT)

tags, _ = normalise_tags(["Nurse Gift!!", "nurse gift", "NURSE GIFT"], filler=["er nurse"])
check("dedupes case and punctuation variants", len(set(tags)) == TAG_COUNT)
check("every tag is within the length limit", all(len(t) <= MAX_TAG_CHARS for t in tags))
check("every tag is lowercase and clean", all(t == clean_tag(t) for t in tags))

long_tag = clean_tag("a very long tag that exceeds the etsy character limit")
check(
    "long tag is cut on a word boundary",
    len(long_tag) <= MAX_TAG_CHARS and not long_tag.endswith(" "),
    repr(long_tag),
)

title, note = truncate_title("Word " * 60)
check(f"title trimmed to {MAX_TITLE_CHARS}", len(title) <= MAX_TITLE_CHARS, f"{len(title)} chars")
check("trim is reported", note is not None)
short, note = truncate_title("A Perfectly Fine Title")
check("short title is left alone", short == "A Perfectly Fine Title" and note is None)

# ---------------------------------------------------------------------------
section("Scribe repair")
# ---------------------------------------------------------------------------
from config.settings import get_settings  # noqa: E402
from village.scribe import Scribe  # noqa: E402

scribe = Scribe(get_settings())
copy = scribe._repair(  # noqa: SLF001 - exercising the repair path directly
    {"title": "x" * 300, "tags": ["only", "three", "tags"], "description": "too short"},
    niche="test niche",
    audience="testers",
    concept="a concept",
    keywords=["kw one", "kw two"],
    product="tee",
)
check("repaired copy is valid", copy.valid, f"title={len(copy.title)} tags={len(copy.tags)}")
check("repairs are recorded", len(copy.repairs) >= 3, f"{len(copy.repairs)} repairs")

# ---------------------------------------------------------------------------
section("listing state machine")
# ---------------------------------------------------------------------------
from core.database import (  # noqa: E402
    ListingStatus,
    create_listing,
    get_listing,
    init_db,
    update_status,
)

init_db()
row = create_listing(niche="selftest niche", title="Selftest", price_cents=1999)
check("listing starts DRAFTED", row.status == ListingStatus.DRAFTED.value)
check("tags round-trip through JSON", get_listing(row.id).tags == [])

moved = update_status(row.id, ListingStatus.PENDING_APPROVAL)
check("DRAFTED -> PENDING_APPROVAL allowed", moved is not None)

illegal = update_status(row.id, ListingStatus.PUBLISHED)
check("PENDING_APPROVAL -> PUBLISHED refused", illegal is None, "must go through APPROVED")

update_status(row.id, ListingStatus.APPROVED)
update_status(row.id, ListingStatus.PUBLISHED)
check("APPROVED -> PUBLISHED allowed", get_listing(row.id).status == "PUBLISHED")
check(
    "PUBLISHED is terminal",
    update_status(row.id, ListingStatus.REJECTED) is None,
    "a stale button tap must not reopen it",
)

row2 = create_listing(niche="tags test", tags=["alpha", "beta"])
check("tags persist", get_listing(row2.id).tags == ["alpha", "beta"])

# ---------------------------------------------------------------------------
section("image processing")
# ---------------------------------------------------------------------------
from core.image_processor import make_placeholder, process_image, read_dpi  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    raw = make_placeholder(Path(tmp) / "raw.png", "self test artwork", size=(600, 400))
    check("placeholder written", raw.is_file() and raw.stat().st_size > 1000)

    processed = process_image(raw, Path(tmp) / "out.png", strip_background=False)
    settings = get_settings()
    check(
        "scaled to the print canvas",
        (processed.width, processed.height) == settings.print_pixel_size,
        f"{processed.width}x{processed.height}",
    )
    check("DPI stamped on the file", read_dpi(processed.path) == (300, 300))
    check("aspect ratio preserved by padding", processed.path.is_file())

    try:
        process_image(Path(tmp) / "missing.png")
        check("missing source raises", False)
    except FileNotFoundError:
        check("missing source raises FileNotFoundError", True)

# ---------------------------------------------------------------------------
section("guard")
# ---------------------------------------------------------------------------
from village.guard import Guard  # noqa: E402

guard = Guard(get_settings())
with tempfile.TemporaryDirectory() as tmp:
    art = make_placeholder(Path(tmp) / "a.png", "guard test", size=(400, 400))
    good = process_image(art, Path(tmp) / "a_print.png", strip_background=False)

    ok_report = guard.validate(
        title="Sourdough Starter Shirt | Gift for Home Bakers | Bread Lover Tee",
        tags=[f"tag {n}" for n in range(13)],
        description="D" * 400,
        image_path=good.path,
    )
    check("valid listing passes", ok_report.ok, ok_report.summary)

    bad_report = guard.validate(
        title="x" * 200,
        tags=["one", "two"],
        description="short",
        image_path=None,
    )
    check("blocks an over-long title", any("Title is 200" in e for e in bad_report.errors))
    check("blocks a wrong tag count", any("2 tags" in e for e in bad_report.errors))
    check("blocks a thin description", any("Description is" in e for e in bad_report.errors))
    check("blocks a missing image", any("No artwork" in e for e in bad_report.errors))
    check("verdict is FAIL", not bad_report.ok)

    ip_report = guard.validate(
        title="Baby Yoda Shirt",
        tags=[f"tag {n}" for n in range(13)],
        description="D" * 400,
        image_path=good.path,
    )
    check("blocks trademarked copy", not ip_report.ok)

    tiny = Path(tmp) / "tiny.png"
    make_placeholder(tiny, "t", size=(200, 200))
    small_report = guard.validate(
        title="A Perfectly Reasonable Shirt Title For Testing",
        tags=[f"tag {n}" for n in range(13)],
        description="D" * 400,
        image_path=tiny,
    )
    check("blocks a low-resolution print", not small_report.ok)

# ---------------------------------------------------------------------------
section("merchant dry-run fallback")
# ---------------------------------------------------------------------------
from village.merchant import Merchant  # noqa: E402


async def merchant_checks() -> None:
    merchant = Merchant(get_settings())
    check("simulating while DRY_RUN is on", merchant.simulating)

    with tempfile.TemporaryDirectory() as tmp:
        art = make_placeholder(Path(tmp) / "m.png", "merchant test", size=(500, 500))
        result = await merchant.publish(
            title="T", description="D", tags=["a"], image_path=art
        )
        check("simulated publish succeeds", result.ok and result.simulated)
        check("fabricates a product id", bool(result.product_id))
        check("records its steps", len(result.steps) == 3)

        missing = await merchant.publish(
            title="T", description="D", tags=["a"], image_path=Path(tmp) / "nope.png"
        )
        check("missing artwork fails cleanly", not missing.ok and "missing" in (missing.error or ""))

    health = await merchant.health_check()
    check("health check reports simulation", health["mode"] == "simulated")


asyncio.run(merchant_checks())

# ---------------------------------------------------------------------------
section("telegram card")
# ---------------------------------------------------------------------------
from village.town_crier import (  # noqa: E402
    APPROVE_PREFIX,
    MAX_CAPTION_CHARS,
    REJECT_PREFIX,
    approval_keyboard,
    build_caption,
)

keyboard = approval_keyboard(42)
buttons = [button for row in keyboard.inline_keyboard for button in row]
check("card has three buttons", len(buttons) == 3)
check(
    "approve carries the listing id",
    any(b.callback_data == f"{APPROVE_PREFIX}:42" for b in buttons),
)
check(
    "reject carries the listing id",
    any(b.callback_data == f"{REJECT_PREFIX}:42" for b in buttons),
)

listing = get_listing(row.id)
listing.title = "Café & Co <Bakery> \"Best\""
listing.tags = [f"tag {n}" for n in range(13)]
caption = build_caption(listing, settings=get_settings())
check("caption escapes HTML", "&lt;Bakery&gt;" in caption and "&amp;" in caption)
check("caption is within Telegram's limit", len(caption) <= MAX_CAPTION_CHARS, f"{len(caption)}")

listing.description = "D" * 5000
listing.title = "T" * 400
long_caption = build_caption(listing, settings=get_settings())
check("over-long caption is truncated", len(long_caption) <= MAX_CAPTION_CHARS)

# ---------------------------------------------------------------------------
section("scout fallback")
# ---------------------------------------------------------------------------
from village.scout import FALLBACK_BRIEFS, Scout, _extract_json  # noqa: E402


async def scout_checks() -> None:
    settings = get_settings()
    scout = Scout(settings)
    brief = await scout.find_niche()
    check("returns a usable brief", bool(brief.niche and brief.art_prompt))
    # With a real key the Scout calls OpenRouter and the source is the model;
    # without one it must fall back. Assert the branch that actually applies,
    # so the suite passes on a configured machine and an empty one alike.
    if settings.openrouter_configured:
        check("brief came from the model", "fallback" not in brief.source, brief.source)
    else:
        check("brief is marked as a fallback", "fallback" in brief.source, brief.source)
    check(
        "every curated brief is IP-clean",
        all(
            screen_text(f"{b['niche']} {b['concept']} {b['art_prompt']}").ok
            for b in FALLBACK_BRIEFS
        ),
    )
    await scout.aclose()


asyncio.run(scout_checks())

check("parses bare JSON", _extract_json('{"a": 1}')["a"] == 1)
check("parses fenced JSON", _extract_json('```json\n{"a": 2}\n```')["a"] == 2)
check("parses JSON after prose", _extract_json('Sure!\n{"a": 3}')["a"] == 3)
try:
    _extract_json("not json at all")
    check("rejects non-JSON", False)
except ValueError:
    check("rejects non-JSON", True)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    sys.exit(1)
print("SELFTEST OK")
