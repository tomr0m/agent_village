"""The Night Scribe: gathers overnight crypto news while the village sleeps.

Reads RSS feeds and CoinGecko, de-duplicates against what is already held, and
writes anything new to ``raw_crypto_events``. It makes no editorial judgement —
that is the Morning Ledger's job. This one only decides what is *new* and
roughly what *kind* of story it is.

Every source is optional. A feed that 500s, a rate-limited API, a machine with
no network — each degrades to "fewer stories tonight" rather than to a failed
scan, because a newsletter built from four sources instead of five is still a
newsletter.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from loguru import logger

from config.settings import Settings, get_settings
from core import events

COINGECKO_MARKETS = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
    "&price_change_percentage=24h"
)

#: Words that mark a story as an exploit rather than market noise. Order does
#: not matter; one hit is enough, because the cost of mis-filing a story as an
#: exploit is that the editor sees it — which is what we want anyway.
EXPLOIT_WORDS = (
    "exploit", "hack", "hacked", "drained", "rug", "rugpull", "breach",
    "vulnerability", "attack", "attacker", "stolen", "theft", "compromise",
    "backdoor", "reentrancy", "flash loan", "phishing", "scam", "malicious",
)

MACRO_WORDS = (
    "fed", "federal reserve", "inflation", "cpi", "rate cut", "rate hike",
    "sec ", "regulation", "regulator", "etf", "treasury", "lawsuit", "congress",
    "tariff", "gdp", "jobs report", "macro",
)

#: How much price movement is worth writing down at all.
NOTABLE_MOVE_PERCENT = 8.0

#: Cap per scan, so one hyperactive feed cannot swamp an edition.
MAX_PER_FEED = 25


@dataclass
class ScanResult:
    """What one night scan found."""

    fetched: int = 0
    stored: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.stored} new of {self.fetched} seen"]
        if self.sources:
            parts.append(
                ", ".join(f"{name} {count}" for name, count in sorted(self.sources.items()))
            )
        if self.errors:
            parts.append(f"{len(self.errors)} source(s) failed")
        return " · ".join(parts)


def classify(headline: str, summary: str = "") -> str:
    """Bucket a story as exploit / macro / market.

    Exploit wins over macro when both match: "SEC sues exchange after $200m
    hack" is a heist story with a regulatory footnote, not the reverse.
    """
    text = f"{headline} {summary}".lower()
    if any(word in text for word in EXPLOIT_WORDS):
        return "exploit"
    if any(word in text for word in MACRO_WORDS):
        return "macro"
    return "market"


def _clean(text: str, limit: int = 600) -> str:
    """Strip tags and collapse whitespace out of feed HTML."""
    without_tags = re.sub(r"<[^>]+>", " ", str(text or ""))
    collapsed = re.sub(r"\s+", " ", without_tags).strip()
    return collapsed[:limit]


def _source_name(url: str) -> str:
    """A short, stable name for a feed, used as the ``source`` column."""
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    return host.split(".")[0][:60] or "unknown"


class NightScribe:
    """Collects overnight stories into the database."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ---- sources ------------------------------------------------------------
    async def fetch_feed(self, url: str) -> list[dict[str, Any]]:
        """One RSS feed, parsed into event dicts."""
        import feedparser  # noqa: PLC0415
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds, follow_redirects=True
        ) as client:
            # A browser-ish agent: several crypto publishers answer 403 to the
            # default httpx string.
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; agent-village/1.0)"},
            )
            response.raise_for_status()
            body = response.content

        parsed = await asyncio.to_thread(feedparser.parse, body)
        source = _source_name(url)

        stories: list[dict[str, Any]] = []
        for entry in (parsed.entries or [])[:MAX_PER_FEED]:
            link = str(getattr(entry, "link", "") or "").strip()
            headline = _clean(getattr(entry, "title", ""), 300)
            if not link or not headline:
                continue
            summary = _clean(
                getattr(entry, "summary", "") or getattr(entry, "description", "")
            )
            stories.append(
                {
                    "source": source,
                    "headline": headline,
                    "summary": summary,
                    "url": link,
                    "category": classify(headline, summary),
                }
            )
        return stories

    async def fetch_market_moves(self) -> list[dict[str, Any]]:
        """BTC/ETH movement plus the sharpest 24h gainers and losers.

        Written as events so the editor sees prices and news in one list rather
        than having to join two shapes together.
        """
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds
        ) as client:
            response = await client.get(COINGECKO_MARKETS)
            response.raise_for_status()
            coins = response.json()

        if not isinstance(coins, list) or not coins:
            return []

        def move(coin: dict[str, Any]) -> float:
            try:
                return float(coin.get("price_change_percentage_24h") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H")
        picked: list[dict[str, Any]] = []

        # The majors always matter, whatever they did.
        for coin in coins:
            if str(coin.get("id")) in ("bitcoin", "ethereum"):
                picked.append(coin)

        ranked = sorted(coins, key=move)
        picked.extend(ranked[:3])          # sharpest losers
        picked.extend(ranked[-3:])         # sharpest gainers

        seen: set[str] = set()
        stories: list[dict[str, Any]] = []
        for coin in picked:
            coin_id = str(coin.get("id") or "")
            if not coin_id or coin_id in seen:
                continue
            seen.add(coin_id)

            change = move(coin)
            major = coin_id in ("bitcoin", "ethereum")
            if not major and abs(change) < NOTABLE_MOVE_PERCENT:
                continue

            symbol = str(coin.get("symbol") or "").upper()
            price = coin.get("current_price")
            direction = "up" if change >= 0 else "down"
            stories.append(
                {
                    "source": "coingecko",
                    "headline": f"{symbol} {direction} {abs(change):.1f}% in 24h at ${price:,}",
                    "summary": (
                        f"{coin.get('name')} ({symbol}) trades at ${price:,}, "
                        f"{direction} {abs(change):.2f}% over 24 hours. "
                        f"Market cap ${coin.get('market_cap') or 0:,}."
                    ),
                    # Deterministic per coin per hour: the same scan twice in
                    # one hour is one row, not two.
                    "url": f"https://www.coingecko.com/en/coins/{coin_id}#{stamp}",
                    "category": "market",
                }
            )
        return stories

    # ---- the scan -----------------------------------------------------------
    async def scan(self) -> ScanResult:
        """One pass over every source. Never raises."""
        result = ScanResult()
        events.agent_working("night_scribe", "Reading the night's wires…", progress=0.2)
        events.log(f"Night Scribe is reading {len(self.settings.crypto_feed_list)} feeds…")

        feeds = self.settings.crypto_feed_list
        gathered: list[dict[str, Any]] = []

        for url in feeds:
            try:
                stories = await self.fetch_feed(url)
            except Exception as exc:  # noqa: BLE001 - one dead feed is not a failure
                name = _source_name(url)
                logger.warning("Feed {} unavailable: {}", name, exc)
                result.errors.append(f"{name}: {exc}")
                continue
            gathered.extend(stories)
            result.sources[_source_name(url)] = len(stories)

        try:
            moves = await self.fetch_market_moves()
            gathered.extend(moves)
            result.sources["coingecko"] = len(moves)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CoinGecko unavailable: {}", exc)
            result.errors.append(f"coingecko: {exc}")

        result.fetched = len(gathered)

        from core.database import insert_crypto_events  # noqa: PLC0415

        result.stored = await asyncio.to_thread(insert_crypto_events, gathered)

        logger.success("Night Scribe: {}", result.summary())
        events.log(f"Night Scribe: {result.summary()}",
                   "warn" if result.errors else "success")
        for failure in result.errors:
            # Named individually: "1 source failed" is not actionable, and a
            # feed that has been dead for a week should be visible as such.
            events.log(f"Feed unavailable — {failure[:120]}", "warn")
        events.agent_done("night_scribe", f"Filed {result.stored} new stories")
        events.agent_output(
            "night_scribe", "scan",
            fetched=result.fetched, stored=result.stored, sources=result.sources,
        )
        return result
