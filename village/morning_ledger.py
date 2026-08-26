"""The Morning Ledger: turns the night's raw stories into one edition.

Three sections, fixed, because a newsletter people open every day is one they
know the shape of:

1. **Overnight Pulse** — three bullets on macro, price and liquidity.
2. **The Daily Heist** — one exploit explained properly: the vulnerability, the
   amount, and the lesson that outlives the incident.
3. **Alpha & Security Tip** — one thing the reader can actually do today.

The model is reached through OpenRouter unless a direct ``ANTHROPIC_API_KEY``
is set. Both speak the OpenAI chat shape, so there is one code path rather than
two.

The whole thing degrades: with no model, no stories, or a mangled response, it
still produces a real edition from what the Night Scribe collected. A morning
with no newsletter is worse than a morning with a thin one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Sequence

from loguru import logger
from openai import AsyncOpenAI

from config.settings import Settings, get_settings
from core import events

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"

LEDGER_PROMPT = """You write "The Morning Ledger", a daily crypto briefing for \
people who hold their own keys. Your reader is technical, sceptical, and has \
already seen the headlines — they read you for what the headlines leave out.

Voice: plain, specific, unexcited. No hype, no "to the moon", no emoji in the \
body text, no "in this edition". Never invent a number, a protocol name, a date \
or an amount that is not in the material you were given. If the material is \
thin, say less rather than padding.

Return ONE JSON object and nothing else:

{
  "title": "the edition's title, under 70 characters, specific to today",
  "market_pulse_md": "EXACTLY 3 markdown bullets, each one line, on macro \
moves, price action and liquidity. Cite the concrete number where you have it.",
  "heist_story_md": "3-5 short markdown paragraphs on ONE exploit. Cover: what \
was attacked, the vulnerability class in plain English, the amount drained if \
known, and the lesson. If nothing was exploited overnight, take a well-known \
historical incident and say plainly that it is a classic, not news.",
  "alpha_tip_md": "1-2 markdown paragraphs. ONE concrete operational security \
practice for self-custody or on-chain use. Something the reader can do today, \
not 'be careful'."
}

RULES:
- Markdown only inside the fields. No HTML, no code fences around the JSON.
- The heist section is the reason people subscribe. Give it the most detail.
- Never give financial advice, never predict a price, never name a token to buy.
- Where you reference a story, name the source outlet so it can be checked.
"""

#: A heist section this short means the model padded or refused.
MIN_HEIST_CHARS = 300


@dataclass
class Edition:
    """One built edition, before it reaches the database."""

    title: str
    market_pulse_md: str
    heist_story_md: str
    alpha_tip_md: str
    publish_date: date
    source: str = "openrouter"
    story_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def full_markdown(self) -> str:
        """The whole edition, in the order a reader meets it."""
        stamp = self.publish_date.strftime("%A %d %B %Y")
        return "\n".join(
            [
                f"# {self.title}",
                f"*The Morning Ledger — {stamp}*",
                "",
                "## Overnight Pulse",
                self.market_pulse_md.strip(),
                "",
                "## The Daily Heist",
                self.heist_story_md.strip(),
                "",
                "## Alpha & Security Tip",
                self.alpha_tip_md.strip(),
                "",
                "---",
                "*Not financial advice. Verify every claim before acting on it.*",
            ]
        )

    def to_fields(self) -> dict[str, Any]:
        """The shape ``create_edition`` wants."""
        return {
            "publish_date": self.publish_date,
            "title": self.title,
            "market_pulse_md": self.market_pulse_md,
            "heist_story_md": self.heist_story_md,
            "alpha_tip_md": self.alpha_tip_md,
            "full_markdown": self.full_markdown,
        }


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull one JSON object out of a model response."""
    import json  # noqa: PLC0415

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


class MorningLedger:
    """Builds the daily edition."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """OpenRouter, or Anthropic directly when a key for it exists."""
        if self._client is None:
            direct = self.settings.anthropic_api_key.strip()
            self._client = AsyncOpenAI(
                base_url=ANTHROPIC_BASE_URL if direct else self.settings.openrouter_base_url,
                api_key=direct or self.settings.openrouter_api_key,
                timeout=self.settings.request_timeout_seconds,
                max_retries=self.settings.max_retries,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def model(self) -> str:
        """The model id, adjusted for whichever endpoint is in use.

        OpenRouter namespaces Anthropic models as ``anthropic/...``; the direct
        API does not accept that prefix.
        """
        name = self.settings.newsletter_model.strip()
        if self.settings.anthropic_api_key.strip() and name.startswith("anthropic/"):
            return name.split("/", 1)[1]
        return name

    # ---- material -----------------------------------------------------------
    def brief_from_events(self, rows: Sequence[Any]) -> str:
        """The night's stories, grouped so the editor can see the shape."""
        buckets: dict[str, list[Any]] = {"exploit": [], "macro": [], "market": []}
        for row in rows:
            buckets.setdefault(row.category, []).append(row)

        lines: list[str] = []
        for name, label in (
            ("exploit", "EXPLOITS AND SECURITY INCIDENTS"),
            ("macro", "MACRO AND REGULATORY"),
            ("market", "MARKET AND PRICE"),
        ):
            group = buckets.get(name) or []
            if not group:
                continue
            lines.append(f"\n### {label} ({len(group)})")
            for row in group[:20]:
                lines.append(f"- [{row.source}] {row.headline}")
                if row.summary:
                    lines.append(f"    {row.summary[:280]}")
        return "\n".join(lines) if lines else "(no stories were collected overnight)"

    # ---- building -----------------------------------------------------------
    async def build(self, hours: int = 12) -> Edition:
        """Write today's edition. Never raises."""
        from core.database import recent_crypto_events  # noqa: PLC0415

        events.agent_working("night_scribe", "Handing the night's file over…", progress=0.7)
        rows = recent_crypto_events(hours=hours)
        today = datetime.now(timezone.utc).date()

        logger.info("Morning Ledger: {} stories from the last {}h", len(rows), hours)

        if not self.settings.newsletter_configured:
            logger.warning("No model key — the Ledger is writing from the wire alone")
            return self._fallback(rows, today, "no api key")

        brief = self.brief_from_events(rows)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LEDGER_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Today is {today.isoformat()}. Here is everything the "
                            f"Night Scribe collected in the last {hours} hours.\n"
                            f"{brief}\n\nWrite today's edition."
                        ),
                    },
                ],
                temperature=0.7,
                max_tokens=3000,
            )
        except Exception as exc:  # noqa: BLE001 - degrade rather than skip a day
            logger.error("Ledger model call failed ({}); writing from the wire", exc)
            return self._fallback(rows, today, f"api error: {exc}")

        try:
            payload = _extract_json(response.choices[0].message.content or "")
            edition = self._build(payload, today, len(rows))
        except Exception as exc:  # noqa: BLE001
            logger.error("Unusable Ledger response ({}); writing from the wire", exc)
            return self._fallback(rows, today, "unparsable response")

        logger.success("Morning Ledger written: {!r}", edition.title)
        return edition

    def _build(self, payload: dict[str, Any], today: date, story_count: int) -> Edition:
        """Validate a model payload into an Edition."""
        def text(key: str, limit: int) -> str:
            return str(payload.get(key, "") or "").strip()[:limit]

        heist = text("heist_story_md", 6000)
        if len(heist) < MIN_HEIST_CHARS:
            # The heist section is the reason people subscribe; a two-line one
            # is a failed generation, not a short news day.
            raise ValueError(f"heist section is only {len(heist)} characters")

        title = text("title", 200)
        if not title:
            raise ValueError("no title")

        return Edition(
            title=title,
            market_pulse_md=text("market_pulse_md", 3000) or "- A quiet night on the wires.",
            heist_story_md=heist,
            alpha_tip_md=text("alpha_tip_md", 3000) or _DEFAULT_TIP,
            publish_date=today,
            source=f"{'anthropic' if self.settings.anthropic_api_key.strip() else 'openrouter'}:{self.model}",
            story_count=story_count,
        )

    def _fallback(self, rows: Sequence[Any], today: date, reason: str) -> Edition:
        """An edition assembled from the wire, with no model involved.

        Deliberately reads as a wire digest rather than imitating the written
        voice: a reader should be able to tell that nobody wrote this one.
        """
        exploits = [r for r in rows if r.category == "exploit"][:4]
        macro = [r for r in rows if r.category == "macro"][:3]
        market = [r for r in rows if r.category == "market"][:6]

        pulse = "\n".join(
            f"- **{r.source}**: {r.headline}" for r in (macro + market)[:3]
        ) or "- The wires were quiet overnight."

        if exploits:
            heist_lines = [
                "The Night Scribe flagged the following security incidents. "
                "No editorial pass was possible this morning, so these are the "
                "raw reports — read the sources before acting on any of it.",
                "",
            ]
            for r in exploits:
                heist_lines.append(f"**{r.headline}**")
                if r.summary:
                    heist_lines.append(r.summary[:400])
                heist_lines.append(f"Source: {r.source} — {r.url}")
                heist_lines.append("")
            heist = "\n".join(heist_lines)
        else:
            heist = (
                "No exploit reports came over the wire overnight, and no "
                "editorial pass was possible this morning.\n\n"
                "A quiet night is worth noting rather than filling. The classic "
                "reading, if you want it, is the 2016 DAO reentrancy incident: "
                "a contract that sent funds before updating its own balance, "
                "which let a caller re-enter the withdrawal repeatedly. The "
                "lesson — update state before making an external call — is the "
                "reason the checks-effects-interactions pattern exists."
            )

        return Edition(
            title=f"The Morning Ledger — {today.strftime('%d %B %Y')}",
            market_pulse_md=pulse,
            heist_story_md=heist,
            alpha_tip_md=_DEFAULT_TIP,
            publish_date=today,
            source=f"fallback ({reason})",
            story_count=len(rows),
            errors=[reason],
        )

    async def publish_draft(self, hours: int = 12, *, dispatch: bool = True) -> Any:
        """Build the edition, store it as a DRAFT, and post it for approval.

        :param dispatch: send the Telegram approval card. True by default so a
            drafted edition never sits unseen; callers that post the card
            themselves pass False rather than sending it twice.
        """
        from core.database import create_edition  # noqa: PLC0415

        edition = await self.build(hours=hours)
        row = create_edition(**edition.to_fields())

        events.agent_done("night_scribe", f"Edition drafted: {edition.title}")
        events.log(
            f"The Morning Ledger is drafted: {edition.title} "
            f"({edition.story_count} stories, {len(edition.full_markdown):,} chars)",
            "success",
        )

        if dispatch:
            await self.dispatch(row)
        return row

    async def dispatch(self, row: Any) -> bool:
        """Post one edition's approval card to Telegram.

        Delegates to the Town Crier rather than talking to Telegram here: the
        card, its buttons and their callbacks belong together, and a second
        implementation would drift from the first.
        """
        if not self.settings.telegram_configured:
            logger.warning(
                "Telegram not configured — edition {} stays a DRAFT. "
                "Approve it with: python main.py --approve-edition {}",
                row.id, row.id,
            )
            return False

        from village.town_crier import TownCrier  # noqa: PLC0415

        sent = await TownCrier(self.settings).dispatch_edition(row.id)
        if sent:
            events.log(f"Edition #{row.id} sent to the editor for approval.", "info")
        return sent


#: Used whenever the model did not supply one. A real, specific practice
#: rather than "be careful", because the section is worthless otherwise.
_DEFAULT_TIP = (
    "Revoke stale token approvals. Every `approve` you have ever signed stays "
    "live until you remove it, and a protocol you used once in 2021 can still "
    "move those tokens if its contract is later compromised.\n\n"
    "Open a revocation tool for each chain you have used, sort approvals by "
    "value, and remove every one you do not currently need — especially "
    "unlimited allowances. It costs gas and takes ten minutes, and it closes "
    "the single most common path between an old contract and your current "
    "balance."
)
