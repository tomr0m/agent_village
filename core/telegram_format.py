"""Markdown to the HTML subset Telegram actually accepts.

Telegram's ``parse_mode=HTML`` is not HTML. It supports ``b i u s a code pre
blockquote`` and nothing else — no headings, no lists, no ``<br>``, no ``<p>``.
An unsupported tag does not degrade, it fails the whole ``sendMessage`` call
with 400.

So headings become bold lines, bullets become "• ", and everything else is
escaped. The alternative — wrapping the edition in ``<pre>`` — renders a
newsletter as a code block, which is what the previous approval path did and
is fine for a draft nobody but the editor sees. It is not fine for a channel.
"""

from __future__ import annotations

import html
import re

#: Telegram rejects a sendMessage body over this. Editions run longer.
TELEGRAM_LIMIT = 4096

#: Leave room for the part counter a split adds.
SAFE_CHUNK = 3800


def _inline(text: str) -> str:
    """Convert inline markdown inside one already-escaped line."""
    # Links first: their label may itself contain emphasis markers.
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        text,
    )
    # Bold before italic, or **x** is read as two italics wrapping nothing.
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_]+)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<![_\w])_([^_\n]+)_(?!_)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    return text


def markdown_to_telegram_html(markdown: str) -> str:
    """Render markdown as Telegram-safe HTML.

    Everything is escaped first and tags are introduced afterwards, so a stray
    ``<`` in the copy cannot close a tag or break the parse.
    """
    lines_out: list[str] = []

    for raw in (markdown or "").splitlines():
        line = raw.rstrip()

        if not line.strip():
            lines_out.append("")
            continue

        # A horizontal rule has no equivalent; a spacer reads the same.
        if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
            lines_out.append("—")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            body = _inline(html.escape(heading.group(2).strip()))
            # The top heading is the title; sub-headings get a blank line above
            # them so sections separate without a rule.
            lines_out.append(f"<b>{body}</b>" if level <= 2 else f"<b>{body}</b>")
            continue

        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet:
            lines_out.append(f"• {_inline(html.escape(bullet.group(1)))}")
            continue

        numbered = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if numbered:
            lines_out.append(
                f"{numbered.group(1)}. {_inline(html.escape(numbered.group(2)))}"
            )
            continue

        quote = re.match(r"^\s*>\s?(.*)$", line)
        if quote:
            lines_out.append(f"<i>{_inline(html.escape(quote.group(1)))}</i>")
            continue

        lines_out.append(_inline(html.escape(line)))

    # Collapse runs of blank lines: markdown uses them for structure, Telegram
    # just shows the gap.
    rendered: list[str] = []
    for line in lines_out:
        if not line and rendered and not rendered[-1]:
            continue
        rendered.append(line)

    return "\n".join(rendered).strip()


def split_for_telegram(text: str, limit: int = SAFE_CHUNK) -> list[str]:
    """Split a long body into sendable chunks.

    Splits on blank lines, then on single newlines, so a message never breaks
    mid-sentence. A tag opened in one chunk and closed in another would fail
    the parse, and paragraph boundaries never fall inside one.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for block in text.split("\n\n"):
        if len(block) > limit:
            # A single huge paragraph: fall back to line boundaries.
            for line in block.splitlines():
                if len(current) + len(line) + 1 > limit:
                    chunks.append(current.strip())
                    current = line
                else:
                    current = f"{current}\n{line}" if current else line
            continue

        if len(current) + len(block) + 2 > limit:
            chunks.append(current.strip())
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]
