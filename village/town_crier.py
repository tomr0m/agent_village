"""The Town Crier: the human-in-the-loop gate, over Telegram.

A finished listing is dispatched as a photo card — the artwork, the title, the
13 tags, the Guard's report — with two inline buttons. Approve hands the listing
to the Merchant and publishes it. Reject closes it out with a reason.

The callback payload carries the listing id, so a card stays actionable across a
worker restart: state lives in the database, never in memory.
"""

from __future__ import annotations

import asyncio
import html
import signal
from pathlib import Path
from typing import Any

from loguru import logger
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from config.settings import Settings, get_settings
from datetime import datetime, timedelta

from core import events
from core.database import (
    Channel,
    DealStatus,
    create_deal,
    get_deal,
    EditionStatus,
    edition_for_date,
    get_edition,
    recent_deals,
    record_deal_pin,
    update_edition_status,
    update_deal_status,
    Listing,
    ListingStatus,
    Short,
    ShortStatus,
    counts_by_status,
    get_listing,
    get_short,
    record_revenue,
    list_by_status,
    recent_listings,
    recent_shorts,
    record_youtube_upload,
    short_counts_by_status,
    shorts_by_status,
    update_short,
    update_short_status,
    update_status,
)
from village.merchant import Merchant, PublishResult
from village.youtube_publisher import UploadResult, YouTubePublisher

#: Callback payloads are ``"<action>:<listing id>"``.
APPROVE_PREFIX = "approve"
REJECT_PREFIX = "reject"
DETAILS_PREFIX = "details"

#: The Bard's cards use their own verbs, so a stale listing button can never
#: be routed into the video handler.
def _next_publish_time(clock: str) -> datetime:
    """The next occurrence of ``HH:MM`` in local time.

    Today if that hour is still ahead, tomorrow otherwise — an edition approved
    at 09:00 goes out tomorrow morning rather than instantly, which is what
    "scheduled for 08:00" has to mean to be worth anything.
    """
    try:
        hour, _, minute = clock.strip().partition(":")
        target_hour, target_minute = int(hour), int(minute or 0)
    except ValueError:
        target_hour, target_minute = 8, 0

    now = datetime.now()
    when = now.replace(
        hour=min(23, max(0, target_hour)),
        minute=min(59, max(0, target_minute)),
        second=0, microsecond=0,
    )
    return when if when > now else when + timedelta(days=1)


#: Morning Ledger cards.
LEDGER_APPROVE_PREFIX = "lapprove"
LEDGER_REGEN_PREFIX = "lregen"
LEDGER_REJECT_PREFIX = "lreject"

#: Deal cards. Distinct from the listing and Short prefixes so a stale card
#: from another agent can never resolve against a deal id.
DEAL_APPROVE_PREFIX = "dapprove"
DEAL_REJECT_PREFIX = "dreject"

SHORT_APPROVE_PREFIX = "vapprove"
SHORT_REROLL_PREFIX = "vreroll"
SHORT_REJECT_PREFIX = "vreject"

#: Telegram rejects captions over 1024 characters.
MAX_CAPTION_CHARS = 1024


def _escape(value: Any) -> str:
    """HTML-escape a value for Telegram's HTML parse mode."""
    return html.escape(str(value or ""), quote=False)


def approval_keyboard(listing_id: int) -> InlineKeyboardMarkup:
    """The two-button decision card, plus a details expander."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"{APPROVE_PREFIX}:{listing_id}"),
                InlineKeyboardButton("🚫 Reject", callback_data=f"{REJECT_PREFIX}:{listing_id}"),
            ],
            [InlineKeyboardButton("🔍 Full details", callback_data=f"{DETAILS_PREFIX}:{listing_id}")],
        ]
    )


def build_caption(listing: Listing, *, settings: Settings) -> str:
    """The card body: what the operator needs to decide, and nothing else."""
    tags = ", ".join(listing.tags)
    mode = "DRY RUN" if settings.dry_run else "LIVE"

    guard_line = (listing.guard_report or "").splitlines()[0] if listing.guard_report else "n/a"
    warnings = [
        line.strip() for line in (listing.guard_report or "").splitlines() if "[WARN]" in line
    ]

    parts = [
        f"<b>Listing #{listing.id}</b>  ·  <code>{_escape(mode)}</code>",
        "",
        f"<b>{_escape(listing.title)}</b>",
        "",
        f"<i>Niche:</i> {_escape(listing.niche)}",
        f"<i>Buyer:</i> {_escape(listing.audience)}",
        f"<i>Price:</i> {_escape(listing.price_display)}",
        "",
        f"<i>Tags ({len(listing.tags)}):</i> {_escape(tags)}",
        "",
        f"<i>Guard:</i> {_escape(guard_line)}",
    ]
    if warnings:
        parts.append(f"<i>Warnings:</i> {len(warnings)}")

    caption = "\n".join(parts)
    if len(caption) > MAX_CAPTION_CHARS:
        caption = caption[: MAX_CAPTION_CHARS - 20].rstrip() + "\n…"
    return caption


def short_keyboard(short_id: int) -> InlineKeyboardMarkup:
    """Approve / reroll / reject, for a Short."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Approve & Upload", callback_data=f"{SHORT_APPROVE_PREFIX}:{short_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔀 Reroll Script", callback_data=f"{SHORT_REROLL_PREFIX}:{short_id}"
                ),
                InlineKeyboardButton(
                    "🚫 Reject", callback_data=f"{SHORT_REJECT_PREFIX}:{short_id}"
                ),
            ],
        ]
    )


def build_short_caption(short: Short, *, settings: Settings) -> str:
    """The card body for a Short: what it is, how long, and how it was made."""
    mode = "DRY RUN" if settings.dry_run else "LIVE"
    tags = " ".join(f"#{tag}" for tag in short.hashtags[:6])
    storyboard = short.render_backend == "storyboard"

    parts = [
        f"<b>🎭 Short #{short.id}</b>  ·  <code>{_escape(mode)}</code>",
        "",
        f"<b>{_escape(short.title)}</b>",
        f"<i>{_escape(short.category)}</i>",
        "",
        f"<i>Hook:</i> {_escape(short.hook)}",
        "",
        f"<i>Runtime:</i> {short.duration_seconds:.0f}s  ·  "
        f"<i>Scenes:</i> {len(short.scenes)}",
        f"<i>Voice:</i> {_escape(short.voice_backend or 'n/a')}  ·  "
        f"<i>Cut:</i> {_escape(short.render_backend or 'n/a')}",
    ]
    if short.voice_backend == "placeholder":
        parts.append("\n⚠️ <b>Silent placeholder track</b> — no TTS was available.")
    if storyboard:
        parts.append("\n⚠️ <b>Storyboard only</b> — no video toolchain on this machine.")
    if tags:
        parts.append(f"\n{_escape(tags)}")

    caption = "\n".join(parts)
    if len(caption) > MAX_CAPTION_CHARS:
        caption = caption[: MAX_CAPTION_CHARS - 20].rstrip() + "\n…"
    return caption


class TownCrier:
    """Dispatches approval cards and services their callbacks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.merchant = Merchant(self.settings)

    # ---- outbound -----------------------------------------------------------
    async def dispatch(self, listing_id: int) -> bool:
        """Send one listing to the operator for approval.

        :returns: whether the card was delivered. A missing Telegram config is a
            warning rather than a failure — the listing stays pending and can be
            approved later from the CLI.
        """
        listing = get_listing(listing_id)
        if listing is None:
            logger.error("Cannot dispatch listing {}: not found", listing_id)
            return False

        if not self.settings.telegram_configured:
            logger.warning(
                "Telegram not configured - listing {} stays PENDING_APPROVAL. "
                "Approve it with: python main.py --approve {}",
                listing_id,
                listing_id,
            )
            return False

        bot = ApplicationBuilder().token(self.settings.telegram_bot_token).build().bot
        caption = build_caption(listing, settings=self.settings)
        keyboard = approval_keyboard(listing.id)
        image_path = Path(listing.processed_image_path or listing.raw_image_path or "")

        try:
            async with bot:
                if image_path.is_file():
                    with image_path.open("rb") as handle:
                        message = await bot.send_photo(
                            chat_id=self.settings.telegram_chat_id,
                            photo=InputFile(handle, filename=image_path.name),
                            caption=caption,
                            parse_mode=ParseMode.HTML,
                            reply_markup=keyboard,
                        )
                else:
                    logger.warning("No artwork on disk for listing {}; sending text", listing_id)
                    message = await bot.send_message(
                        chat_id=self.settings.telegram_chat_id,
                        text=caption,
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard,
                    )
        except TelegramError as exc:
            logger.error("Telegram dispatch failed for listing {}: {}", listing_id, exc)
            return False

        update_status(
            listing_id,
            ListingStatus.PENDING_APPROVAL,
            reason="Awaiting operator decision in Telegram.",
            telegram_message_id=message.message_id,
        )
        logger.success("Dispatched listing {} to Telegram", listing_id)
        return True

    async def dispatch_short(self, short_id: int) -> bool:
        """Send a finished Short to the operator as a playable video card.

        Telegram caps bot uploads at 50 MB. A vertical 40-second H.264 short is
        typically 3-8 MB, but a storyboard PNG or an oversized render is sent as
        a document instead of silently failing.
        """
        short = get_short(short_id)
        if short is None:
            logger.error("Cannot dispatch short {}: not found", short_id)
            return False

        if not self.settings.telegram_configured:
            logger.warning(
                "Telegram not configured - short {} stays PENDING_APPROVAL. "
                "Approve it with: python main.py --approve-short {}",
                short_id,
                short_id,
            )
            return False

        media = Path(short.video_path or "")
        if not media.is_file():
            logger.error("Short {} has no media on disk", short_id)
            return False

        size_mb = media.stat().st_size / (1024 * 1024)
        caption = build_short_caption(short, settings=self.settings)
        keyboard = short_keyboard(short.id)
        bot = ApplicationBuilder().token(self.settings.telegram_bot_token).build().bot

        try:
            async with bot:
                if media.suffix.lower() == ".mp4" and size_mb <= 48:
                    with media.open("rb") as handle:
                        message = await bot.send_video(
                            chat_id=self.settings.telegram_chat_id,
                            video=InputFile(handle, filename=media.name),
                            caption=caption,
                            parse_mode=ParseMode.HTML,
                            reply_markup=keyboard,
                            supports_streaming=True,
                            width=self.settings.video_width,
                            height=self.settings.video_height,
                            duration=int(short.duration_seconds) or None,
                        )
                else:
                    # Too large for send_video, or a storyboard PNG.
                    reason = (
                        f"({size_mb:.0f} MB — over Telegram's video limit)"
                        if size_mb > 48
                        else "(storyboard, not a video)"
                    )
                    logger.warning("Sending short {} as a document {}", short_id, reason)
                    with media.open("rb") as handle:
                        message = await bot.send_document(
                            chat_id=self.settings.telegram_chat_id,
                            document=InputFile(handle, filename=media.name),
                            caption=f"{caption}\n\n<i>Sent as a file {_escape(reason)}</i>",
                            parse_mode=ParseMode.HTML,
                            reply_markup=keyboard,
                        )
        except TelegramError as exc:
            logger.error("Telegram dispatch failed for short {}: {}", short_id, exc)
            return False

        update_short(short_id, telegram_message_id=message.message_id)
        logger.success("Dispatched short {} to Telegram ({:.1f} MB)", short_id, size_mb)
        return True

    # ---- inbound ------------------------------------------------------------
    async def on_decision(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an Approve/Reject tap."""
        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()

        action, _, raw_id = query.data.partition(":")
        try:
            listing_id = int(raw_id)
        except ValueError:
            logger.warning("Malformed callback payload: {!r}", query.data)
            return

        listing = get_listing(listing_id)
        if listing is None:
            await self._replace_markup(query, f"Listing #{listing_id} no longer exists.")
            return

        if listing.status_enum is not ListingStatus.PENDING_APPROVAL:
            await self._replace_markup(
                query, f"Listing #{listing_id} was already {listing.status}."
            )
            return

        if action == REJECT_PREFIX:
            update_status(
                listing_id, ListingStatus.REJECTED, reason="Rejected by the operator in Telegram."
            )
            await self._replace_markup(query, f"🚫 Listing #{listing_id} rejected.")
            logger.info("Operator rejected listing {}", listing_id)
            return

        if action != APPROVE_PREFIX:
            return

        approved = update_status(
            listing_id, ListingStatus.APPROVED, reason="Approved by the operator in Telegram."
        )
        if approved is None:
            await self._replace_markup(query, f"Listing #{listing_id} could not be approved.")
            return

        await self._replace_markup(query, f"✅ Listing #{listing_id} approved — publishing…")

        result = await self.merchant.publish(
            title=approved.title,
            description=approved.description,
            tags=approved.tags,
            image_path=approved.processed_image_path or approved.raw_image_path or "",
            price_cents=approved.price_cents,
        )

        if result.ok:
            update_status(
                listing_id,
                ListingStatus.PUBLISHED,
                reason=result.summary(),
                printify_image_id=result.image_id,
                printify_product_id=result.product_id,
                external_url=result.external_url,
                dry_run=1 if result.simulated else 0,
            )
            record_revenue(
                Channel.ETSY,
                approved.price_cents,
                note=approved.title[:120],
                source_key=f"listing:{listing_id}",
                dry_run=result.simulated,
            )
            tail = " (simulated)" if result.simulated else ""
            await self._send(
                context,
                f"📦 Listing #{listing_id} published{tail}.\n"
                f"Product: <code>{_escape(result.product_id)}</code>\n"
                f"{_escape(result.external_url or '')}",
            )
        else:
            update_status(listing_id, ListingStatus.FAILED, reason=result.summary())
            await self._send(
                context, f"⚠️ Listing #{listing_id} failed to publish: {_escape(result.error)}"
            )

    async def on_short_decision(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Approve & Upload / Reroll Script / Reject on a Short card."""
        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()

        action, _, raw_id = query.data.partition(":")
        try:
            short_id = int(raw_id)
        except ValueError:
            logger.warning("Malformed short callback: {!r}", query.data)
            return

        short = get_short(short_id)
        if short is None:
            await self._replace_markup(query, f"Short #{short_id} no longer exists.")
            return

        if short.status_enum is not ShortStatus.PENDING_APPROVAL:
            await self._replace_markup(query, f"Short #{short_id} was already {short.status}.")
            return

        if action == SHORT_REJECT_PREFIX:
            update_short_status(
                short_id, ShortStatus.REJECTED, reason="Rejected by the operator in Telegram."
            )
            await self._replace_markup(query, f"🚫 Short #{short_id} rejected.")
            return

        if action == SHORT_REROLL_PREFIX:
            await self._replace_markup(query, f"🔀 Rerolling short #{short_id}…")
            await self._send(context, f"🎭 The Bard is rewriting #{short_id}. This takes a minute.")

            # Run in the background: a Telegram callback must return promptly,
            # and a full reroll is a script, a voice take, images and an encode.
            async def rerun() -> None:
                from village.bard import BardAgent  # noqa: PLC0415 - avoids a cycle

                bard = BardAgent(self.settings)
                try:
                    result = await bard.reroll(short_id)
                finally:
                    await bard.aclose()
                if not result.ok:
                    await self._send(
                        context, f"⚠️ Reroll of #{short_id} failed: {_escape('; '.join(result.errors))}"
                    )

            asyncio.create_task(rerun())
            return

        if action != SHORT_APPROVE_PREFIX:
            return

        approved = update_short_status(
            short_id, ShortStatus.APPROVED, reason="Approved by the operator in Telegram."
        )
        if approved is None:
            await self._replace_markup(query, f"Short #{short_id} could not be approved.")
            return

        await self._replace_markup(query, f"✅ Short #{short_id} approved.")

        publisher = YouTubePublisher(self.settings)
        can_upload, blocked_because = publisher.status()

        if not can_upload:
            # Not an error: uploading is opt-in, and the file is still ready to
            # post by hand. Say plainly why nothing was published so the
            # operator is not left guessing whether it silently failed.
            update_short_status(
                short_id,
                ShortStatus.PUBLISHED,
                reason=f"Approved for manual upload ({blocked_because}).",
            )
            _seed_short_revenue(short_id, self.settings)
            await self._send(
                context,
                f"📼 Short #{short_id} is approved and ready to upload.\n"
                f"<b>{_escape(approved.title)}</b>\n"
                f"<code>{_escape(approved.video_path or '')}</code>\n\n"
                f"<i>Not sent to YouTube: {_escape(blocked_because)}</i>\n\n"
                f"{_escape(approved.description)}\n\n"
                + " ".join(f"#{_escape(tag)}" for tag in approved.hashtags[:8]),
            )
            return

        await self._send(context, f"📡 Uploading #{short_id} to YouTube…")

        # In the background: a Telegram callback has to return promptly and an
        # upload is minutes of I/O.
        async def publish() -> None:
            await self.upload_to_youtube(short_id, notify=context)

        asyncio.create_task(publish())

    async def upload_to_youtube(
        self, short_id: int, notify: ContextTypes.DEFAULT_TYPE | None = None
    ) -> UploadResult:
        """Upload an approved Short and record where it went.

        Shared by the Telegram button and the CLI, so both paths write the same
        row and post the same link.
        """
        short = get_short(short_id)
        if short is None:
            return UploadResult(ok=False, error=f"short #{short_id} no longer exists")

        publisher = YouTubePublisher(self.settings)
        result = await publisher.upload_short(
            short.video_path or "",
            title=short.title,
            description=short.description,
            tags=short.hashtags,
        )

        if not result.ok:
            # The video is fine — only the upload failed — so the Short stays
            # APPROVED and can be retried rather than being marked FAILED.
            logger.error("Upload of short {} failed: {}", short_id, result.error)
            update_short_status(
                short_id, ShortStatus.APPROVED,
                reason=f"Upload failed: {result.error}",
            )
            if notify is not None:
                await self._send(
                    notify,
                    f"⚠️ Short #{short_id} did NOT upload.\n"
                    f"{_escape(result.error)}\n\n"
                    f"The file is still here: <code>{_escape(short.video_path or '')}</code>\n"
                    f"Retry with: <code>python main.py --upload-short {short_id}</code>",
                )
            return result

        record_youtube_upload(
            short_id,
            video_id=result.video_id,
            url=result.watch_url or result.url,
            privacy=result.privacy_status,
            simulated=result.simulated,
        )
        update_short_status(
            short_id, ShortStatus.PUBLISHED,
            reason=(
                "Simulated upload (dry run)." if result.simulated
                else f"Uploaded to YouTube as {result.video_id} [{result.privacy_status}]."
            ),
        )
        _seed_short_revenue(short_id, self.settings)

        link = result.watch_url or result.url
        events.toast(f"Short #{short_id} published: {link}", "success")

        if notify is not None:
            note = ""
            if result.simulated:
                note = "\n<i>DRY RUN — nothing was actually uploaded.</i>"
            elif result.privacy_status != "public":
                note = (
                    f"\n<i>Uploaded as {_escape(result.privacy_status)} — "
                    "only you can see it until you change that on the channel.</i>"
                )
            await self._send(
                notify,
                f"🎬 Short #{short_id} is live.\n"
                f"<b>{_escape(short.title)}</b>\n"
                f'<a href="{_escape(link)}">{_escape(link)}</a>{note}',
            )

        return result

    async def cmd_ledger(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/ledger — today's edition, or build one if there is not one yet."""
        edition = edition_for_date()
        if edition is not None:
            await self.dispatch_edition(edition.id, context)
            return

        await self._send(context, "📰 No edition today yet. Writing one…")

        async def build() -> None:
            from village.morning_ledger import MorningLedger  # noqa: PLC0415

            ledger = MorningLedger(self.settings)
            try:
                row = await ledger.publish_draft(dispatch=False)
            except Exception as exc:  # noqa: BLE001 - report, never crash the bot
                logger.exception("Ledger build failed")
                await self._send(context, f"⚠️ The Ledger failed: {_escape(str(exc))}")
                return
            finally:
                await ledger.aclose()
            await self.dispatch_edition(row.id, context)

        asyncio.create_task(build())

    async def dispatch_edition(
        self, edition_id: int, context: ContextTypes.DEFAULT_TYPE | None = None
    ) -> bool:
        """Post one Morning Ledger edition for approval.

        Telegram caps a message at 4096 characters and an edition runs longer,
        so the card carries the pulse and the opening of the heist, with the
        full text available through the dashboard. Truncating mid-word in a
        newsletter about precision would be its own kind of error.
        """
        if not self.settings.telegram_configured:
            logger.warning(
                "Telegram not configured — edition {} stays a DRAFT. "
                "Approve it with: python main.py --approve-edition {}",
                edition_id, edition_id,
            )
            return False

        edition = get_edition(edition_id)
        if edition is None:
            logger.error("Cannot dispatch edition {}: not found", edition_id)
            return False

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        f"✅ Approve & Schedule for {self.settings.ledger_publish_time}",
                        callback_data=f"{LEDGER_APPROVE_PREFIX}:{edition_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        "🔄 Regenerate", callback_data=f"{LEDGER_REGEN_PREFIX}:{edition_id}"
                    ),
                    InlineKeyboardButton(
                        "🚫 Reject", callback_data=f"{LEDGER_REJECT_PREFIX}:{edition_id}"
                    ),
                ],
            ]
        )

        heist = edition.heist_story_md.strip()
        preview = heist[:700] + ("…" if len(heist) > 700 else "")
        body = (
            f"📰 <b>{_escape(edition.title)}</b>\n"
            f"<i>The Morning Ledger — {edition.publish_date}</i>\n\n"
            f"<b>Overnight Pulse</b>\n{_escape(edition.market_pulse_md.strip())}\n\n"
            f"<b>The Daily Heist</b>\n{_escape(preview)}\n\n"
            f"<i>{len(edition.full_markdown):,} characters in full.</i>"
        )

        try:
            if context is not None:
                message = await context.bot.send_message(
                    chat_id=self.settings.telegram_chat_id, text=body,
                    parse_mode=ParseMode.HTML, reply_markup=keyboard,
                    disable_web_page_preview=True,
                )
            else:
                bot = ApplicationBuilder().token(self.settings.telegram_bot_token).build().bot
                async with bot:
                    message = await bot.send_message(
                        chat_id=self.settings.telegram_chat_id, text=body,
                        parse_mode=ParseMode.HTML, reply_markup=keyboard,
                        disable_web_page_preview=True,
                    )
        except TelegramError as exc:
            logger.error("Could not dispatch edition {}: {}", edition_id, exc)
            return False

        update_edition_status(
            edition_id, EditionStatus.DRAFT, reason="Awaiting the editor's verdict.",
            telegram_message_id=getattr(message, "message_id", None),
        )
        logger.success("Dispatched edition {} to Telegram", edition_id)
        return True

    async def on_edition_decision(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Approve & Schedule / Regenerate / Reject on a Ledger card."""
        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()

        action, _, raw_id = query.data.partition(":")
        try:
            edition_id = int(raw_id)
        except ValueError:
            logger.warning("Malformed edition callback: {!r}", query.data)
            return

        edition = get_edition(edition_id)
        if edition is None:
            await self._replace_markup(query, f"Edition #{edition_id} no longer exists.")
            return

        if edition.status_enum is not EditionStatus.DRAFT:
            await self._replace_markup(
                query, f"Edition #{edition_id} was already {edition.status.lower()}."
            )
            return

        if action == LEDGER_REJECT_PREFIX:
            update_edition_status(
                edition_id, EditionStatus.REJECTED, reason="Rejected by the editor."
            )
            await self._replace_markup(query, f"🚫 Edition #{edition_id} rejected.")
            return

        if action == LEDGER_REGEN_PREFIX:
            await self._replace_markup(query, f"🔄 Rewriting edition #{edition_id}…")
            await self._send(context, "📰 The Ledger is being rewritten. One moment.")

            async def rerun() -> None:
                from village.morning_ledger import MorningLedger  # noqa: PLC0415

                ledger = MorningLedger(self.settings)
                try:
                    row = await ledger.publish_draft(dispatch=False)
                finally:
                    await ledger.aclose()
                await self.dispatch_edition(row.id, context)

            asyncio.create_task(rerun())
            return

        if action != LEDGER_APPROVE_PREFIX:
            return

        # Schedule for the configured hour. If that time has already passed
        # today the send goes out on the next daemon tick rather than waiting
        # a whole day — an approved edition should not sit unsent.
        when = _next_publish_time(self.settings.ledger_publish_time)
        approved = update_edition_status(
            edition_id, EditionStatus.APPROVED,
            reason=f"Approved; scheduled for {when:%Y-%m-%d %H:%M}.",
            scheduled_for=when,
        )
        if approved is None:
            await self._replace_markup(query, f"Edition #{edition_id} could not be approved.")
            return

        await self._replace_markup(
            query, f"✅ Edition #{edition_id} approved for {when:%H:%M}."
        )
        await self._send(
            context,
            f"📅 <b>{_escape(approved.title)}</b> is scheduled for "
            f"{when:%H:%M on %d %B}.\n"
            f"<i>The daemon sends it; leave it running.</i>",
        )

    async def publish_edition(
        self, edition_id: int, context: ContextTypes.DEFAULT_TYPE | None = None
    ) -> bool:
        """Deliver an approved edition. Called by the 08:00 scheduler."""
        edition = get_edition(edition_id)
        if edition is None:
            return False
        if edition.status_enum is not EditionStatus.APPROVED:
            logger.debug("Edition {} is {}, not APPROVED", edition_id, edition.status)
            return False

        body = edition.full_markdown
        # Telegram's 4096-character ceiling, split on blank lines so a section
        # never breaks mid-sentence.
        chunks: list[str] = []
        current = ""
        for block in body.split("\n\n"):
            if len(current) + len(block) + 2 > 3800:
                chunks.append(current)
                current = block
            else:
                current = f"{current}\n\n{block}" if current else block
        if current:
            chunks.append(current)

        sent = True
        for index, chunk in enumerate(chunks, start=1):
            tail = f"\n\n<i>({index}/{len(chunks)})</i>" if len(chunks) > 1 else ""
            if not await self.announce(f"<pre>{_escape(chunk)}</pre>{tail}"):
                sent = False
                break

        if sent:
            update_edition_status(
                edition_id, EditionStatus.PUBLISHED, reason="Delivered to Telegram."
            )
            events.toast(f"The Morning Ledger went out: {edition.title}", "success")
        return sent

    async def broadcast_edition(self, edition_id: int) -> tuple[bool, str]:
        """Post a finished edition to the public channel.

        Separate from :meth:`publish_edition`, which sends to the operator's
        private chat as a monospaced block. A channel post is the published
        artefact, so the markdown is rendered into Telegram's HTML subset and
        split on paragraph boundaries.

        :returns: ``(sent, detail)``.
        """
        from core.telegram_format import (  # noqa: PLC0415
            markdown_to_telegram_html, split_for_telegram,
        )

        if not self.settings.channel_configured:
            return False, (
                "TELEGRAM_CHANNEL_ID is not set (and TELEGRAM_BOT_TOKEN must be too)"
            )

        edition = get_edition(edition_id)
        if edition is None:
            return False, f"no edition {edition_id}"

        body = markdown_to_telegram_html(edition.full_markdown)
        chunks = split_for_telegram(body)
        channel = self.settings.telegram_channel_id.strip()

        bot = ApplicationBuilder().token(self.settings.telegram_bot_token).build().bot
        try:
            async with bot:
                for index, chunk in enumerate(chunks, start=1):
                    tail = (
                        f"\n\n<i>({index}/{len(chunks)})</i>" if len(chunks) > 1 else ""
                    )
                    await bot.send_message(
                        chat_id=channel,
                        text=chunk + tail,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
        except TelegramError as exc:
            detail = self._explain_channel(exc, channel)
            logger.error("Channel publish failed: {}", detail)
            return False, detail

        logger.success(
            "Edition {} published to {} in {} message(s)", edition_id, channel, len(chunks)
        )
        return True, f"published to {channel} in {len(chunks)} message(s)"

    @staticmethod
    def _explain_channel(exc: Exception, channel: str) -> str:
        """Turn Telegram's terse channel errors into the fix."""
        text = str(exc).lower()
        if "chat not found" in text:
            return (
                f"Telegram cannot find {channel}. Check the @name, and note the "
                "bot must be a MEMBER of the channel before it can post."
            )
        if "not enough rights" in text or "administrator" in text:
            return (
                f"The bot is in {channel} but cannot post. Make it an "
                "administrator with 'Post Messages' enabled."
            )
        if "can't parse entities" in text:
            return f"Telegram rejected the HTML: {exc}"
        return str(exc)

    async def announce(self, text: str) -> bool:
        """Send a plain message to the operator, outside any approval flow.

        Used by agents that have something to say but nothing to approve — the
        Overseer's advisories, for one.
        """
        if not self.settings.telegram_configured:
            logger.warning("Cannot announce: Telegram is not configured")
            return False

        bot = ApplicationBuilder().token(self.settings.telegram_bot_token).build().bot
        try:
            async with bot:
                await bot.send_message(
                    chat_id=self.settings.telegram_chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        except TelegramError as exc:
            logger.error("Announcement failed: {}", exc)
            return False
        return True

    async def cmd_scout(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/scout [niche] — curate one Amazon recommendation and post the card.

        Research runs in the background: a Telegram handler must return
        promptly, and a model call takes seconds.
        """
        niche = " ".join(context.args).strip() if context.args else ""
        target = niche or self.settings.amazon_default_niche

        await self._send(context, f"🔎 Scouting deals for <b>{_escape(target)}</b>…")

        async def research() -> None:
            from village.dealscout import DealScout  # noqa: PLC0415 - avoids a cycle

            scout = DealScout(self.settings)
            try:
                deal = await scout.find_deal(niche or None)
            except ValueError as exc:
                await self._send(context, f"🚫 {_escape(str(exc))}")
                return
            except Exception as exc:  # noqa: BLE001 - report, never crash the bot
                logger.exception("Scout failed")
                await self._send(context, f"⚠️ The Scout failed: {_escape(str(exc))}")
                return
            finally:
                await scout.aclose()

            row = create_deal(deal.to_dict(), dry_run=self.settings.dry_run)
            await self.dispatch_deal(row.id, deal, context=context)

        asyncio.create_task(research())

    #: /deal is an alias — the two names are equally natural for this.
    cmd_deal = cmd_scout

    async def cmd_deals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/deals — the most recent curated recommendations and their state."""
        rows = recent_deals(10)
        if not rows:
            await self._send(context, "No deals yet. Send /scout to curate one.")
            return

        lines = ["<b>Recent deals</b>", ""]
        for row in rows:
            mark = {
                DealStatus.APPROVED.value: "✅",
                DealStatus.REJECTED.value: "🚫",
                DealStatus.PUBLISHED.value: "📣",
            }.get(row.status, "•")
            lines.append(
                f"{mark} <code>#{row.id}</code> {_escape(row.product[:44])} "
                f"<i>{_escape(row.status.lower())}</i>"
            )
        await self._send(context, "\n".join(lines))

    async def dispatch_deal(
        self,
        deal_id: int,
        deal: Any,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> bool:
        """Post one deal card with Approve & Post / Reject.

        Shared by /scout and the CLI, so both produce the same card.
        """
        if not self.settings.telegram_configured:
            return False

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✅ Approve & Post",
                        callback_data=f"{DEAL_APPROVE_PREFIX}:{deal_id}",
                    ),
                    InlineKeyboardButton(
                        "❌ Reject", callback_data=f"{DEAL_REJECT_PREFIX}:{deal_id}"
                    ),
                ]
            ]
        )

        body = f"<b>Deal #{deal_id}</b>\n\n{deal.as_telegram_html()}"
        try:
            if context is not None:
                message = await context.bot.send_message(
                    chat_id=self.settings.telegram_chat_id,
                    text=body,
                    parse_mode=ParseMode.HTML,
                    reply_markup=keyboard,
                    disable_web_page_preview=True,
                )
            else:
                # No running application (the CLI path): build a one-shot bot,
                # exactly as dispatch() does for listings.
                bot = ApplicationBuilder().token(
                    self.settings.telegram_bot_token
                ).build().bot
                async with bot:
                    message = await bot.send_message(
                        chat_id=self.settings.telegram_chat_id,
                        text=body,
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard,
                        disable_web_page_preview=True,
                    )
        except Exception as exc:  # noqa: BLE001 - the row is stored either way
            logger.error("Could not dispatch deal {}: {}", deal_id, exc)
            return False

        update_deal_status(
            deal_id, DealStatus.PENDING_APPROVAL, reason="Sent to Telegram."
        )
        if getattr(message, "message_id", None):
            from core.database import session_scope, Deal as DealRow  # noqa: PLC0415

            with session_scope() as session:
                row = session.get(DealRow, deal_id)
                if row is not None:
                    row.telegram_message_id = message.message_id

        logger.success("Dispatched deal {} to Telegram", deal_id)
        return True

    async def on_deal_decision(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle Approve & Post / Reject on a deal card."""
        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()

        action, _, raw_id = query.data.partition(":")
        try:
            deal_id = int(raw_id)
        except ValueError:
            logger.warning("Malformed deal callback: {!r}", query.data)
            return

        row = get_deal(deal_id)
        if row is None:
            await self._replace_markup(query, f"Deal #{deal_id} no longer exists.")
            return

        if row.status_enum is not DealStatus.PENDING_APPROVAL:
            await self._replace_markup(
                query, f"Deal #{deal_id} was already {row.status.lower()}."
            )
            return

        if action == DEAL_REJECT_PREFIX:
            update_deal_status(
                deal_id, DealStatus.REJECTED, reason="Rejected by the operator."
            )
            await self._replace_markup(query, f"🚫 Deal #{deal_id} rejected.")
            return

        if action != DEAL_APPROVE_PREFIX:
            return

        approved = update_deal_status(
            deal_id, DealStatus.APPROVED, reason="Approved by the operator."
        )
        if approved is None:
            await self._replace_markup(query, f"Deal #{deal_id} could not be approved.")
            return

        await self._replace_markup(query, f"✅ Deal #{deal_id} approved.")

        update_deal_status(
            deal_id, DealStatus.PUBLISHED, reason="Approved for posting."
        )

        pinned = await self.pin_deal(deal_id, context=context)

        # The copy is handed back either way: Pinterest is one destination, and
        # the operator may well want to post this somewhere else too.
        tail = ""
        if pinned and pinned.ok and pinned.simulated:
            tail = "\n\n📌 <i>Pin simulated (dry run).</i>"
        elif pinned and pinned.ok:
            # The Make path hands off asynchronously: its reply carries no Pin
            # id, so claiming "Pinned" with a link would be inventing one.
            pin_url = getattr(pinned, "url", "") or ""
            if pin_url.startswith("https://www.pinterest.com/pin/"):
                tail = f'\n\n📌 <a href="{_escape(pin_url)}">Pinned</a>'
            else:
                tail = "\n\n📌 <i>Sent to Make — the scenario posts the Pin.</i>"
        elif pinned and not pinned.ok:
            tail = f"\n\n⚠️ Not pinned: {_escape(pinned.error)}"

        await self._send(
            context,
            f"📣 Deal #{deal_id} is approved and ready to post.\n\n"
            f"{_escape(approved.product)}\n"
            f'<a href="{_escape(approved.affiliate_url)}">{_escape(approved.affiliate_url)}</a>'
            f"{tail}",
        )

    async def pin_deal(
        self, deal_id: int, context: ContextTypes.DEFAULT_TYPE | None = None
    ) -> Any:
        """Post an approved deal to Pinterest, if that is switched on.

        Shared by the Telegram button and ``--pin-deal``. Returns None when
        Pinterest is not enabled at all, so the caller can tell "off" apart
        from "tried and failed".
        """
        from core.webhooks import MakeWebhook  # noqa: PLC0415
        from village.pinterest_publisher import PinterestPublisher  # noqa: PLC0415

        webhook = MakeWebhook(self.settings)
        publisher = PinterestPublisher(self.settings)

        # Make first when it is configured. It holds its own Pinterest
        # connection, so it works while the direct v5 app is unapproved — and
        # running both would post the same Pin twice.
        via_make = webhook.configured
        if not via_make:
            usable, reason = publisher.status()
            if not usable:
                logger.debug("Not pinning deal {}: {}", deal_id, reason)
                return None

        row = get_deal(deal_id)
        if row is None:
            return None

        if row.pinterest_pin_id:
            logger.info("Deal {} is already pinned as {}", deal_id, row.pinterest_pin_id)
            return None

        from village.dealscout import Deal as DealPayload  # noqa: PLC0415

        deal = DealPayload.from_dict(row.payload)

        if not via_make:
            result = await publisher.post_deal(deal)
            if result.ok and not result.simulated:
                record_deal_pin(deal_id, pin_id=result.pin_id, url=result.url)
            elif not result.ok:
                logger.error("Could not pin deal {}: {}", deal_id, result.error)
            return result

        # ---- Make.com ------------------------------------------------------
        # Compose the same card the direct path would post, so whichever route
        # is in use the Pin looks identical.
        card = self.settings.storage_dir / "pins" / f"deal-{deal_id}.jpg"
        art_url = ""
        try:
            from village.pinterest_publisher import (  # noqa: PLC0415
                PIN_SIZE, render_pin_image, source_pin_background,
            )

            card.parent.mkdir(parents=True, exist_ok=True)
            if not card.is_file():
                art, art_url = await source_pin_background(
                    deal, card.with_name(f"deal-{deal_id}-bg.jpg"), self.settings
                )
                await asyncio.to_thread(render_pin_image, deal, card, PIN_SIZE, art)
        except Exception as exc:  # noqa: BLE001 - send the copy without the card
            logger.error("Could not build the Pin card for deal {}: {}", deal_id, exc)
            card = None

        # image_url, in order of what it actually points at:
        #  1. the finished card on a publicly reachable dashboard — the real
        #     artwork, disclosure and all;
        #  2. the raw background's own public URL — a valid image, but WITHOUT
        #     the hook, price or the affiliate disclosure burned in;
        #  3. nothing, in which case the scenario must use image_base64.
        image_url = self.settings.public_url(f"/api/deals/{deal_id}/pin.jpg")
        if not image_url and art_url:
            logger.info(
                "image_url will be the raw background, not the finished card. "
                "Set PUBLIC_BASE_URL, or map image_base64 in the scenario, to "
                "post the card with its disclosure."
            )
            image_url = art_url
        elif not image_url:
            logger.info(
                "image_url is empty: no PUBLIC_BASE_URL, and this artwork has no "
                "third-party-fetchable URL. The scenario must use image_base64 "
                "(toBinary(...; base64)) to get the picture."
            )

        result = await webhook.post_deal(deal, deal_id, card, image_url)

        if result.ok and not result.simulated:
            # Make posts asynchronously and the webhook reply carries no Pin id,
            # so record the handoff rather than inventing one. This is also what
            # stops a second approval re-sending the same deal.
            record_deal_pin(
                deal_id,
                pin_id=f"make-{deal_id}",
                url=str(getattr(deal, "affiliate_url", "") or ""),
            )
        elif not result.ok:
            logger.error("Make webhook failed for deal {}: {}", deal_id, result.error)

        return result

    async def cmd_shorts(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List the most recent Shorts and their state."""
        rows = recent_shorts(limit=10)
        if not rows:
            await update.effective_message.reply_text("The Bard has not written anything yet.")
            return
        lines = [f"{row.summary()} — {row.duration_seconds:.0f}s" for row in rows]
        await update.effective_message.reply_text("🎭 Recent Shorts\n" + "\n".join(lines))

    async def cmd_pending_shorts(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Re-send every Short still awaiting a decision."""
        pending = shorts_by_status(ShortStatus.PENDING_APPROVAL, limit=5)
        if not pending:
            await update.effective_message.reply_text("No Shorts are waiting on you.")
            return
        for short in pending:
            await self.dispatch_short(short.id)

    async def on_details(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the full description and Guard report as a follow-up message."""
        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()

        _, _, raw_id = query.data.partition(":")
        listing = get_listing(int(raw_id)) if raw_id.isdigit() else None
        if listing is None:
            return

        body = (
            f"<b>Listing #{listing.id} — full detail</b>\n\n"
            f"<b>Concept</b>\n{_escape(listing.concept)}\n\n"
            f"<b>Art prompt</b>\n<code>{_escape(listing.art_prompt[:600])}</code>\n\n"
            f"<b>Guard report</b>\n<pre>{_escape(listing.guard_report or 'n/a')}</pre>\n\n"
            f"<b>Description</b>\n{_escape(listing.description[:1500])}"
        )
        await self._send(context, body, chat_id=query.message.chat_id if query.message else None)

    # ---- commands -----------------------------------------------------------
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        mode = "DRY RUN (nothing is published for real)" if self.settings.dry_run else "LIVE"
        await update.effective_message.reply_text(
            "🏘 Agent Village is listening.\n"
            f"Mode: {mode}\n\n"
            "/pending — listings awaiting your decision\n"
            "/status — counts by state\n"
            "/last — the most recent listings\n"
            "/shorts — the Bard's recent videos\n"
            "/pendingshorts — re-send Shorts awaiting a decision"
        )

    async def cmd_pending(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        pending = list_by_status(ListingStatus.PENDING_APPROVAL, limit=10)
        if not pending:
            await update.effective_message.reply_text("Nothing is waiting on you.")
            return
        for listing in pending:
            await update.effective_message.reply_text(
                build_caption(listing, settings=self.settings),
                parse_mode=ParseMode.HTML,
                reply_markup=approval_keyboard(listing.id),
            )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        counts = counts_by_status()
        if not counts:
            await update.effective_message.reply_text("No listings yet.")
            return
        lines = [f"{status}: {count}" for status, count in sorted(counts.items())]
        shorts = short_counts_by_status()
        if shorts:
            lines.append("")
            lines.append("Shorts:")
            lines.extend(f"  {status}: {count}" for status, count in sorted(shorts.items()))
        await update.effective_message.reply_text("📊 Village state\n" + "\n".join(lines))

    async def cmd_last(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        rows = recent_listings(limit=10)
        if not rows:
            await update.effective_message.reply_text("No listings yet.")
            return
        await update.effective_message.reply_text(
            "🕒 Recent listings\n" + "\n".join(row.summary() for row in rows)
        )

    # ---- helpers ------------------------------------------------------------
    async def _replace_markup(self, query: Any, note: str) -> None:
        """Retire a card's buttons and append the outcome to its caption."""
        try:
            if query.message and query.message.caption is not None:
                await query.edit_message_caption(
                    caption=f"{query.message.caption}\n\n{note}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=None,
                )
            elif query.message:
                await query.edit_message_text(
                    text=f"{query.message.text}\n\n{note}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=None,
                )
        except TelegramError as exc:
            # A double-tap edits an unchanged message; that is not worth an alert.
            logger.debug("Could not edit the approval card: {}", exc)

    async def _send(
        self, context: ContextTypes.DEFAULT_TYPE, text: str, chat_id: str | int | None = None
    ) -> None:
        try:
            await context.bot.send_message(
                chat_id=chat_id or self.settings.telegram_chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except TelegramError as exc:
            logger.error("Could not send Telegram message: {}", exc)

    # ---- application --------------------------------------------------------
    def build_application(self) -> Application:
        """Wire the handlers onto a polling application."""
        if not self.settings.telegram_bot_token.strip():
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is not set. Add it to .env, or run generation "
                "without --bot and approve from the CLI."
            )

        application = ApplicationBuilder().token(self.settings.telegram_bot_token).build()
        application.add_handler(CommandHandler("start", self.cmd_start))
        application.add_handler(CommandHandler("help", self.cmd_start))
        application.add_handler(CommandHandler("pending", self.cmd_pending))
        application.add_handler(CommandHandler("status", self.cmd_status))
        application.add_handler(CommandHandler("last", self.cmd_last))
        application.add_handler(CommandHandler("ledger", self.cmd_ledger))
        application.add_handler(CommandHandler("scout", self.cmd_scout))
        application.add_handler(CommandHandler("deal", self.cmd_deal))
        application.add_handler(CommandHandler("deals", self.cmd_deals))
        application.add_handler(CommandHandler("shorts", self.cmd_shorts))
        application.add_handler(CommandHandler("pendingshorts", self.cmd_pending_shorts))
        application.add_handler(
            CallbackQueryHandler(self.on_decision, pattern=rf"^({APPROVE_PREFIX}|{REJECT_PREFIX}):\d+$")
        )
        application.add_handler(
            CallbackQueryHandler(self.on_details, pattern=rf"^{DETAILS_PREFIX}:\d+$")
        )
        application.add_handler(
            CallbackQueryHandler(
                self.on_deal_decision,
                pattern=rf"^({DEAL_APPROVE_PREFIX}|{DEAL_REJECT_PREFIX}):\d+$",
            )
        )
        application.add_handler(
            CallbackQueryHandler(
                self.on_edition_decision,
                pattern=(
                    rf"^({LEDGER_APPROVE_PREFIX}|{LEDGER_REGEN_PREFIX}"
                    rf"|{LEDGER_REJECT_PREFIX}):\d+$"
                ),
            )
        )
        application.add_handler(
            CallbackQueryHandler(
                self.on_short_decision,
                pattern=rf"^({SHORT_APPROVE_PREFIX}|{SHORT_REROLL_PREFIX}|{SHORT_REJECT_PREFIX}):\d+$",
            )
        )
        application.add_error_handler(self._on_error)
        return application

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Telegram handler error: {}", context.error)

    def run_polling(self) -> None:
        """Block, servicing callbacks until interrupted.

        Deliberately does NOT call ``Application.run_polling``. That helper
        manages the event loop itself, and to do so it calls
        ``asyncio.get_event_loop()`` — which Python 3.12 deprecated and 3.14
        turned into ``RuntimeError: There is no current event loop`` when no
        loop is running and none has been set.

        Setting a loop first would paper over it, but the loop would still be
        owned by a library internal. Driving the documented
        initialize/start/stop lifecycle under ``asyncio.run()`` keeps ownership
        here and works the same on every Python from 3.10 up.
        """
        try:
            asyncio.run(self._serve_forever())
        except KeyboardInterrupt:  # pragma: no cover - operator interrupt
            # asyncio.run re-raises Ctrl-C after cancelling; the shutdown in
            # _serve_forever has already run by this point.
            logger.info("Town Crier stopped.")

    async def _serve_forever(self) -> None:
        """Poll until a signal or Ctrl-C, then shut down cleanly."""
        application = self.build_application()
        stopping = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stopping.set)
            except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
                # Without signal handlers Ctrl-C still arrives as
                # KeyboardInterrupt, which asyncio.run propagates.
                pass

        await application.initialize()
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

        me = application.bot.username or application.bot.id
        logger.success("Town Crier polling as @{} — Ctrl-C to stop.", me)

        try:
            await stopping.wait()
        except asyncio.CancelledError:  # pragma: no cover - loop teardown
            pass
        finally:
            logger.info("Town Crier shutting down…")
            await TownCrier.stop_application(application)

    async def run_polling_async(self) -> Application:
        """Start polling inside an existing event loop.

        Used by the daemon, which runs the bot alongside the generation
        schedule. The caller owns shutdown.
        """
        application = self.build_application()
        await application.initialize()
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.success("Town Crier polling for approvals…")
        return application

    @staticmethod
    async def stop_application(application: Application) -> None:
        """Shut a polling application down cleanly."""
        try:
            if application.updater and application.updater.running:
                await application.updater.stop()
            if application.running:
                await application.stop()
            await application.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.warning("Error during Telegram shutdown: {}", exc)


async def approve_from_cli(
    listing_id: int, *, settings: Settings | None = None
) -> PublishResult:
    """Approve and publish a listing without Telegram.

    Keeps the village usable when no bot token is configured, and gives the
    dry-run path a way to exercise publication from a test.
    """
    active = settings or get_settings()
    listing = get_listing(listing_id)
    if listing is None:
        raise ValueError(f"No listing {listing_id}")
    if listing.status_enum not in {ListingStatus.PENDING_APPROVAL, ListingStatus.FAILED}:
        raise ValueError(f"Listing {listing_id} is {listing.status}, not awaiting approval")

    update_status(listing_id, ListingStatus.APPROVED, reason="Approved from the CLI.")
    merchant = Merchant(active)
    result = await merchant.publish(
        title=listing.title,
        description=listing.description,
        tags=listing.tags,
        image_path=listing.processed_image_path or listing.raw_image_path or "",
        price_cents=listing.price_cents,
    )

    if result.ok:
        update_status(
            listing_id,
            ListingStatus.PUBLISHED,
            reason=result.summary(),
            printify_image_id=result.image_id,
            printify_product_id=result.product_id,
            external_url=result.external_url,
            dry_run=1 if result.simulated else 0,
        )
        record_revenue(
            Channel.ETSY,
            listing.price_cents,
            note=listing.title[:120],
            source_key=f"listing:{listing_id}",
            dry_run=result.simulated,
        )
    else:
        update_status(listing_id, ListingStatus.FAILED, reason=result.summary())
    return result


async def approve_short_from_cli(short_id: int, *, settings: Settings | None = None) -> bool:
    """Approve a Short without Telegram, marking it ready to upload."""
    short = get_short(short_id)
    if short is None:
        raise ValueError(f"No short {short_id}")
    if short.status_enum is not ShortStatus.PENDING_APPROVAL:
        raise ValueError(f"Short {short_id} is {short.status}, not awaiting approval")

    if update_short_status(short_id, ShortStatus.APPROVED, reason="Approved from the CLI.") is None:
        return False
    published = update_short_status(
        short_id,
        ShortStatus.PUBLISHED,
        reason="Approved for upload. YouTube publishing is manual by design.",
    )
    if published is not None:
        _seed_short_revenue(short_id, settings or get_settings())
    return published is not None


def _seed_short_revenue(short_id: int, settings: Settings) -> None:
    """Open a ledger line for a newly published Short.

    It starts at zero — no views have been recorded — but it puts the video on
    the books so the YouTube channel is visible in the treasury immediately,
    and so ``--record-views`` has a line to update.
    """
    short = get_short(short_id)
    if short is None:
        return
    rpm = short.rpm_cents or settings.youtube_rpm_cents
    update_short(short_id, rpm_cents=rpm)
    record_revenue(
        Channel.YOUTUBE,
        int(round(short.views / 1000 * rpm)),
        note=f"{short.views:,} views @ {rpm}c RPM",
        source_key=f"short:{short_id}",
        estimated=True,
        dry_run=bool(short.dry_run),
    )


def reject_short_from_cli(short_id: int, reason: str = "Rejected from the CLI.") -> bool:
    """Reject a pending Short without Telegram."""
    return update_short_status(short_id, ShortStatus.REJECTED, reason=reason) is not None


def reject_from_cli(listing_id: int, reason: str = "Rejected from the CLI.") -> bool:
    """Reject a pending listing without Telegram."""
    return update_status(listing_id, ListingStatus.REJECTED, reason=reason) is not None
