"""Post approved deals to Pinterest with the v5 API.

Same shape as :mod:`village.youtube_publisher`: opt-in, never posts under
DRY_RUN, reports failures rather than raising, and turns the platform's terse
errors into something that says what to do about them.

**On the image.** A Pin cannot exist without one, and a curated deal has no
photograph — the Scout recommends a product *class*, not a specific listing it
has seen. The obvious shortcut is to generate a photoreal product image, and
this deliberately does not: an AI image of a product that is not the product
being linked to, on a Pin that earns commission, misrepresents what someone is
clicking through to buy. Instead each Pin gets a designed **text card** built
here with Pillow — clearly a graphic, honest about being one, and carrying the
hook, the product and the price range that the Pin is actually about.

**On disclosure.** Pinterest's merchant guidelines require affiliate links to
be disclosed on the Pin itself, so the disclosure goes in the description and
is not optional.

**On the board.** Configured by NAME, because that is what is readable in the
Pinterest UI. The name is resolved to a board id once per process and cached.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import Settings, get_settings

API_ROOT = "https://api.pinterest.com/v5"

#: Pinterest recommends 2:3. Anything else is cropped in the feed.
PIN_SIZE = (1000, 1500)

#: Platform limits. Exceeding either is a 400, so they are enforced here.
MAX_PIN_TITLE = 100
MAX_PIN_DESCRIPTION = 800

#: Faces to try for the card, per weight, most-wanted first.
#:
#: NOT the Shorts font. ``video_engine.load_font`` leads with Impact, which is
#: correct for a burned-in caption over video and looks like a 2009 meme on a
#: Pinterest card. Inter is listed first because it is the modern default and
#: someone may well have installed it; Helvetica Neue is the fallback that
#: exists on every Mac, and its .ttc carries real weights rather than faking
#: bold by smearing.
#:
#: Each entry is (path, collection index).
PIN_FONTS: dict[str, tuple[tuple[str, int], ...]] = {
    "bold": (
        ("/Library/Fonts/Inter-Bold.ttf", 0),
        (str(Path.home() / "Library/Fonts/Inter-Bold.ttf"), 0),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
        ("/System/Library/Fonts/Helvetica.ttc", 1),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
        ("C:/Windows/Fonts/arialbd.ttf", 0),
    ),
    "medium": (
        ("/Library/Fonts/Inter-Medium.ttf", 0),
        (str(Path.home() / "Library/Fonts/Inter-Medium.ttf"), 0),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 10),
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
        ("C:/Windows/Fonts/arial.ttf", 0),
    ),
    "regular": (
        ("/Library/Fonts/Inter-Regular.ttf", 0),
        (str(Path.home() / "Library/Fonts/Inter-Regular.ttf"), 0),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
        ("C:/Windows/Fonts/arial.ttf", 0),
    ),
}


def pin_font(size: int, weight: str = "bold") -> Any:
    """A clean sans face at ``size``, or Pillow's bitmap font as a last resort."""
    from PIL import ImageFont  # noqa: PLC0415

    for path, index in PIN_FONTS.get(weight, ()):
        if not Path(path).is_file():
            continue
        try:
            return ImageFont.truetype(path, size, index=index)
        except Exception:  # noqa: BLE001 - unreadable face, try the next
            continue

    logger.debug("No sans face found for weight {!r}; using the bitmap font", weight)
    return ImageFont.load_default()


#: Required on affiliate Pins by Pinterest's own guidelines.
AFFILIATE_NOTE = "Affiliate link — I may earn a commission from purchases."

#: Transient statuses worth one retry.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3


class PinterestError(RuntimeError):
    """Raised when a Pin cannot be created."""


@dataclass
class PinResult:
    """The outcome of one Pin attempt."""

    ok: bool
    pin_id: str = ""
    url: str = ""
    board_id: str = ""
    simulated: bool = False
    error: str = ""

    def summary(self) -> str:
        if not self.ok:
            return f"pin failed: {self.error}"
        if self.simulated:
            return "pin simulated (dry run)"
        return f"pinned {self.pin_id}"


#: The look the Pin background is asked for. Deliberately a LIFESTYLE SCENE and
#: not a hero product shot: a generated close-up of "the product" would read as
#: a photograph of the thing being sold, which it is not. A styled desk with
#: nothing claiming to be the item is honest context.
PIN_ART_DIRECTIVE = (
    "Bright airy lifestyle photograph, editorial interior styling, soft natural "
    "window light from the side, warm neutral palette, wood and matte surfaces, "
    "styled props and visible texture, uncluttered negative space in the lower "
    "third of the frame, everything in sharp focus front to back, crisp fine "
    "detail, shot on 35mm at f/8, photorealistic, high resolution. "
    "Vertical 2:3 portrait composition. "
    "No text, letters, captions, watermarks, logos or brand marks anywhere. "
    "No people's faces. No packaging, no boxes, no labels."
)


async def source_pin_background(
    deal: Any, destination: Path, settings: Settings
) -> tuple[Path | None, str]:
    """Find a background photograph for a Pin card.

    Order matters, and so does what each source actually is:

    1. **Pexels** — a real licensed photograph. Preferred whenever a key
       exists, because a photograph of a desk really is one.
    2. **Generated art** — a plausible invention, used as ambient context.

    Neither is the product being linked to. Amazon's own product images need
    PA-API access and may not be scraped, so the card never claims to show the
    item; it shows the setting the item belongs in.

    :returns: ``(path, source_url)``. The URL is empty unless the provider
        served one publicly — it is the raw background, never the finished
        card, and callers must not present it as the Pin artwork.
    """
    query = str(getattr(deal, "search_terms", "") or getattr(deal, "product", ""))

    from core.stock_video import StockPhotoLibrary  # noqa: PLC0415

    photos = StockPhotoLibrary(settings)
    if photos.configured:
        found = await photos.fetch(query, destination)
        if found is not None:
            return found, photos.last_source_url

    from core.scene_art import SceneArtist  # noqa: PLC0415

    # A Pin is ONE image that sits in a feed being judged on how it looks, so
    # it is worth the better generator: OpenRouter returns 1024x1024 against
    # Pollinations' 576x1024 free-tier cap, for a fraction of a cent. Shorts
    # keep their own order — they need five images per video, where the free
    # tier's price matters more than its resolution.
    art_settings = settings
    if settings.openrouter_configured:
        art_settings = settings.model_copy(
            update={"image_provider_order": "openrouter,pollinations"}
        )

    scene = f"{query} in a styled modern workspace, product not visible"
    try:
        painted = await SceneArtist(art_settings).paint(
            scene, destination, directive=PIN_ART_DIRECTIVE
        )
    except Exception as exc:  # noqa: BLE001 - a plain card still works
        logger.warning("Could not source Pin art: {}", exc)
        return None, ""

    # The drawn placeholder is worse than no image at all here: a Pin with a
    # concentric-ring graphic behind the text looks broken, where a clean
    # typographic card does not.
    if painted.simulated:
        return None, ""
    return painted.path, painted.source_url


def _scrim(size: tuple[int, int], start: float = 0.42) -> Any:
    """A top-to-bottom darkening mask, transparent above ``start``.

    Text over a photograph is unreadable without one, and a hard panel edge
    looks pasted on. The ramp is eased rather than linear so the transition is
    invisible.
    """
    from PIL import Image  # noqa: PLC0415

    width, height = size
    mask = Image.new("L", (1, height))
    for y in range(height):
        position = y / max(1, height - 1)
        if position <= start:
            value = 0.0
        else:
            eased = (position - start) / (1 - start)
            value = eased ** 1.6            # slow start, deep finish
        mask.putpixel((0, y), int(245 * value))
    return mask.resize((width, height))


def render_pin_image(
    deal: Any,
    destination: Path,
    size: tuple[int, int] = PIN_SIZE,
    background: Path | str | None = None,
) -> Path:
    """Draw the deal as a 2:3 Pin card over a photograph.

    Pinterest is a visual feed, so the image carries the card and the type sits
    on it: full-bleed background, an eased scrim darkening toward the bottom,
    and the copy in the lower half where the scrim is heaviest.

    Falls back to a clean typographic card when no background could be sourced,
    rather than putting text over something that looks broken.
    """
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter  # noqa: PLC0415

    width, height = size
    canvas = Image.new("RGB", size, (24, 20, 16))

    photo = Path(background) if background else None
    has_photo = bool(photo and photo.is_file())

    if has_photo:
        with Image.open(photo) as handle:
            art = handle.convert("RGB")
        # Cover-crop: bars on a Pin look like a mistake.
        scale = max(width / art.width, height / art.height)
        art = art.resize(
            (max(1, int(art.width * scale)), max(1, int(art.height * scale))),
            Image.Resampling.LANCZOS,
        )

        # Free generators cap well below 1000x1500 — Pollinations serves
        # 576x1024 — so the background routinely arrives needing a 1.7x
        # upscale, and Lanczos alone leaves it visibly soft. Sharpen in
        # proportion to how far it was stretched.
        if scale > 1.2:
            strength = min(2.0, scale - 1.0)
            art = art.filter(
                ImageFilter.UnsharpMask(
                    radius=1.5 * strength, percent=int(95 * strength), threshold=3
                )
            )

        left = (art.width - width) // 2
        top = (art.height - height) // 2
        canvas = art.crop((left, top, left + width, top + height))

        # A little contrast and warmth so the photograph reads as styled rather
        # than as a washed-out stock frame.
        canvas = ImageEnhance.Contrast(canvas).enhance(1.12)
        canvas = ImageEnhance.Color(canvas).enhance(1.08)

        # Darken toward the bottom so the copy has somewhere to live.
        shade = Image.new("RGB", size, (14, 11, 9))
        canvas = Image.composite(shade, canvas, _scrim(size))

    draw = ImageDraw.Draw(canvas)

    margin = 78
    inner = width - margin * 2

    def wrap(text: str, font: Any) -> list[str]:
        words, lines, line = str(text or "").split(), [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=font) > inner and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines

    def shadowed(xy: tuple[int, int], text: str, font: Any, fill: tuple[int, int, int]) -> None:
        """Type over a photograph needs its own contrast, not the scrim's."""
        x, y = xy
        draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=fill)

    hook_font = pin_font(58, "bold")
    product_font = pin_font(34, "medium")
    price_font = pin_font(42, "bold")
    small_font = pin_font(26, "regular")

    hook_lines = wrap(getattr(deal, "hook", ""), hook_font)[:4]
    product_lines = wrap(getattr(deal, "product", ""), product_font)[:3]
    cons = list(getattr(deal, "cons", []) or [])
    con_lines = wrap(f"Trade-off: {cons[0]}", small_font)[:2] if cons else []

    block = (
        len(hook_lines) * 74
        + 34
        + len(product_lines) * 50
        + 96                                   # price chip
        + (len(con_lines) * 36 + 18 if con_lines else 0)
    )

    # Anchored to the bottom, above the disclosure: that is where the eye lands
    # last and where the gradient is deepest.
    cursor = height - 150 - block

    # A second, tighter gradient directly behind the copy. The global scrim
    # stops the picture blowing out; this is what stops a bright or busy patch
    # of photograph landing under a line of text. It fades in over its top 45%
    # so there is no visible edge where it starts.
    if has_photo:
        panel_top = max(0, cursor - 70)
        panel_height = height - panel_top
        panel = Image.new("RGBA", (width, panel_height), (10, 8, 6, 0))
        panel_draw = ImageDraw.Draw(panel)
        for offset in range(panel_height):
            ramp = offset / max(1, panel_height - 1)
            alpha = int(215 * min(1.0, (ramp / 0.45) ** 1.4))
            panel_draw.line([(0, offset), (width, offset)], fill=(10, 8, 6, alpha))
        canvas = canvas.convert("RGBA")
        canvas.alpha_composite(panel, (0, panel_top))
        canvas = canvas.convert("RGB")
        draw = ImageDraw.Draw(canvas)

    # Thin frame last, so the gradient cannot paint over it.
    draw.rectangle([0, 0, width - 1, height - 1], outline=(214, 176, 96), width=5)

    for line in hook_lines:
        shadowed((margin, cursor), line, hook_font, (255, 255, 255))
        cursor += 74

    cursor += 12
    draw.rectangle([margin, cursor, margin + 120, cursor + 5], fill=(214, 176, 96))
    cursor += 22

    for line in product_lines:
        shadowed((margin, cursor), line, product_font, (232, 224, 208))
        cursor += 50

    cursor += 18
    price = str(getattr(deal, "price_range", "") or "")
    if price:
        # A filled chip, so the number survives any background behind it.
        chip_w = int(draw.textlength(price, font=price_font)) + 44
        draw.rectangle([margin, cursor, margin + chip_w, cursor + 66], fill=(214, 176, 96))
        draw.text((margin + 22, cursor + 8), price, font=price_font, fill=(26, 21, 16))
    cursor += 96

    for line in con_lines:
        shadowed((margin, cursor), line, small_font, (236, 200, 130))
        cursor += 36

    note_font = pin_font(22, "regular")
    for index, line in enumerate(wrap(AFFILIATE_NOTE, note_font)[:2]):
        shadowed(
            (margin, height - 108 + index * 30), line, note_font, (206, 194, 176)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=88)
    return destination


def build_pin_text(deal: Any) -> tuple[str, str]:
    """The Pin's title and description, inside Pinterest's limits."""
    title = str(getattr(deal, "product", "") or "Deal").strip()[:MAX_PIN_TITLE]

    parts: list[str] = []
    hook = str(getattr(deal, "hook", "") or "").strip()
    if hook:
        parts.append(hook)

    benefits = list(getattr(deal, "benefits", []) or [])[:3]
    if benefits:
        parts.append("\n".join(f"• {item}" for item in benefits))

    verdict = str(getattr(deal, "verdict", "") or "").strip()
    if verdict:
        parts.append(verdict)

    parts.append(AFFILIATE_NOTE)

    description = "\n\n".join(parts)
    if len(description) > MAX_PIN_DESCRIPTION:
        # Trim the body, never the disclosure.
        keep = MAX_PIN_DESCRIPTION - len(AFFILIATE_NOTE) - 4
        description = description[:keep].rstrip() + "\n\n" + AFFILIATE_NOTE
    return title, description


class PinterestPublisher:
    """Creates Pins for approved deals."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._board_id: str | None = None

    # ---- configuration ------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self.settings.pinterest_configured

    def status(self) -> tuple[bool, str]:
        """Whether a Pin could be posted now, and why not if it could not."""
        if not self.settings.pinterest_enabled:
            return False, "PINTEREST_ENABLED is off"
        if not self.settings.pinterest_access_token.strip():
            return False, "PINTEREST_ACCESS_TOKEN is not set"
        if not self.settings.pinterest_board_name.strip():
            return False, "PINTEREST_BOARD_NAME is not set"
        return True, ""

    # ---- api ----------------------------------------------------------------
    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """One v5 call, with a retry on the transient statuses."""
        import httpx  # noqa: PLC0415

        url = f"{API_ROOT}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.settings.pinterest_access_token.strip()}",
            "Content-Type": "application/json",
        }

        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout_seconds
            ) as client:
                try:
                    response = await client.request(
                        method, url, headers=headers, json=payload
                    )
                except Exception as exc:  # noqa: BLE001 - network flake
                    last = exc
                    if attempt < MAX_ATTEMPTS:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise PinterestError(str(exc)) from exc

            if response.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                logger.warning(
                    "Pinterest returned {}; retry {}/{}",
                    response.status_code, attempt, MAX_ATTEMPTS,
                )
                await asyncio.sleep(2 ** attempt)
                continue

            if response.status_code >= 400:
                raise PinterestError(self._explain(response.status_code, response.text))

            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise PinterestError("Pinterest returned a non-JSON body") from exc

        raise PinterestError(str(last) if last else "exhausted retries")

    async def resolve_board(self) -> str:
        """The board id for the configured board name.

        Resolved once per process. A name is what the operator can read off the
        Pinterest UI; the API only takes ids.
        """
        if self._board_id:
            return self._board_id

        wanted = self.settings.pinterest_board_name.strip().casefold()
        payload = await self._request("GET", "boards?page_size=100")
        boards = payload.get("items") or []

        for board in boards:
            if str(board.get("name", "")).strip().casefold() == wanted:
                self._board_id = str(board.get("id"))
                logger.debug("Board {!r} -> {}", board.get("name"), self._board_id)
                return self._board_id

        available = ", ".join(repr(b.get("name")) for b in boards) or "none"
        raise PinterestError(
            f"No board named {self.settings.pinterest_board_name!r}. "
            f"Boards on this account: {available}"
        )

    async def post_deal(self, deal: Any, image_path: Path | str | None = None) -> PinResult:
        """Create one Pin for a curated deal.

        :param deal: anything with hook / product / price_range / affiliate_url.
        :param image_path: an existing card image; one is rendered if omitted.
        :returns: a :class:`PinResult`; never raises for an expected failure.
        """
        link = str(getattr(deal, "affiliate_url", "") or "")
        if not link:
            return PinResult(ok=False, error="the deal has no affiliate link")

        if self.settings.dry_run:
            logger.info("DRY RUN — not pinning {!r}", getattr(deal, "product", ""))
            return PinResult(
                ok=True, pin_id="dry-run", url="", simulated=True,
            )

        usable, reason = self.status()
        if not usable:
            return PinResult(ok=False, error=reason)

        # Render the card first: a failure here is local and cheap to report,
        # and there is no point resolving a board for a Pin that cannot exist.
        try:
            target = Path(image_path) if image_path else (
                self.settings.storage_dir / "pins" /
                f"pin-{abs(hash(link)) % 10**8}.jpg"
            )
            # Own the directory here rather than relying on whichever source
            # happens to run: the background and the card both land in it.
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                art, _ = await source_pin_background(
                    deal, target.with_name(target.stem + "-bg.jpg"), self.settings
                )
                await asyncio.to_thread(render_pin_image, deal, target, PIN_SIZE, art)
            encoded = base64.b64encode(target.read_bytes()).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not build the Pin image: {}", exc)
            return PinResult(ok=False, error=f"image: {exc}")

        title, description = build_pin_text(deal)

        try:
            board_id = await self.resolve_board()
            payload = await self._request(
                "POST",
                "pins",
                {
                    "board_id": board_id,
                    "title": title,
                    "description": description,
                    "link": link,
                    "media_source": {
                        "source_type": "image_base64",
                        "content_type": "image/jpeg",
                        "data": encoded,
                    },
                },
            )
        except PinterestError as exc:
            logger.error("Pinterest post failed: {}", exc)
            return PinResult(ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, never raised at callers
            logger.exception("Pinterest post raised")
            return PinResult(ok=False, error=str(exc))

        pin_id = str(payload.get("id") or "")
        if not pin_id:
            return PinResult(ok=False, error="Pinterest accepted the Pin but returned no id")

        result = PinResult(
            ok=True,
            pin_id=pin_id,
            url=f"https://www.pinterest.com/pin/{pin_id}/",
            board_id=board_id,
        )
        logger.success("Pinned to {!r}: {}", self.settings.pinterest_board_name, result.url)
        return result

    async def check(self) -> tuple[bool, str]:
        """Verify the token and the board without posting anything."""
        if not self.settings.pinterest_access_token.strip():
            return False, "PINTEREST_ACCESS_TOKEN is not set"
        try:
            board_id = await self.resolve_board()
        except PinterestError as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return True, f"board {self.settings.pinterest_board_name!r} -> {board_id}"

    # ---- errors -------------------------------------------------------------
    @staticmethod
    def _explain(status: int, body: str) -> str:
        """Turn a v5 error into the sentence that fixes it."""
        message = body
        try:
            parsed = json.loads(body)
            message = parsed.get("message") or body
        except Exception:  # noqa: BLE001 - the raw body still gets reported
            pass

        lowered = message.lower()

        if "consumer type is not supported" in lowered:
            # This reads like a token problem and is not one. Every v5 endpoint
            # returns it when the app behind the token has only Trial access.
            return (
                "Pinterest rejected the app, not the token: 'consumer type is "
                "not supported'. The developer app behind this token only has "
                "TRIAL access, which cannot call the v5 API for a real account. "
                "Apply for Standard access at developers.pinterest.com "
                "(App > Request access), then mint a fresh token. "
                f"(HTTP {status})"
            )
        if status == 401:
            return (
                f"Pinterest rejected the credentials ({message}). The token may "
                "have expired — v5 access tokens are short-lived and need "
                "refreshing."
            )
        if status == 403:
            return (
                f"Pinterest refused the request ({message}). Check the token's "
                "scopes include boards:read and pins:write."
            )
        if status == 429:
            return f"Pinterest rate limit reached ({message}). Try again later."
        return f"HTTP {status}: {message}"
