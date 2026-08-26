"""Outbound webhooks — currently Make.com, for posting Pins.

Make holds its own Pinterest connection, so it sidesteps the developer-app
approval that blocks the direct v5 API. This project's job is therefore to hand
Make a complete, self-describing payload and let the scenario do the posting.

**On the image.** Pinterest fetches the picture from its own servers, so a Pin
needs a public HTTPS URL and nothing else will do — not a local path, and not
base64 in the payload. ``image_url`` is therefore always populated, falling back
through the composed card (when PUBLIC_BASE_URL makes it reachable), the stock
background's CDN URL, and finally a generic Unsplash stand-in.

That last case is a real compromise and worth knowing about: the Pin shows a
stock photo of somebody else's desk rather than the card this project designed,
and the affiliate disclosure survives only because it is also in the caption.
``image_source`` in the payload names which branch was taken, so a scenario can
route or skip on it.

Nothing here posts under DRY_RUN, and every failure is reported rather than
raised: a webhook that did not fire must not lose the deal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import Settings, get_settings

#: Make's webhook responds with this tiny string on success. Anything else is
#: worth surfacing verbatim — a misconfigured scenario answers 200 with text.
MAKE_ACCEPTED = "accepted"

@dataclass
class WebhookResult:
    """The outcome of one webhook call."""

    ok: bool
    status: int = 0
    body: str = ""
    simulated: bool = False
    error: str = ""
    payload_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.simulated:
            return "webhook simulated (dry run)"
        if not self.ok:
            return f"webhook failed: {self.error}"
        return f"webhook accepted (HTTP {self.status})"


#: Pinterest will not accept a Pin without a fetchable image, so image_url can
#: never be empty. These stand in when nothing specific to the deal exists.
#:
#: Pexels' CDN, NOT Unsplash. Unsplash returns 200 to a browser but Pinterest
#: answers error 235 ("this image is broken") for it, and the likely reason is
#: visible in the URL: ``/photo-1593642632823-8f785ba67e45`` has no file
#: extension at all. Every URL below ends in ``.jpeg``, resolves in ZERO
#: redirects, and was checked to return 200 with an empty User-Agent and with
#: Pinterest's own crawler string.
#:
#: Each id was also looked at before being listed here. One otherwise-fine
#: candidate (267389) was dropped for showing social-media brand logos, which
#: have no business on an affiliate Pin.
PEXELS_SIZING = "?auto=compress&cs=tinysrgb&w=1000&h=1500&fit=crop"

STOCK_FALLBACKS: dict[str, tuple[int, ...]] = {
    # desk lamp + laptop, office room, overhead desk, iMac on a desk
    "workspace": (265072, 245219, 4050315, 1029757),
    # coding on a laptop, macbook on dark wood, patch cables, laptop and plant
    "tech": (1181244, 218863, 1054397, 374074),
    # a desk covered in devices, laptop with plant
    "gadgets": (356056, 374074),
    # furniture showroom, overhead desk
    "lifestyle": (276528, 4050315),
}

#: Used when the category is unknown.
STOCK_DEFAULT = STOCK_FALLBACKS["workspace"]


def stock_image_url(photo_id: int) -> str:
    """A direct, extension-bearing Pexels CDN URL."""
    return (
        f"https://images.pexels.com/photos/{photo_id}/"
        f"pexels-photo-{photo_id}.jpeg{PEXELS_SIZING}"
    )


def stock_fallback(deal: Any, deal_id: int = 0) -> str:
    """A public, correctly sized stand-in image for a deal.

    Chosen by category then by deal id rather than at random, so re-sending the
    same deal produces the same Pin image instead of a different one each time.
    """
    category = str(getattr(deal, "category", "") or "").strip().lower()
    pool = STOCK_FALLBACKS.get(category, STOCK_DEFAULT) or STOCK_DEFAULT
    return stock_image_url(pool[deal_id % len(pool)])


async def verify_image_url(url: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Fetch ``url`` the way Pinterest will, and say whether it would work.

    Pinterest reports a broken image as error 235 long after the webhook has
    returned 200, so a URL that cannot be fetched fails silently unless it is
    checked here. The request deliberately carries no browser User-Agent: a
    host that only serves browsers is exactly the failure being screened for.

    :returns: ``(ok, reason)``; reason is empty when ok.
    """
    import httpx  # noqa: PLC0415

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            # GET rather than HEAD: some CDNs answer HEAD with 405 while
            # serving the image perfectly well.
            response = await client.get(url, headers={"User-Agent": "agent-village/1.0"})
    except Exception as exc:  # noqa: BLE001
        return False, f"unreachable ({type(exc).__name__})"

    if response.status_code != 200:
        return False, f"HTTP {response.status_code}"

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        return False, f"content-type is {content_type or 'unset'}, not an image"

    if len(response.content) < 1024:
        return False, f"only {len(response.content)} bytes"

    return True, ""


async def resolve_image_url(
    deal: Any, deal_id: int, card_url: str = "", art_url: str = ""
) -> tuple[str, str]:
    """The public HTTPS image a Pin will use, and where it came from.

    Order, first one that actually FETCHES wins:

    1. ``deal.image_url`` — a real product image, if one is ever known.
    2. ``card_url`` — the composed card on a publicly reachable dashboard.
       The only option carrying the hook, price and disclosure.
    3. ``art_url`` — the background's own CDN URL.
    4. Stock — a generic stand-in, so the field is never empty.

    Each candidate is fetched before being chosen. Pinterest reports a broken
    image as error 235 well after the webhook returned 200, so an unverified
    URL fails silently and looks like a working integration.

    :returns: ``(url, source)``; source names which branch won, for the log.
    """
    def public(value: Any) -> str:
        text = str(value or "").strip()
        # A local path is not a URL, and neither is http on someone else's
        # laptop: Pinterest fetches this from its own servers.
        return text if text.startswith("https://") else ""

    candidates = [
        (public(getattr(deal, "image_url", "")), "deal.image_url"),
        (public(card_url), "pin card (PUBLIC_BASE_URL)"),
        (public(art_url), "stock background"),
        (stock_fallback(deal, deal_id), "stock fallback"),
    ]

    for url, source in candidates:
        if not url:
            continue
        ok, reason = await verify_image_url(url)
        if ok:
            return url, source
        logger.warning("Rejecting {} for the Pin ({}): {}", source, reason, url[:90])

    # Every candidate failed, including the stock pool — almost certainly no
    # outbound network. Send the stock URL anyway rather than an empty field:
    # Pinterest can retry a URL, it cannot retry nothing.
    fallback = stock_fallback(deal, deal_id)
    logger.error(
        "No Pin image could be verified; sending the stock URL unverified. "
        "Check outbound network access."
    )
    return fallback, "stock fallback (unverified)"


async def build_deal_payload(
    deal: Any,
    deal_id: int,
    image_path: Path | str | None = None,
    image_url: str = "",
) -> dict[str, Any]:
    """The JSON a Make scenario receives for one approved deal.

    The five fields a Pinterest module needs — ``deal_id``, ``title``,
    ``caption``, ``image_url``, ``affiliate_url`` — come first and are always
    present. ``image_url`` is guaranteed to be a public HTTPS URL: Pinterest
    fetches the image from its own servers and rejects a Pin it cannot load.

    The image is NOT inlined as base64. It was, and that made the payload
    ~300 KB of which Pinterest could use none: the module wants a URL. The
    local card is still written to disk and still served over PUBLIC_BASE_URL
    when that is configured.

    :param image_path: kept for the caller's convenience; only its existence is
        reported, so a scenario can tell "the card exists but is not reachable"
        apart from "there is no card".
    """
    from village.pinterest_publisher import AFFILIATE_NOTE, build_pin_text  # noqa: PLC0415

    title, description = build_pin_text(deal)
    resolved, source = await resolve_image_url(deal, deal_id, image_url)

    payload: dict[str, Any] = {
        # ---- what the Pinterest module maps -------------------------------
        "deal_id": deal_id,
        "title": title,
        "caption": description,
        "image_url": resolved,
        "affiliate_url": str(getattr(deal, "affiliate_url", "") or ""),
        # ---- context, for scenarios that want to compose their own copy ----
        "description": description,
        "product": str(getattr(deal, "product", "") or ""),
        "hook": str(getattr(deal, "hook", "") or ""),
        "price_range": str(getattr(deal, "price_range", "") or ""),
        "category": str(getattr(deal, "category", "") or ""),
        "niche": str(getattr(deal, "niche", "") or ""),
        "disclosure": AFFILIATE_NOTE,
        "image_source": source,
    }

    path = Path(image_path) if image_path else None
    payload["card_available"] = bool(path and path.is_file())

    if source.startswith("stock fallback"):
        # Worth saying in the payload as well as the log: a scenario may prefer
        # to skip a Pin rather than post a stock photo of somebody else's desk.
        logger.info(
            "Deal {} has no specific image; using a stock stand-in. Set "
            "PUBLIC_BASE_URL to pin the composed card instead.", deal_id,
        )

    return payload


class MakeWebhook:
    """Posts approved deals to a Make.com scenario."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def configured(self) -> bool:
        return self.settings.make_webhook_configured

    def status(self) -> tuple[bool, str]:
        """Whether the webhook could fire now, and why not if it could not."""
        if not self.settings.make_pinterest_webhook_url.strip():
            return False, "MAKE_PINTEREST_WEBHOOK_URL is not set"
        if not self.configured:
            return False, "MAKE_PINTEREST_WEBHOOK_URL is not a http(s) URL"
        return True, ""

    async def post_deal(
        self,
        deal: Any,
        deal_id: int,
        image_path: Path | str | None = None,
        image_url: str = "",
    ) -> WebhookResult:
        """Send one approved deal to Make.

        Never raises: a webhook is a side effect of approval, and approval has
        already happened by the time this runs.
        """
        usable, reason = self.status()
        if not usable:
            return WebhookResult(ok=False, error=reason)

        payload = await build_deal_payload(deal, deal_id, image_path, image_url)

        if self.settings.dry_run:
            logger.info("DRY RUN — not calling the Make webhook for deal {}", deal_id)
            return WebhookResult(
                ok=True, simulated=True, payload_keys=sorted(payload)
            )

        import httpx  # noqa: PLC0415

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout_seconds
            ) as client:
                response = await client.post(
                    self.settings.make_pinterest_webhook_url.strip(), json=payload
                )
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            logger.error("Make webhook failed for deal {}: {}", deal_id, exc)
            return WebhookResult(ok=False, error=str(exc), payload_keys=sorted(payload))

        body = (response.text or "").strip()

        if response.status_code >= 400:
            detail = self._explain(response.status_code, body)
            logger.error("Make webhook rejected deal {}: {}", deal_id, detail)
            return WebhookResult(
                ok=False, status=response.status_code, body=body[:300],
                error=detail, payload_keys=sorted(payload),
            )

        logger.success(
            "Make webhook accepted deal {} (HTTP {})", deal_id, response.status_code
        )
        return WebhookResult(
            ok=True, status=response.status_code, body=body[:300],
            payload_keys=sorted(payload),
        )

    @staticmethod
    def _explain(status: int, body: str) -> str:
        """Turn Make's terse failures into something worth acting on."""
        if status == 400 and "not registered" in body.lower():
            # The classic: the scenario has never run, so Make has not learned
            # the payload shape and rejects everything.
            return (
                "Make has no data structure for this webhook yet. Open the "
                "scenario, click the webhook module, choose 'Redetermine data "
                "structure', then send one Pin — Make learns the fields from "
                f"the first payload. ({body[:120]})"
            )
        if status == 404:
            return (
                "Make returned 404 — the webhook URL is wrong, or the scenario "
                "was deleted. Copy the URL again from the webhook module."
            )
        if status == 410:
            return "Make returned 410 — this webhook has been removed."
        if status == 429:
            return "Make rate limit reached; the scenario is over its plan's quota."
        return f"HTTP {status}: {body[:200]}"
