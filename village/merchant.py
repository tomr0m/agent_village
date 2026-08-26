"""The Merchant: publishes an approved listing to Printify.

Three calls, in order: upload the artwork, create the product against a blueprint
and print provider, then publish it to the connected Etsy shop.

Every one of them routes through :meth:`Merchant._simulate` when the village is
in dry-run mode **or** when Printify credentials are absent. That automatic
fallback is what lets the entire pipeline — including the Telegram approve
button and the resulting database transition — be exercised end to end with no
store connected. A simulated result is always flagged as such, on the return
value and on the listing row, so nothing downstream mistakes it for a real sale.
"""

from __future__ import annotations

import asyncio
import base64
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from config.settings import Settings, get_settings


@dataclass
class PublishResult:
    """The outcome of a publish attempt."""

    ok: bool
    simulated: bool
    image_id: str | None = None
    product_id: str | None = None
    external_url: str | None = None
    error: str | None = None
    steps: list[str] = field(default_factory=list)

    def summary(self) -> str:
        prefix = "SIMULATED" if self.simulated else "LIVE"
        if not self.ok:
            return f"{prefix} publish failed: {self.error}"
        return f"{prefix} publish ok - product {self.product_id}"


def _fake_id(prefix: str) -> str:
    """A stable-looking identifier for simulated resources."""
    body = "".join(random.choices(string.hexdigits.lower()[:16], k=24))
    return f"{prefix}_{body}"


class Merchant:
    """Printify API client with an automatic dry-run fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ---- mode ---------------------------------------------------------------
    @property
    def simulating(self) -> bool:
        """Should this run avoid the network entirely?"""
        if self.settings.dry_run:
            return True
        if not self.settings.printify_configured:
            logger.warning(
                "Printify credentials absent - falling back to simulation despite DRY_RUN=false"
            )
            return True
        return False

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.printify_base_url,
            headers={
                "Authorization": f"Bearer {self.settings.printify_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AgentVillage/1.0",
            },
            timeout=self.settings.request_timeout_seconds,
        )

    # ---- public API ---------------------------------------------------------
    async def publish(
        self,
        *,
        title: str,
        description: str,
        tags: list[str],
        image_path: Path | str,
        price_cents: int | None = None,
    ) -> PublishResult:
        """Upload, create and publish one product.

        Never raises: a transport failure is returned as a failed result so the
        caller can record it on the listing and move on.
        """
        price = price_cents or self.settings.listing_price_cents
        path = Path(image_path)

        if not path.is_file():
            return PublishResult(
                ok=False,
                simulated=self.simulating,
                error=f"artwork missing at {path}",
            )

        if self.simulating:
            return await self._simulate(title, path, price)

        steps: list[str] = []
        try:
            async with self._client() as client:
                image_id = await self._upload_image(client, path)
                steps.append(f"uploaded image {image_id}")

                variant_ids = await self._fetch_variant_ids(client)
                steps.append(f"resolved {len(variant_ids)} variant(s)")

                product_id = await self._create_product(
                    client,
                    title=title,
                    description=description,
                    tags=tags,
                    image_id=image_id,
                    variant_ids=variant_ids,
                    price_cents=price,
                )
                steps.append(f"created product {product_id}")

                external_url = await self._publish_product(client, product_id)
                steps.append("published to the connected shop")

        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:400] if exc.response is not None else ""
            message = f"HTTP {exc.response.status_code if exc.response else '?'}: {body}"
            logger.error("Printify rejected the request: {}", message)
            return PublishResult(ok=False, simulated=False, error=message, steps=steps)
        except Exception as exc:  # noqa: BLE001 - transport, DNS, timeout
            logger.error("Printify publish failed: {}", exc)
            return PublishResult(ok=False, simulated=False, error=str(exc), steps=steps)

        logger.success("Published product {} to shop {}", product_id, self.settings.printify_shop_id)
        return PublishResult(
            ok=True,
            simulated=False,
            image_id=image_id,
            product_id=product_id,
            external_url=external_url,
            steps=steps,
        )

    # ---- live calls ---------------------------------------------------------
    async def _upload_image(self, client: httpx.AsyncClient, path: Path) -> str:
        """POST /uploads/images.json — base64 upload, returns the image id."""
        payload = {
            "file_name": path.name,
            "contents": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        response = await client.post("/uploads/images.json", json=payload)
        response.raise_for_status()
        data = response.json()
        image_id = str(data.get("id", "")).strip()
        if not image_id:
            raise RuntimeError(f"upload returned no id: {data}")
        return image_id

    async def _fetch_variant_ids(self, client: httpx.AsyncClient, limit: int = 12) -> list[int]:
        """Resolve sellable variants for the configured blueprint and provider."""
        endpoint = (
            f"/catalog/blueprints/{self.settings.printify_blueprint_id}"
            f"/print_providers/{self.settings.printify_print_provider_id}/variants.json"
        )
        response = await client.get(endpoint)
        response.raise_for_status()
        variants = response.json().get("variants", [])
        ids = [int(variant["id"]) for variant in variants if "id" in variant][:limit]
        if not ids:
            raise RuntimeError(
                f"blueprint {self.settings.printify_blueprint_id} has no variants for "
                f"provider {self.settings.printify_print_provider_id}"
            )
        return ids

    async def _create_product(
        self,
        client: httpx.AsyncClient,
        *,
        title: str,
        description: str,
        tags: list[str],
        image_id: str,
        variant_ids: list[int],
        price_cents: int,
    ) -> str:
        """POST /shops/{shop}/products.json — returns the product id."""
        payload: dict[str, Any] = {
            "title": title,
            "description": description,
            "tags": tags,
            "blueprint_id": self.settings.printify_blueprint_id,
            "print_provider_id": self.settings.printify_print_provider_id,
            "variants": [
                {"id": variant_id, "price": price_cents, "is_enabled": True}
                for variant_id in variant_ids
            ],
            "print_areas": [
                {
                    "variant_ids": variant_ids,
                    "placeholders": [
                        {
                            "position": "front",
                            "images": [
                                {
                                    "id": image_id,
                                    # Centred, full-width placement.
                                    "x": 0.5,
                                    "y": 0.5,
                                    "scale": 1.0,
                                    "angle": 0,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        response = await client.post(
            f"/shops/{self.settings.printify_shop_id}/products.json", json=payload
        )
        response.raise_for_status()
        product_id = str(response.json().get("id", "")).strip()
        if not product_id:
            raise RuntimeError("product creation returned no id")
        return product_id

    async def _publish_product(self, client: httpx.AsyncClient, product_id: str) -> str | None:
        """POST /shops/{shop}/products/{id}/publish.json."""
        payload = {
            "title": True,
            "description": True,
            "images": True,
            "variants": True,
            "tags": True,
            "keyFeatures": True,
            "shipping_template": True,
        }
        response = await client.post(
            f"/shops/{self.settings.printify_shop_id}/products/{product_id}/publish.json",
            json=payload,
        )
        response.raise_for_status()
        # Printify answers 200 with an empty body; the storefront URL arrives
        # later over its webhook, so the product page is the best link we have.
        return (
            f"https://printify.com/app/store/{self.settings.printify_shop_id}"
            f"/products/{product_id}"
        )

    # ---- simulation ---------------------------------------------------------
    async def _simulate(self, title: str, path: Path, price_cents: int) -> PublishResult:
        """Fabricate a successful publish without touching the network."""
        # A short pause keeps the simulated timeline honest in the logs and lets
        # any progress UI actually render the step.
        await asyncio.sleep(0.2)

        image_id = _fake_id("img")
        product_id = _fake_id("prod")
        steps = [
            f"[dry-run] uploaded {path.name} ({path.stat().st_size / 1024:.1f} KB) -> {image_id}",
            f"[dry-run] created product {product_id} "
            f"(blueprint {self.settings.printify_blueprint_id}, "
            f"provider {self.settings.printify_print_provider_id}, "
            f"price {price_cents / 100:.2f})",
            "[dry-run] published to the simulated shop",
        ]
        for step in steps:
            logger.info(step)

        return PublishResult(
            ok=True,
            simulated=True,
            image_id=image_id,
            product_id=product_id,
            external_url=f"https://example.invalid/dry-run/{product_id}",
            steps=steps,
        )

    async def health_check(self) -> dict[str, Any]:
        """Report whether live publishing would work right now."""
        if self.simulating:
            return {
                "mode": "simulated",
                "reason": "DRY_RUN enabled"
                if self.settings.dry_run
                else "Printify credentials missing",
                "reachable": None,
            }
        try:
            async with self._client() as client:
                response = await client.get("/shops.json")
                response.raise_for_status()
                shops = response.json()
        except Exception as exc:  # noqa: BLE001
            return {"mode": "live", "reachable": False, "error": str(exc)}
        return {"mode": "live", "reachable": True, "shops": len(shops)}
