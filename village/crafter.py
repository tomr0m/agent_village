"""The Crafter: turns an art prompt into a PNG via OpenRouter.

OpenRouter does not expose OpenAI's ``/images/generations`` endpoint. Image
output comes back through **chat completions** from an image-capable model, with
``modalities: ["image", "text"]`` requested and the result delivered as base64
data URLs on ``message.images[]``. This client speaks that protocol and falls
back to a locally drawn placeholder whenever a real generation is unavailable —
no key, dry-run mode, or an API failure — so the pipeline always yields an asset.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from config.settings import Settings, get_settings
from core.image_processor import make_placeholder

#: Prepended to every prompt so the model returns printable art rather than a
#: product photo, which is what image models default to for apparel prompts.
PRINT_DIRECTIVE = (
    "Produce a single flat graphic suitable for direct-to-garment printing. "
    "Isolated on a plain solid white background. Centered composition with even "
    "margins. Bold shapes and high contrast. No mockup, no garment, no model, "
    "no photograph, no drop shadow, no watermark, no signature, no border. "
    "Do not render any text beyond what the prompt explicitly asks for."
)

#: ``data:image/png;base64,....``
_DATA_URL = re.compile(r"^data:image/(?P<kind>[a-zA-Z0-9.+-]+);base64,(?P<payload>.+)$", re.DOTALL)


@dataclass(frozen=True)
class CraftedImage:
    """A generated (or simulated) artwork on disk."""

    path: Path
    prompt: str
    model: str
    simulated: bool
    notes: str = ""

    @property
    def bytes_written(self) -> int:
        return self.path.stat().st_size if self.path.is_file() else 0


def _timestamp_slug(text: str, limit: int = 40) -> str:
    """A filesystem-safe, time-stamped stem for an asset."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "art").lower()).strip("-")[:limit] or "art"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slug}"


def _decode_data_url(value: str) -> tuple[bytes, str] | None:
    """Decode a base64 data URL into raw bytes and a file extension."""
    match = _DATA_URL.match(value.strip())
    if not match:
        return None
    kind = match.group("kind").lower()
    extension = "jpg" if kind in {"jpeg", "jpg"} else re.sub(r"[^a-z0-9]", "", kind) or "png"
    try:
        return base64.b64decode(match.group("payload"), validate=False), extension
    except (binascii.Error, ValueError) as exc:
        logger.warning("Could not decode image data URL: {}", exc)
        return None


def _harvest_images(message: Any) -> list[str]:
    """Pull every image URL out of a chat message, whatever shape it arrives in.

    OpenRouter normalises to ``message.images[].image_url.url``, but the SDK may
    hand back a pydantic model or a plain dict depending on the provider, and
    some providers put the payload in the content parts instead.
    """
    urls: list[str] = []

    def push(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            urls.append(value)

    def read(container: Any, key: str) -> Any:
        if isinstance(container, dict):
            return container.get(key)
        return getattr(container, key, None)

    for image in read(message, "images") or []:
        holder = read(image, "image_url")
        push(read(holder, "url") if holder is not None else None)
        if holder is None:
            push(read(image, "url"))

    content = read(message, "content")
    if isinstance(content, list):
        for part in content:
            if read(part, "type") in {"image_url", "output_image", "image"}:
                holder = read(part, "image_url")
                push(read(holder, "url") if holder is not None else read(part, "url"))

    return urls


class Crafter:
    """Generates the listing artwork."""

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

    async def craft(self, prompt: str, *, label: str = "art") -> CraftedImage:
        """Render one image for ``prompt``.

        Never raises for an API problem: a failure produces a simulated asset and
        says so on the result, so the operator sees the whole pipeline run and
        the Guard can still reason about the output.
        """
        stem = _timestamp_slug(label)

        if self.settings.dry_run:
            return self._simulate(prompt, stem, "dry run")
        if not self.settings.openrouter_configured:
            return self._simulate(prompt, stem, "no OPENROUTER_API_KEY")

        full_prompt = f"{prompt.strip()}\n\n{PRINT_DIRECTIVE}"
        logger.info("Crafter generating with {}", self.settings.image_model)

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.image_model,
                messages=[{"role": "user", "content": full_prompt}],
                # OpenRouter's flag for requesting image output on a chat call.
                extra_body={"modalities": ["image", "text"]},
                max_tokens=4096,
            )
        except Exception as exc:  # noqa: BLE001 - degrade rather than abort
            logger.error("Image generation failed ({}); simulating instead", exc)
            return self._simulate(prompt, stem, f"api error: {exc}")

        if not response.choices:
            logger.error("Image model returned no choices; simulating instead")
            return self._simulate(prompt, stem, "empty response")

        urls = _harvest_images(response.choices[0].message)
        if not urls:
            logger.error(
                "Model {} returned no image; it may not support image output",
                self.settings.image_model,
            )
            return self._simulate(prompt, stem, "no image in response")

        decoded = _decode_data_url(urls[0])
        if decoded is None:
            # A provider may hand back an https URL rather than inline base64.
            if urls[0].startswith("http"):
                downloaded = await self._download(urls[0], stem)
                if downloaded is not None:
                    return CraftedImage(
                        path=downloaded,
                        prompt=full_prompt,
                        model=self.settings.image_model,
                        simulated=False,
                        notes="downloaded from remote url",
                    )
            logger.error("Could not decode the returned image; simulating instead")
            return self._simulate(prompt, stem, "undecodable image")

        payload, extension = decoded
        destination = self.settings.storage_dir / f"{stem}.{extension}"
        destination.write_bytes(payload)
        logger.success("Crafted {} ({:.1f} KB)", destination.name, len(payload) / 1024)

        return CraftedImage(
            path=destination,
            prompt=full_prompt,
            model=self.settings.image_model,
            simulated=False,
        )

    # ---- internals ----------------------------------------------------------
    async def _download(self, url: str, stem: str) -> Path | None:
        """Fetch an image the provider returned by reference."""
        import httpx  # noqa: PLC0415 - only needed on this branch

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not download generated image: {}", exc)
            return None

        extension = "png"
        content_type = response.headers.get("content-type", "")
        if "jpeg" in content_type or "jpg" in content_type:
            extension = "jpg"
        elif "webp" in content_type:
            extension = "webp"

        destination = self.settings.storage_dir / f"{stem}.{extension}"
        destination.write_bytes(response.content)
        logger.success("Downloaded {} ({:.1f} KB)", destination.name, len(response.content) / 1024)
        return destination

    def _simulate(self, prompt: str, stem: str, reason: str) -> CraftedImage:
        """Draw a stand-in asset locally."""
        destination = self.settings.storage_dir / f"{stem}-simulated.png"
        make_placeholder(destination, prompt[:120])
        logger.info("Crafter simulated {} ({})", destination.name, reason)
        return CraftedImage(
            path=destination,
            prompt=prompt,
            model="simulated",
            simulated=True,
            notes=reason,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
