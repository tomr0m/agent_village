"""Cinematic still generation for Short scenes.

Separate from :mod:`village.crafter` on purpose. The Crafter makes print-on-demand
artwork and appends a directive demanding a flat graphic on a plain white
background with no photographic detail — exactly right for a t-shirt and exactly
wrong for a moody 9:16 film frame. Sharing that code meant every scene in every
Short was being asked for the opposite of what it needed.

Providers are tried in the order named by ``IMAGE_PROVIDER_ORDER``, each
falling through to the next, and the drawn placeholder is always the floor:

* **Pollinations** — free, no key, no account. Caps at 576x1024 and takes ~45s
  a scene, but it cannot fail because a model id was renamed, which is why it
  leads by default.
* **OpenRouter** — whatever ``IMAGE_MODEL`` names, if there is a key. Only a
  handful of models on OpenRouter emit images at all; asking any other id for
  one returns a bare 404, so the error path below names the working families.
* **Hugging Face** — SDXL at a real 1024x1792, if ``HF_API_TOKEN`` is set.
  This is where to go for Stability models: OpenRouter hosts none.
* **Local placeholder** — last resort only, when nothing else is reachable.

The point of the chain is that the concentric-circle placeholder is now what you
get when nothing else is reachable, instead of the default for anyone without an
image API key.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import quote

from loguru import logger

from config.settings import Settings, get_settings

#: Appended to every scene prompt. This is the look: a still from a prestige
#: documentary, not an illustration.
CINEMATIC_DIRECTIVE = (
    "Dark moody cinematic photograph, Unreal Engine 5 render aesthetic, "
    "octane render, dramatic volumetric lighting with visible light shafts, "
    "deep shadows and selective highlights, atmospheric haze, desaturated "
    "muted colour grade with a single warm accent, 35mm anamorphic lens, "
    "shallow depth of field, subtle film grain, ultra detailed, 8k, "
    "photorealistic, dramatic composition. "
    "Vertical 9:16 aspect ratio, tall portrait orientation, full-bleed. "
    "No text, letters, captions, watermarks, logos or borders anywhere."
)

#: Pollinations serves this regardless of what is requested on the free tier.
#: Stated here so the upscale in the assembler is a deliberate decision rather
#: than a surprise.
POLLINATIONS_SIZE = (576, 1024)

#: SDXL's tallest supported 9:16-ish bucket.
HF_SIZE = (1024, 1792)

#: Prefixes of the OpenRouter model families that emit images. Used only to
#: make a 404 actionable — the real list lives on OpenRouter and changes, so
#: this is a hint in an error message, never a validation gate.
IMAGE_CAPABLE_HINTS: tuple[str, ...] = (
    "google/gemini-2.5-flash-image",
    "google/gemini-3-pro-image",
    "google/gemini-3.1-flash-image",
    "openai/gpt-5-image",
)

HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"

#: Generation on a cold model can genuinely take this long; the default request
#: timeout is tuned for chat completions and is far too short.
IMAGE_TIMEOUT_SECONDS = 180

#: Anything smaller than this is an error page or a truncated download.
MIN_IMAGE_BYTES = 4096

#: Pollinations rate-limits per IP. Sequential requests are queued server-side
#: and simply take a while (~45s each is normal on the free tier); requests
#: issued in parallel get 429 instead. So scenes are generated ONE AT A TIME on
#: purpose — gathering them concurrently makes the whole batch fail rather than
#: making any of it faster.
RATE_LIMITED_STATUS = frozenset({429, 502, 503})
RATE_LIMIT_ATTEMPTS = 3
RATE_LIMIT_BACKOFF = 20


@dataclass
class SceneImage:
    """One rendered scene still."""

    path: Path
    prompt: str
    provider: str
    width: int = 0
    height: int = 0
    simulated: bool = False
    notes: str = ""
    #: A publicly fetchable URL for this image, when the provider served one.
    #: Pollinations does; a base64 data URL from OpenRouter does not. Anything
    #: downstream that needs a URL rather than bytes reads this.
    source_url: str = ""

    @property
    def real(self) -> bool:
        """Whether an actual image model produced this."""
        return not self.simulated


class SceneArtist:
    """Renders scene stills, falling through providers until one works."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _available(self) -> list[tuple[str, str]]:
        """The (name, label) of every backend that is actually usable now."""
        usable: list[tuple[str, str]] = []
        for name in self.settings.image_providers:
            if name == "openrouter" and self.settings.openrouter_configured:
                usable.append((name, f"openrouter:{self.settings.image_model}"))
            elif name == "huggingface" and self.settings.hf_api_token.strip():
                usable.append((name, "huggingface:sdxl"))
            elif name == "pollinations" and self.settings.pollinations_enabled:
                usable.append((name, "pollinations:flux"))
        return usable

    def describe_chain(self) -> list[str]:
        """Which providers will be tried, in order. For the CLI banner."""
        return [label for _, label in self._available()] + ["placeholder"]

    async def paint(
        self, prompt: str, destination: Path, directive: str | None = None
    ) -> SceneImage:
        """Render one still to ``destination``.

        Never raises: every provider failure falls through, and the last resort
        always produces a file so the caller has something to use.

        :param directive: overrides the cinematic style. A Short wants a dark
            film frame; a Pinterest card wants bright lifestyle photography, and
            the two cannot share one directive.
        """
        full = f"{prompt.strip()}\n\n{directive or CINEMATIC_DIRECTIVE}"
        destination.parent.mkdir(parents=True, exist_ok=True)

        Renderer = Callable[[str, Path], Awaitable[SceneImage | None]]
        renderers: dict[str, Renderer] = {
            "openrouter": self._via_openrouter,
            "huggingface": self._via_huggingface,
            "pollinations": self._via_pollinations,
        }
        attempts = [(name, renderers[name]) for name, _ in self._available()]

        for name, render in attempts:
            try:
                image = await render(full, destination)
            except Exception as exc:  # noqa: BLE001 - fall through to the next
                if name == "openrouter":
                    logger.warning(
                        "Scene art via openrouter failed: {}", self._explain(exc)
                    )
                else:
                    logger.warning("Scene art via {} failed: {}", name, exc)
                continue
            if image is not None:
                return image

        logger.warning(
            "No image provider produced a scene; drawing a placeholder. "
            "Set OPENROUTER_API_KEY or HF_API_TOKEN, or leave Pollinations "
            "enabled, for real artwork."
        )
        return self._placeholder(full, destination)

    # ---- providers ----------------------------------------------------------
    async def _via_openrouter(self, prompt: str, destination: Path) -> SceneImage | None:
        """Whatever image model OpenRouter is pointed at."""
        from openai import AsyncOpenAI  # noqa: PLC0415

        from village.crafter import _decode_data_url, _harvest_images  # noqa: PLC0415

        client = AsyncOpenAI(
            base_url=self.settings.openrouter_base_url,
            api_key=self.settings.openrouter_api_key,
            timeout=IMAGE_TIMEOUT_SECONDS,
            max_retries=self.settings.max_retries,
        )
        try:
            response = await client.chat.completions.create(
                model=self.settings.image_model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"modalities": ["image", "text"]},
                max_tokens=4096,
            )
        finally:
            await client.close()

        if not response.choices:
            return None

        urls = _harvest_images(response.choices[0].message)
        if not urls:
            return None

        decoded = _decode_data_url(urls[0])
        if decoded is None:
            if not urls[0].startswith("http"):
                return None
            payload = await self._get_bytes(urls[0])
        else:
            payload = decoded[0]

        if not payload or len(payload) < MIN_IMAGE_BYTES:
            return None

        return self._write(payload, destination, prompt, f"openrouter:{self.settings.image_model}")

    async def _via_huggingface(self, prompt: str, destination: Path) -> SceneImage | None:
        """SDXL through the HF inference API. Real 1024x1792."""
        import httpx  # noqa: PLC0415

        width, height = HF_SIZE
        async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                HF_URL,
                headers={
                    "Authorization": f"Bearer {self.settings.hf_api_token.strip()}",
                    "Accept": "image/png",
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.0,
                        "negative_prompt": (
                            "text, watermark, logo, caption, signature, border, "
                            "frame, blurry, low quality, cartoon, illustration, "
                            "flat colour, white background"
                        ),
                    },
                    "options": {"wait_for_model": True},
                },
            )
            if response.status_code == 503:
                # The model is warming up. Worth one wait — HF reports how long.
                logger.info("Hugging Face model is loading; waiting once")
                await asyncio.sleep(20)
                response = await client.post(
                    HF_URL,
                    headers={"Authorization": f"Bearer {self.settings.hf_api_token.strip()}"},
                    json={"inputs": prompt, "options": {"wait_for_model": True}},
                )
            response.raise_for_status()
            payload = response.content

        if len(payload) < MIN_IMAGE_BYTES:
            return None
        return self._write(payload, destination, prompt, "huggingface:sdxl")

    async def _via_pollinations(self, prompt: str, destination: Path) -> SceneImage | None:
        """Free generation, no key and no account.

        The service caps the free tier at 576x1024 whatever is requested — the
        aspect is a correct 9:16, the resolution simply is not 1080p, and the
        assembler upscales with a sharpen pass to compensate.
        """
        width, height = POLLINATIONS_SIZE
        # A random seed per call, or every Short about the same topic gets the
        # identical picture back — the endpoint is deterministic on seed.
        url = POLLINATIONS_URL.format(prompt=quote(prompt[:1500], safe="")) + (
            f"?width={width}&height={height}&model=flux&nologo=true"
            f"&seed={random.randint(1, 2**31 - 1)}"
        )

        import httpx  # noqa: PLC0415

        for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
            try:
                payload = await self._get_bytes(url)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in RATE_LIMITED_STATUS and attempt < RATE_LIMIT_ATTEMPTS:
                    delay = RATE_LIMIT_BACKOFF * attempt
                    logger.info(
                        "Pollinations returned {} (rate limit); retrying in {}s "
                        "[{}/{}]", status, delay, attempt, RATE_LIMIT_ATTEMPTS,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            if payload and len(payload) >= MIN_IMAGE_BYTES:
                # Deliberately NOT recorded as a source_url. The endpoint is
                # deterministic on its seed, so the URL does keep serving the
                # same picture — but it answers 403 to third parties, so
                # handing it to something like a Make scenario produces a
                # broken image rather than this one.
                return self._write(payload, destination, prompt, "pollinations:flux")

            # A rate-limit body is a few hundred bytes of JSON, not an image.
            if attempt < RATE_LIMIT_ATTEMPTS:
                await asyncio.sleep(RATE_LIMIT_BACKOFF * attempt)

        return None

    async def validate_openrouter_model(self) -> tuple[bool, str]:
        """Check the configured model actually emits images.

        OpenRouter publishes every model's output modalities, so this is a
        cheap preflight that turns "404 on every scene, mid-render" into one
        line before anything starts. Best effort: an unreachable catalogue
        reports unknown rather than blocking a run.
        """
        model = self.settings.image_model
        try:
            payload = await self._get_bytes("https://openrouter.ai/api/v1/models")
            catalogue = json.loads((payload or b"{}").decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - never block on the catalogue
            return True, f"could not verify ({exc})"

        entries = {
            entry.get("id"): entry
            for entry in catalogue.get("data", [])
            if isinstance(entry, dict)
        }
        entry = entries.get(model)
        if entry is None:
            capable = sorted(
                identifier for identifier, item in entries.items()
                if "image" in (item.get("architecture") or {}).get("output_modalities", [])
                and not identifier.startswith("openrouter/")
            )
            return False, (
                f"{model!r} is not on OpenRouter. Image-capable ids: "
                + ", ".join(capable[:6])
            )

        modalities = (entry.get("architecture") or {}).get("output_modalities", [])
        if "image" not in modalities:
            return False, f"{model!r} exists but outputs {modalities}, not images"
        return True, "ok"

    def _explain(self, exc: Exception) -> str:
        """Turn OpenRouter's bare 404 into the sentence that fixes it.

        A 404 here almost always means the configured id is not an
        image-capable model — usually because it was renamed and the old id
        simply stopped existing, which is what happened to
        ``google/gemini-2.5-flash-image-preview``. The default message ("404")
        gives no hint of that.
        """
        text = str(exc)
        model = self.settings.image_model

        if "404" not in text and "not found" not in text.lower():
            return text

        if not model.startswith(IMAGE_CAPABLE_HINTS):
            return (
                f"IMAGE_MODEL={model!r} is not an image-capable model on "
                "OpenRouter (404). Only a few families emit images: "
                + ", ".join(IMAGE_CAPABLE_HINTS)
                + ". Note there are no Stability/SDXL models on OpenRouter — "
                "set HF_API_TOKEN to use those. Falling back."
            )
        return (
            f"IMAGE_MODEL={model!r} returned 404. The id may have been renamed "
            "or retired; check https://openrouter.ai/models for the current "
            "name. Falling back."
        )

    # ---- helpers ------------------------------------------------------------
    async def _get_bytes(self, url: str) -> bytes | None:
        """GET a URL and return the body, or None."""
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(
            timeout=IMAGE_TIMEOUT_SECONDS, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    def _write(
        self,
        payload: bytes,
        destination: Path,
        prompt: str,
        provider: str,
        source_url: str = "",
    ) -> SceneImage:
        """Persist bytes and report what actually landed on disk."""
        destination.write_bytes(payload)

        width = height = 0
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(destination) as handle:
                width, height = handle.size
        except Exception as exc:  # noqa: BLE001 - the file is still usable
            logger.debug("Could not read the size of {}: {}", destination.name, exc)

        logger.success(
            "Scene art from {} — {}x{}, {:.0f} KB",
            provider, width, height, len(payload) / 1024,
        )
        return SceneImage(
            path=destination, prompt=prompt, provider=provider,
            width=width, height=height, source_url=source_url,
        )

    def _placeholder(self, prompt: str, destination: Path) -> SceneImage:
        """The old drawn stand-in. Only when nothing else worked."""
        from core.image_processor import make_placeholder  # noqa: PLC0415

        width, height = self.settings.video_size
        make_placeholder(destination, prompt[:120], size=(width, height))
        return SceneImage(
            path=destination, prompt=prompt, provider="placeholder",
            width=width, height=height, simulated=True,
            notes="no image provider was reachable",
        )
