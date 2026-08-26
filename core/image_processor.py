"""Turn a raw generated image into a print-ready asset.

Two jobs, in order:

1. **Background removal** with ``rembg``, so the artwork sits on transparency
   rather than on whatever backdrop the model imagined.
2. **Print scaling**: fit the art inside the configured print area and stamp the
   file at the target DPI, because a print-on-demand provider reads physical
   size from the DPI metadata, not from the pixel count alone.

``rembg`` drags in ``onnxruntime`` and downloads a model on first use, so it is
imported lazily. If it is unavailable — a fresh dry run, a slim container — the
step degrades to a transparent-safe pass-through and says so, rather than taking
the pipeline down.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFilter

from config.settings import get_settings

#: Pillow refuses very large images by default as a decompression-bomb guard.
#: A 300 DPI 12x16in canvas is 3600x4800, comfortably inside a raised ceiling.
Image.MAX_IMAGE_PIXELS = 200_000_000


@dataclass(frozen=True)
class ProcessedImage:
    """The result of the processing chain."""

    path: Path
    width: int
    height: int
    dpi: int
    background_removed: bool
    notes: tuple[str, ...] = ()

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


def _load(path: Path) -> Image.Image:
    """Open an image and normalise it to RGBA."""
    with Image.open(path) as handle:
        return handle.convert("RGBA")


def remove_background(image: Image.Image) -> tuple[Image.Image, bool]:
    """Strip the background with ``rembg``.

    :returns: the image and whether removal actually ran.
    """
    try:
        from rembg import remove  # noqa: PLC0415 - deliberately lazy
    except Exception as exc:  # noqa: BLE001 - any import failure is non-fatal
        logger.warning("rembg unavailable ({}); keeping the original background", exc)
        return image, False

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    try:
        cut_out = remove(buffer.getvalue())
    except Exception as exc:  # noqa: BLE001 - model download or runtime failure
        logger.warning("Background removal failed ({}); keeping the original", exc)
        return image, False

    with Image.open(io.BytesIO(cut_out)) as handle:
        return handle.convert("RGBA"), True


def trim_transparent(image: Image.Image, padding: int = 8) -> Image.Image:
    """Crop to the opaque content, then re-pad.

    Cutting to the subject before scaling is what makes the art fill the print
    area instead of floating in a sea of transparency.
    """
    alpha = image.getchannel("A")
    box = alpha.getbbox()
    if box is None:
        logger.warning("Image is fully transparent; skipping trim")
        return image

    cropped = image.crop(box)
    if padding <= 0:
        return cropped

    padded = Image.new(
        "RGBA", (cropped.width + padding * 2, cropped.height + padding * 2), (0, 0, 0, 0)
    )
    padded.paste(cropped, (padding, padding), cropped)
    return padded


def fit_to_print_area(
    image: Image.Image, target_width: int, target_height: int
) -> Image.Image:
    """Scale the art to fill the print canvas without distorting it.

    The aspect ratio is preserved and the result is centred on a transparent
    canvas of exactly the requested size, so every asset the merchant uploads has
    identical dimensions.
    """
    if image.width == 0 or image.height == 0:
        raise ValueError("Cannot scale a zero-sized image")

    scale = min(target_width / image.width, target_height / image.height)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))

    # LANCZOS both up and down: upscaling generated art is the common case here,
    # and it holds edges better than bicubic at these ratios.
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    if scale > 1.5:
        # A gentle sharpen counteracts the softness a large upscale introduces.
        resized = resized.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=3))

    canvas = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    canvas.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
        resized,
    )
    return canvas


def process_image(
    source: Path | str,
    destination: Path | str | None = None,
    *,
    strip_background: bool | None = None,
) -> ProcessedImage:
    """Run the full chain and write a print-ready PNG.

    :param source: the raw generated image.
    :param destination: output path; defaults to ``<source stem>_print.png``.
    :param strip_background: overrides the ``REMOVE_BACKGROUND`` setting.
    :raises FileNotFoundError: when the source does not exist.
    """
    settings = get_settings()
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"No image at {source_path}")

    output = Path(destination) if destination else source_path.with_name(
        f"{source_path.stem}_print.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    image = _load(source_path)
    logger.debug("Loaded {} ({}x{})", source_path.name, image.width, image.height)

    should_strip = settings.remove_background if strip_background is None else strip_background
    removed = False
    if should_strip:
        image, removed = remove_background(image)
        if removed:
            image = trim_transparent(image)
            notes.append("background removed")
        else:
            notes.append("background kept (rembg unavailable)")
    else:
        notes.append("background removal disabled")

    target_width, target_height = settings.print_pixel_size
    image = fit_to_print_area(image, target_width, target_height)

    image.save(
        output,
        format="PNG",
        dpi=(settings.target_dpi, settings.target_dpi),
        optimize=True,
    )

    result = ProcessedImage(
        path=output,
        width=image.width,
        height=image.height,
        dpi=settings.target_dpi,
        background_removed=removed,
        notes=tuple(notes),
    )
    logger.info(
        "Processed {} -> {} ({}x{} @ {} DPI, {})",
        source_path.name,
        output.name,
        result.width,
        result.height,
        result.dpi,
        "; ".join(result.notes),
    )
    return result


def read_dpi(path: Path | str) -> tuple[int, int] | None:
    """Read the DPI a file declares, or ``None`` when it declares none."""
    try:
        with Image.open(path) as handle:
            dpi = handle.info.get("dpi")
    except Exception as exc:  # noqa: BLE001 - unreadable file is not fatal here
        logger.warning("Could not read DPI from {}: {}", path, exc)
        return None
    if not dpi:
        return None
    return (int(round(dpi[0])), int(round(dpi[1])))


def make_placeholder(
    destination: Path | str,
    text: str,
    *,
    size: tuple[int, int] = (1024, 1024),
    background: tuple[int, int, int, int] = (24, 24, 32, 255),
    accent: tuple[int, int, int, int] = (99, 102, 241, 255),
) -> Path:
    """Draw a stand-in artwork for dry runs.

    Deliberately looks synthetic — concentric rings and the concept text — so a
    simulated asset can never be mistaken for a generated one in the storage
    directory.
    """
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGBA", size, background)
    draw = ImageDraw.Draw(image)

    centre_x, centre_y = size[0] // 2, size[1] // 2
    for step in range(9):
        radius = int(min(size) * 0.46) - step * 42
        if radius <= 0:
            break
        alpha = 210 - step * 20
        draw.ellipse(
            [centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius],
            outline=(accent[0], accent[1], accent[2], max(alpha, 40)),
            width=6,
        )

    caption = (text or "DRY RUN").strip()
    words = caption.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > 22 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    lines = lines[:4] or ["DRY RUN"]

    line_height = 34
    start_y = centre_y - (len(lines) * line_height) // 2
    for index, line in enumerate(lines):
        # Default bitmap font: no font file to ship, and legible at this size.
        draw.text(
            (centre_x, start_y + index * line_height),
            line,
            fill=(240, 240, 245, 255),
            anchor="mm",
        )

    draw.text(
        (centre_x, size[1] - 60),
        "SIMULATED ASSET - DRY RUN",
        fill=(accent[0], accent[1], accent[2], 255),
        anchor="mm",
    )

    image.save(output, format="PNG")
    logger.debug("Wrote placeholder artwork {}", output.name)
    return output
