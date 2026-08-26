"""The Guard: the last automated check before a human is asked to approve.

Everything the Guard tests is objective — a character count, a file's pixel
dimensions, a term on the blocked list. Taste is the human's job; correctness is
the Guard's. A listing that fails here never reaches Telegram, so the operator's
attention is spent on judgement rather than on catching a 141-character title.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from loguru import logger

from config.settings import Settings, get_settings
from core.image_processor import read_dpi
from core.trademark_guard import ScreenResult, screen_many
from village.scribe import (
    MAX_TAG_CHARS,
    MAX_TITLE_CHARS,
    MIN_DESCRIPTION_CHARS,
    TAG_COUNT,
)

#: Below this, a direct-to-garment print looks soft on a large front graphic.
MIN_PRINT_PIXELS = 1500


@dataclass
class GuardReport:
    """The verdict, with everything the operator needs to act on it."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def summary(self) -> str:
        if self.ok and not self.warnings:
            return f"PASS - {self.checks_run} checks, no findings."
        if self.ok:
            return f"PASS with {len(self.warnings)} warning(s)."
        return f"FAIL - {len(self.errors)} blocking issue(s)."

    def render(self) -> str:
        """Multi-line report, stored on the listing and shown in Telegram."""
        lines = [self.summary]
        for error in self.errors:
            lines.append(f"  [BLOCK] {error}")
        for warning in self.warnings:
            lines.append(f"  [WARN]  {warning}")
        return "\n".join(lines)


class Guard:
    """Validates a finished listing against every objective rule."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(
        self,
        *,
        title: str,
        tags: Sequence[str],
        description: str,
        image_path: Path | str | None,
        art_prompt: str = "",
        niche: str = "",
    ) -> GuardReport:
        """Run every check and return a single verdict."""
        errors: list[str] = []
        warnings: list[str] = []
        checks = 0

        checks += self._check_title(title, errors, warnings)
        checks += self._check_tags(tags, errors, warnings)
        checks += self._check_description(description, errors, warnings)
        checks += self._check_image(image_path, errors, warnings)
        checks += self._check_ip(
            {
                "title": title,
                "tags": " ".join(tags),
                "description": description,
                "art_prompt": art_prompt,
                "niche": niche,
            },
            errors,
            warnings,
        )

        report = GuardReport(
            ok=not errors, errors=errors, warnings=warnings, checks_run=checks
        )
        log = logger.success if report.ok else logger.error
        log("Guard: {}", report.summary)
        for line in [*errors, *warnings]:
            logger.debug("  {}", line)
        return report

    # ---- individual checks --------------------------------------------------
    def _check_title(self, title: str, errors: list[str], warnings: list[str]) -> int:
        value = (title or "").strip()
        if not value:
            errors.append("Title is empty.")
            return 1
        if len(value) > MAX_TITLE_CHARS:
            errors.append(f"Title is {len(value)} chars; Etsy allows {MAX_TITLE_CHARS}.")
        if len(value) < 30:
            warnings.append(f"Title is only {len(value)} chars; short titles rank poorly.")
        if value.isupper():
            warnings.append("Title is entirely uppercase; Etsy penalises shouting.")
        if '"' in value:
            warnings.append("Title contains a double quote, which Etsy renders awkwardly.")
        if value.count(",") > 2:
            warnings.append("Title is comma-separated; pipes or dashes read better.")
        return 5

    def _check_tags(
        self, tags: Sequence[str], errors: list[str], warnings: list[str]
    ) -> int:
        values = [str(tag) for tag in tags]
        if len(values) != TAG_COUNT:
            errors.append(f"Listing has {len(values)} tags; Etsy requires exactly {TAG_COUNT}.")

        for tag in values:
            if not tag.strip():
                errors.append("A tag is empty.")
            elif len(tag) > MAX_TAG_CHARS:
                errors.append(f"Tag {tag!r} is {len(tag)} chars; the limit is {MAX_TAG_CHARS}.")
            if any(char for char in tag if not (char.isalnum() or char == " ")):
                errors.append(f"Tag {tag!r} contains punctuation; Etsy allows letters and spaces.")
            if tag != tag.lower():
                warnings.append(f"Tag {tag!r} is not lowercase.")

        duplicates = {tag for tag in values if values.count(tag) > 1}
        if duplicates:
            errors.append(f"Duplicate tags: {', '.join(sorted(duplicates))}.")

        single_words = [tag for tag in values if " " not in tag.strip()]
        if len(single_words) > 6:
            warnings.append(
                f"{len(single_words)} single-word tags; phrase tags match more searches."
            )
        return 5

    def _check_description(
        self, description: str, errors: list[str], warnings: list[str]
    ) -> int:
        value = (description or "").strip()
        if len(value) < MIN_DESCRIPTION_CHARS:
            errors.append(
                f"Description is {len(value)} chars; at least {MIN_DESCRIPTION_CHARS} expected."
            )
        if "<" in value and ">" in value:
            warnings.append("Description appears to contain HTML, which Etsy strips.")
        if "http://" in value or "https://" in value:
            warnings.append("Description contains a URL; Etsy restricts off-site links.")
        return 3

    def _check_image(
        self, image_path: Path | str | None, errors: list[str], warnings: list[str]
    ) -> int:
        if not image_path:
            errors.append("No artwork attached to the listing.")
            return 1

        path = Path(image_path)
        if not path.is_file():
            errors.append(f"Artwork missing on disk: {path}")
            return 1

        if path.stat().st_size < 10_000:
            errors.append(f"Artwork is only {path.stat().st_size} bytes; that is not printable.")

        try:
            from PIL import Image  # noqa: PLC0415 - keeps import cost off the hot path

            with Image.open(path) as handle:
                width, height = handle.size
                mode = handle.mode
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Artwork could not be opened: {exc}")
            return 2

        if min(width, height) < MIN_PRINT_PIXELS:
            errors.append(
                f"Artwork is {width}x{height}; the short edge must be at least "
                f"{MIN_PRINT_PIXELS}px for a front print."
            )

        expected_width, expected_height = self.settings.print_pixel_size
        if (width, height) != (expected_width, expected_height):
            warnings.append(
                f"Artwork is {width}x{height}, not the configured print canvas "
                f"{expected_width}x{expected_height}."
            )

        if mode != "RGBA":
            warnings.append(f"Artwork mode is {mode}; RGBA preserves transparency for DTG.")

        dpi = read_dpi(path)
        if dpi is None:
            warnings.append("Artwork declares no DPI; the provider will assume 72.")
        elif min(dpi) < self.settings.target_dpi:
            warnings.append(
                f"Artwork declares {dpi[0]}x{dpi[1]} DPI, below the {self.settings.target_dpi} target."
            )
        return 6

    def _check_ip(
        self, fields: dict[str, str], errors: list[str], warnings: list[str]
    ) -> int:
        result: ScreenResult = screen_many(fields)
        for hit in result.blocking:
            errors.append(f"Trademark risk in {hit.category}: {hit.term!r}.")
        for hit in result.warnings:
            detail = f" ({hit.detail})" if hit.detail else ""
            warnings.append(f"Risky phrasing in {hit.category}: {hit.term!r}{detail}.")
        return len(fields)
