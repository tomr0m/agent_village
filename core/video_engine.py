"""Assemble a vertical Short from still images, narration and a script.

Three things happen here, in order:

1. **Ken Burns.** Each scene is cropped through a window that shrinks (or
   grows) over its runtime, so a still reads as motion. Alternating the
   direction per scene stops the sequence feeling mechanical.
2. **Subtitles.** Cues are rendered to transparent PNGs with Pillow and overlaid
   as timed images. That is deliberate: MoviePy's ``TextClip`` shells out to
   ImageMagick, which is the single most common reason a video pipeline that
   "works on my machine" dies in a container. Pillow is already a dependency.
3. **Mux.** The narration is laid under the picture and the whole thing is
   encoded to a yuv420p H.264 mp4 — the profile every platform will accept.

Three render backends, tried in order, so the pipeline always produces
*something*:

* ``moviepy`` — if importable.
* ``ffmpeg`` — the CLI, driven directly. Preferred when present: it is faster,
  and it is what MoviePy shells out to anyway.
* ``storyboard`` — no video toolchain at all: writes a contact-sheet PNG of the
  scenes with the narration burned in, clearly labelled as a storyboard. The
  run completes and the operator sees exactly what would have been rendered.
"""

from __future__ import annotations

import json
import math
import difflib
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field, field
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config.settings import Settings, get_settings

#: A scene shorter than this reads as a glitch rather than a Ken Burns move,
#: so cuts are never placed closer together than this even when the narration
#: rattles through two scenes in a second.
MIN_SCENE_SECONDS = 1.6

#: The Ken Burns moves, cycled so consecutive scenes never repeat one.
#:
#: Each entry is (zoom-direction, x-drift, y-drift) as zoompan expressions in
#: terms of ``on`` (current output frame) and ``FRAMES`` (the scene's length),
#: substituted below. The drift expressions are written against the *scaled*
#: source, so ``iw``/``ih`` are the oversampled dimensions and the visible
#: window is ``iw/zoom`` wide — panning means moving that window across the
#: slack between the two.
KEN_BURNS_MOVES: tuple[tuple[str, str, str], ...] = (
    # Push in, centred: the default "something is coming" move.
    ("in", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"),
    # Pull back while drifting right: reveals context.
    ("out", "(iw-iw/zoom)*(on/FRAMES)", "(ih-ih/zoom)/2"),
    # Push in while rising: lifts the eye up the frame.
    ("in", "(iw-iw/zoom)/2", "(ih-ih/zoom)*(1-on/FRAMES)"),
    # Pull back drifting left: the mirror of the second, so a four-scene Short
    # ends on a move that does not echo the one before it.
    ("out", "(iw-iw/zoom)*(1-on/FRAMES)", "(ih-ih/zoom)/2"),
    # Push in, sinking: settles onto the closing detail.
    ("in", "(iw-iw/zoom)/2", "(ih-ih/zoom)*(on/FRAMES)"),
)


def ken_burns_filter(
    index: int, frames: int, zoom: float, width: int, height: int, fps: int
) -> str:
    """The filtergraph that animates one still into a moving clip.

    Two traps here, both of which cost real time to find:

     * ``crop`` cannot do this. A filter's output dimensions are fixed at
       configuration time, so an expression that shrinks the crop window per
       frame makes libx264 refuse to open the encoder. ``zoompan`` exists
       precisely because it rescales to a constant ``s``.
     * Do NOT combine ``-loop 1 -t`` with ``d=frames``. Looping produces N input
       frames and zoompan then expands EACH of them into N more, so a
       14-second scene renders 400x the work it needs. One input frame in,
       ``d`` frames out.

    The source is pre-scaled to just ``(1 + zoom)`` of the output, so the
    tightest crop is still a 1:1 pixel match and nothing is upscaled, without
    paying to rescale a 4K intermediate every frame.
    """
    direction, x_expr, y_expr = KEN_BURNS_MOVES[index % len(KEN_BURNS_MOVES)]

    source_w = int(width * (1 + zoom)) // 2 * 2
    source_h = int(height * (1 + zoom)) // 2 * 2

    span = max(1, frames - 1)
    if direction == "in":
        zoom_expr = f"1+{zoom:.4f}*on/{span}"
    else:
        zoom_expr = f"1+{zoom:.4f}*(1-on/{span})"

    x_expr = x_expr.replace("FRAMES", str(span))
    y_expr = y_expr.replace("FRAMES", str(span))

    return (
        f"scale={source_w}:{source_h}:flags=lanczos,"
        f"zoompan=z='{zoom_expr}':d={frames}"
        f":x='{x_expr}':y='{y_expr}'"
        f":s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )

#: Where the subtitle band sits, as a fraction of frame height. Shorts put UI
#: over the bottom fifth, so captions ride just above centre.
SUBTITLE_CENTRE = 0.60

#: Fonts to try, in order, before falling back to Pillow's bitmap default.
FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


@dataclass
class WordTiming:
    """One spoken word and exactly when the voice says it.

    Produced by the TTS provider (edge-tts emits WordBoundary events), so these
    are measured, not estimated. ``start``/``end`` are seconds from the top of
    the narration track.
    """

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _token(word: str) -> str:
    """A word reduced to what two sources can be expected to agree on.

    The script says ``"1518,"`` and the voice engine reports ``1518``; casing
    and punctuation differ constantly. Comparing on letters and digits alone is
    what makes the alignment below survive that.
    """
    return "".join(ch for ch in word.lower() if ch.isalnum())


def align_words(script_words: Sequence[str], timings: Sequence[WordTiming]) -> list[int | None]:
    """Map each script word to the index of the timing that speaks it.

    The two sequences are close but never identical: the TTS engine splits
    hyphenated words, expands numerals, and drops anything it reads as
    punctuation. A positional zip would therefore drift — and drift in the
    middle of a narration puts every later caption on the wrong word.

    ``difflib`` handles exactly this shape of problem: it finds the matching
    runs and tolerates the insertions and deletions between them. Script words
    with no counterpart map to ``None`` and are interpolated by the caller.

    :returns: one entry per script word, each a timing index or ``None``.
    """
    left = [_token(w) for w in script_words]
    right = [_token(t.text) for t in timings]
    mapping: list[int | None] = [None] * len(left)

    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping


@dataclass
class Cue:
    """One subtitle card and the window it is on screen for."""

    text: str
    start: float
    end: float
    #: The measured words this card was built from, when there were any. Kept
    #: so the card can highlight whichever one is being spoken right now.
    words: list[WordTiming] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.05, self.end - self.start)


@dataclass
class RenderResult:
    """What came out of the assembler."""

    ok: bool
    path: Path | None
    backend: str
    duration: float = 0.0
    scenes: int = 0
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            return f"render failed ({self.backend}): {self.error}"
        return f"{self.backend} rendered {self.scenes} scene(s), {self.duration:.1f}s"


# ---------------------------------------------------------------------------
# Toolchain discovery
# ---------------------------------------------------------------------------


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def has_moviepy() -> bool:
    try:
        import moviepy  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001 - a broken install is the same as absent
        return False


def available_backends() -> list[str]:
    """Which renderers this machine can actually use, best first."""
    backends: list[str] = []
    if ffmpeg_path():
        backends.append("ffmpeg")
    if has_moviepy():
        backends.append("moviepy")
    backends.append("storyboard")
    return backends


def probe_duration(path: Path | str) -> float:
    """Length of a media file in seconds, or 0.0 when it cannot be read."""
    probe = ffprobe_path()
    target = Path(path)
    if not target.is_file():
        return 0.0

    if probe:
        try:
            output = subprocess.run(
                [
                    probe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            return float(output.stdout.strip())
        except Exception as exc:  # noqa: BLE001
            logger.debug("ffprobe could not read {}: {}", target.name, exc)

    # Pydub can read a duration without ffprobe when it has an audio backend.
    try:
        from pydub import AudioSegment  # noqa: PLC0415

        return len(AudioSegment.from_file(target)) / 1000.0
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """First available bold face, or Pillow's bitmap font."""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:  # noqa: BLE001 - unreadable font, try the next
                continue
    logger.warning("No TrueType font found; subtitles will use the bitmap font")
    return ImageFont.load_default()


def build_cues(
    narration: str,
    total_seconds: float,
    *,
    words_per_cue: int = 3,
    start: float = 0.0,
) -> list[Cue]:
    """Split narration into timed caption cards.

    Timing is proportional to each cue's character count rather than its word
    count, because "a" and "extraordinary" do not take the same time to say.
    Without real word-level timestamps from the TTS provider this is an
    estimate, and it is a much better estimate than an even split.
    """
    words = (narration or "").split()
    if not words or total_seconds <= 0:
        return []

    chunks: list[str] = []
    for index in range(0, len(words), max(1, words_per_cue)):
        chunks.append(" ".join(words[index : index + words_per_cue]))

    weights = [max(1, len(chunk)) for chunk in chunks]
    total_weight = sum(weights)

    cues: list[Cue] = []
    cursor = start
    for chunk, weight in zip(chunks, weights):
        span = total_seconds * (weight / total_weight)
        cues.append(Cue(text=chunk.upper(), start=cursor, end=cursor + span))
        cursor += span
    return cues


def cues_from_timings(
    timings: Sequence[WordTiming],
    *,
    words_per_cue: int = 3,
    max_gap: float = 0.35,
) -> list[Cue]:
    """Group measured word timings into caption cards.

    Each card starts exactly when its first word is spoken and ends when its
    last one finishes, so the caption cannot drift away from the audio however
    long the narration runs.

    A card is held on screen through any gap shorter than ``max_gap`` before the
    next one, which stops captions from strobing between words. Longer gaps are
    real pauses and are left blank.
    """
    if not timings:
        return []

    cues: list[Cue] = []
    step = max(1, words_per_cue)

    for index in range(0, len(timings), step):
        group = timings[index : index + step]
        text = " ".join(word.text for word in group).strip()
        if not text:
            continue
        cues.append(
            Cue(
                text=text.upper(),
                start=group[0].start,
                end=group[-1].end,
                words=list(group),
            )
        )

    # Close short gaps so one card runs into the next.
    for current, following in zip(cues, cues[1:]):
        if 0 < following.start - current.end <= max_gap:
            current.end = following.start

    return cues


def _timestamp(seconds: float, separator: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (WebVTT)."""
    total = max(0.0, seconds)
    hours, remainder = divmod(int(total), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((total - int(total)) * 1000))
    if millis == 1000:  # rounding carried into the next second
        millis, secs = 0, secs + 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def cues_to_srt(cues: Sequence[Cue]) -> str:
    """The cue list as SubRip.

    Written from the same cues that are burned into the picture, so the sidecar
    file and the on-screen captions cannot disagree — which is the whole point
    of having a sidecar to check sync against.
    """
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n"
            f"{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n"
            f"{cue.text}\n"
        )
    return "\n".join(blocks)


def cues_to_vtt(cues: Sequence[Cue]) -> str:
    """The cue list as WebVTT, which is what YouTube and browsers ingest."""
    blocks = ["WEBVTT\n"]
    for cue in cues:
        blocks.append(
            f"{_timestamp(cue.start, '.')} --> {_timestamp(cue.end, '.')}\n"
            f"{cue.text}\n"
        )
    return "\n".join(blocks)


def _wrap_words(
    words: Sequence[str],
    font: Any,
    draw: Any,
    max_width: int,
) -> list[list[int]]:
    """Group word indices into lines that fit ``max_width``.

    Wrapping is done on MEASURED pixel widths rather than a character count,
    and it returns indices rather than strings, because the caller has to draw
    each word separately to colour one of them — which means it needs to know
    exactly which word sits where.
    """
    if not words:
        return []

    def width_of(text: str) -> int:
        return int(draw.textlength(text, font=font))

    space = width_of(" ")
    lines: list[list[int]] = []
    current: list[int] = []
    used = 0

    for index, word in enumerate(words):
        word_width = width_of(word)
        extra = word_width if not current else space + word_width
        if current and used + extra > max_width:
            lines.append(current)
            current, used = [index], word_width
        else:
            current.append(index)
            used += extra

    if current:
        lines.append(current)
    return lines


def render_cue_image(
    text: str,
    size: tuple[int, int],
    *,
    font_size: int | None = None,
    accent: str = "#ffd24a",
    highlight: int | None = None,
) -> Image.Image:
    """One subtitle card as a transparent RGBA image.

    Heavy black stroke plus a drop shadow, which is what keeps captions legible
    over any frame — a plain fill disappears the moment the image behind it is
    bright.

    :param highlight: index of the word to draw in ``accent`` rather than
        white. This is the karaoke effect: the same card is rendered once per
        word, each timed to when that word is actually spoken. Layout is
        computed from the full text every time, so the line never reflows as
        the highlight moves along it.
    """
    width, height = size
    scale = font_size or max(48, int(width * 0.085))
    font = load_font(scale)

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    words = (text or "").split()
    if not words:
        return canvas

    max_width = int(width * 0.82)
    lines = _wrap_words(words, font, draw, max_width)

    line_height = int(scale * 1.22)
    block_height = line_height * len(lines)
    top = int(height * SUBTITLE_CENTRE) - block_height // 2
    stroke = max(4, scale // 11)
    space_width = draw.textlength(" ", font=font)

    for row, indices in enumerate(lines):
        y = top + row * line_height
        widths = [draw.textlength(words[i], font=font) for i in indices]
        total = sum(widths) + space_width * (len(indices) - 1)
        cursor = (width - total) / 2

        for position, word_index in enumerate(indices):
            word = words[word_index]
            active = highlight is not None and word_index == highlight

            # Shadow first, then the stroked fill over it.
            draw.text(
                (cursor + 4, y + 5), word, font=font,
                fill=(0, 0, 0, 170), anchor="la",
            )
            draw.text(
                (cursor, y), word, font=font,
                fill=accent if active else "#ffffff",
                anchor="la", stroke_width=stroke, stroke_fill="#0a0a0a",
            )
            cursor += widths[position] + space_width

    return canvas


# ---------------------------------------------------------------------------
# Scene preparation
# ---------------------------------------------------------------------------


def prepare_scene_image(source: Path, destination: Path, size: tuple[int, int]) -> Path:
    """Cover-crop an image to the vertical frame and grade it cinematically.

    Generated art rarely arrives at 9:16, so it is scaled to cover and centre
    cropped rather than letterboxed — bars look like a mistake on a phone.
    """
    width, height = size
    with Image.open(source) as handle:
        image = handle.convert("RGB")

    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )

    # Free image endpoints cap well below 1080p — Pollinations serves 576x1024
    # whatever is asked of it — so a scene routinely arrives needing close to a
    # 2x upscale. Lanczos alone leaves that visibly soft, and softness is far
    # more obvious once a Ken Burns move is pushing into it. An unsharp pass
    # scaled to how far the image was stretched puts the edges back without
    # crunching art that was already full size.
    if scale > 1.25:
        strength = min(2.0, scale - 1.0)
        resized = resized.filter(
            ImageFilter.UnsharpMask(
                radius=1.6 * strength, percent=int(85 * strength), threshold=3
            )
        )

    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    cropped = resized.crop((left, top, left + width, top + height))

    # Vignette: darken the edges so the eye lands on the centre, and so the
    # subtitles always sit on a slightly darker field.
    vignette = Image.new("L", (width, height), 0)
    ImageDraw.Draw(vignette).ellipse(
        [-width * 0.25, -height * 0.12, width * 1.25, height * 1.12], fill=255
    )
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=width * 0.12))
    darkened = Image.composite(cropped, Image.new("RGB", (width, height), (8, 8, 12)), vignette)

    destination.parent.mkdir(parents=True, exist_ok=True)
    darkened.save(destination, format="PNG")
    return destination


# ---------------------------------------------------------------------------
# The assembler
# ---------------------------------------------------------------------------


class VideoEngine:
    """Turns scenes plus narration into a finished vertical mp4."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ---- public ------------------------------------------------------------
    def render(
        self,
        *,
        scenes: Sequence[dict[str, Any]],
        audio_path: Path | str | None,
        output: Path | str,
        narration: str = "",
        backend: str | None = None,
        word_timings: Sequence[WordTiming] | None = None,
    ) -> RenderResult:
        """Assemble the Short.

        :param scenes: dicts carrying at least ``image_path`` and ``narration``.
        :param audio_path: the narration track; may be ``None``.
        :param output: destination .mp4
        :param backend: force one of ``ffmpeg`` / ``moviepy`` / ``storyboard``.
        :param word_timings: measured word timestamps from the TTS provider.
            When present, scene cuts and captions are placed on the audio itself
            rather than estimated from text length.
        """
        usable = [s for s in scenes if s.get("image_path") and Path(s["image_path"]).is_file()]
        if not usable:
            return RenderResult(
                ok=False, path=None, backend="none", error="no scene images on disk"
            )

        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)

        audio = Path(audio_path) if audio_path and Path(audio_path).is_file() else None
        duration = probe_duration(audio) if audio else 0.0
        if duration <= 0:
            duration = float(self.settings.shorts_target_seconds)
            if audio:
                logger.warning("Could not read the narration length; assuming {}s", duration)

        words = list(word_timings or [])
        timings = self._distribute(usable, duration, words)
        chosen = backend or available_backends()[0]
        logger.info(
            "Rendering {} scene(s) over {:.1f}s with the {} backend ({})",
            len(usable), duration, chosen,
            f"cut on {len(words)} measured word timings" if words
            else "cuts estimated from text length",
        )

        prepared = self._prepare_scenes(usable, target.parent)

        try:
            if chosen == "ffmpeg":
                result = self._render_ffmpeg(
                    prepared, timings, audio, target, narration, duration, words
                )
            elif chosen == "moviepy":
                result = self._render_moviepy(
                    prepared, timings, audio, target, narration, duration, words
                )
            else:
                result = self._render_storyboard(prepared, target, narration)
        except Exception as exc:  # noqa: BLE001 - fall through to the next backend
            logger.error("{} backend failed: {}", chosen, exc)
            remaining = [b for b in available_backends() if b != chosen]
            if remaining:
                logger.info("Falling back to the {} backend", remaining[0])
                return self.render(
                    scenes=scenes,
                    audio_path=audio_path,
                    output=output,
                    narration=narration,
                    backend=remaining[0],
                )
            return RenderResult(ok=False, path=None, backend=chosen, error=str(exc))

        return result

    def thumbnail(self, video_or_image: Path | str, destination: Path | str) -> Path | None:
        """Grab a poster frame, for the dashboard's preview card."""
        source = Path(video_or_image)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)

        if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            try:
                with Image.open(source) as handle:
                    image = handle.convert("RGB")
                image.thumbnail((540, 960), Image.Resampling.LANCZOS)
                image.save(target, format="JPEG", quality=88)
                return target
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not build a thumbnail: {}", exc)
                return None

        binary = ffmpeg_path()
        if not binary or not source.is_file():
            return None
        try:
            subprocess.run(
                [binary, "-y", "-loglevel", "error", "-ss", "1", "-i", str(source),
                 "-frames:v", "1", "-vf", "scale=540:-2", str(target)],
                capture_output=True, timeout=60, check=True,
            )
            return target if target.is_file() else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Thumbnail extraction failed: {}", exc)
            return None

    # ---- internals ----------------------------------------------------------
    def _distribute(
        self,
        scenes: Sequence[dict[str, Any]],
        total: float,
        words: Sequence[WordTiming] = (),
    ) -> list[float]:
        """How long each scene stays on screen, summing to ``total``.

        With measured word timings the cut lands on the audio: a scene holds
        until the first word of the next scene is spoken. Without them, fall
        back to splitting by how much each scene has to say.
        """
        if words:
            measured = self._measured_spans(scenes, total, words)
            if measured:
                return measured

        weights = [max(1, len((scene.get("narration") or "").strip())) for scene in scenes]
        pool = sum(weights)
        # Never let a scene fall under 2s: below that the Ken Burns move reads
        # as a glitch rather than a movement.
        raw = [max(2.0, total * (w / pool)) for w in weights]
        scale = total / sum(raw)
        return [value * scale for value in raw]

    def _scene_word_windows(
        self, scenes: Sequence[dict[str, Any]], words: Sequence[WordTiming]
    ) -> list[list[WordTiming]]:
        """Which measured words belong to which scene.

        The timings arrive as one flat stream for the whole narration, so the
        scenes have to be located inside it. Their narration text concatenated
        IS that stream, which makes this a sequence-alignment problem rather
        than a guess — see :func:`align_words`.
        """
        script_words: list[str] = []
        owners: list[int] = []
        for index, scene in enumerate(scenes):
            for word in (scene.get("narration") or "").split():
                script_words.append(word)
                owners.append(index)

        if not script_words:
            return [[] for _ in scenes]

        mapping = align_words(script_words, words)
        buckets: list[list[WordTiming]] = [[] for _ in scenes]
        for position, timing_index in enumerate(mapping):
            if timing_index is not None:
                buckets[owners[position]].append(words[timing_index])
        return buckets

    def _measured_spans(
        self,
        scenes: Sequence[dict[str, Any]],
        total: float,
        words: Sequence[WordTiming],
    ) -> list[float] | None:
        """Scene lengths taken from the audio, or None if alignment was too poor.

        A scene holds until the first word of the next scene is spoken, and the
        last scene runs to the end of the track, so the picture covers the audio
        exactly rather than stopping on the final syllable.
        """
        buckets = self._scene_word_windows(scenes, words)

        matched = sum(len(bucket) for bucket in buckets)
        if matched < max(1, len(words) // 2):
            # Fewer than half the words placed: the script and the audio have
            # diverged (a rewritten narration, a different language). An
            # estimate beats a confidently wrong cut.
            logger.warning(
                "Only {}/{} words aligned to scenes; estimating scene lengths instead",
                matched, len(words),
            )
            return None

        count = len(scenes)

        # Where the audio says each cut belongs. cuts[i] ends scene i, and the
        # final entry is the end of the track.
        cuts: list[float] = []
        for index in range(count - 1):
            following = next((bucket for bucket in buckets[index + 1 :] if bucket), None)
            if following:
                cut = following[0].start
            elif buckets[index]:
                cut = buckets[index][-1].end
            else:
                cut = total * (index + 1) / count
            cuts.append(min(max(cut, 0.0), total))
        cuts.append(total)

        # Monotonic: a scene with no aligned words must not cut backwards.
        for index in range(1, count):
            cuts[index] = max(cuts[index], cuts[index - 1])

        # Hold each scene long enough for the Ken Burns move to read — but only
        # where the track is long enough to afford it. Rescaling to satisfy an
        # impossible minimum would slide every cut off the audio, which is the
        # exact failure this method exists to prevent, so when the narration is
        # too fast for the minimum the measured cuts win and the minimum goes.
        if MIN_SCENE_SECONDS * count <= total:
            previous = 0.0
            for index in range(count - 1):
                cuts[index] = max(cuts[index], previous + MIN_SCENE_SECONDS)
                previous = cuts[index]
            # Forward pass can push cuts past the end; pull them back from the
            # tail, which is always feasible given the check above.
            for index in range(count - 2, -1, -1):
                cuts[index] = min(cuts[index], cuts[index + 1] - MIN_SCENE_SECONDS)

        spans: list[float] = []
        cursor = 0.0
        for cut in cuts:
            spans.append(max(0.0, cut - cursor))
            cursor = cut
        return spans

    def _prepare_scenes(
        self, scenes: Sequence[dict[str, Any]], workdir: Path
    ) -> list[dict[str, Any]]:
        """Cover-crop and grade every scene image into the frame size."""
        prepared: list[dict[str, Any]] = []
        frames = workdir / "frames"
        frames.mkdir(parents=True, exist_ok=True)

        for index, scene in enumerate(scenes):
            source = Path(scene["image_path"])
            destination = frames / f"scene{index:02d}.png"
            try:
                prepare_scene_image(source, destination, self.settings.video_size)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not prepare {}: {}", source.name, exc)
                continue
            prepared.append({**scene, "frame_path": destination})
        return prepared

    # ---- ffmpeg -------------------------------------------------------------
    def _render_ffmpeg(
        self,
        scenes: Sequence[dict[str, Any]],
        timings: Sequence[float],
        audio: Path | None,
        output: Path,
        narration: str,
        duration: float,
        words: Sequence[WordTiming] = (),
    ) -> RenderResult:
        """Drive the ffmpeg CLI directly.

        Each scene is encoded on its own with a zoompan move, the clips are
        concatenated, then subtitles and audio go on in a final pass. Several
        small invocations beat one enormous filtergraph: it is far easier to see
        which scene broke, and it sidesteps ffmpeg's filter-count limits.
        """
        binary = ffmpeg_path()
        if not binary:
            raise RuntimeError("ffmpeg is not on PATH")

        width, height = self.settings.video_size
        fps = self.settings.video_fps
        workdir = output.parent / "work"
        workdir.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []

        # Round on the CUMULATIVE timeline rather than per scene. Rounding each
        # scene independently loses up to half a frame every cut, and those
        # errors add: eight scenes can put the picture four frames away from the
        # audio by the end. Taking the difference between rounded boundaries
        # makes the frame counts sum to exactly round(duration * fps).
        span_frames: list[int] = []
        cursor = 0.0
        emitted = 0
        for seconds in timings:
            cursor += seconds
            boundary = int(round(cursor * fps))
            span_frames.append(max(2, boundary - emitted))
            emitted += span_frames[-1]

        # A cross-dissolve CONSUMES footage: xfade overlaps its two inputs, so
        # the chain ends up (n-1) * transition shorter than the sum of its
        # parts. Every clip but the last is therefore rendered one transition
        # longer, which puts the total back exactly where the audio needs it and
        # leaves every scene boundary — and so every caption — untouched.
        transition_frames = self._transition_frames(len(scenes), span_frames, fps)
        frame_plan = [
            count + (transition_frames if index < len(span_frames) - 1 else 0)
            for index, count in enumerate(span_frames)
        ]

        for index, (scene, frames) in enumerate(zip(scenes, frame_plan)):
            clip = workdir / f"clip{index:02d}.mp4"
            source = scene.get("video_path") or scene["frame_path"]

            if scene.get("video_path"):
                # Stock footage already moves; a Ken Burns push on top of a
                # camera move reads as a mistake. Fit it to the frame and let
                # the original motion carry the scene.
                filtergraph = (
                    f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},fps={fps},format=yuv420p"
                )
                command = [
                    binary, "-y", "-loglevel", "error",
                    "-stream_loop", "-1", "-i", str(source),
                    "-an", "-vf", filtergraph, "-frames:v", str(frames),
                ]
            else:
                filtergraph = ken_burns_filter(
                    index, frames, self.settings.video_zoom, width, height, fps
                )
                command = [
                    binary, "-y", "-loglevel", "error",
                    "-i", str(source),
                    "-vf", filtergraph, "-frames:v", str(frames),
                ]

            self._run(
                command + [
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", str(fps), str(clip),
                ],
                f"scene {index}",
            )
            clips.append(clip)

        silent = workdir / "silent.mp4"
        if transition_frames and len(clips) > 1:
            self._join_with_dissolves(
                binary, clips, span_frames, transition_frames, fps, silent
            )
        else:
            # Hard cuts: the concat demuxer needs no re-encode at all.
            listing = workdir / "clips.txt"
            listing.write_text(
                "".join(f"file '{clip.as_posix()}'\n" for clip in clips),
                encoding="utf-8",
            )
            self._run(
                [binary, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", str(listing), "-c", "copy", str(silent)],
                "concat",
            )

        # Subtitles as timed PNG overlays, then the narration underneath.
        cues = self._scene_cues(scenes, timings, words)
        overlays = self._write_cue_images(cues, workdir)

        command = [binary, "-y", "-loglevel", "error", "-i", str(silent)]
        for overlay in overlays:
            command += ["-i", str(overlay["path"])]
        if audio:
            command += ["-i", str(audio)]

        filters: list[str] = []
        if overlays:
            source = "[0:v]"
            for index, overlay in enumerate(overlays):
                label = f"[v{index}]"
                filters.append(
                    f"{source}[{index + 1}:v]overlay=0:0:"
                    f"enable='between(t,{overlay['start']:.3f},{overlay['end']:.3f})'{label}"
                )
                source = label
            command += ["-map", source]
        else:
            command += ["-map", "0:v"]

        if audio:
            # Three separate things keep the voice locked to the picture:
            #
            #  * `aresample=async=1:first_pts=0` pins the track to t=0. Without
            #    it a container whose first packet carries a non-zero PTS shifts
            #    the whole narration by that offset.
            #  * `apad` extends the audio with silence so `-shortest` always
            #    trims the pad and never the final syllable — the video is a
            #    whole number of frames and can land a few ms short of the mp3.
            #  * `-fps_mode cfr` forbids ffmpeg from dropping or duplicating
            #    frames to fit, which is drift by another name.
            audio_index = len(overlays) + 1
            filters.append(f"[{audio_index}:a]aresample=async=1:first_pts=0,apad[a]")
            command += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-shortest"]

        if filters:
            command += ["-filter_complex", ";".join(filters)]

        command += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(fps),
            "-movflags", "+faststart", str(output),
        ]
        self._run(command, "mux")

        shutil.rmtree(workdir, ignore_errors=True)
        final = probe_duration(output) or duration
        return RenderResult(
            ok=output.is_file(),
            path=output,
            backend="ffmpeg",
            duration=final,
            scenes=len(scenes),
            notes=[f"{len(cues)} subtitle cue(s)"],
        )

    def _transition_frames(
        self, scene_count: int, span_frames: Sequence[int], fps: int
    ) -> int:
        """Dissolve length in frames, or 0 to cut hard.

        A dissolve cannot be longer than the scenes it joins, or xfade runs out
        of footage mid-blend. The shortest scene sets the ceiling, at a third of
        its length so there is still a held frame either side of the blend.
        """
        wanted = int(round(self.settings.video_transition_seconds * fps))
        if wanted <= 0 or scene_count < 2:
            return 0

        ceiling = max(0, min(span_frames) // 3)
        allowed = min(wanted, ceiling)
        if allowed < 2:
            # Below two frames a dissolve is indistinguishable from a cut and
            # only costs a re-encode.
            return 0
        if allowed < wanted:
            logger.debug(
                "Shortening the dissolve to {} frames; the shortest scene is {}",
                allowed, min(span_frames),
            )
        return allowed

    def _join_with_dissolves(
        self,
        binary: str,
        clips: Sequence[Path],
        span_frames: Sequence[int],
        transition_frames: int,
        fps: int,
        output: Path,
    ) -> None:
        """Chain the clips together with cross-dissolves.

        xfade takes an ``offset``: the point in the accumulated timeline where
        the blend starts. Because each clip was rendered one transition longer
        than its span, the running sum of the SPANS is exactly where each
        boundary falls in the finished video — so the dissolve begins on the
        scene change the audio dictated, and the result is exactly as long as
        the sum of the spans.
        """
        transition = transition_frames / fps

        command: list[str] = [binary, "-y", "-loglevel", "error"]
        for clip in clips:
            command += ["-i", str(clip)]

        steps: list[str] = []
        source = "[0:v]"
        offset = 0.0
        for index in range(1, len(clips)):
            offset += span_frames[index - 1] / fps
            label = f"[x{index}]"
            steps.append(
                f"{source}[{index}:v]xfade=transition=fade"
                f":duration={transition:.6f}:offset={offset:.6f}{label}"
            )
            source = label

        command += [
            "-filter_complex", ";".join(steps),
            "-map", source,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-fps_mode", "cfr",
            str(output),
        ]
        self._run(command, "dissolve")

    def _run(self, command: list[str], label: str) -> None:
        """Run a subprocess and surface ffmpeg's own error text on failure."""
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-4:]
            raise RuntimeError(f"{label}: ffmpeg exited {result.returncode}: {' | '.join(tail)}")

    def _scene_cues(
        self,
        scenes: Sequence[dict[str, Any]],
        timings: Sequence[float],
        words: Sequence[WordTiming] = (),
    ) -> list[Cue]:
        """Build the full cue list.

        With measured timings every card sits on the word it transcribes, so
        the captions cannot walk away from the voice. Otherwise each scene's
        text is spread across its own window as an estimate.
        """
        if words:
            return cues_from_timings(
                words, words_per_cue=self.settings.subtitle_words_per_cue
            )

        cues: list[Cue] = []
        cursor = 0.0
        for scene, seconds in zip(scenes, timings):
            cues.extend(
                build_cues(
                    scene.get("narration", ""),
                    seconds,
                    words_per_cue=self.settings.subtitle_words_per_cue,
                    start=cursor,
                )
            )
            cursor += seconds
        return cues

    def _write_cue_images(self, cues: Sequence[Cue], workdir: Path) -> list[dict[str, Any]]:
        """Render the caption overlays and the window each is shown for.

        A cue with measured words becomes one image PER WORD — the same card
        each time, with a different word in the accent colour — so the
        highlight tracks the voice. Without measured words there is nothing to
        track, and the cue is drawn once with no highlight at all rather than
        guessing which word is being said.
        """
        overlays: list[dict[str, Any]] = []
        directory = workdir / "cues"
        directory.mkdir(parents=True, exist_ok=True)
        size = self.settings.video_size

        for index, cue in enumerate(cues):
            if not cue.words:
                path = directory / f"cue{index:03d}.png"
                render_cue_image(cue.text, size).save(path, format="PNG")
                overlays.append({"path": path, "start": cue.start, "end": cue.end})
                continue

            for position, word in enumerate(cue.words):
                path = directory / f"cue{index:03d}w{position:02d}.png"
                render_cue_image(cue.text, size, highlight=position).save(path, format="PNG")

                # Hold each highlight until the next word starts, so the card
                # never blinks out during the gap between two words. The last
                # one runs to the end of the cue, which already absorbs any
                # short pause before the next card.
                is_last = position == len(cue.words) - 1
                end = cue.end if is_last else cue.words[position + 1].start
                overlays.append(
                    {"path": path, "start": word.start, "end": max(end, word.end)}
                )

        return overlays

    # ---- moviepy ------------------------------------------------------------
    def _render_moviepy(
        self,
        scenes: Sequence[dict[str, Any]],
        timings: Sequence[float],
        audio: Path | None,
        output: Path,
        narration: str,
        duration: float,
        words: Sequence[WordTiming] = (),
    ) -> RenderResult:
        """MoviePy path, using Pillow-rendered captions rather than TextClip."""
        from moviepy import (  # noqa: PLC0415
            AudioFileClip,
            CompositeVideoClip,
            ImageClip,
            concatenate_videoclips,
        )

        width, height = self.settings.video_size
        zoom = self.settings.video_zoom
        clips = []

        for index, (scene, seconds) in enumerate(zip(scenes, timings)):
            base = ImageClip(str(scene["frame_path"])).with_duration(seconds)
            zoom_in = index % 2 == 0

            def scaler(t, seconds=seconds, zoom_in=zoom_in):
                progress = min(1.0, max(0.0, t / max(0.01, seconds)))
                return 1 + zoom * (progress if zoom_in else 1 - progress)

            moved = base.resized(scaler).with_position(("center", "center"))
            clips.append(CompositeVideoClip([moved], size=(width, height)).with_duration(seconds))

        video = concatenate_videoclips(clips, method="compose")

        cues = self._scene_cues(scenes, timings, words)
        workdir = output.parent / "work"
        overlays = []
        for overlay in self._write_cue_images(cues, workdir):
            overlays.append(
                ImageClip(str(overlay["path"]))
                .with_start(overlay["start"])
                .with_duration(max(0.05, overlay["end"] - overlay["start"]))
                .with_position((0, 0))
            )

        composite = CompositeVideoClip([video, *overlays], size=(width, height))
        if audio:
            track = AudioFileClip(str(audio))
            composite = composite.with_audio(track).with_duration(
                min(composite.duration, track.duration)
            )

        composite.write_videofile(
            str(output),
            fps=self.settings.video_fps,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            logger=None,
        )
        composite.close()
        shutil.rmtree(workdir, ignore_errors=True)

        return RenderResult(
            ok=output.is_file(),
            path=output,
            backend="moviepy",
            duration=probe_duration(output) or duration,
            scenes=len(scenes),
            notes=[f"{len(cues)} subtitle cue(s)"],
        )

    # ---- storyboard ---------------------------------------------------------
    def _render_storyboard(
        self, scenes: Sequence[dict[str, Any]], output: Path, narration: str
    ) -> RenderResult:
        """No video toolchain: emit a labelled contact sheet instead.

        Honest degradation. The file is a PNG beside the intended .mp4 path and
        says STORYBOARD across the top, so nobody mistakes it for a render.
        """
        width, height = self.settings.video_size
        thumb_w = width // 2
        thumb_h = height // 2
        sheet = Image.new("RGB", (thumb_w * len(scenes), thumb_h + 140), (14, 12, 18))
        draw = ImageDraw.Draw(sheet)
        font = load_font(34)
        small = load_font(24)

        for index, scene in enumerate(scenes):
            with Image.open(scene["frame_path"]) as handle:
                frame = handle.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            sheet.paste(frame, (index * thumb_w, 90))

            caption = textwrap.fill((scene.get("narration") or "")[:180], width=38)
            draw.text((index * thumb_w + 16, thumb_h + 100), caption, font=small, fill="#e8dcc0")
            draw.text((index * thumb_w + 16, 54), f"SCENE {index + 1}", font=small, fill="#ffd24a")

        draw.text((16, 12), "STORYBOARD — no video toolchain available", font=font, fill="#ff8a5c")

        target = output.with_suffix(".storyboard.png")
        sheet.save(target, format="PNG")
        logger.warning("Wrote a storyboard instead of a video: {}", target.name)

        return RenderResult(
            ok=True,
            path=target,
            backend="storyboard",
            duration=0.0,
            scenes=len(scenes),
            notes=["install ffmpeg or moviepy to render real video"],
        )


def describe_toolchain() -> dict[str, Any]:
    """What this machine can do, for the CLI banner and /api/health."""
    return {
        "ffmpeg": ffmpeg_path(),
        "ffprobe": ffprobe_path(),
        "moviepy": has_moviepy(),
        "backends": available_backends(),
        "preferred": available_backends()[0],
    }


if __name__ == "__main__":  # pragma: no cover - manual probe
    print(json.dumps(describe_toolchain(), indent=2))
