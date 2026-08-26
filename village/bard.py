"""The Bard: writes, voices, illustrates and cuts a faceless vertical Short.

Four steps, each of which degrades rather than fails:

1. **Script** — an OpenRouter turn returning strict JSON: a three-second hook,
   N scenes each with its own narration and image prompt, and an open loop that
   refuses to resolve. Falls back to a curated script with no key.
2. **Voice** — edge-tts (free, no key), OpenAI TTS, or ElevenLabs. With none
   available it synthesises a silent track of the right length so the video
   still cuts to the correct duration.
3. **Scenes** — the Crafter renders each image prompt at 9:16.
4. **Assembly** — :mod:`core.video_engine` applies Ken Burns, burns in the
   captions and muxes the narration.

The Bard writes about the dead, the unexplained and the strange. That is a
category with a specific hazard: it is very easy to state a rumour as a fact.
The script prompt makes the distinction explicit and the output is screened for
the usual IP traps before a single image is generated.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import struct
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
from openai import AsyncOpenAI

from config.settings import Settings, get_settings
from core import events
from core.database import (
    ShortStatus,
    create_short,
    get_short,
    update_short,
    update_short_status,
)
from core.trademark_guard import screen_many
from core.scene_art import SceneArtist
from core.stock_video import StockVideoLibrary
from core.video_engine import (
    VideoEngine,
    WordTiming,
    cues_from_timings,
    cues_to_srt,
    cues_to_vtt,
    describe_toolchain,
    ffprobe_path,
    probe_duration,
)
from village.crafter import Crafter

#: The three formats that carry a faceless channel.
CATEGORIES: tuple[str, ...] = ("Dark History", "Unsolved Mysteries", "Mind-Blowing Facts")

#: How many times to ask the TTS provider before giving up. edge-tts streams
#: over a websocket to Microsoft, where a mid-stream drop is ordinary; one
#: failed attempt is not a reason to publish a silent video.
TTS_ATTEMPTS = 3

#: Seconds to wait after a failed attempt, multiplied by the attempt number.
TTS_RETRY_BACKOFF = 1.5

#: Below this, a narration file is a truncated stream rather than a read.
MIN_NARRATION_SECONDS = 1.0

#: A Short with fewer beats than this is a still image with a voice over it.
MIN_SCENES = 4

#: Words per second of finished narration.
#:
#: Measured, not assumed: en-US-ChristopherNeural at TTS_RATE=-8% reads
#: 129 words in 57.7s and 186 words in 83.4s — 2.23 either way. The 2.5 this
#: used to assume is a 12% underestimate, which is the difference between a
#: 53-second Short and one YouTube reclassifies as a regular video.
WORDS_PER_SECOND = 2.23

#: YouTube only treats a video under 60 seconds as a Short. Budget against 55
#: so an over-long scene still lands inside the limit.
SHORTS_HARD_LIMIT_SECONDS = 55

#: The hook and the conclusion are capped in the prompt; the scenes divide up
#: whatever the budget has left after them.
HOOK_WORDS = 14
CONCLUSION_WORDS = 20

#: Never ask for scenes so short they cannot carry a sentence.
MIN_SCENE_WORDS = 15

SCRIPT_PROMPT = """You write faceless vertical short-form video. Your channel covers \
dark history, unsolved mysteries and mind-blowing facts. You have three seconds \
to stop a thumb, and you never waste them.

Return ONE JSON object and nothing else:

{
  "title": "under 60 characters, the YouTube title",
  "category": "Dark History | Unsolved Mysteries | Mind-Blowing Facts",
  "topic": "the specific subject in 3-8 words",
  "hook": "the first line. Under 14 words. Spoken in about 3 seconds.",
  "scenes": [
    {
      "narration": "about {scene_words} words of spoken narration for this scene",
      "image_prompt": "a complete image-generation prompt, see the rules below",
      "on_screen": "3-6 words for the punchy caption emphasis",
      "stock_query": "2-4 plain words to search stock footage, see the rules below"
    }
  ],
  "conclusion": "the final line. An OPEN LOOP: a question or an unresolved detail \
that makes the viewer sit through it again. Under 20 words.",
  "description": "2-3 sentences for the video description",
  "hashtags": ["5 to 8 lowercase hashtags without the # symbol"]
}

HOOK RULES — this is the whole video's job in one line:
- Open on the strangest concrete detail, never on context. Name the year, the \
place and the thing that happened in the first breath; do not warm up to it. \
"Today we're looking at a strange event in history" is the failure mode.
- No greetings. No "did you know". No "in this video". No channel branding.
- Present tense where you can. It puts the viewer inside the moment.

SCENE RULES:
- Write EXACTLY {scene_count} scenes.
- Each scene is about {scene_words} words. Count them.
- HARD LIMIT: the hook, every scene and the conclusion together must not exceed \
{max_words} words. That is not a style note — over it, the video runs past 60 \
seconds and YouTube stops treating it as a Short at all. Count the total \
before you answer, and cut the weakest sentence if you are over.
- Target about {target_seconds} seconds, which is roughly {word_budget} words.
- Each scene advances the story. Never restate the previous scene.
- Escalate: scene 1 sets the scene, the middle turns it, the last one lands the \
detail that makes the conclusion land.

IMAGE PROMPT RULES — each is rendered as a VERTICAL 9:16 cinematic still:
- Write a FILM FRAME, not an illustration. Name the subject, the framing (wide \
establishing / medium / extreme close-up), the light source and its direction, \
the era, the weather, the texture, and what the camera is doing.
- Every scene must look DIFFERENT from the others: alternate wide and close, \
interior and exterior, high angle and low. Four near-identical frames is the \
single fastest way to lose a viewer.
- Lean dark and atmospheric. Volumetric shafts of light, deep shadow, haze, \
rain, dust, smoke, candle or lantern or moonlight. Desaturated with ONE warm \
accent — a lamp, a fire, a window.
- Concrete nouns only. "A rusted iron door half open onto a flooded stairwell, \
lit by a single failing bulb" beats "a mysterious scary place".
- No text, no words, no captions, no watermarks, no modern logos in the image.
- No recognisable living people. No real brand marks. Historical figures long \
dead are fine; describe them, do not name a likeness to copy.
- Never depict gore, corpses in detail, or real violence against identifiable \
people. Atmosphere carries this genre; explicit imagery gets a channel struck.
- Do NOT write style boilerplate — no "cinematic", "8k", "photorealistic", \
"Unreal Engine", no camera-brand names. That is appended automatically and \
repeating it wastes the prompt and confuses the model.

STOCK QUERY RULES — used to look up B-roll that may replace the still:
- Plain searchable nouns for the SETTING or TEXTURE, not the story. "stormy sea waves", "abandoned lighthouse", "old paper documents", "fog forest".
- Never a proper noun, a date, or a person. Stock libraries do not have footage of your specific event, and a query that names one returns nothing.
- It must still make sense under the narration if the still is swapped for it.

TRUTH RULES — non-negotiable for this genre:
- State facts as facts and theories as theories. If something is disputed, say \
"historians still argue", "one theory holds", "no one has ever confirmed".
- Never invent a statistic, a date, a name, or a quote. If you are not certain \
of a number, describe it qualitatively instead.
- The open loop must come from genuine unresolved ambiguity, not from a \
manufactured cliffhanger.

Output raw JSON. No markdown, no code fences, no commentary."""

#: Angles the Bard picks from when no topic is given.
#:
#: These are ANGLES, not scripts: each one is broad enough that the model still
#: chooses the specific story, and narrow enough that it cannot fall back on
#: the same famous anecdote every time. Asking an unprompted model for "a
#: strange historical event" reliably returns the two or three best-known ones.
TOPIC_SEEDS: tuple[str, ...] = (
    # --- unsolved historical mysteries ---
    "a disappearance at sea where the vessel was found intact",
    "a body found somewhere it could not possibly have reached",
    "a message or signal nobody has ever decoded",
    "an entire settlement abandoned with meals still on the table",
    "a historical figure whose death has never been satisfactorily explained",
    "an artefact found in the wrong century's layer of ground",
    "a building whose original purpose is still argued over",
    "a person who arrived somewhere with no verifiable past",
    "a photograph or recording nobody can account for",
    # --- dark psychology / bizarre human behaviour ---
    "a mass delusion that swept through a town",
    "an experiment that would never be approved today",
    "a crowd that did something no individual in it would have done",
    "a con that worked because the truth was less believable",
    "a group that kept believing after the prophecy failed",
    "a memory millions of people share that never happened",
    "a job that quietly destroys the people who do it",
    "a decision people make against their own interests, every time",
    "an obedience or conformity finding that still unsettles researchers",
    "a hoax the perpetrator could never stop",
    # --- cosmic / mind-blowing science ---
    "a place in the universe where physics stops behaving",
    "a number so large it breaks the intuition it was meant to build",
    "something enormous that is invisible from where we stand",
    "a signal from space that has still not been explained",
    "an object in the solar system that should not be there",
    "a scale comparison that makes the everyday world feel wrong",
    "a substance with a property that sounds invented",
    "a form of life surviving somewhere nothing should",
    "a limit of what can ever be observed, and why",
    "a thing that happens constantly and nobody can feel",
    # --- dark history ---
    "an industrial accident nobody was held responsible for",
    "a medical practice that was standard and is now horrifying",
    "a law that produced the exact opposite of its intent",
    "a war deception so strange it sounds fictional",
    "a famine or shortage caused by a decision rather than a harvest",
    "an execution or punishment that went wrong in public",
    "a survivor whose account contradicts the official record",
    "a place that was quietly evacuated and never reopened",
)

#: How many recent titles to keep the Bard away from.
RECENT_TOPIC_MEMORY = 12

#: Used when OpenRouter is unavailable. Real scripts, so the rest of the
#: pipeline is exercised against realistic shapes and durations.
FALLBACK_SCRIPTS: tuple[dict[str, Any], ...] = (
    {
        "title": "The Town That Danced Itself To Death",
        "category": "Dark History",
        "topic": "the 1518 dancing plague of Strasbourg",
        "hook": "In July 1518, a woman steps into a Strasbourg street and starts to dance.",
        "scenes": [
            {
                "narration": "She does not stop. Not that night, not the next day. Within a "
                "month the number dancing with her is closer to four hundred.",
                "image_prompt": "A lone woman dancing barefoot in a narrow medieval "
                "European street at dusk, timber-framed houses leaning overhead, "
                "onlookers half-lit in doorways",
                "on_screen": "SHE NEVER STOPPED",
            },
            {
                "narration": "The authorities make a decision that still seems insane. They rule "
                "the afflicted must dance it out, and clear the guildhalls for "
                "them.",
                "image_prompt": "A crowded medieval guildhall interior with a wooden "
                "stage, musicians playing, dozens of exhausted figures moving in "
                "candlelight, dust in the air",
                "on_screen": "THEY HIRED MUSICIANS",
            },
            {
                "narration": "People collapse. Accounts describe deaths from exhaustion. "
                "Historians still argue the cause: ergot poisoning, mass hysteria, "
                "or that specific desperate summer.",
                "image_prompt": "Empty medieval town square at dawn after a crowd has "
                "gone, scattered shoes and torn cloth on wet cobblestones, long shadows",
                "on_screen": "NO ONE AGREES WHY",
            },
        {
                    "narration": "The city even hired musicians to play alongside them, believing "
                "the dancing had to be driven out rather than stopped.",
                    "image_prompt": (
                        "Interior of a candlelit medieval guildhall at night, empty wooden floor scuffed raw in a wide circle, abandoned fiddle and drum on a bench, single shaft of moonlight through a high window, dust hanging in the air, low angle"
                    ),
                    "on_screen": "THEY HIRED MUSICIANS",
                },
            ],
        "conclusion": "The dancing stopped as suddenly as it began. Nobody has ever "
        "explained why it started.",
        "description": "In 1518, hundreds of people in Strasbourg danced for weeks and "
        "no one has ever agreed on the cause. Ergot poisoning, mass hysteria, or "
        "something else entirely.",
        "hashtags": ["darkhistory", "history", "strasbourg", "1518", "unexplained", "historyfacts"],
    },
    {
        "title": "The Signal That Lasted 72 Seconds",
        "category": "Unsolved Mysteries",
        "topic": "the 1977 Wow! signal",
        "hook": "For seventy-two seconds in 1977, a radio telescope heard something.",
        "scenes": [
            {
                "narration": "Ohio State's Big Ear swept the sky on a schedule, printing its "
                "readings as columns of numbers. Most are noise. On August "
                "fifteenth, one is not.",
                "image_prompt": "A vast 1970s radio telescope array at night under a "
                "dense star field, control building lights glowing faintly below",
                "on_screen": "SEVENTY-TWO SECONDS",
            },
            {
                "narration": "The signal rises far above the background, holds, and fades "
                "exactly as a fixed point in space would as the Earth turns beneath "
                "it.",
                "image_prompt": "Close overhead view of a continuous-feed computer printout "
                "on a desk in a dim 1970s office, a single column circled in red pen, "
                "a coffee cup at the edge of frame",
                "on_screen": "ONE WORD: WOW",
            },
            {
                "narration": "A volunteer astronomer circles it on the printout and writes one "
                "word in the margin. Wow. Nothing like it has ever been heard "
                "again.",
                "image_prompt": "A single small telescope silhouette against an enormous "
                "empty night sky, a dark patch of space at the top of frame",
                "on_screen": "IT NEVER REPEATED",
            },
        {
                    "narration": "Instruments have pointed at that patch of sky for decades. Comets "
                "were proposed and dismissed. One clean reading, then silence.",
                    "image_prompt": (
                        "Extreme close-up of a continuous-feed printout, a single column of handwritten characters circled in red ink, harsh desk lamp raking across the paper grain, everything beyond the page in darkness"
                    ),
                    "on_screen": "NEVER AGAIN",
                },
            ],
        "conclusion": "We have listened to that exact spot for over forty years. It has "
        "never spoken again.",
        "description": "In 1977 the Big Ear radio telescope recorded a 72-second narrowband "
        "signal that has never been detected again. No explanation has ever stuck.",
        "hashtags": ["unsolved", "space", "wowsignal", "seti", "mystery", "unexplained"],
    },
    {
        "title": "There Is A Cloud That Weighs A Million Pounds",
        "category": "Mind-Blowing Facts",
        "topic": "the real mass of a cumulus cloud",
        "hook": "The white cloud drifting over you right now weighs about a million pounds.",
        "scenes": [
            {
                "narration": "A typical cumulus cloud is roughly a kilometre on each side, and "
                "the water in it is spread as impossibly small droplets.",
                "image_prompt": "A single towering white cumulus cloud filling a deep blue "
                "summer sky, seen from below",
                "on_screen": "ONE MILLION POUNDS",
            },
            {
                "narration": "Multiply that tiny mass by the number of droplets and the figure "
                "stops being intuitive. The cloud holds hundreds of tonnes of "
                "water.",
                "image_prompt": "Extreme close macro view of suspended water droplets "
                "catching backlight, dark background, shallow depth of field",
                "on_screen": "IT STILL FLOATS",
            },
            {
                "narration": "It floats anyway. The air beneath is warmer and rising, and it "
                "pushes up harder than those droplets fall.",
                "image_prompt": "Heavy rain beginning to fall from a dark cloud base over "
                "an empty field, first drops striking dry ground",
                "on_screen": "THEN IT FALLS",
            },
        {
                    "narration": "Gravity never stops pulling on them. It is simply losing to the "
                "column of rising air underneath. Rain is that argument finally "
                "being lost.",
                    "image_prompt": (
                        "Extreme close-up of sunlit water droplets suspended in mist against a black background, each one catching a pinpoint of warm light, shallow depth of field, everything behind falling into darkness"
                    ),
                    "on_screen": "IT IS FALLING",
                },
            ],
        "conclusion": "Every clear day, a few hundred tonnes of water is hanging directly "
        "over your head.",
        "description": "A cumulus cloud holds hundreds of tonnes of water and floats "
        "anyway. Here is why it stays up, and what changes when it comes down.",
        "hashtags": ["facts", "science", "weather", "mindblowing", "didyouknow", "clouds"],
    },
)


@dataclass
class Scene:
    """One beat of the Short."""

    narration: str
    image_prompt: str
    on_screen: str = ""
    stock_query: str = ""
    seconds: float = 0.0
    image_path: str | None = None
    #: A stock B-roll clip standing in for the still, when one was fetched.
    video_path: str | None = None
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "narration": self.narration,
            "image_prompt": self.image_prompt,
            "on_screen": self.on_screen,
            "stock_query": self.stock_query,
            "seconds": self.seconds,
            "image_path": self.image_path,
            "video_path": self.video_path,
            "simulated": self.simulated,
        }


@dataclass
class ShortScript:
    """A validated script, ready to voice and illustrate."""

    title: str
    category: str
    topic: str
    hook: str
    scenes: list[Scene]
    conclusion: str
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    source: str = "openrouter"

    @property
    def narration(self) -> str:
        """The full read, in order: hook, every scene, then the open loop."""
        parts = [self.hook, *(scene.narration for scene in self.scenes), self.conclusion]
        return " ".join(part.strip() for part in parts if part and part.strip())

    @property
    def word_count(self) -> int:
        return len(self.narration.split())

    @property
    def estimated_seconds(self) -> float:
        """Projected runtime at the measured narration rate."""
        return self.word_count / WORDS_PER_SECOND

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "topic": self.topic,
            "hook": self.hook,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "conclusion": self.conclusion,
            "description": self.description,
            "hashtags": list(self.hashtags),
            "source": self.source,
            "words": self.word_count,
            "estimated_seconds": round(self.estimated_seconds, 1),
        }


@dataclass
class VoiceTrack:
    """A rendered narration and everything the assembler needs to cut to it.

    ``words`` is empty for providers that do not report boundaries; the
    assembler falls back to estimating from text length in that case.
    """

    path: Path | None
    backend: str
    words: list[WordTiming] = field(default_factory=list)
    subtitles: Path | None = None

    @property
    def timed(self) -> bool:
        """Whether this track carries measured word timestamps."""
        return bool(self.words)


@dataclass
class BardResult:
    """What one Short production produced."""

    ok: bool
    short_id: int | None = None
    title: str = ""
    video_path: Path | None = None
    audio_path: Path | None = None
    subtitle_path: Path | None = None
    duration: float = 0.0
    render_backend: str = ""
    voice_backend: str = ""
    dispatched: bool = False
    errors: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            return f"FAILED: {'; '.join(self.errors) or 'unknown error'}"
        gate = "sent to Telegram" if self.dispatched else "awaiting CLI approval"
        return f"short #{self.short_id} '{self.title}' ({self.duration:.0f}s) — {gate}"


def _extract_json(raw: str) -> dict[str, Any]:
    """Tolerant JSON extraction, matching the Scout's and Scribe's."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in response") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON was not an object")
    return parsed


class BardAgent:
    """Produces one faceless vertical Short, end to end."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.crafter = Crafter(self.settings)
        self.video = VideoEngine(self.settings)
        self.stock = StockVideoLibrary(self.settings)
        self.artist = SceneArtist(self.settings)
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

    # ======================================================================
    # 1. Script
    # ======================================================================

    def recent_titles(self, limit: int = RECENT_TOPIC_MEMORY) -> list[str]:
        """Titles of the Shorts most recently made.

        Read from the database rather than kept in a side file: the Shorts
        table already IS the history, and a separate record would drift from it
        the first time a row was deleted.
        """
        try:
            from core.database import recent_shorts  # noqa: PLC0415

            return [row.title for row in recent_shorts(limit=limit) if row.title]
        except Exception as exc:  # noqa: BLE001 - never block a run over history
            logger.debug("Could not read recent titles: {}", exc)
            return []

    def pick_topic(self) -> str:
        """An angle to write about when the caller did not name one.

        Seeds are weighted away from anything close to a recent title. An
        unprompted model reliably returns the two or three most famous stories
        in a category — the seed is what stops that, and the recent-title check
        is what stops the seed itself repeating.
        """
        recent = [title.casefold() for title in self.recent_titles()]

        def stale(seed: str) -> bool:
            # Cheap overlap test: a seed shares a distinctive word with a title
            # the channel already used.
            words = {w for w in seed.casefold().split() if len(w) > 5}
            return any(any(word in title for word in words) for title in recent)

        fresh = [seed for seed in TOPIC_SEEDS if not stale(seed)]
        if not fresh:
            # Everything looks recent, which means the memory is longer than the
            # seed bank. Fall back to the full list rather than refusing to run.
            logger.debug("Every topic seed matched a recent title; using them all")
            fresh = list(TOPIC_SEEDS)

        return random.choice(fresh)

    async def write_script(self, topic: str | None = None) -> ShortScript:
        """Produce a validated script.

        Never raises: an unusable model response falls back to a curated script
        so the voice, image and assembly steps are always exercised.
        """
        scene_count = self.settings.shorts_scene_count
        target = self.settings.shorts_target_seconds

        if not self.settings.openrouter_configured:
            logger.warning("OPENROUTER_API_KEY unset — the Bard is reciting from memory")
            return self._fallback_script("no api key")

        # Every number below comes from one rate, so the per-scene figure and
        # the total can no longer contradict each other. They used to: the
        # prompt asked for ~100 words in total AND 25-40 words in each of four
        # scenes, and the model followed the per-scene number every time.
        budget = int(target * WORDS_PER_SECOND)
        max_words = int(SHORTS_HARD_LIMIT_SECONDS * WORDS_PER_SECOND)
        scene_words = max(
            MIN_SCENE_WORDS,
            (budget - HOOK_WORDS - CONCLUSION_WORDS) // max(1, scene_count),
        )

        system = (
            SCRIPT_PROMPT.replace("{scene_count}", str(scene_count))
            .replace("{scene_words}", str(scene_words))
            .replace("{target_seconds}", str(target))
            .replace("{word_budget}", str(budget))
            .replace("{max_words}", str(max_words))
        )
        # No topic means "choose one", not "choose the same one". Without a seed
        # the model returns its handful of favourites over and over.
        chosen = topic or self.pick_topic()
        avoid = self.recent_titles()

        user = (
            f"Write one short about: {chosen}."
            if topic
            else f"Write one short on this angle: {chosen}\n\n"
            f"Pick ONE specific real case that fits it. Category: "
            f"{random.choice(CATEGORIES)}. "
            "Choose something specific and genuinely strange, not a famous story "
            "everyone has already seen."
        )

        if avoid:
            # The model has no memory between runs, so the channel's own back
            # catalogue has to be handed to it or it will cheerfully write the
            # same Short a third time.
            user += (
                "\n\nThis channel has ALREADY published the following. Do not "
                "write about any of them, or about the same underlying event:\n"
                + "\n".join(f"- {title}" for title in avoid)
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.95,
                max_tokens=2400,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Script call failed ({}); reciting from memory", exc)
            return self._fallback_script(f"api error: {exc}")

        content = (response.choices[0].message.content or "") if response.choices else ""
        try:
            payload = _extract_json(content)
        except ValueError as exc:
            logger.error("Script was unparseable ({}); reciting from memory", exc)
            return self._fallback_script("unparseable response")

        script = self._coerce_script(payload)
        if script is None:
            logger.error("Script was incomplete; reciting from memory")
            return self._fallback_script("incomplete script")

        return self._screen_script(script)

    def _coerce_script(self, payload: dict[str, Any]) -> ShortScript | None:
        """Validate a model payload into a script, repairing what it can."""

        def text(key: str, limit: int = 400) -> str:
            value = payload.get(key, "")
            return value.strip()[:limit] if isinstance(value, str) else ""

        raw_scenes = payload.get("scenes")
        if not isinstance(raw_scenes, list) or not raw_scenes:
            return None

        scenes: list[Scene] = []
        for entry in raw_scenes:
            if not isinstance(entry, dict):
                continue
            narration = str(entry.get("narration", "")).strip()
            prompt = str(entry.get("image_prompt", "")).strip()
            if not narration or not prompt:
                continue
            scenes.append(
                Scene(
                    narration=narration,
                    image_prompt=prompt,
                    on_screen=str(entry.get("on_screen", "")).strip()[:60],
                    stock_query=str(entry.get("stock_query", "")).strip()[:80],
                )
            )

        if not scenes:
            return None

        # Hold the scene count to the configured shape: too many scenes and each
        # one is on screen too briefly for the Ken Burns move to read.
        wanted = self.settings.shorts_scene_count
        if len(scenes) > wanted:
            logger.info("Script had {} scenes; trimming to {}", len(scenes), wanted)
            scenes = scenes[:wanted]
        elif len(scenes) < MIN_SCENES:
            # Too few beats and the picture sits still through the whole read,
            # which is the single most common reason a Short is scrolled past.
            # Rejecting sends it back for a retry rather than shipping it.
            logger.warning(
                "Script returned {} scene(s); at least {} are needed for the "
                "video to move. Rejecting the draft.", len(scenes), MIN_SCENES,
            )
            return None

        hook = text("hook", 200)
        if not hook:
            return None

        category = text("category", 60)
        if category not in CATEGORIES:
            category = CATEGORIES[0]

        hashtags = payload.get("hashtags")
        tags = (
            [re.sub(r"[^a-z0-9]", "", str(t).lower())[:30] for t in hashtags][:8]
            if isinstance(hashtags, list)
            else []
        )

        return ShortScript(
            title=text("title", 100) or text("topic", 60) or "Untitled Short",
            category=category,
            topic=text("topic", 120) or text("title", 120),
            hook=hook,
            scenes=scenes,
            conclusion=text("conclusion", 300),
            description=text("description", 900),
            hashtags=[tag for tag in tags if tag],
        )

    def _screen_script(self, script: ShortScript) -> ShortScript:
        """Refuse a script that trips the IP screen rather than illustrating it."""
        result = screen_many(
            {
                "title": script.title,
                "hook": script.hook,
                "narration": script.narration,
                "prompts": " ".join(scene.image_prompt for scene in script.scenes),
            }
        )
        if result.ok:
            return script

        logger.warning("The Bard's script tripped the IP screen: {}", result.reason())
        return self._fallback_script(f"ip screen: {result.reason()}")

    def _fallback_script(self, reason: str) -> ShortScript:
        """A curated script, chosen at random so repeated runs differ."""
        chosen = random.choice(FALLBACK_SCRIPTS)
        logger.info("Bard fallback script: {} ({})", chosen["title"], reason)
        return ShortScript(
            title=chosen["title"],
            category=chosen["category"],
            topic=chosen["topic"],
            hook=chosen["hook"],
            scenes=[Scene(**scene) for scene in chosen["scenes"]],
            conclusion=chosen["conclusion"],
            description=chosen["description"],
            hashtags=list(chosen["hashtags"]),
            source=f"fallback ({reason})",
        )

    async def add_stock_footage(self, scenes: Sequence[Scene], workdir: Path) -> int:
        """Replace some scenes' stills with real vertical B-roll.

        Off unless ``PEXELS_API_KEY`` is set, capped by ``STOCK_VIDEO_MAX_CLIPS``,
        and non-fatal in every failure mode: the still stays on the scene and
        the assembler animates it exactly as before.

        The still is kept on the scene either way — it is what the thumbnail is
        cut from, and what the storyboard backend falls back to.

        :returns: how many scenes ended up with footage.
        """
        if not self.stock.configured:
            return 0

        try:
            clips = await self.stock.fetch_many(
                [scene.stock_query for scene in scenes], workdir / "broll"
            )
        except Exception as exc:  # noqa: BLE001 - B-roll is never worth failing over
            logger.warning("Stock footage lookup failed: {}", exc)
            return 0

        for index, clip in clips.items():
            scenes[index].video_path = str(clip.path)
            logger.info(
                "Scene {}/{} uses stock footage — {}",
                index + 1, len(scenes), clip.credit,
            )

        return len(clips)

    # ======================================================================
    # 2. Voice
    # ======================================================================

    async def speak(self, narration: str, destination: Path) -> VoiceTrack:
        """Render narration to an audio file.

        :returns: a :class:`VoiceTrack` carrying the path, the backend that
            produced it, and — for providers that report them — the word-level
            timestamps the assembler cuts and captions against. The backend is
            recorded on the row so a silent placeholder is never mistaken for a
            real read.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        provider = self.settings.tts_provider

        # Checked before the attempt so a missing package or key is reported as
        # itself, rather than as a generic "unavailable" after the fact.
        usable, reason = self.settings.tts_status()
        if not usable:
            logger.error(
                "Voice provider {!r} cannot speak: {}. The Short will be SILENT.",
                provider, reason,
            )
            return VoiceTrack(
                path=self._speak_placeholder(narration, destination),
                backend="placeholder",
            )

        if provider == "edge":
            # The only provider that reports word boundaries, so it is the only
            # one that can be handled as a stream rather than a file.
            track = await self._speak_edge(narration, destination)
            if track:
                return track
            backend = "edge-tts"
        else:
            speakers = {
                "openai": (self._speak_openai, "openai"),
                "elevenlabs": (self._speak_elevenlabs, "elevenlabs"),
            }
            render, backend = speakers[provider]
            path = await render(narration, destination)
            if path:
                return VoiceTrack(path=path, backend=backend)

        logger.error(
            "{} produced no audio after {} attempt(s); the Short will be "
            "SILENT. Re-run with LOG_LEVEL=DEBUG for the underlying error.",
            backend, TTS_ATTEMPTS,
        )
        return VoiceTrack(
            path=self._speak_placeholder(narration, destination),
            backend="placeholder",
        )

    async def _speak_edge(self, narration: str, destination: Path) -> VoiceTrack | None:
        """edge-tts: free, no key, good documentary voices.

        Synthesised with ``stream()`` rather than ``save()``. Both write the
        same audio, but only the stream carries the WordBoundary events, and
        those events are what let the captions and the scene cuts sit on the
        voice instead of on an estimate of how long the words take to say.

        The websocket to Microsoft's endpoint can drop mid-stream, so a failed
        attempt is retried with a short backoff rather than costing the Short
        its narration. Results are checked by decoded duration: a truncated
        stream can still leave a plausibly sized file on disk.
        """
        try:
            import edge_tts  # noqa: PLC0415
        except ImportError as exc:
            # Loud: this is the difference between a narrated Short and a
            # silent one, and it is fixed by a single pip install.
            logger.error(
                "edge-tts is not installed in this interpreter ({}). "
                "Install it with:  pip install 'edge-tts>=6.1.0'   — or run "
                "./setup.sh. Without it the Short has no narration.",
                exc,
            )
            return None

        for attempt in range(1, TTS_ATTEMPTS + 1):
            words: list[WordTiming] = []
            try:
                communicate = edge_tts.Communicate(
                    narration,
                    self.settings.tts_voice,
                    rate=self.settings.tts_rate,
                    pitch=self.settings.tts_pitch,
                    boundary="WordBoundary",
                )
                with destination.open("wb") as handle:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            handle.write(chunk["data"])
                        elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                            # Same arithmetic edge_tts.SubMaker applies, kept
                            # here because the timings drive scene cuts too, and
                            # SubMaker only ever yields one cue per word — too
                            # short to put on screen as a caption.
                            words.append(
                                WordTiming(
                                    text=chunk["text"],
                                    # Offsets arrive in 100-nanosecond ticks.
                                    start=chunk["offset"] / 1e7,
                                    end=(chunk["offset"] + chunk["duration"]) / 1e7,
                                )
                            )
            except Exception as exc:  # noqa: BLE001 - network or voice-name failure
                logger.warning(
                    "edge-tts attempt {}/{} failed: {}", attempt, TTS_ATTEMPTS, exc
                )
            else:
                spoken = self._verify_audio(destination)
                if spoken:
                    subtitles = self._write_subtitles(words, destination)
                    logger.success(
                        "Narration voiced by edge-tts as {} "
                        "({:.0f} KB, {:.1f}s, {} word timings)",
                        self.settings.tts_voice,
                        destination.stat().st_size / 1024, spoken, len(words),
                    )
                    return VoiceTrack(
                        path=destination,
                        backend="edge-tts",
                        words=words,
                        subtitles=subtitles,
                    )
                logger.warning(
                    "edge-tts attempt {}/{} returned an unusable file",
                    attempt, TTS_ATTEMPTS,
                )

            # Nothing usable was written; do not leave a stub for the next try
            # to mistake for success.
            destination.unlink(missing_ok=True)
            if attempt < TTS_ATTEMPTS:
                await asyncio.sleep(TTS_RETRY_BACKOFF * attempt)

        return None

    def _write_subtitles(self, words: Sequence[WordTiming], audio: Path) -> Path | None:
        """Write the measured captions beside the audio as .srt and .vtt.

        Grouped exactly as the burned-in captions are, so the sidecars are a
        faithful record of what is on screen rather than a second, subtly
        different set of timings. edge-tts reports one boundary per word, and a
        50ms single-word cue is not something any player can display usefully.

        Neither file is consumed by the renderer. They exist so sync can be
        checked in an editor, and so the Short can be uploaded with real closed
        captions instead of auto-generated ones.

        :returns: the .srt path, or None when there is nothing to write.
        """
        if not words:
            return None

        cues = cues_from_timings(
            words, words_per_cue=self.settings.subtitle_words_per_cue
        )
        if not cues:
            return None

        srt_path = audio.with_suffix(".srt")
        try:
            srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
            audio.with_suffix(".vtt").write_text(cues_to_vtt(cues), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 - never fail a render over a sidecar
            logger.debug("Could not write subtitles: {}", exc)
            return None

        return srt_path

    @staticmethod
    def _verify_audio(path: Path) -> float:
        """Length of a rendered narration, or 0.0 if it is not usable.

        Size alone is not enough: a websocket that drops after the MP3 header
        leaves a few KB of nothing. ffprobe reads the real duration, and when
        ffprobe is absent the size check is kept as a weaker fallback.
        """
        if not path.is_file():
            return 0.0

        duration = probe_duration(path)
        if duration >= MIN_NARRATION_SECONDS:
            return duration

        if duration == 0.0 and ffprobe_path() is None:
            # No probe available; fall back to the old heuristic rather than
            # discarding audio that is probably fine.
            return 1.0 if path.stat().st_size > 4096 else 0.0

        logger.debug("Rejected {} — only {:.2f}s of audio", path.name, duration)
        return 0.0

    async def _speak_openai(self, narration: str, destination: Path) -> Path | None:
        """OpenAI's speech endpoint. Note this uses OPENAI_API_KEY, not
        OpenRouter — OpenRouter has no TTS route."""
        if not self.settings.openai_api_key.strip():
            logger.debug("OPENAI_API_KEY unset; cannot use OpenAI TTS")
            return None

        try:
            client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.request_timeout_seconds,
            )
            async with client.audio.speech.with_streaming_response.create(
                model=self.settings.openai_tts_model,
                voice=self.settings.openai_tts_voice,
                input=narration,
                response_format="mp3",
            ) as response:
                await response.stream_to_file(str(destination))
            await client.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenAI TTS failed: {}", exc)
            return None

        return destination if destination.is_file() and destination.stat().st_size > 1024 else None

    async def _speak_elevenlabs(self, narration: str, destination: Path) -> Path | None:
        """ElevenLabs REST, called directly so the SDK is not a dependency."""
        if not (self.settings.elevenlabs_api_key.strip() and self.settings.elevenlabs_voice_id.strip()):
            logger.debug("ElevenLabs is not configured")
            return None

        import httpx  # noqa: PLC0415

        url = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{self.settings.elevenlabs_voice_id}"
        )
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(
                    url,
                    headers={
                        "xi-api-key": self.settings.elevenlabs_api_key,
                        "accept": "audio/mpeg",
                        "content-type": "application/json",
                    },
                    json={
                        "text": narration,
                        "model_id": self.settings.elevenlabs_model,
                        "voice_settings": {"stability": 0.45, "similarity_boost": 0.8},
                    },
                )
                response.raise_for_status()
                destination.write_bytes(response.content)
        except Exception as exc:  # noqa: BLE001
            logger.error("ElevenLabs failed: {}", exc)
            return None

        return destination if destination.is_file() and destination.stat().st_size > 1024 else None

    def _speak_placeholder(self, narration: str, destination: Path) -> Path:
        """A silent WAV of the estimated read length.

        Written with the stdlib so it works with nothing installed. It gives the
        assembler a real duration to cut against, and the ``placeholder`` voice
        backend on the row makes it obvious no one actually spoke.
        """
        seconds = max(5.0, len(narration.split()) / 2.5)
        target = destination.with_suffix(".wav")
        rate = 24000

        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            # A very quiet tone rather than pure digital silence: some encoders
            # drop an all-zero track entirely.
            frames = bytearray()
            for index in range(int(rate * seconds)):
                sample = int(180 * math.sin(2 * math.pi * 55 * index / rate))
                frames += struct.pack("<h", sample)
            handle.writeframes(bytes(frames))

        logger.warning("No TTS available — wrote a {:.0f}s placeholder track", seconds)
        return target

    # ======================================================================
    # 3. Scenes
    # ======================================================================

    async def illustrate(self, script: ShortScript, workdir: Path) -> list[Scene]:
        """Render every scene image, sequentially.

        Goes through the SceneArtist rather than the Crafter. The Crafter exists
        to make print-on-demand artwork and appends a directive demanding a flat
        graphic on a plain white background with no photographic detail — which
        is correct for a t-shirt and the exact opposite of a moody film frame.
        Every Short was being illustrated against that directive.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        scenes_dir = workdir / "scenes"

        for index, scene in enumerate(script.scenes, start=1):
            events.agent_working(
                "bard",
                f"Painting scene {index}/{len(script.scenes)}…",
                progress=0.4 + 0.2 * (index / len(script.scenes)),
            )

            painted = await self.artist.paint(
                scene.image_prompt, scenes_dir / f"scene{index:02d}.jpg"
            )
            scene.image_path = str(painted.path)
            scene.simulated = painted.simulated

            logger.info(
                "Scene {}/{}: {} via {} ({}x{}){}",
                index, len(script.scenes), painted.path.name, painted.provider,
                painted.width, painted.height,
                " — PLACEHOLDER" if painted.simulated else "",
            )

        return script.scenes

    # ======================================================================
    # The full production
    # ======================================================================

    async def produce(self, topic: str | None = None, *, dispatch: bool = True) -> BardResult:
        """Write, voice, illustrate, cut and dispatch one Short."""
        result = BardResult(ok=False)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        # ---- 1. script ------------------------------------------------------
        events.agent_working("bard", "Working the story into a shape…", progress=0.1)
        script = await self.write_script(topic)
        result.title = script.title
        result.stages.append(f"script: {script.title} ({script.word_count} words, {script.source})")
        logger.info(
            "Script: {!r} — {} scenes, ~{:.0f}s",
            script.title, len(script.scenes), script.estimated_seconds,
        )

        short = create_short(
            topic=script.topic,
            category=script.category,
            title=script.title,
            hook=script.hook,
            conclusion=script.conclusion,
            narration=script.narration,
            description=script.description,
            hashtags=script.hashtags,
            scenes=[scene.to_dict() for scene in script.scenes],
            dry_run=1 if self.settings.dry_run else 0,
            status=ShortStatus.DRAFTED.value,
            status_reason=f"Script from {script.source}.",
        )
        result.short_id = short.id
        events.agent_output(
            "bard", "script",
            short_id=short.id, title=script.title, hook=script.hook,
            category=script.category, scenes=len(script.scenes),
            words=script.word_count, source=script.source,
        )

        workdir = self.settings.shorts_dir / f"{stamp}-{short.id}"
        workdir.mkdir(parents=True, exist_ok=True)

        # ---- 2. voice -------------------------------------------------------
        events.agent_working("bard", "Recording the narration…", progress=0.3)
        try:
            track = await self.speak(script.narration, workdir / "narration.mp3")
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"voice: {exc}")
            events.agent_error("bard", "The voice failed.", str(exc))
            update_short_status(short.id, ShortStatus.FAILED, reason=f"Voice failed: {exc}")
            return result

        audio_path = track.path
        result.audio_path = audio_path
        result.subtitle_path = track.subtitles
        result.voice_backend = track.backend
        spoken = probe_duration(audio_path) if audio_path else 0.0

        sync = f", {len(track.words)} word timings" if track.timed else ", estimated timing"
        result.stages.append(f"voice: {track.backend} ({spoken:.1f}s{sync})")

        # YouTube only treats a video as a Short UNDER 60 seconds. A read that
        # runs long is not a failure — the file is still good — but it silently
        # changes what the platform does with it, so it is worth shouting about.
        if spoken >= 60:
            warning = (
                f"The narration runs {spoken:.0f}s. YouTube only classifies videos "
                "under 60s as Shorts; this will be published as a regular video. "
                "Lower SHORTS_TARGET_SECONDS or shorten the script."
            )
            logger.warning(warning)
            events.toast(warning, "warn")
            result.stages.append(f"warning: {spoken:.0f}s exceeds the 60s Shorts limit")
        elif spoken > self.settings.shorts_target_seconds * 1.25:
            logger.info(
                "Narration ran {:.0f}s against a {}s target — still a valid Short.",
                spoken, self.settings.shorts_target_seconds,
            )
        update_short(
            short.id, audio_path=str(audio_path) if audio_path else None,
            voice_backend=track.backend,
        )

        # ---- 3. scenes ------------------------------------------------------
        try:
            scenes = await self.illustrate(script, workdir)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"scenes: {exc}")
            events.agent_error("bard", "The scenes failed.", str(exc))
            update_short_status(short.id, ShortStatus.FAILED, reason=f"Scenes failed: {exc}")
            return result

        result.stages.append(f"scenes: {len(scenes)} rendered")

        # ---- 3b. optional B-roll --------------------------------------------
        # Never fatal: a Short illustrated entirely with stills is the baseline
        # this whole step is trying to improve on, so any failure just leaves
        # the stills in place.
        stock_used = await self.add_stock_footage(scenes, workdir)
        if stock_used:
            result.stages.append(f"b-roll: {stock_used} stock clip(s)")

        update_short(short.id, scenes=[scene.to_dict() for scene in scenes])

        # ---- 4. assembly ----------------------------------------------------
        events.agent_working("bard", "Cutting the video…", progress=0.75)
        update_short_status(short.id, ShortStatus.RENDERING, reason="Assembling the video.")

        render = await asyncio.to_thread(
            self.video.render,
            scenes=[scene.to_dict() for scene in scenes],
            audio_path=audio_path,
            output=workdir / "short.mp4",
            narration=script.narration,
            word_timings=track.words,
        )

        if not render.ok:
            result.errors.append(f"render: {render.error}")
            events.agent_error("bard", "The cut failed.", render.error or "")
            update_short_status(short.id, ShortStatus.FAILED, reason=render.summary())
            return result

        result.video_path = render.path
        result.duration = render.duration
        result.render_backend = render.backend
        result.stages.append(f"render: {render.summary()}")

        thumbnail = await asyncio.to_thread(
            self.video.thumbnail, render.path, workdir / "thumbnail.jpg"
        )

        update_short(
            short.id,
            video_path=str(render.path),
            thumbnail_path=str(thumbnail) if thumbnail else None,
            duration_seconds=render.duration,
            render_backend=render.backend,
        )
        events.agent_output(
            "bard", "video",
            short_id=short.id,
            title=script.title,
            video_url=f"/api/shorts/{short.id}/video",
            thumbnail_url=f"/api/shorts/{short.id}/thumbnail" if thumbnail else None,
            duration=round(render.duration, 1),
            backend=render.backend,
            voice=track.backend,
            storyboard=render.backend == "storyboard",
        )

        # ---- 5. hand to the human ------------------------------------------
        update_short_status(
            short.id, ShortStatus.PENDING_APPROVAL, reason="Awaiting the operator's verdict."
        )

        if dispatch:
            events.agent_working("bard", "Sending it to the herald…", progress=0.95)
            from village.town_crier import TownCrier  # noqa: PLC0415 - avoids a cycle

            result.dispatched = await TownCrier(self.settings).dispatch_short(short.id)
            result.stages.append(
                "telegram: dispatched" if result.dispatched else "telegram: not configured"
            )

        events.agent_done("bard", f"'{script.title}' is ready for review.")
        events.listing_event({})  # nudge the dashboard's stats refresh
        result.ok = True
        logger.success("Bard complete — {}", result.summary())
        return result

    async def reroll(self, short_id: int, topic: str | None = None) -> BardResult:
        """Rewrite and re-cut an existing short, keeping its history.

        The old row is marked back to DRAFTED and a new production runs; the
        reroll counter on the original records that a human sent it back.
        """
        existing = get_short(short_id)
        if existing is None:
            result = BardResult(ok=False)
            result.errors.append(f"no short {short_id}")
            return result

        update_short(short_id, reroll_count=existing.reroll_count + 1)
        update_short_status(
            short_id, ShortStatus.DRAFTED, reason="Rerolled by the operator."
        )
        logger.info("Rerolling short {} (attempt {})", short_id, existing.reroll_count + 1)
        return await self.produce(topic or existing.topic)

    def toolchain(self) -> dict[str, Any]:
        """What the Bard can actually do on this machine."""
        chain = describe_toolchain()
        chain["tts_provider"] = self.settings.tts_provider
        chain["tts_configured"] = self.settings.tts_configured
        try:
            import edge_tts  # noqa: F401,PLC0415

            chain["edge_tts"] = True
        except Exception:  # noqa: BLE001
            chain["edge_tts"] = False
        return chain

    async def aclose(self) -> None:
        await self.crafter.aclose()
        if self._client is not None:
            await self._client.close()
            self._client = None
