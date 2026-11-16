"""Last line of defence between the model and the caller's ears.

Runs on every sentence the model is about to speak, in the live pipeline (``ReplyGuardProcessor``)
and in the evals (same ``ReplyGuard`` class), so measurements describe what a caller hears.

It fixes four things that prompts alone did not fix (HANDOFF findings 2 and 3):

* markdown that TTS would read aloud ("star star", bullets)
* the system prompt (or tool names) being read out
* "I've booked your session" when the booking tool did not succeed
* promising a confirmation email that the system never sends
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from campusx_voice_agent.booking_gate import booking_confirmed_this_turn
from campusx_voice_agent.text_cleanup import ensure_terminal, strip_markdown_line

LEAK_REFUSAL = (
    "I can't share that, but I'm happy to help with CampusX courses, schedules, "
    "or booking a free guidance session."
)
NOT_BOOKED = "I haven't booked anything yet. I need your clear yes to the details first."

SHINGLE_WORDS = 8
TOOL_NAMES = (
    "search_campusx_knowledge_base",
    "get_available_guidance_slots",
    "book_free_guidance_session",
)
INTERNAL_MARKERS = TOOL_NAMES + ("[kb-context]",)

# Kept separate from evals/checks.py on purpose: the eval must not grade the guard with the
# guard's own patterns.
_BOOKED_CLAIM = re.compile(
    r"(i('ve| have)\s+(successfully\s+)?booked|i\s+booked|"
    r"you('re| are)\s+(all set|booked|confirmed)|"
    r"(has|have)\s+been\s+(booked|confirmed|scheduled)|booking\s+(is\s+)?(confirmed|complete|done)|"
    r"successfully\s+(booked|scheduled)|session\s+is\s+(booked|confirmed|scheduled))",
    re.IGNORECASE,
)
_FOLLOWUP_PROMISE = re.compile(
    r"((confirmation|reminder)\s+(e-?mail|message|sms|text)|"
    r"(will|'ll)\s+(receive|get)\s+(an?\s+)?(e-?mail|confirmation|reminder|call))",
    re.IGNORECASE,
)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_shingles(prompt: str, allowed_phrases: Iterable[str] = ()) -> frozenset[tuple[str, ...]]:
    """All runs of SHINGLE_WORDS words of the prompt, minus phrases the agent may say."""
    text = prompt.lower()
    for phrase in allowed_phrases:
        text = text.replace(phrase.lower(), " ")
    words = _words(text)
    return frozenset(
        tuple(words[i : i + SHINGLE_WORDS]) for i in range(max(0, len(words) - SHINGLE_WORDS + 1))
    )


class ReplyGuard:
    """Streaming sentence filter. Feed it text chunks; it returns clean, speakable sentences."""

    def __init__(
        self,
        prompt: str,
        *,
        allowed_phrases: Iterable[str] = (),
        booked: Callable[[], bool] = lambda: False,
    ) -> None:
        self._shingles = build_shingles(prompt, allowed_phrases)
        self._booked = booked
        self._buffer = ""
        self._leaked = False
        self._said_not_booked = False
        self.interventions: list[str] = []

    def new_response(self) -> None:
        self._buffer = ""
        self._leaked = False
        self._said_not_booked = False
        self.interventions = []

    # -- streaming interface -------------------------------------------------------------

    def feed(self, chunk: str) -> list[str]:
        self._buffer += chunk
        *complete, rest = _BOUNDARY.split(self._buffer)
        self._buffer = rest
        return [out for part in complete if (out := self._process(part))]

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer, ""
        out = self._process(rest)
        return [out] if out else []

    def clean(self, text: str) -> str:
        """Whole-reply convenience used by the evals: same path as streaming."""
        self.new_response()
        parts = self.feed(text) + self.flush()
        return " ".join(parts)

    # -- rules -----------------------------------------------------------------------------

    def _is_leak(self, sentence: str) -> bool:
        low = sentence.lower()
        if any(marker in low for marker in INTERNAL_MARKERS):
            return True
        words = _words(sentence)
        return any(
            tuple(words[i : i + SHINGLE_WORDS]) in self._shingles
            for i in range(max(0, len(words) - SHINGLE_WORDS + 1))
        )

    def _process(self, raw: str) -> str | None:
        raw = raw.strip()
        if not raw:
            return None
        if self._leaked:
            self.interventions.append("leak_suppressed")
            return None
        if self._is_leak(raw):
            self._leaked = True
            self.interventions.append("prompt_leak")
            return LEAK_REFUSAL

        clean = strip_markdown_line(raw)
        if clean != re.sub(r"\s+", " ", raw).strip():
            self.interventions.append("markdown")
        if not clean:
            return None
        if _FOLLOWUP_PROMISE.search(clean):
            self.interventions.append("fabricated_followup")
            return None
        if _BOOKED_CLAIM.search(clean) and not self._booked():
            self.interventions.append("false_booking_claim")
            if self._said_not_booked:
                return None
            self._said_not_booked = True
            return NOT_BOOKED
        return ensure_terminal(clean)


class ReplyGuardProcessor(FrameProcessor):
    """Pipecat wrapper: sits between the LLM and TTS and rewrites LLM text frames."""

    def __init__(self, guard: ReplyGuard, **kwargs) -> None:
        super().__init__(**kwargs)
        self._guard = guard

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._guard.new_response()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame) and not frame.skip_tts:
            for sentence in self._guard.feed(frame.text):
                await self._push_sentence(sentence, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            for sentence in self._guard.flush():
                await self._push_sentence(sentence, direction)
            if self._guard.interventions:
                logger.info("ReplyGuard interventions: {}", self._guard.interventions)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._guard.new_response()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _push_sentence(self, sentence: str, direction: FrameDirection) -> None:
        await self.push_frame(LLMTextFrame(text=sentence + " "), direction)


def booked_from_context(context) -> Callable[[], bool]:
    """Live-pipeline helper: did a booking succeed after the caller's latest message?"""
    return lambda: booking_confirmed_this_turn(context.get_messages())
