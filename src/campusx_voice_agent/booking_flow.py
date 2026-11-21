"""Offer the booking tool only when booking is allowed, and write the read-back in code.

Second real run (qwen2.5:7b): in 6 of 12 booking flows the model called
``book_free_guidance_session`` right after the caller picked a slot, before any read-back. The gate
refused every one of them, but each refusal cost a whole extra model round (booking latency p50
3.9 s -> 5.0 s, max 12.5 s). A model cannot call a tool it is not shown, so the tool is now hidden
until the transcript shows:

1. an assistant read-back that contains the email address and asks a question, and
2. the caller's next message being an explicit yes.

Third real run: that alone broke bookings (1 of 7 should-book flows booked), because the model kept
leaving the email out of its own read-back, so the tool never unlocked. So the READ-BACK is now
produced by code: ``confirm_booking_details`` validates the four details and returns the exact
sentence, and the reply guard speaks it instead of whatever the model wrote (``forced_readback``).
The model only has to call a tool with arguments, which it does reliably.

While the booking tool is hidden and the caller has answered the slot offers, a short system note
says what to do next. ``booking_gate.check_booking_allowed`` still runs inside the booking tool and
stays the final check (a hallucinated call to a hidden tool is executed by Pipecat's handler and
refused there).
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from campusx_voice_agent.booking_gate import (
    _last_index,
    _normalized_spoken,
    booking_confirmed_this_turn,
    is_explicit_yes,
    message_text,
)

BOOKING_TOOL = "book_free_guidance_session"
CONFIRM_TOOL = "confirm_booking_details"
BOOKING_MARKER = "[BOOKING-STATE]"
_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}")

LOCKED_HINT = (
    f"{BOOKING_MARKER}\n"
    "The booking tool is locked. If the caller has chosen a slot, call confirm_booking_details now "
    "with their four details; the system reads them back, so do not say them yourself. The booking "
    "tool unlocks only after the caller says a clear yes to that read-back."
)


def _is_booking_state(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "system"
        and message_text(message).startswith(BOOKING_MARKER)
    )


def _slots_were_offered(messages: list[Any]) -> bool:
    return any(
        isinstance(m, dict) and m.get("role") == "tool" and "available_slots" in message_text(m)
        for m in messages
    )


def _ever_booked(messages: list[Any]) -> bool:
    return any(
        isinstance(m, dict)
        and m.get("role") == "tool"
        and '"booked": true' in message_text(m).lower()
        for m in messages
    )


def booking_tool_unlocked(messages: list[Any]) -> bool:
    """True only right after a read-back (with an email, ending in a question) and a clear yes."""
    user_idx = _last_index(messages, "user")
    if user_idx is None or not is_explicit_yes(message_text(messages[user_idx])):
        return False
    assistant_idx = _last_index(messages, "assistant", before=user_idx)
    if assistant_idx is None:
        return False
    readback = message_text(messages[assistant_idx])
    return "?" in readback and _EMAIL.search(_normalized_spoken(readback)) is not None


def _caller_replied_after_slots(messages: list[Any]) -> bool:
    """Slots were offered in an earlier turn and the caller has answered since."""
    last_offer = None
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "tool" and "available_slots" in message_text(m):
            last_offer = i
    if last_offer is None:
        return False
    return any(isinstance(m, dict) and m.get("role") == "user" for m in messages[last_offer + 1 :])


def forced_readback(messages: list[Any] | None) -> str | None:
    """The code-written read-back, if confirm_booking_details succeeded this turn."""
    if not messages:
        return None
    user_idx = _last_index(list(messages), "user")
    found = None
    for m in messages[(user_idx + 1) if user_idx is not None else 0 :]:
        if not (isinstance(m, dict) and m.get("role") == "tool"):
            continue
        text = message_text(m)
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("ready") is True
            and payload.get("say_exactly")
        ):
            found = str(payload["say_exactly"])
    return found


def apply_booking_state(messages: list[Any]) -> dict[str, Any]:
    """Refresh the booking note in ``messages`` (in place); say which booking tools are offered."""
    messages[:] = [m for m in messages if not _is_booking_state(m)]
    done = _ever_booked(messages) or booking_confirmed_this_turn(messages)
    unlocked = booking_tool_unlocked(messages) and not done
    confirm_visible = _caller_replied_after_slots(messages) and not done
    hint = not unlocked and confirm_visible
    if hint:
        messages.append({"role": "system", "content": LOCKED_HINT})
    return {"unlocked": unlocked, "confirm_visible": confirm_visible, "hint": hint}


def allowed_tool_names(all_names: list[str], state: dict[str, Any] | bool) -> list[str]:
    """Tools the model is shown. ``state`` is apply_booking_state's result (a bool: 'unlocked')."""
    if isinstance(state, bool):
        state = {"unlocked": state, "confirm_visible": True}
    return [
        n
        for n in all_names
        if (n != BOOKING_TOOL or state["unlocked"])
        and (n != CONFIRM_TOOL or state["confirm_visible"])
    ]


class BookingStateProcessor(FrameProcessor):
    """Pipecat wrapper: before each LLM run, show the booking tool only when it is unlocked."""

    def __init__(self, tools: list, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tools = list(tools)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            try:
                messages = list(frame.context.get_messages())
                state = apply_booking_state(messages)
                frame.context.set_messages(messages)
                names = allowed_tool_names([t.__name__ for t in self._tools], state)
                frame.context.set_tools([t for t in self._tools if t.__name__ in names])
                logger.info(
                    "Booking tool {} (hint={})",
                    "UNLOCKED" if state["unlocked"] else "hidden",
                    state["hint"],
                )
            except Exception as exc:  # noqa: BLE001 - never let this take the call down
                logger.exception("Booking state failed: {}", exc)
        await self.push_frame(frame, direction)
