"""Code-level rules for booking, checked against the real conversation transcript.

The model may *ask* to book at any time. The tool only books when, in the actual transcript:

1. the caller's latest message is an explicit yes (not "yes but change...", not "ok"),
2. the assistant's previous message read back the same name, email and slot that are being
   booked (so a name correction can never be silently dropped), and
3. the slot was really offered by the availability tool earlier in this conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

MAX_SLOTS_OFFERED = 3
MAX_YES_WORDS = 14

_YES_START = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|sure|correct|confirmed?|affirmative|absolutely|definitely|exactly"
    r"|please do|do it|go ahead|book it|that'?s (?:right|correct)|that is (?:right|correct)"
    r"|sounds good|all correct|right)\b",
    re.IGNORECASE,
)
_NOT_A_CLEAN_YES = re.compile(
    r"\b(?:no|not|don'?t|do not|wait|hold on|actually|but|however|instead|different|change|wrong"
    r"|incorrect|mistake|fix|let me|maybe|later|cancel|stop)\b",
    re.IGNORECASE,
)
_BOOKED_TRUE = re.compile(r"booked['\"]?\s*:\s*true", re.IGNORECASE)

_REJECTION_TEXT = {
    "no_explicit_yes": (
        "The caller has not said an explicit yes to the details yet.",
        "Do not book. Read back the name, email, goal and exact slot, ask 'Shall I book it? "
        "Please say yes or no.', and call this tool again only after the caller clearly says yes.",
    ),
    "readback_mismatch": (
        "The details being booked do not match what was read back to the caller.",
        "Do not book. Read back the correct name, email, goal and exact slot, ask 'Shall I book "
        "it? Please say yes or no.', and wait for the caller's yes.",
    ),
    "slot_not_offered": (
        "That slot was not offered by the availability tool in this conversation.",
        "Call get_available_guidance_slots, offer at most three slots it returns, and let the "
        "caller choose one.",
    ),
    "no_transcript": (
        "The conversation could not be verified.",
        "Do not book. Ask the caller to confirm the details again.",
    ),
}


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str = "ok"
    missing: tuple[str, ...] = ()

    def as_result(self) -> dict[str, Any]:
        error, next_step = _REJECTION_TEXT.get(self.reason, ("Booking not allowed.", ""))
        result: dict[str, Any] = {
            "booked": False,
            "error": error,
            "reason": self.reason,
            "next_step": next_step,
        }
        if self.missing:
            result["not_read_back"] = list(self.missing)
        return result


def message_text(message: Any) -> str:
    """Plain text of a chat message whose content may be a string or a list of parts."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def is_explicit_yes(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or len(cleaned.split()) > MAX_YES_WORDS:
        return False
    return bool(_YES_START.match(cleaned)) and not _NOT_A_CLEAN_YES.search(cleaned)


def _normalized_spoken(text: str) -> str:
    """Lowercase, undo spoken email words, and drop spaces so emails compare reliably."""
    low = text.lower()
    low = re.sub(r"\s+dot\s+", ".", low)
    low = re.sub(r"\s+at\s+(?=[a-z0-9]+\.)", "@", low)
    low = re.sub(r"\s+at the rate\s+", "@", low)
    return re.sub(r"\s+", "", low)


def _slot_datetime(slot: str) -> datetime | None:
    try:
        return datetime.strptime(slot.removesuffix(" IST"), "%A, %d %B %Y at %I:%M %p")
    except ValueError:
        return None


_TIME_IN_TEXT = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b")


def _times_in(text: str) -> set[tuple[int, int, str]]:
    return {(int(h), int(m or 0), ap) for h, m, ap in _TIME_IN_TEXT.findall(text.lower())}


def _mentions_slot(readback: str, slot: str) -> bool:
    """The read-back must give the slot's time (with am/pm) and its weekday or its date."""
    low = readback.lower()
    when = _slot_datetime(slot)
    if when is None:
        return slot.lower() in low
    time_ok = (when.hour % 12 or 12, when.minute, "a" if when.hour < 12 else "p") in _times_in(low)
    weekday_ok = when.strftime("%A").lower() in low
    month_ok = when.strftime("%B").lower() in low or when.strftime("%b").lower() in low
    day_ok = re.search(rf"\b0?{when.day}(?:st|nd|rd|th)?\b", low) is not None
    return time_ok and (weekday_ok or (month_ok and day_ok))


_WEEKDAYS_MONTHS = (
    "monday tuesday wednesday thursday friday saturday sunday "
    "january february march april may june "
    "july august september october november december"
).split()
# Capitalised words that may follow a name in a read-back without being part of the name.
_NAME_BOUNDARY = frozenset(
    "email e-mail goal and for at on is shall please the your with to in learning wants want "
    "interested session slot booking".split()
    + _WEEKDAYS_MONTHS
)


def _name_read_back_whole(readback: str, name: str) -> bool:
    """The full name is spoken, and not followed by a further name part ("Ravi Kumar Reddy").

    A caller who corrects their name must hear the corrected name before saying yes; booking the
    old one afterwards would store the wrong person (HANDOFF finding 2).
    """
    parts = [p for p in re.findall(r"[a-z]+", name.lower()) if len(p) > 1]
    if not parts:
        return False
    text = re.sub(r"\S+@\S+", " ", readback)  # a surname inside the email is not a name
    pattern = r"\b" + r"[\s,.-]+".join(re.escape(p) for p in parts) + r"\b"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        following = re.match(
            r"\s+([A-Z][a-z'-]+)", text[match.end() :]
        )  # case-sensitive on purpose
        if not following or following.group(1).lower() in _NAME_BOUNDARY:
            return True
    return False


def readback_missing(readback: str, *, name: str, email: str, slot: str) -> list[str]:
    """Which of name / email / slot are NOT in the assistant's read-back."""
    missing: list[str] = []
    if not _name_read_back_whole(readback, name):
        missing.append("name")
    spoken = _normalized_spoken(readback)
    address = email.strip().lower().replace(" ", "")
    local, _, domain = address.partition("@")
    if not (address in spoken or (local and domain and local in spoken and domain in spoken)):
        missing.append("email")
    if not _mentions_slot(readback, slot):
        missing.append("slot")
    return missing


def _last_index(messages: list[Any], role: str, before: int | None = None) -> int | None:
    stop = len(messages) if before is None else before
    for i in range(stop - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == role and message_text(m).strip():
            return i
    return None


def slot_was_offered(messages: list[Any], slot: str) -> bool:
    return any(
        isinstance(m, dict) and m.get("role") == "tool" and slot in message_text(m)
        for m in messages
    )


def check_booking_allowed(
    messages: list[Any] | None, *, name: str, email: str, slot: str
) -> GateDecision:
    """Decide whether the booking tool may write to the database right now."""
    if not messages:
        return GateDecision(False, "no_transcript")
    user_idx = _last_index(messages, "user")
    if user_idx is None or not is_explicit_yes(message_text(messages[user_idx])):
        return GateDecision(False, "no_explicit_yes")
    assistant_idx = _last_index(messages, "assistant", before=user_idx)
    readback = message_text(messages[assistant_idx]) if assistant_idx is not None else ""
    missing = readback_missing(readback, name=name, email=email, slot=slot)
    if missing:
        return GateDecision(False, "readback_mismatch", tuple(missing))
    if not slot_was_offered(messages, slot):
        return GateDecision(False, "slot_not_offered")
    return GateDecision(True)


def booking_confirmed_this_turn(messages: list[Any] | None) -> bool:
    """True if a booking tool call succeeded after the caller's latest message."""
    if not messages:
        return False
    user_idx = _last_index(list(messages), "user")
    start = (user_idx + 1) if user_idx is not None else 0
    return any(
        isinstance(m, dict) and m.get("role") == "tool" and _BOOKED_TRUE.search(message_text(m))
        for m in messages[start:]
    )
