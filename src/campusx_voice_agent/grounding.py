"""Ground every caller question in the knowledge base BEFORE the model answers.

The 7B model almost never chose to call the search tool (2 of 34 knowledge questions in the
baseline) and answered from memory instead. Now the code searches on every turn and attaches
the result as a system message right after the caller's message. The model can still call the
search tool for a second look.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from campusx_voice_agent.booking_gate import message_text
from campusx_voice_agent.knowledge_base import _tokens, search_knowledge_base

KB_MARKER = "[KB-CONTEXT]"
MAX_PASSAGES = 3
MIN_TERMS_FOR_STANDALONE_QUERY = 2

_QUESTION_LIKE = re.compile(
    r"(\?|^\W*(?:what|which|who|whom|when|where|why|how|is|are|was|were|do|does|did|can|could|will|"
    r"would|should|tell|explain|any|anything|list|give me|show)\b)",
    re.IGNORECASE,
)


def _is_kb_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "system"
        and message_text(message).startswith(KB_MARKER)
    )


def _user_texts(messages: list[Any]) -> list[str]:
    return [message_text(m) for m in messages if isinstance(m, dict) and m.get("role") == "user"]


def build_query(messages: list[Any]) -> str:
    """The caller's latest words; a short follow-up ("and trees?") borrows the previous question."""
    texts = [t.strip() for t in _user_texts(messages) if t.strip()]
    if not texts:
        return ""
    latest = texts[-1]
    if len(_tokens(latest)) < MIN_TERMS_FOR_STANDALONE_QUERY and len(texts) >= 2:
        return f"{texts[-2]} {latest}"
    return latest


def format_found(result: dict[str, Any]) -> str:
    lines = [
        KB_MARKER,
        "Knowledge base passages for the caller's latest message. "
        "State CampusX facts only from these:",
    ]
    for i, item in enumerate(result["results"][:MAX_PASSAGES], start=1):
        snippet = item["snippet"].replace("**", "").replace("__", "").replace("`", "")
        lines.append(f"{i}. {snippet} (source: {item['source']})")
    unmatched = result.get("unmatched_terms") or []
    if unmatched:
        lines.append(
            f"Not in the knowledge base at all: {', '.join(unmatched)}. "
            "Do not claim anything about these."
        )
    return "\n".join(lines)


def format_not_found(result: dict[str, Any]) -> str:
    lines = [
        KB_MARKER,
        "No knowledge base passage matches the caller's latest message. "
        "If they asked for a CampusX fact, "
        "say you cannot verify it and point them to support@campusx.in. Do not guess.",
    ]
    unmatched = result.get("unmatched_terms") or []
    if unmatched:
        lines.append(f"Not in the knowledge base at all: {', '.join(unmatched)}.")
    return "\n".join(lines)


def inject_kb_context(messages: list[Any]) -> dict[str, Any]:
    """Replace any earlier KB message with a fresh one for the latest caller message.

    Mutates ``messages`` in place and returns a small record for logging and the evals.
    Idempotent: running it twice on the same turn gives the same list.
    """
    messages[:] = [m for m in messages if not _is_kb_message(m)]
    query = build_query(messages)
    info: dict[str, Any] = {
        "query": query,
        "found": False,
        "injected": None,
        "sources": [],
        "top_score": None,
    }
    if not query:
        return info

    result = search_knowledge_base(query, limit=MAX_PASSAGES)
    latest = (_user_texts(messages) or [""])[-1]
    if result["found"]:
        info.update(found=True, injected="passages")
        info["sources"] = [r["source"] for r in result["results"]]
        info["top_score"] = result["results"][0]["score"]
        text = format_found(result)
    elif _QUESTION_LIKE.search(latest):
        info["injected"] = "not_found"
        text = format_not_found(result)
    else:
        return info  # small talk, names, emails, slot choices: nothing to look up
    messages.append({"role": "system", "content": text})
    info["unmatched_terms"] = result.get("unmatched_terms", [])
    return info


class KnowledgeContextProcessor(FrameProcessor):
    """Pipecat wrapper: attach knowledge-base passages to the context before the LLM runs."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            try:
                messages = list(frame.context.get_messages())
                info = inject_kb_context(messages)
                frame.context.set_messages(messages)
                logger.info(
                    "KB lookup: found={} injected={} query={!r}",
                    info["found"],
                    info["injected"],
                    info["query"],
                )
            except Exception as exc:  # noqa: BLE001 - never let retrieval take the call down
                logger.exception("KB lookup failed: {}", exc)
        await self.push_frame(frame, direction)
