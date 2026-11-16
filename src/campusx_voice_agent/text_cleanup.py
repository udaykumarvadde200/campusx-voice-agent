"""Make model text safe to speak: remove markdown, keep the words."""

from __future__ import annotations

import re

_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022+]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
_QUOTE = re.compile(r"^\s*>\s?")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_ITALIC_UNDER = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])")
_TERMINAL = ".!?:;"


def strip_markdown_line(line: str) -> str:
    """Remove markdown from one line. Returns "" if nothing speakable is left."""
    text = _LIST_MARKER.sub("", line)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _BOLD.sub(r"\2", text)
    text = _ITALIC_STAR.sub(r"\1", text)
    text = _ITALIC_UNDER.sub(r"\1", text)
    text = text.replace("|", ", ")
    text = re.sub(r"[*`#]+", "", text)  # stray markers
    text = re.sub(r"^\s*[-=_]{3,}\s*$", "", text)  # horizontal rules
    return re.sub(r"\s+", " ", text).strip(" ,")


def strip_markdown(text: str) -> str:
    """Whole-text version: code fences dropped, one spoken unit per line, order kept."""
    text = _FENCE.sub(" ", text)
    lines = [strip_markdown_line(line) for line in text.splitlines()]
    return " ".join(ensure_terminal(line) for line in lines if line)


def ensure_terminal(sentence: str) -> str:
    """Give a spoken unit closing punctuation so TTS pauses after it."""
    sentence = sentence.strip()
    if sentence and sentence[-1] not in _TERMINAL:
        return sentence + "."
    return sentence
