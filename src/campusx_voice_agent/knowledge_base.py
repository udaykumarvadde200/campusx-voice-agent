"""Local Markdown knowledge-base retrieval for the CampusX voice agent.

What changed in Section B (see HANDOFF.md, finding 4):

* Chunking: every table row becomes its own chunk (so late rows such as 31 August are never
  cut off), long sections are split into windows instead of being truncated, and every chunk
  carries its file name and heading path as context.
* Scoring: BM25 over stemmed content words. Stopwords and conversational filler are removed.
* "Not found" is possible: a chunk is only returned when it covers enough of what the caller
  asked (``MIN_COVERAGE``). Numbers in the query (dates, session numbers) must match exactly,
  and a number alone is never enough. Without this a question about "9 August" matches the
  heading "## 9. Recursion".
* Sections that are written for the agent (how to use this file, example Q&A, purpose notes)
  are not facts about CampusX and are never returned.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from campusx_voice_agent.config import DATA_DIR

SUPPORTED_EXTENSIONS = {".md", ".txt"}
DEFAULT_KB_DIR = DATA_DIR / "knowledge_base"

# A chunk is returned only if it matches at least this share (by idf weight) of the caller's
# content words. Chosen on evals/datasets/retrieval.jsonl and checked on
# evals/heldout/retrieval_heldout.jsonl; see tests/test_brain_fixes.py for the sensitivity check.
MIN_COVERAGE = 0.5
MAX_CHUNK_CHARS = 800
MAX_SNIPPET_CHARS = 1000

# Headings of sections that are instructions or examples for the agent, not CampusX facts.
NON_FACT_HEADING = re.compile(
    r"(how the voice agent should use|knowledge base vs system prompt|knowledge base purpose"
    r"|^purpose$|example questions)",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
# Speech-to-text splits names the knowledge base writes as one word.
_ALIASES = (
    (re.compile(r"\bllm\s+ops\b"), "llmops"),
    (re.compile(r"\bgen\s+ai\b"), "genai"),
)
_CONTRACTION_S = re.compile(r"(?<=[a-z])['\u2019]s\b")
_CONTRACTION_OTHER = re.compile(r"['\u2019](?:re|ve|ll|d|m)\b")
_CONTRACTION_NT = re.compile(r"n['\u2019]t\b")

# Spoken numbers ("twenty seven", "fourth") become digits so they match "27" and "4" in the
# knowledge base. "one", "first" and "second" are left alone: they are mostly not numbers.
_UNITS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ORDINAL_WORDS = {
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_ORDINAL_UNITS = {"first": 1, "second": 2, **_ORDINAL_WORDS}
_NUMBER_WORD = re.compile(
    r"\b(?:("
    + "|".join(_TENS)
    + r")(?:[\s-]+("
    + "|".join(list(_UNITS) + ["one"])
    + r"|"
    + "|".join(_ORDINAL_UNITS)
    + r"))?|("
    + "|".join(_UNITS)
    + r")|("
    + "|".join(_ORDINAL_WORDS)
    + r"))\b"
)


def _spoken_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tens, unit, plain, ordinal = match.groups()
        if tens:
            value = _TENS[tens]
            if unit:
                value += 1 if unit == "one" else _UNITS.get(unit, _ORDINAL_UNITS.get(unit, 0))
            return str(value)
        if plain:
            return str(_UNITS[plain])
        return str(_ORDINAL_WORDS[ordinal])

    return _NUMBER_WORD.sub(repl, text)


# Function words, question words and spoken filler. Course-content verbs that callers use
# generically ("do you teach / cover / include ...") are listed too: they say nothing about
# WHICH topic is asked. Deliberately NOT listed: "long", "much", "many", "time", "cost",
# "price", "fee", "weeks": those carry the question ("how long", "how much") and must count
# as unmatched when the knowledge base has no such information.
STOPWORDS = frozenset(
    """
    a an the and or but nor so if then than as of to in on at by for from with without about into
    over under up down out off again also too very just only even still yet
    i me my mine we us our you your yours he she it its they them their this that these those
    am is are was were be been being do does did doing done have has had having
    can could may might must shall should will would
    what which who whom whose when where why how
    there here any some all each every both either neither no not
    tell say said ask please hey hi hello ok okay yes yeah sure thanks thank
    want need know wanna gonna get got let lets go going
    guys guy stuff thing things bit kind sort like really actually basically
    learn learning learned teach teaches taught teaching cover covers covered covering
    include includes included including offer offers offered provide provides provided
    available talk mention discuss
    anything something everything anyone someone everyone else other another one ones
    suitable good great useful helpful right best worth fit mean means meaning
    """.split()
)


@dataclass(frozen=True)
class KnowledgeMatch:
    source: str
    title: str
    snippet: str
    score: float
    coverage: float = 1.0

    def as_dict(self) -> dict[str, str | float]:
        return {
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class _Chunk:
    source: str
    title: str
    snippet: str
    terms: tuple[str, ...]  # stemmed content words, in order, with repeats (term frequency)
    pairs: frozenset[tuple[str, str]] = frozenset()  # adjacent word pairs, for phrase matches


@dataclass
class _Corpus:
    chunks: list[_Chunk]
    df: dict[str, int] = field(default_factory=dict)
    avg_len: float = 1.0

    def idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))


# --------------------------------------------------------------------------- text helpers


def _knowledge_base_dir() -> Path:
    configured = os.getenv("CAMPUSX_KNOWLEDGE_BASE_DIR")
    return Path(configured) if configured else DEFAULT_KB_DIR


def _stem(word: str) -> str:
    """Tiny deterministic stemmer: graphs/graph, trees/tree, timing/time, cancelled/cancel."""
    if word.isdigit() or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _normalize(text: str) -> str:
    lowered = text.lower()
    lowered = _CONTRACTION_NT.sub(" not", lowered)
    lowered = _CONTRACTION_S.sub("", lowered)
    lowered = _CONTRACTION_OTHER.sub("", lowered)
    lowered = _spoken_numbers(lowered)
    for pattern, replacement in _ALIASES:
        lowered = pattern.sub(replacement, lowered)
    return _ORDINAL.sub(r"\1", lowered)


def _tokens(text: str) -> set[str]:
    """Distinct stemmed content words of ``text`` (stopwords removed)."""
    return set(_content_terms(text))


def _content_terms(text: str) -> list[str]:
    words = _TOKEN_PATTERN.findall(_normalize(text))
    return [_stem(w) for w in words if w not in STOPWORDS]


def _title_for(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _sections(text: str) -> list[str]:
    """Heading-level sections (kept for the retrieval eval's diagnosis)."""
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _compact(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."


# --------------------------------------------------------------------------- chunking


def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[\s|:\-]+", line.strip())) and "-" in line


def _split_body(body_lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split a section body into ("table", lines) and ("text", lines) blocks."""
    blocks: list[tuple[str, list[str]]] = []
    for line in body_lines:
        kind = "table" if _is_table_line(line) else "text"
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(line)
        else:
            blocks.append((kind, [line]))
    return blocks


def _text_windows(lines: list[str]) -> list[str]:
    """Group paragraphs into windows of at most MAX_CHUNK_CHARS (never cut mid-paragraph)."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append("\n".join(current))
                current = []
        elif line.strip() == "---":
            continue
        else:
            current.append(line)
    if current:
        paragraphs.append("\n".join(current))

    windows: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if buffer and len(candidate) > MAX_CHUNK_CHARS:
            windows.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        windows.append(buffer)
    return windows


def _make_chunk(path: Path, title: str, breadcrumb: str, body: str) -> _Chunk:
    context = f"{path.stem.replace('_', ' ')} | {breadcrumb}"
    snippet = _compact(f"{breadcrumb}: {body}" if breadcrumb else body)
    terms = tuple(_content_terms(f"{context} {body}"))
    return _Chunk(
        source=path.name,
        title=title,
        snippet=snippet,
        terms=terms,
        pairs=frozenset(zip(terms, terms[1:], strict=False)),
    )


def _chunk_file(path: Path, text: str) -> list[_Chunk]:
    title = _title_for(path, text)
    chunks: list[_Chunk] = []
    stack: dict[int, str] = {}
    body: list[str] = []

    def flush() -> None:
        headings = [stack[level] for level in sorted(stack)]
        if not any(body) or any(NON_FACT_HEADING.search(h) for h in headings):
            return
        breadcrumb = " > ".join(headings)
        for kind, lines in _split_body(body):
            if kind == "table":
                rows = [ln for ln in lines if not _is_separator(ln)]
                if len(rows) < 2:
                    continue
                header = _cells(rows[0])
                for row in rows[1:]:
                    cells = _cells(row)
                    pairs = [f"{h}: {c}" for h, c in zip(header, cells, strict=False) if c]
                    chunks.append(_make_chunk(path, title, breadcrumb, " | ".join(pairs)))
            else:
                for window in _text_windows(lines):
                    chunks.append(_make_chunk(path, title, breadcrumb, window))

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if match:
            flush()
            body = []
            level = len(match.group(1))
            for deeper in [lv for lv in stack if lv >= level]:
                del stack[deeper]
            stack[level] = match.group(2).strip()
        else:
            body.append(line)
    flush()
    return chunks


_CACHE: dict[tuple, _Corpus] = {}


def _load_corpus(root: Path) -> _Corpus:
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    signature = (str(root), tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files))
    cached = _CACHE.get(signature)
    if cached is not None:
        return cached

    chunks: list[_Chunk] = []
    for path in files:
        chunks.extend(_chunk_file(path, path.read_text(encoding="utf-8")))
    df: dict[str, int] = {}
    for chunk in chunks:
        for term in set(chunk.terms):
            df[term] = df.get(term, 0) + 1
    avg_len = sum(len(c.terms) for c in chunks) / max(1, len(chunks))
    corpus = _Corpus(chunks=chunks, df=df, avg_len=avg_len or 1.0)
    _CACHE.clear()  # keep only the current knowledge base in memory
    _CACHE[signature] = corpus
    return corpus


# --------------------------------------------------------------------------- scoring


def _bm25(
    corpus: _Corpus,
    query_terms: set[str],
    query_pairs: set[tuple[str, str]],
    chunk: _Chunk,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    length_norm = 1 - b + b * len(chunk.terms) / corpus.avg_len
    score = 0.0
    for term in query_terms:
        tf = chunk.terms.count(term)
        if tf:
            score += corpus.idf(term) * tf * (k1 + 1) / (tf + k1 * length_norm)
    # Phrase bonus: "3 august" in the question beats a row that merely contains a 3 somewhere.
    for first, second in query_pairs & chunk.pairs:
        score += 0.5 * (corpus.idf(first) + corpus.idf(second))
    return score


def _coverage(corpus: _Corpus, query_terms: set[str], chunk_terms: set[str]) -> float:
    """Share of the caller's content words (weighted by rarity) that this chunk contains."""
    total = sum(corpus.idf(t) for t in query_terms)
    if total <= 0:
        return 0.0
    return sum(corpus.idf(t) for t in query_terms if t in chunk_terms) / total


def search_knowledge_base(
    query: str,
    *,
    limit: int = 4,
    kb_dir: Path | None = None,
    min_coverage: float | None = None,
) -> dict[str, object]:
    """Search local CampusX knowledge files and return short verified snippets.

    ``found`` is False when nothing covers enough of the question. ``unmatched_terms`` lists the
    caller's content words that appear nowhere in the knowledge base, so the agent can say what
    it could not verify instead of guessing.
    """
    cleaned_query = query.strip()
    root = kb_dir or _knowledge_base_dir()
    gate = MIN_COVERAGE if min_coverage is None else min_coverage
    base_result: dict[str, object] = {
        "query": cleaned_query,
        "knowledge_base_dir": str(root),
        "current_date": date.today().isoformat(),
        "timezone": "Asia/Kolkata",
    }

    if not cleaned_query:
        return {
            **base_result,
            "found": False,
            "message": "No search query was provided.",
            "results": [],
        }
    if not root.exists():
        return {
            **base_result,
            "found": False,
            "message": "Knowledge base directory does not exist.",
            "results": [],
        }

    corpus = _load_corpus(root)
    query_terms = _tokens(cleaned_query)
    if not query_terms:
        return {
            **base_result,
            "found": False,
            "message": "The query has no searchable words.",
            "results": [],
            "unmatched_terms": [],
        }

    ordered = _content_terms(cleaned_query)
    query_pairs = set(zip(ordered, ordered[1:], strict=False))
    numbers = {t for t in query_terms if t.isdigit()}
    words = query_terms - numbers
    if not words:  # "9" or "13" on its own says nothing about what the caller wants
        return {
            **base_result,
            "found": False,
            "message": "The query has only numbers; there is nothing to look up.",
            "results": [],
            "unmatched_terms": [],
        }
    vocabulary = set(corpus.df)
    unmatched = sorted(t for t in query_terms if t not in vocabulary)

    scored: list[tuple[float, float, _Chunk]] = []
    for chunk in corpus.chunks:
        chunk_terms = set(chunk.terms)
        if numbers and not numbers <= chunk_terms:
            continue  # dates and session numbers must match exactly
        if words and not (words & chunk_terms):
            continue  # a number alone ("9") is not evidence
        coverage = _coverage(corpus, query_terms, chunk_terms)
        if coverage < gate:
            continue
        scored.append((_bm25(corpus, query_terms, query_pairs, chunk), coverage, chunk))

    scored.sort(key=lambda item: (-item[0], item[2].source, item[2].snippet))
    matches = [
        KnowledgeMatch(
            source=c.source,
            title=c.title,
            snippet=c.snippet,
            score=round(s, 3),
            coverage=round(cov, 3),
        )
        for s, cov, c in scored[: max(1, limit)]
    ]
    return {
        **base_result,
        "found": bool(matches),
        "message": (
            "Use these verified knowledge-base snippets when answering."
            if matches
            else "No verified knowledge-base result matched the query."
        ),
        "results": [match.as_dict() for match in matches],
        "unmatched_terms": unmatched,
    }
