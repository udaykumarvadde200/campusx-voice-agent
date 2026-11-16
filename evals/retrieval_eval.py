"""Retrieval quality of the local knowledge-base search (the agent's tool and fallback path)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from campusx_voice_agent import knowledge_base as kb  # noqa: E402


def _contains_all(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return all(k in low for k in keywords)


def _section_has_keywords(kb_dir: Path, source: str, keywords: list[str]) -> tuple[bool, bool]:
    """(some section contains the keywords, and its returned snippet would keep them)."""
    text = (kb_dir / source).read_text(encoding="utf-8")
    # "kept" now means: some retrievable chunk of that file holds all the keywords in its snippet.
    chunks = kb._load_corpus(kb_dir).chunks
    kept = any(c.source == source and _contains_all(c.snippet, keywords) for c in chunks)
    for section in kb._sections(text):
        if _contains_all(section, keywords):
            return True, kept
    return False, False


def evaluate_item(item: dict, k: int, kb_dir: Path) -> dict:
    full = kb.search_knowledge_base(item["query"], limit=10**6, kb_dir=kb_dir)["results"]
    ranked = full
    top_score = ranked[0]["score"] if ranked else 0
    rec = {
        "id": item["id"],
        "query": item["query"],
        "answerable": item["answerable"],
        "top_score": top_score,
        "n_results": len(ranked),
        "top_sources": [r["source"] for r in ranked[:k]],
        "top_scores": [r["score"] for r in ranked[:k]],
    }
    if not item["answerable"]:
        return rec

    gold = item["gold_keywords"]
    rank = next(
        (i for i, r in enumerate(ranked, 1) if r["source"] == item["gold_source"] and _contains_all(r["snippet"], gold)),
        None,
    )
    rec["rank"] = rank
    rec["hit_at_1"] = rank == 1
    rec["hit_at_k"] = rank is not None and rank <= k
    rec["rr"] = (1.0 / rank) if rank else 0.0
    if rec["hit_at_k"]:
        rec["diagnosis"] = "hit"
    elif rank is not None:
        rec["diagnosis"] = "found_but_ranked_below_k"
    else:
        exists, kept = _section_has_keywords(kb_dir, item["gold_source"], gold)
        if exists and not kept:
            rec["diagnosis"] = "snippet_truncated"  # right section, but the 900-char cut dropped the answer
        elif exists:
            rec["diagnosis"] = "section_not_scored"  # query shares no tokens with the right section
        else:
            rec["diagnosis"] = "gold_not_in_source"  # dataset label problem
    return rec


def best_threshold(answerable: list[float], unanswerable: list[float]) -> dict:
    """Best score cut-off for 'is this answerable?' on this dataset (accuracy)."""
    if not answerable or not unanswerable:
        return {}
    best = {"threshold": 0, "accuracy": 0.0}
    n = len(answerable) + len(unanswerable)
    for t in sorted(set(answerable + unanswerable)):
        correct = sum(s >= t for s in answerable) + sum(s < t for s in unanswerable)
        acc = correct / n
        if acc > best["accuracy"]:
            best = {"threshold": t, "accuracy": round(acc, 3)}
    return best


def run_retrieval(dataset: list[dict], k: int = 4, kb_dir: Path | None = None) -> dict:
    kb_dir = kb_dir or kb._knowledge_base_dir()
    records = [evaluate_item(item, k, kb_dir) for item in dataset]
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    diagnoses: dict[str, int] = {}
    for r in ans:
        diagnoses[r["diagnosis"]] = diagnoses.get(r["diagnosis"], 0) + 1
    ans_scores = [r["top_score"] for r in ans]
    una_scores = [r["top_score"] for r in una]
    summary = {
        "k": k,
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "hit_at_1": round(sum(r["hit_at_1"] for r in ans) / max(1, len(ans)), 3),
        f"hit_at_{k}": round(sum(r["hit_at_k"] for r in ans) / max(1, len(ans)), 3),
        "mrr": round(sum(r["rr"] for r in ans) / max(1, len(ans)), 3),
        "diagnoses": diagnoses,
        "mean_top_score_answerable": round(sum(ans_scores) / max(1, len(ans_scores)), 2),
        "mean_top_score_unanswerable": round(sum(una_scores) / max(1, len(una_scores)), 2),
        "unanswerable_with_any_result": sum(r["n_results"] > 0 for r in una),
        "answerable_vs_unanswerable_cutoff": best_threshold(ans_scores, una_scores),
    }
    return {"summary": summary, "records": records}
