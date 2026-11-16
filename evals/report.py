"""Turn raw per-case records into summary.json and summary.md."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return round(s[idx], 2)


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval, so a 12-case suite is not read as more precise than it is."""
    if not total:
        return (float("nan"), float("nan"))
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0, centre - half), 3), round(min(1, centre + half), 3))


def summarize_agent(records: list[dict]) -> dict:
    out: dict = {"n_runs": len(records)}
    if not records:
        return out
    tp = sum(r["task_pass"] for r in records)
    sp = sum(r["style_pass"] for r in records)
    out["task_pass"] = {"passed": tp, "total": len(records), "ci95": wilson(tp, len(records))}
    out["style_pass"] = {"passed": sp, "total": len(records)}

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    out["by_category"] = {
        c: {"passed": sum(x["task_pass"] for x in rs), "total": len(rs)} for c, rs in sorted(by_cat.items())
    }

    by_check: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        # one verdict per check name per case (a case passes a check name only if all instances pass)
        per_case: dict[str, bool] = {}
        for c in r["checks"]:
            per_case[c["name"]] = per_case.get(c["name"], True) and c["passed"]
        for name, ok in per_case.items():
            by_check[name].append(ok)
    out["by_check"] = {n: {"passed": sum(v), "total": len(v)} for n, v in sorted(by_check.items())}

    lat = [t["latency_s"] for r in records for t in r["turns"]]
    out["latency_s"] = {"p50": percentile(lat, 50), "p95": percentile(lat, 95), "max": round(max(lat), 2), "turns": len(lat)}
    out["round_limit_hits"] = sum(t.get("hit_round_limit", False) for r in records for t in r["turns"])

    # Section B: how much of the result comes from the safety nets rather than the model itself.
    turns = [t for r in records for t in r["turns"]]
    out["guard_interventions"] = dict(Counter(i for t in turns for i in t.get("guard", [])))
    lookups = [t["retrieval"] for t in turns if t.get("retrieval")]
    out["auto_retrieval"] = {
        "turns": len(turns),
        "passages_attached": sum(bool(x.get("found")) for x in lookups),
        "not_found_note": sum(x.get("injected") == "not_found" for x in lookups),
    }
    out["spoken_differs_from_raw"] = sum(1 for t in turns if t.get("raw_text") not in (None, t["text"]))
    return out


def failures(records: list[dict], limit: int = 40) -> list[dict]:
    rows = []
    for r in records:
        bad = [c for c in r["checks"] if not c["passed"] and not c["style"]]
        if bad:
            last = r["turns"][-1]["text"] if r["turns"] else ""
            rows.append({"id": r["id"], "category": r["category"], "problems": [f"{c['name']}: {c['detail']}" for c in bad], "reply": last[:300]})
    return rows[:limit]


def render_markdown(meta: dict, scen: dict | None, flows: dict | None, retr: dict | None,
                    scen_fail: list[dict], flow_fail: list[dict]) -> str:
    L: list[str] = []
    L.append(f"# Eval run: {meta['run_id']}\n")
    L.append(f"- Model: `{meta.get('model')}` | temperature {meta.get('temperature')} | repeats {meta.get('repeats')}")
    L.append(f"- Cold-start (first request incl. model load): {meta.get('warmup_s', 'n/a')} s")
    L.append(f"- Dataset hashes: {meta.get('dataset_hashes')}\n")

    if retr:
        s = retr["summary"]
        k = s["k"]
        L.append("## Retrieval (local keyword search)\n")
        L.append(f"- Answerable queries: {s['n_answerable']} | hit@1 {s['hit_at_1']} | hit@{k} {s[f'hit_at_{k}']} | MRR {s['mrr']}")
        L.append(f"- Diagnoses of answerable queries: {s['diagnoses']}")
        L.append(f"- Unanswerable queries returning results anyway: {s['unanswerable_with_any_result']}/{s['n_unanswerable']}")
        L.append(f"- Mean top score: answerable {s['mean_top_score_answerable']} vs unanswerable {s['mean_top_score_unanswerable']}")
        L.append(f"- Best score cut-off for 'can answer?': {s['answerable_vs_unanswerable_cutoff']}\n")

    for title, s in (("Agent scenarios", scen), ("Booking flows", flows)):
        if not s:
            continue
        tp = s["task_pass"]
        lo, hi = tp["ci95"]
        L.append(f"## {title}\n")
        L.append(f"- Task pass: {tp['passed']}/{tp['total']} ({pct(tp['passed'], tp['total'])}), 95% CI {lo}-{hi}")
        L.append(f"- Style pass (length, no markdown, no leaked tool JSON): {s['style_pass']['passed']}/{s['style_pass']['total']}")
        lat = s["latency_s"]
        L.append(f"- Brain latency per user turn (LLM + tools, no STT/TTS): p50 {lat['p50']} s, p95 {lat['p95']} s, max {lat['max']} s")
        if "auto_retrieval" in s and s["auto_retrieval"]["turns"]:
            ar = s["auto_retrieval"]
            L.append(f"- Auto retrieval: passages attached on {ar['passages_attached']}/{ar['turns']} turns, "
                     f"'nothing matched' note on {ar['not_found_note']}")
            L.append(f"- Reply guard changed the spoken text on {s.get('spoken_differs_from_raw', 0)} turns: "
                     f"{s.get('guard_interventions') or 'no interventions'}")
        if s.get("round_limit_hits"):
            L.append(f"- Turns that hit the tool-round limit with no reply: {s['round_limit_hits']}")
        L.append("\n| Category | Passed | Total | Rate |\n|---|---|---|---|")
        for c, v in s["by_category"].items():
            L.append(f"| {c} | {v['passed']} | {v['total']} | {pct(v['passed'], v['total'])} |")
        L.append("\n| Check | Passed | Total | Rate |\n|---|---|---|---|")
        for c, v in s["by_check"].items():
            L.append(f"| {c} | {v['passed']} | {v['total']} | {pct(v['passed'], v['total'])} |")
        L.append("")

    for title, rows in (("Failed scenarios", scen_fail), ("Failed booking flows", flow_fail)):
        if rows:
            L.append(f"## {title}\n")
            for r in rows:
                L.append(f"- **{r['id']}** ({r['category']}): " + "; ".join(r["problems"]))
                L.append(f"  - last reply: {r['reply']!r}")
            L.append("")
    return "\n".join(L)
