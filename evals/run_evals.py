"""CLI: run the baseline evals against the local Ollama model.

    python evals/run_evals.py --suite all --model qwen2.5:7b --tag baseline
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))

import checks  # noqa: E402
import harness  # noqa: E402
import report  # noqa: E402
import retrieval_eval  # noqa: E402

DATASETS = EVALS_DIR / "datasets"
RESULTS = EVALS_DIR / "results"


def load_jsonl(name: str) -> list[dict]:
    with open(DATASETS / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def ollama_root(base_url: str) -> str:
    return base_url.removesuffix("/v1").rstrip("/")


def ollama_version(base_url: str) -> str:
    try:
        with urllib.request.urlopen(f"{ollama_root(base_url)}/api/version", timeout=5) as r:
            return json.load(r).get("version", "unknown")
    except Exception:  # noqa: BLE001
        return "unreachable"


def select(rows: list[dict], ids: str | None, limit: int | None) -> list[dict]:
    if ids:
        wanted = {i.strip() for i in ids.split(",")}
        rows = [r for r in rows if r["id"] in wanted or r.get("category") in wanted]
    return rows[:limit] if limit else rows


def log(msg: str) -> None:
    print(msg, flush=True)


async def run_scenarios(cases, chat, schema, repeats, **brain) -> list[dict]:
    records = []
    for case in cases:
        for rep in range(repeats):
            harness.new_isolated_db()
            turns = await harness.run_conversation(case["turns"], chat, schema, **brain)
            results = checks.scenario_checks(case, turns)
            task = [c for c in results if not c.style]
            style = [c for c in results if c.style]
            rec = {
                "suite": "scenario", "id": case["id"], "category": case["category"], "repeat": rep,
                "turns": turns, "checks": [c.as_dict() for c in results],
                "task_pass": all(c.passed for c in task), "style_pass": all(c.passed for c in style),
            }
            records.append(rec)
            mark = "PASS" if rec["task_pass"] else "FAIL"
            log(f"[scenario] {mark} {case['id']} (style {'ok' if rec['style_pass'] else 'bad'}) {turns[-1]['latency_s']}s")
    return records


async def run_flows(cases, chat, schema, repeats, **brain) -> list[dict]:
    records = []
    for case in cases:
        for rep in range(repeats):
            db = harness.new_isolated_db()
            turns = await harness.run_conversation(case["turns"], chat, schema, **brain)
            rows = harness.read_bookings(db)
            results = checks.booking_checks(case, turns, rows)
            task = [c for c in results if not c.style]
            style = [c for c in results if c.style]
            rec = {
                "suite": "booking", "id": case["id"], "category": "booking_flow", "repeat": rep,
                "turns": turns, "bookings": rows, "checks": [c.as_dict() for c in results],
                "task_pass": all(c.passed for c in task), "style_pass": all(c.passed for c in style),
            }
            records.append(rec)
            mark = "PASS" if rec["task_pass"] else "FAIL"
            log(f"[booking]  {mark} {case['id']}")
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


async def main_async(args) -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{args.model.replace(':', '-').replace('/', '-')}_{args.tag}"
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)

    kb_dir = harness.PROJECT_DIR / "data" / "knowledge_base"
    meta = {
        "run_id": run_id, "model": args.model, "base_url": args.base_url, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "seed": args.seed, "repeats": args.repeats,
        "ollama_version": ollama_version(args.base_url), "python": platform.python_version(),
        "dataset_hashes": {p.name: sha(p) for p in sorted(DATASETS.glob("*.jsonl"))},
        "kb_hashes": {p.name: sha(p) for p in sorted(kb_dir.glob("*.md"))},
        "prompt_hash": hashlib.sha1(harness.GUIDANCE_SYSTEM_PROMPT.encode()).hexdigest()[:10],
    }

    brain = {
        "grounding": not args.no_grounding,
        "guard": not args.no_guard,
        "booking_state": not args.no_booking_state,
    }
    meta["brain"] = brain  # what was switched on for this run (needed to read an ablation correctly)

    retr = scen_sum = flow_sum = None
    scen_rec: list[dict] = []
    flow_rec: list[dict] = []

    if args.suite in ("all", "retrieval"):
        ds = select(load_jsonl("retrieval.jsonl"), args.ids, args.limit)
        retr = retrieval_eval.run_retrieval(ds, k=args.k, kb_dir=kb_dir)
        (out / "retrieval.json").write_text(json.dumps(retr, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"[retrieval] hit@1={retr['summary']['hit_at_1']} MRR={retr['summary']['mrr']}")

    if args.suite in ("all", "scenarios", "booking"):
        chat = harness.make_ollama_chat(args.base_url, args.model, args.temperature, args.max_tokens, args.seed)
        schema = harness.tools_schema()
        t0 = time.perf_counter()
        await chat([{"role": "user", "content": "Hello"}], schema)  # warm-up; measures cold start
        meta["warmup_s"] = round(time.perf_counter() - t0, 2)
        log(f"[warmup] first request took {meta['warmup_s']}s")

        if args.suite in ("all", "scenarios"):
            cases = select(load_jsonl("scenarios.jsonl"), args.ids, args.limit)
            scen_rec = await run_scenarios(cases, chat, schema, args.repeats, **brain)
            write_jsonl(out / "agent.jsonl", scen_rec)
            scen_sum = report.summarize_agent(scen_rec)
        if args.suite in ("all", "booking"):
            cases = select(load_jsonl("booking_flows.jsonl"), args.ids, args.limit)
            flow_rec = await run_flows(cases, chat, schema, args.repeats, **brain)
            write_jsonl(out / "booking.jsonl", flow_rec)
            flow_sum = report.summarize_agent(flow_rec)

    summary = {"meta": meta, "scenarios": scen_sum, "booking": flow_sum,
               "retrieval": retr["summary"] if retr else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = report.render_markdown(meta, scen_sum, flow_sum, retr,
                                report.failures(scen_rec), report.failures(flow_rec))
    (out / "summary.md").write_text(md, encoding="utf-8")
    log(f"\nResults written to {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["all", "retrieval", "scenarios", "booking"], default="all")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--temperature", type=float, default=0.3, help="same as the live agent")
    p.add_argument("--max-tokens", type=int, default=220, help="same as the live agent")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--repeats", type=int, default=1, help="run each case N times (LLMs are noisy)")
    p.add_argument("--k", type=int, default=4, help="retrieval cut-off (the tool returns 4)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default=None, help="comma-separated case ids or categories")
    p.add_argument("--no-grounding", action="store_true", help="ablation: no search before the model answers")
    p.add_argument("--no-guard", action="store_true", help="ablation: speak the model's raw text")
    p.add_argument("--no-booking-state", action="store_true", help="ablation: always offer the booking tool")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default=str(RESULTS))
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
