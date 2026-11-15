"""Re-score saved eval records with the current checks/datasets, without calling the model.

Use it after fixing a check or a dataset pattern, so an old baseline can be judged by the same rules
as new runs:

    python evals/rescore.py evals/baselines/qwen2.5-7b-baseline
    -> writes evals/baselines/qwen2.5-7b-baseline-rescored/
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))

import checks  # noqa: E402
import report  # noqa: E402

DATASETS = EVALS_DIR / "datasets"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rescore_file(src: Path, cases: dict[str, dict], kind: str) -> list[dict]:
    out = []
    for rec in load_jsonl(src):
        case = cases[rec["id"]]
        if kind == "scenario":
            results = checks.scenario_checks(case, rec["turns"])
        else:
            results = checks.booking_checks(case, rec["turns"], rec.get("bookings", []))
        rec["checks"] = [c.as_dict() for c in results]
        rec["task_pass"] = all(c.passed for c in results if not c.style)
        rec["style_pass"] = all(c.passed for c in results if c.style)
        out.append(rec)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n", encoding="utf-8")


def main(run_dir: Path) -> Path:
    out_dir = run_dir.with_name(run_dir.name + "-rescored")
    out_dir.mkdir(exist_ok=True)
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["meta"]
    meta["run_id"] = meta["run_id"] + " (rescored)"
    meta["dataset_hashes"] = {p.name: hashlib.sha1(p.read_bytes()).hexdigest()[:10] for p in sorted(DATASETS.glob("*.jsonl"))}
    meta["rescored_with_checks"] = hashlib.sha1((EVALS_DIR / "checks.py").read_bytes()).hexdigest()[:10]

    scen_rec, flow_rec = [], []
    if (run_dir / "agent.jsonl").exists():
        cases = {c["id"]: c for c in load_jsonl(DATASETS / "scenarios.jsonl")}
        scen_rec = rescore_file(run_dir / "agent.jsonl", cases, "scenario")
        write_jsonl(out_dir / "agent.jsonl", scen_rec)
    if (run_dir / "booking.jsonl").exists():
        cases = {c["id"]: c for c in load_jsonl(DATASETS / "booking_flows.jsonl")}
        flow_rec = rescore_file(run_dir / "booking.jsonl", cases, "booking")
        write_jsonl(out_dir / "booking.jsonl", flow_rec)

    retr = None
    if (run_dir / "retrieval.json").exists():
        retr = json.loads((run_dir / "retrieval.json").read_text(encoding="utf-8"))
        (out_dir / "retrieval.json").write_text(json.dumps(retr, indent=1, ensure_ascii=False), encoding="utf-8")

    scen_sum = report.summarize_agent(scen_rec) if scen_rec else None
    flow_sum = report.summarize_agent(flow_rec) if flow_rec else None
    summary = {"meta": meta, "scenarios": scen_sum, "booking": flow_sum, "retrieval": retr["summary"] if retr else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = report.render_markdown(meta, scen_sum, flow_sum, retr, report.failures(scen_rec), report.failures(flow_rec))
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    # what changed relative to the original verdicts
    old = {}
    for name in ("agent.jsonl", "booking.jsonl"):
        if (run_dir / name).exists():
            for r in load_jsonl(run_dir / name):
                old[(r["suite"], r["id"], r["repeat"])] = r["task_pass"]
    flips = [(r["suite"], r["id"], old[(r["suite"], r["id"], r["repeat"])], r["task_pass"])
             for r in scen_rec + flow_rec if old.get((r["suite"], r["id"], r["repeat"])) != r["task_pass"]]
    print(f"Rescored {len(scen_rec) + len(flow_rec)} records -> {out_dir}")
    print("Verdict changes (suite, id, old, new):", flips if flips else "none")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
