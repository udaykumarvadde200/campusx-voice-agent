"""Paired comparison of two runs: which cases flipped, plus an exact sign test.

    python evals/compare.py evals/results/<before> evals/results/<after>
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path


def load(run: Path) -> dict[tuple, dict]:
    rows = {}
    for name in ("agent.jsonl", "booking.jsonl"):
        f = run / name
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                rows[(r["suite"], r["id"], r["repeat"])] = r
    return rows


def sign_test_p(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs (McNemar, exact)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a_dir: Path, b_dir: Path) -> str:
    A, B = load(a_dir), load(b_dir)
    keys = sorted(set(A) & set(B))
    fixed = [k for k in keys if not A[k]["task_pass"] and B[k]["task_pass"]]
    broke = [k for k in keys if A[k]["task_pass"] and not B[k]["task_pass"]]
    pa = sum(A[k]["task_pass"] for k in keys)
    pb = sum(B[k]["task_pass"] for k in keys)
    lines = [
        f"Paired cases: {len(keys)} (only in A: {len(set(A) - set(B))}, only in B: {len(set(B) - set(A))})",
        f"A task pass: {pa}/{len(keys)} | B task pass: {pb}/{len(keys)}",
        f"Fixed by B: {len(fixed)} | Broken by B: {len(broke)} | exact sign-test p = {sign_test_p(len(broke), len(fixed)):.4f}",
        "",
    ]
    if fixed:
        lines.append("Fixed: " + ", ".join(f"{k[1]}" for k in fixed))
    if broke:
        lines.append("Broken: " + ", ".join(f"{k[1]}" for k in broke))
    sa = json.loads((a_dir / "summary.json").read_text(encoding="utf-8"))["meta"]
    sb = json.loads((b_dir / "summary.json").read_text(encoding="utf-8"))["meta"]
    if sa["dataset_hashes"] != sb["dataset_hashes"]:
        lines.append("\nWARNING: dataset files differ between runs; results are not strictly comparable.")
    if sa.get("kb_hashes") != sb.get("kb_hashes"):
        lines.append("WARNING: knowledge-base files differ between runs.")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(compare(Path(sys.argv[1]), Path(sys.argv[2])))
