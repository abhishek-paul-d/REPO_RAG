# ruff: noqa: SIM115, B905, E731, RUF015, B023
"""Paired analysis of the header run: header_on minus header_off, per policy and subset.

Usage: PYTHONPATH=. uv run python eval/analyze_header_run.py eval/retrieval_results/phase4_header_run
"""

import json
import sys
from pathlib import Path

from eval.retrieval.statistics import paired_intervals
from repo_rag.config import Settings

P = Path(sys.argv[1])
s = Settings()


def load(v):
    r = []
    for d in ["psf__requests", "expressjs__express", "pallets__flask"]:
        r += json.load(open(P / f"{d}__header_{v}" / "per_question.json"))
    return r


off, on = load("off"), load("on")
f = lambda m: f"{m['estimate']:+.3f} [{m['low']:+.3f},{m['high']:+.3f}]"
subs = {
    "all120": lambda r: True,
    "orig90": lambda r: not r["paraphrase"],
    "para30": lambda r: r["paraphrase"],
    "ambiguous": lambda r: r["stratum"] == "ambiguous",
    "multi_block": lambda r: r["stratum"] == "multi_block",
    "implementation": lambda r: r["stratum"] == "implementation",
    "test": lambda r: r["stratum"] == "test",
    "example": lambda r: r["stratum"] == "example",
}
repos = {
    "requests": lambda r: r["id"].startswith("requests"),
    "flask": lambda r: r["id"].startswith("flask"),
    "express": lambda r: r["id"].startswith("express"),
}
for m in ["hybrid", "plain"]:
    print(f"\n=== {m}: header_on minus header_off  (MRR | complete proof | better/worse)")
    for name, fn in {**subs, **repos}.items():
        a = sorted([r for r in on if r["method"] == m and fn(r)], key=lambda r: r["id"])
        b = sorted([r for r in off if r["method"] == m and fn(r)], key=lambda r: r["id"])
        assert [x["id"] for x in a] == [x["id"] for x in b]
        iv = paired_intervals(a, b, ["mrr@10", "any_complete_proof@10"], s)
        bt = sum(x["mrr@10"] > y["mrr@10"] + 1e-9 for x, y in zip(a, b))
        w = sum(x["mrr@10"] < y["mrr@10"] - 1e-9 for x, y in zip(a, b))
        print(
            f"  {name:15} n={len(a):3d} {f(iv['mrr@10'])} | {f(iv['any_complete_proof@10'])} | {bt}/{w}"
        )
for q in [
    "requests-comparison-026",
    "requests-comparison-027",
    "requests-comparison-026-para",
    "requests-comparison-027-para",
]:
    g = lambda rows, m: [r["mrr@10"] for r in rows if r["id"] == q and r["method"] == m][0]
    print(
        q,
        "hybrid",
        g(off, "hybrid"),
        "->",
        g(on, "hybrid"),
        "| plain",
        g(off, "plain"),
        "->",
        g(on, "plain"),
    )
