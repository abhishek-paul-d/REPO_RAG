# ruff: noqa: SIM115, B905
"""Paired analysis of the policy run: every policy minus `plain`, by question subset.

Usage: PYTHONPATH=. uv run python eval/analyze_policy_run.py eval/retrieval_results/phase4_policy_run
"""

import collections
import json
import sys
from pathlib import Path

from eval.retrieval.statistics import paired_intervals
from repo_rag.config import Settings

P = Path(sys.argv[1])
s = Settings()
rows = []
for r in ["psf__requests", "expressjs__express", "pallets__flask"]:
    rows += json.load(open(P / r / "per_question.json"))
by = collections.defaultdict(dict)
for x in rows:
    by[x["id"]][x["method"]] = x
pol = ["tiebreak", "intent", "source_aware", "source_aware_tiebreak"]
subs = {
    "all120": lambda r: True,
    "orig90": lambda r: not r["paraphrase"],
    "para30": lambda r: r["paraphrase"],
    "test(orig+para)": lambda r: r["stratum"] == "test",
    "impl(orig+para)": lambda r: r["stratum"] == "implementation",
    "ambiguous": lambda r: r["stratum"] == "ambiguous",
    "multi_block": lambda r: r["stratum"] == "multi_block",
    "example": lambda r: r["stratum"] == "example",
    "test para": lambda r: r["stratum"] == "test" and r["paraphrase"],
    "impl para": lambda r: r["stratum"] == "implementation" and r["paraphrase"],
    "impl orig": lambda r: r["stratum"] == "implementation" and not r["paraphrase"],
}


def f(m):
    return f"{m['estimate']:+.3f} [{m['low']:+.3f},{m['high']:+.3f}]"


base = [by[i]["plain"] for i in by]
print(
    "plain  MRR by subset:",
    {
        k: round(sum(r["mrr@10"] for r in base if fn(r)) / max(1, sum(fn(r) for r in base)), 3)
        for k, fn in subs.items()
    },
)
for p in pol:
    print(f"\n=== {p} minus plain   (MRR [95% CI] | proof | better/worse)")
    for k, fn in subs.items():
        ids = [i for i in by if fn(by[i]["plain"])]
        a = [by[i][p] for i in ids]
        b = [by[i]["plain"] for i in ids]
        if not ids:
            continue
        iv = paired_intervals(a, b, ["mrr@10", "any_complete_proof@10"], s)
        better = sum(x["mrr@10"] > y["mrr@10"] + 1e-9 for x, y in zip(a, b))
        worse = sum(x["mrr@10"] < y["mrr@10"] - 1e-9 for x, y in zip(a, b))
        print(
            f"  {k:16} n={len(ids):3d} {f(iv['mrr@10'])} | {f(iv['any_complete_proof@10'])} | {better}/{worse}"
        )
# detector
print("\ndetected intent x stratum x paraphrase:")
c = collections.Counter(
    (r["stratum"], r["paraphrase"], r["detected_intent"])
    for r in base
    if r["stratum"] in ("test", "implementation")
)
for k, v in sorted(c.items()):
    print(" ", k, v)
