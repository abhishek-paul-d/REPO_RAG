# ruff: noqa: SIM115, B905, E731
"""Offline test of evidence expansion on SAVED top-10 lists (no GPU).

Settings were fixed before running: keep=5, seeds=3, per_seed=2, modes none/adjacent/callee/both.
Usage: PYTHONPATH=. uv run python eval/analyze_expansion.py eval/retrieval_results/phase4_header_run
Reads the header-off runs (production chunks, no `# defines:` line) and rescores each saved
hybrid and plain-rerank ranking after expansion.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

from eval.retrieval.ambiguous import read_cases, score_case
from eval.retrieval.followup import production_corpus
from eval.retrieval.statistics import paired_intervals
from repo_rag.config import Settings
from repo_rag.retrieval.expand import ChunkIndex, expand

RUNS = Path(sys.argv[1])
REPOS = {
    "requests": ("psf__requests", "data/repos/psf__requests"),
    "express": ("expressjs__express", "data/eval_repos/expressjs__express"),
    "flask": ("pallets__flask", "data/eval_repos/pallets__flask"),
}
MODES = ("none", "adjacent", "callee", "both")
settings = Settings()
rows: list[dict] = []
for short, (folder, checkout) in REPOS.items():
    cases = read_cases(Path(f"eval/datasets/phase4_intent_v1/{short}_qa.jsonl"))
    q0 = cases[0].question
    docs = production_corpus(Path(checkout), q0.repo, q0.commit_sha, settings)
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    index = ChunkIndex(docs)
    traces = json.load(open(RUNS / f"{folder}__header_off" / "traces.json"))
    saved = {(t["id"], t["method"]): t["final_order"] for t in traces}
    for case in cases:
        for method in ("hybrid", "plain"):
            ranked = [by_id[c] for c in saved[(case.question.id, method)]]
            for mode in MODES:
                final = expand(ranked, index, mode=mode)
                score = score_case(case, final, settings)
                rows.append(
                    {
                        "id": case.question.id,
                        "group": case.question.group,
                        "split": "test",
                        "stratum": case.stratum,
                        "method": f"{method}:{mode}",
                        "base": method,
                        "mode": mode,
                        "body_chars": sum(len(d.page_content) for d in final),
                        **score,
                    }
                )

f = lambda m: f"{m['estimate']:+.3f} [{m['low']:+.3f},{m['high']:+.3f}]"
for base in ("hybrid", "plain"):
    print(f"\n=== {base}: MRR@10 | complete proof@10 | mean chars returned")
    for mode in MODES:
        sel = [r for r in rows if r["base"] == base and r["mode"] == mode]
        print(
            f"  {mode:9} MRR {st.mean(r['mrr@10'] for r in sel):.3f}  "
            f"proof {st.mean(r['any_complete_proof@10'] for r in sel):.3f}  "
            f"chars {st.mean(r['body_chars'] for r in sel):,.0f}"
        )
    for mode in MODES[1:]:
        print(f"  --- {mode} minus none (MRR | proof | better/worse on proof)")
        for name, fn in {
            "all120": lambda r: True,
            "multi_block": lambda r: r["stratum"] == "multi_block",
            "ambiguous": lambda r: r["stratum"] == "ambiguous",
            "implementation": lambda r: r["stratum"] == "implementation",
            "test": lambda r: r["stratum"] == "test",
        }.items():
            a = sorted(
                (r for r in rows if r["base"] == base and r["mode"] == mode and fn(r)),
                key=lambda r: r["id"],
            )
            b = sorted(
                (r for r in rows if r["base"] == base and r["mode"] == "none" and fn(r)),
                key=lambda r: r["id"],
            )
            iv = paired_intervals(a, b, ["mrr@10", "any_complete_proof@10"], settings)
            up = sum(x["any_complete_proof@10"] > y["any_complete_proof@10"] for x, y in zip(a, b))
            down = sum(
                x["any_complete_proof@10"] < y["any_complete_proof@10"] for x, y in zip(a, b)
            )
            print(
                f"    {name:15} n={len(a):3d} {f(iv['mrr@10'])} | "
                f"{f(iv['any_complete_proof@10'])} | {up}/{down}"
            )
