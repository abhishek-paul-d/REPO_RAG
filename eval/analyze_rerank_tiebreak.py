# ruff: noqa: E741, SIM115, B905, B007  (research script kept terse; output is printed, nothing imports it)
"""Offline test of the rerank first-stage tie-break on SAVED pair scores (no GPU).

Usage: PYTHONPATH=. uv run python eval/analyze_rerank_tiebreak.py <extracted phase4_comparison_results dir>
Needs the pinned checkouts under data/. Scores v2 (129 q, rerank over 20) and v3 (90 q) by
re-ordering the saved candidate pool with log-odds + lam * -ln(first-stage rank).
"""

import gzip
import json
import math
import statistics as st
import sys
from pathlib import Path

from eval.retrieval.dataset import read_questions
from eval.retrieval.followup import production_corpus
from eval.retrieval.metrics import score_question
from repo_rag.config import get_settings

s = get_settings()
P = Path(sys.argv[1])
R = Path("eval/retrieval_results")
repos = {
    "psf__requests": ("psf/requests", "data/repos/psf__requests", "requests"),
    "expressjs__express": ("expressjs/express", "data/eval_repos/expressjs__express", "express"),
    "pallets__flask": ("pallets/flask", "data/eval_repos/pallets__flask", "flask"),
}
LAMS = [0, 0.1, 0.25, 0.5, 1, 2, 4]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def reorder(pool, ps, lam):
    sc = [(logit(p) + lam * -math.log(i + 1), i) for i, (c, p) in enumerate(zip(pool, ps))]
    return [pool[i] for _, i in sorted(sc, key=lambda x: (-x[0], x[1]))]


out = {"v2": {l: [] for l in LAMS}, "v3": {l: [] for l in LAMS}}
# ---- v2
for rd, (repo, co, short) in repos.items():
    sha = json.load(open(R / f"phase4_v2_full/{rd}/run.json"))["commit_sha"]
    docs = {d.metadata["chunk_id"]: d for d in production_corpus(Path(co), repo, sha, s)}
    qs = {q.id: q for q in read_questions(Path(f"eval/datasets/phase4_v2/{short}_qa.jsonl"))}
    for l in gzip.open(R / f"phase4_v2_full/{rd}/traces.jsonl.gz", "rt"):
        t = json.loads(l)
        if t["method"] != "rerank_20" or t["repetition"] != 0:
            continue
        pool = [p["chunk_id"] for p in t["pairs"]]
        ps = [p["score"] for p in t["pairs"]]
        assert pool == t["fused_order"][: len(pool)]
        for lam in LAMS:
            order = reorder(pool, ps, lam)
            sc = score_question(
                qs[t["id"]],
                [docs[c] for c in order[:10]],
                s.eval_cutoffs,
                coverage_threshold=s.eval_coverage_threshold,
            )
            out["v2"][lam].append((t["id"], sc["mrr@10"], sc["complete_gold@10"], t["cohort"]))
# ---- v3
lab = {}
for r in ["requests", "express", "flask"]:
    for l in open(P / f"labels/{r}_qa.jsonl"):
        d = json.loads(l)
        lab[d["question"]["id"]] = d


def cov(g, cs):
    ls = set()
    for c in cs:
        if c["path"] == g["path"]:
            ls |= set(range(max(c["s"], g["start_line"]), min(c["e"], g["end_line"]) + 1))
    return len(ls) / (g["end_line"] - g["start_line"] + 1)


def sc3(proofs, order):
    rank = 0
    for i, c in enumerate(order, 1):
        if any(
            c["path"] == g["path"] and c["s"] <= g["end_line"] and c["e"] >= g["start_line"]
            for p in proofs
            for g in p
        ):
            rank = i
            break
    return (1 / rank if rank and rank <= 10 else 0), int(
        any(all(cov(g, order[:10]) >= 0.9 for g in p) for p in proofs)
    )


for rd, (repo, co, short) in repos.items():
    sha = json.load(open(P / rd / "run.json"))["commit_sha"]
    m = {d.metadata["chunk_id"]: d.metadata for d in production_corpus(Path(co), repo, sha, s)}
    for t in json.load(open(P / rd / "traces.json")):
        if t["method"] != "rerank20" or t["repetition"] != 0:
            continue
        pool = [p["chunk_id"] for p in t["pairs"]]
        ps = [p["score"] for p in t["pairs"]]
        fused = [f["chunk_id"] for f in t["fused"]]
        assert pool == fused[: len(pool)], t["id"]
        for lam in LAMS:
            order = [
                {"path": m[c]["path"], "s": m[c]["start_line"], "e": m[c]["end_line"]}
                for c in reorder(pool, ps, lam)[:10]
            ]
            a, b = sc3(lab[t["id"]]["accepted_evidence"], order)
            out["v3"][lam].append((t["id"], a, b, lab[t["id"]]["stratum"]))
for ds in out:
    base = {r[0]: r for r in out[ds][0]}
    print(f"\n== {ds}: n={len(base)}   lam  MRR  rank1  complete  better worse (vs lam=0)")
    for lam in LAMS:
        rows = out[ds][lam]
        b = sum(r[1] > base[r[0]][1] + 1e-9 for r in rows)
        w = sum(r[1] < base[r[0]][1] - 1e-9 for r in rows)
        print(
            f"  {lam:>4} {st.mean(r[1] for r in rows):.4f} {sum(r[1] == 1 for r in rows):3d}  {st.mean(r[2] for r in rows):.3f}   {b:2d} {w:2d}"
        )
