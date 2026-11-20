"""Offline BM25 check of commit diff ordering on Flask (2026-11-21). No GPU, no re-embedding.

Rebuilds the commit search text for 3,829 Flask commits under four diff orderings and scores BM25
alone on the 170 pilot queries. Dense search is NOT scored (re-embedding needs a GPU).

    uv run python eval/diff_order_bm25.py
"""

# ruff: noqa: E741, SIM115, B023
import dataclasses
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, "D:/Repo_chatbot/src")
import numpy as np

from repo_rag.config import Settings
from repo_rag.history.commit_indexer import CommitRecord, FileChange, Hunk, commit_document
from repo_rag.store.bm25 import Bm25Index, code_tokens

db = sqlite3.connect("file:D:/Repo_chatbot/data_flask/metadata.db?mode=ro", uri=True)
recs = []
for (rj,) in db.execute("select record_json from history_commits where repo='pallets/flask'"):
    d = json.loads(rj)
    files = tuple(
        FileChange(
            **{
                **f,
                "hunks": tuple(
                    Hunk(*h) if not isinstance(h, dict) else Hunk(**h) for h in f["hunks"]
                ),
                "symbols": tuple(f["symbols"]),
            }
        )
        for f in d["files"]
    )
    recs.append(CommitRecord(**{**d, "files": files}))
queries = [
    json.loads(l)
    for l in open(
        "D:/Repo_chatbot/eval/commit_summary_pilot/pallets__flask/queries.jsonl", encoding="utf-8"
    )
]
print(len(recs), "commits", len(queries), "queries")

DOC = (".rst", ".md", ".txt", ".cfg", ".ini", ".toml", ".yml", ".yaml", ".in")


def kind(p):
    q = p.lower()
    if "test" in q.split("/")[0] or "/tests/" in "/" + q or Path(q).name.startswith("test_"):
        return 1
    if (
        q.endswith(DOC)
        or q.startswith(("docs/", "doc/", "examples/"))
        or Path(q).name in ("changes", "authors", "license", "makefile")
    ):
        return 2
    return 0


def variant(name):
    out = []
    for r in recs:
        files = r.files
        chars = 800
        if name in ("source_first", "source_first_cap"):
            files = tuple(sorted(files, key=lambda f: (kind(f.path), f.path)))
        if name == "source_first_cap":
            files = tuple(dataclasses.replace(f, patch=f.patch[:400]) for f in files)
        if name == "chars2000":
            chars = 2000
        s = Settings(commit_diff_chars=chars)
        doc = commit_document(dataclasses.replace(r, files=files), s)
        doc.metadata["chunk_id"] = r.sha[:12]
        out.append(doc)
    return out


def evaluate(docs):
    idx = Bm25Index.build(docs)
    ranks = []
    for q in queries:
        gold = {g[:12] for g in q["gold"]}
        sc = np.asarray(idx.bm25.get_scores(code_tokens(q["query"])), dtype=float)
        order = np.argsort(-sc, kind="stable")
        r = next((i + 1 for i, j in enumerate(order) if idx.ids[j] in gold), None)
        ranks.append(r)
    return ranks


res = {}
base_docs = None
for name in ("baseline", "source_first", "source_first_cap", "chars2000"):
    docs = variant(name)
    if name == "baseline":
        base_docs = docs
    changed = sum(a.page_content != b.page_content for a, b in zip(base_docs, docs, strict=False))
    ranks = evaluate(docs)
    res[name] = ranks

    def h(k):
        return sum(1 for r in ranks if r and r <= k)

    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    print(
        f"{name:17s} changed_docs={changed:4d} hit@1={h(1)} @5={h(5)} @15={h(15)} @30={h(30)} MRR={mrr:.3f}"
    )
b = res["baseline"]
for name in ("source_first", "source_first_cap", "chars2000"):
    up = sum(1 for x, y in zip(b, res[name], strict=False) if (y or 999) < (x or 999))
    dn = sum(1 for x, y in zip(b, res[name], strict=False) if (y or 999) > (x or 999))
    print(name, "rank improved on", up, "worse on", dn)
# how often is the .py fix pushed out of the first 800 chars at baseline?
gold_recs = {r.sha[:12]: r for r in recs}
crowd = total = 0
for q in queries:
    for g in q["gold"]:
        r = gold_recs.get(g[:12])
        if not r:
            continue
        srcs = [f for f in r.files if kind(f.path) == 0 and f.path.endswith(".py")]
        if not srcs:
            continue
        total += 1
        text = "\n".join(f"--- {f.old_path}\n+++ {f.path}\n{f.patch}" for f in r.files)[:800]
        if not any(f"+++ {f.path}" in text for f in srcs):
            crowd += 1
print(
    f"gold commits touching non-test .py: {total}; none of their .py diff inside first 800 chars: {crowd}"
)
