"""Sort the missed commit questions of two agent runs: where was the gold commit?

For every question a run failed, the agent's own first `search_commits` query is replayed against
the real index and the rank of the gold commit is recorded, together with the commits the model
named instead (subject, date, files shared with the gold commit). Buckets used by the report:

    A  gold in the top 5          (shown in full, but not named)
    B  gold at rank 6-15          (shown as a one-line row)
    C  gold at rank 16-30         (outside the shortlist)
    D  gold not in the top 30

    uv run python eval/commit_failure_sort.py OUT.json
    uv run python eval/commit_failure_sort.py OUT.json --repo pallets/flask \
        --data-dir D:/Repo_chatbot/data_flask --dataset eval/datasets/agent_eval_commit_full_pallets__flask_v1.jsonl \
        --run control_a=eval/agent_eval_results/<result>.json

2026-11-19 result on the two baseline runs (113 questions): of 35 misses (commit_explain) 4 were A,
8 B, 5 C and 18 D; of 39 (default prompt) 7, 11, 5 and 16. Retrieval is the main loss, not the model's
choice from a good shortlist. Read-only: it needs `data/` with the commit index and metadata.db.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever  # noqa: E402
from repo_rag.store.embeddings import get_embeddings  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402

RES = ROOT / "eval/agent_eval_results"
T = "_agent_eval_commit_full_v1_extra3_notests_chars24000_in12288_"
RUNS = {
    "explain": ("commit_explain", "20261118T233628"),
    "default": ("commit_pr_commits", "20261118T233118"),
}
ARGS = argparse.ArgumentParser()
ARGS.add_argument("out", type=Path)
ARGS.add_argument("--repo", default="psf/requests")
ARGS.add_argument("--data-dir", type=Path, default=ROOT / "data")
ARGS.add_argument("--dataset", default="eval/datasets/agent_eval_commit_full_v1.jsonl")
ARGS.add_argument("--run", action="append", default=[], help="name=result.json (repeatable)")
A = ARGS.parse_args()
OUT = A.out

gold, question = {}, {}
for line in (ROOT / A.dataset).read_text(encoding="utf-8").splitlines():
    d = json.loads(line)
    gold[d["id"]] = [s[:12] for s in d["expect"]["shas"]]
    question[d["id"]] = d["question"]

db = sqlite3.connect(f"file:{A.data_dir.as_posix()}/metadata.db?mode=ro", uri=True)
REPO = A.repo


def subject(sha):
    r = db.execute(
        "select message, commit_date from history_commits where repo=? and sha like ?",
        (REPO, sha + "%"),
    ).fetchone()
    return (r[0].splitlines()[0][:90] if r and r[0] else "?", r[1][:10] if r else "?")


def files(sha):
    return {
        r[0]
        for r in db.execute(
            "select path from history_files where repo=? and sha like ?", (REPO, sha + "%")
        )
    }


def prs(sha):
    return {
        r[0]
        for r in db.execute(
            "select pr_number from issue_links where repo=? and commit_sha like ? and pr_number>0",
            (REPO, sha + "%"),
        )
    }


s = Settings(data_dir=A.data_dir)
retr = HybridRetriever(VectorStore(s, get_embeddings(s)), s)


def rank(query, shas, k=30):
    hits = retr.search(REPO, query, kind="commits", k=k)
    for pos, h in enumerate(hits, 1):
        if str(h.document.metadata["commit_sha"])[:12] in shas:
            return pos
    return None


out = {}
runs = {n: ROOT / f for n, f in (x.split("=", 1) for x in A.run)} if A.run else None
for name, spec in (runs or RUNS).items():
    p = spec if runs else RES / f"agent_eval_production_{spec[0]}{T}{spec[1]}.json"
    rows = {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}
    for i, r in rows.items():
        if r["passed"]:
            continue
        g = gold[i]
        q = next(
            (c["arguments"].get("query", "") for c in r["calls"] if c["name"] == "search_commits"),
            None,
        )
        cited = []
        for m in re.findall(r"\b[0-9a-f]{7,40}\b", r["model_answer"] or ""):
            if (
                m[:12] not in cited
                and db.execute(
                    "select 1 from history_commits where repo=? and sha like ?", (REPO, m + "%")
                ).fetchone()
            ):
                cited.append(m[:12])
        gf, gp = files(g[0]), prs(g[0])
        info = []
        for c in cited:
            sj, dt = subject(c)
            info.append(
                {
                    "sha": c,
                    "subject": sj,
                    "date": dt,
                    "same_pr": bool(gp & prs(c)) if gp else None,
                    "shared_files": sorted(files(c) & gf)[:4],
                }
            )
        e = out.setdefault(
            i,
            {
                "id": i,
                "gold": g,
                "gold_subject": subject(g[0]),
                "gold_pr": sorted(gp),
                "gold_files": sorted(gf)[:6],
                "runs": {},
            },
        )
        e["runs"][name] = {
            "abstain": r["abstain"],
            "n_search": sum(1 for c in r["calls"] if c["name"] == "search_commits"),
            "gold_rank_first_query": rank(q, g) if q else None,
            "cited": info,
            "answer": (r["model_answer"] or "")[:300],
        }
        e["title"] = question[i].split("\n\n", 1)[-1][:160].replace("\n", " ")
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(out), "distinct missed questions")
