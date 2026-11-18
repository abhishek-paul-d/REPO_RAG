"""Issue -> fixing PR search: does `search --kind pull_requests` surface the PR that closed it?

Pairs come from GitHub's own closing links (an issue and the PR that closed it), not from
text, so no hand labelling. The query is the issue's title plus the start of its body; the
corpus is every synced PR. The issue's own number is never in the query.

    uv run python eval/run_pr_search_eval.py encode/httpx --data-dir <dir> \\
        [--strategies bm25,hybrid] [--out DIR]

`bm25` needs no model. `hybrid` (dense + BM25) needs `repo-rag index-prs` to have run with the
real embedding model, which takes a GPU: about 0.1 docs/s on a laptop CPU. `hybrid_fused` is
what `search` now does for PRs and issues: it also searches the title alone and fuses the two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repo_rag.config import Settings
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.history.pr_index import pr_document
from repo_rag.store.bm25 import Bm25Index

QUERY_BODY_CHARS = 1000
KS = (1, 5, 10)
BOOTSTRAP = 10_000


def build_pairs(settings: Settings, slug: str) -> tuple[list[dict], list]:
    """(pairs, prs): one pair per closed issue whose closing PR is in the synced set."""
    with MetadataDB(settings.metadata_db_path) as db:
        prs = db.find_prs(slug, limit=1_000_000)
        issues = db.find_issues(slug, limit=1_000_000)
        pairs = []
        for issue in sorted(issues, key=lambda i: i.number):
            gold = [p.number for p in db.prs_closing_issue(slug, issue.number)]
            if gold:
                query = f"{issue.title}\n\n{issue.body[:QUERY_BODY_CHARS]}".strip()
                pairs.append({"issue": issue.number, "query": query, "gold": gold})
    return pairs, prs


def bm25_ranker(prs: list, settings: Settings):
    docs = [pr_document(p, settings) for p in prs]
    index = Bm25Index.build(docs)
    number = {str(d.metadata["chunk_id"]): int(d.metadata["number"]) for d in docs}
    return lambda query, n: [number[cid] for cid, _ in index.search(query, n)]


def hybrid_ranker(
    slug: str, settings: Settings, *, title_fusion: bool = False, rerank: bool = False
):
    from repo_rag.retrieval.retrievers import HybridRetriever
    from repo_rag.store.embeddings import get_embeddings
    from repo_rag.store.vectorstore import VectorStore

    settings = settings.model_copy(
        update={"title_query_fusion": title_fusion, "reranker_mode": "on" if rerank else "off"}
    )
    retriever = HybridRetriever(VectorStore(settings, get_embeddings(settings)), settings)
    return lambda query, n: [
        int(h.document.metadata["number"])
        for h in retriever.search(slug, query, kind="pull_requests", k=n, rerank=rerank)
    ]


def first_gold_rank(ranked: list[int], gold: list[int]) -> int | None:
    return next((r for r, n in enumerate(ranked, start=1) if n in gold), None)


def summarize(ranks: list[int | None], seed: int = 42) -> dict:
    n = len(ranks)
    rng = np.random.default_rng(seed)
    rows = {}
    scores = {f"hit@{k}": np.array([r is not None and r <= k for r in ranks], float) for k in KS}
    scores["mrr@10"] = np.array([1 / r if r is not None and r <= 10 else 0.0 for r in ranks])
    for name, values in scores.items():
        draws = rng.integers(0, n, size=(BOOTSTRAP, n))
        low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
        rows[name] = {"value": float(values.mean()), "ci95": [float(low), float(high)]}
    return {"questions": n, **rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("slug")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--strategies", default="bm25")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    settings = Settings(data_dir=args.data_dir)
    pairs, prs = build_pairs(settings, args.slug)
    if not pairs:
        raise SystemExit("no closed issues with a synced closing PR")
    report: dict = {"repo": args.slug, "pull_requests": len(prs), "pairs": len(pairs)}
    lines = [
        f"{args.slug}: {len(pairs)} issue -> closing-PR pairs over {len(prs)} PRs "
        f"(random Hit@5 would be about {5 / len(prs):.3f})."
    ]
    for name in args.strategies.split(","):
        if name == "bm25":
            rank = bm25_ranker(prs, settings)
        elif name == "hybrid":
            rank = hybrid_ranker(args.slug, settings)
        elif name == "hybrid_fused":  # full query + title-only query, fused
            rank = hybrid_ranker(args.slug, settings, title_fusion=True)
        elif name == "hybrid_fused_rerank":  # GPU only: Qwen reranker over the fused top 50
            rank = hybrid_ranker(args.slug, settings, title_fusion=True, rerank=True)
        else:
            raise SystemExit(f"unknown strategy {name!r}")
        ranks = [first_gold_rank(rank(p["query"], max(KS)), p["gold"]) for p in pairs]
        result = summarize(ranks)
        report[name] = result
        cells = ", ".join(
            f"{m} {result[m]['value']:.3f} [{result[m]['ci95'][0]:.3f}, {result[m]['ci95'][1]:.3f}]"
            for m in ("hit@1", "hit@5", "hit@10", "mrr@10")
        )
        lines.append(f"- {name}: {cells}")
    print("\n".join(lines))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
