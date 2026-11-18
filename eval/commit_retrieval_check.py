"""Laptop-only check of the commit search behind the agent's `search_commits` tool.

Runs every psf/requests bug -> fix pair of `phase5_bugfix_v2` (141: 113 test, 28 dev) through the
same hybrid search the agent uses (no reranker, no model) and reports where the gold commit
ranks. The agent eval's commit questions are 10 of these pairs; this is the whole set, so a
retrieval change of a few questions becomes visible.

The query is the bug text itself, cut as in the agent eval's question (`BUG_CHARS`). The agent
sends a query the model writes, so this is a proxy for the search step, not for the whole agent.

    uv run python eval/commit_retrieval_check.py [--limit-depth 50] [--out result.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever  # noqa: E402
from repo_rag.store.embeddings import get_embeddings  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402

REPO = "psf/requests"
PAIRS = ROOT / "eval/datasets/phase5_bugfix_v2/repos/psf__requests/pairs.jsonl"
BUG_CHARS = 1200  # same cut as eval/agent_eval.py
AGENT_TOP = 5  # what the model sees


def gold_prefixes(pair: dict) -> list[str]:
    gold = pair["gold"]
    shas = ast.literal_eval(gold) if isinstance(gold, str) else gold
    return [s[:12] for s in shas]


def rank_of(hits: list, golds: list[str]) -> int | None:
    for position, hit in enumerate(hits, start=1):
        if str(hit.document.metadata["commit_sha"])[:12] in golds:
            return position
    return None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ranks = [r["rank"] for r in rows]

    def hit(k: int) -> int:
        return sum(1 for x in ranks if x is not None and x <= k)

    mrr = sum(1 / x for x in ranks if x is not None) / n if n else 0.0
    return {
        "n": n,
        "hit@5": hit(5),
        "hit@10": hit(10),
        "hit@20": hit(20),
        "hit@50": hit(50),
        "not_in_depth": sum(1 for x in ranks if x is None),
        "mrr": round(mrr, 3),
    }


def run_config(retriever: HybridRetriever, pairs: list[dict], depth: int) -> tuple[dict, list]:
    rows = []
    for pair in pairs:
        golds = gold_prefixes(pair)
        hits = retriever.search(
            REPO, pair["query"][:BUG_CHARS], kind="commits", k=depth, rerank=False
        )
        rows.append(
            {"id": pair["id"], "split": pair["split"], "gold": golds, "rank": rank_of(hits, golds)}
        )
    result = {
        "all": summarize(rows),
        "test": summarize([r for r in rows if r["split"] == "test"]),
        "dev": summarize([r for r in rows if r["split"] == "dev"]),
    }
    return result, rows


# (name, bm25_weight, title fusion for commits). The first is what the agent uses today.
SWEEP = [
    ("baseline_bm25_0.3", 0.3, False),
    ("title_fusion_bm25_0.3", 0.3, True),
    ("dense_only", 0.0, False),
    ("bm25_0.1", 0.1, False),
    ("bm25_0.5", 0.5, False),
    ("bm25_1.0", 1.0, False),
    ("title_fusion_bm25_0.5", 0.5, True),
    ("title_fusion_bm25_1.0", 1.0, True),
]


def memoize_dense(retriever: HybridRetriever) -> None:
    """The dense query is the slow part and identical across configs, so compute it once."""
    original = retriever.store.search
    cache: dict = {}

    def cached(kind, repo, query, n, exact=None):  # type: ignore[no-untyped-def]
        key = (kind, repo, query, n, tuple(sorted((exact or {}).items())))
        if key not in cache:
            cache[key] = original(kind, repo, query, n, exact)
        return cache[key]

    retriever.store.search = cached  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-depth", type=int, default=50, help="search depth k")
    parser.add_argument("--out", type=Path, help="write the rows and summary here")
    parser.add_argument("--sweep", action="store_true", help="run every config in SWEEP")
    args = parser.parse_args()

    settings = Settings()
    retriever = HybridRetriever(VectorStore(settings, get_embeddings(settings)), settings)
    pairs = [json.loads(x) for x in PAIRS.read_text(encoding="utf-8").splitlines()]
    if not args.sweep:
        result, rows = run_config(retriever, pairs, args.limit_depth)
        print(json.dumps(result, indent=2))
        if args.out:
            args.out.write_text(
                json.dumps({"summary": result, "rows": rows}, indent=2), encoding="utf-8"
            )
        return
    from repo_rag.retrieval import retrievers

    memoize_dense(retriever)
    out: dict[str, dict] = {}
    for name, bm25_weight, title in SWEEP:
        retriever.settings = Settings(bm25_weight=bm25_weight)
        retrievers.TITLE_FUSION_KINDS = (
            ("pull_requests", "issues", "commits") if title else ("pull_requests", "issues")
        )
        result, rows = run_config(retriever, pairs, args.limit_depth)
        out[name] = {"summary": result, "rows": rows}
        s = result
        print(
            f"{name:26s} all {s['all']['hit@5']:3d}/{s['all']['hit@10']:3d}/{s['all']['hit@50']:3d} "
            f"mrr {s['all']['mrr']:.3f} | dev {s['dev']['hit@5']}/{s['dev']['hit@10']} "
            f"mrr {s['dev']['mrr']:.3f} | test {s['test']['hit@5']}/{s['test']['hit@10']}/"
            f"{s['test']['hit@50']} mrr {s['test']['mrr']:.3f}",
            flush=True,
        )
    if args.out:
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
