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

from repo_rag.agent.tools import strip_question_framing  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever, weighted_rrf  # noqa: E402
from repo_rag.store.embeddings import get_embeddings  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402

REPO = "psf/requests"
PAIRS = ROOT / "eval/datasets/phase5_bugfix_v2/repos/psf__requests/pairs.jsonl"
AGENT_DATASET = ROOT / "eval/datasets/agent_eval_commit_full_v1.jsonl"
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


def summarize_replay(ranks: list[int | None]) -> dict:
    n = len(ranks)

    def hit(k: int) -> int:
        return sum(1 for x in ranks if x is not None and x <= k)

    mrr = sum(1 / x for x in ranks if x is not None) / n if n else 0.0
    return {
        "n": n,
        "hit@1": hit(1),
        "hit@5": hit(5),
        "hit@10": hit(10),
        "hit@15": hit(15),
        "hit@30": hit(30),
        "not_in_30": sum(1 for x in ranks if x is None),
        "mrr": round(mrr, 3),
    }


def run_agent_replay(result_path: Path, data_dir: Path | None, out: Path | None) -> None:
    """Replay the 113 commit questions from an agent eval JSON across weight/fusion settings."""
    questions = [json.loads(x) for x in AGENT_DATASET.read_text(encoding="utf-8").splitlines()]
    agent_rows = {r["id"]: r for r in json.loads(result_path.read_text(encoding="utf-8"))["rows"]}
    base_settings = Settings(data_dir=data_dir) if data_dir is not None else Settings()
    retriever = HybridRetriever(
        VectorStore(base_settings, get_embeddings(base_settings)), base_settings
    )
    memoize_dense(retriever)

    configs = [
        ("model_bm25_0.3", "model", 0.3, 30, False),
        ("model_bm25_0.1", "model", 0.1, 30, False),
        ("model_bm25_0.0", "model", 0.0, 30, False),
        ("fused_d15_raw_bm25_0.3", "fused", 0.3, 15, False),
        ("fused_d50_clean_bm25_0.3", "fused", 0.3, 50, True),
        ("fused_d50_clean_bm25_0.1", "fused", 0.1, 50, True),
        ("fused_d50_clean_bm25_0.0", "fused", 0.0, 50, True),
    ]
    results: dict[str, dict] = {}
    for name, mode, weight, depth, clean_q in configs:
        retriever.settings = base_settings.model_copy(update={"commit_bm25_weight": weight})
        rows = []
        for item in questions:
            qid = item["id"]
            golds = [s[:12] for s in item["expect"]["shas"]]
            row = agent_rows.get(qid, {})
            model_q = next(
                (
                    c["arguments"].get("query", "")
                    for c in row.get("calls", [])
                    if c.get("name") == "search_commits"
                ),
                "",
            )
            if mode == "model":
                hits = (
                    retriever.search(REPO, model_q, kind="commits", k=depth, rerank=False)
                    if model_q
                    else []
                )
            else:
                ask_q = strip_question_framing(item["question"]) if clean_q else item["question"]
                q_hits = retriever.search(REPO, ask_q, kind="commits", k=depth, rerank=False)
                if model_q:
                    m_hits = retriever.search(REPO, model_q, kind="commits", k=depth, rerank=False)
                    by_id = {h.chunk_id: h for h in q_hits} | {h.chunk_id: h for h in m_hits}
                    fused = weighted_rrf(
                        {
                            "model": [h.chunk_id for h in m_hits],
                            "question": [h.chunk_id for h in q_hits],
                        },
                        {"question": base_settings.agent_commit_fuse_question_weight},
                        base_settings.rrf_k,
                    )
                    hits = [by_id[cid] for cid, _ in fused][:30]
                else:
                    hits = q_hits[:30]
            rows.append({"id": qid, "gold": golds, "rank": rank_of(hits[:30], golds)})
        summary = summarize_replay([r["rank"] for r in rows])
        results[name] = {"summary": summary, "rows": rows}
        print(
            f"{name:26s} @1={summary['hit@1']:2d} @5={summary['hit@5']:2d} "
            f"@10={summary['hit@10']:2d} @15={summary['hit@15']:2d} "
            f"@30={summary['hit@30']:2d} miss={summary['not_in_30']:2d} "
            f"mrr={summary['mrr']:.3f}",
            flush=True,
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-depth", type=int, default=50, help="search depth k")
    parser.add_argument("--out", type=Path, help="write the rows and summary here")
    parser.add_argument("--sweep", action="store_true", help="run every config in SWEEP")
    parser.add_argument("--data-dir", type=Path, default=None, help="override Settings.data_dir")
    parser.add_argument(
        "--agent-replay",
        type=Path,
        default=None,
        help="replay the 113 first search_commits queries from an agent_eval JSON",
    )
    args = parser.parse_args()

    if args.agent_replay is not None:
        run_agent_replay(args.agent_replay, args.data_dir, args.out)
        return

    settings = Settings(data_dir=args.data_dir) if args.data_dir is not None else Settings()
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
