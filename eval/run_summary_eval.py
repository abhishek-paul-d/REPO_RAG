"""Do the repo-map summaries help find the right FILE? BM25 always; hybrid with --data-dir.

    uv run python eval/run_summary_eval.py eval/datasets/requests_qa.jsonl \\
        data/repos/psf__requests [--out DIR]

Each question's gold chunks name files. Three rankers return files best first:
  chunks     BM25 over code chunks, collapsed to the best-ranked file for each path
  summaries  BM25 over the file summaries
  fused      weighted-RRF of the two file rankings (the same fusion the retriever uses)
The checkout must be at each question's commit_sha. `--data-dir` adds the same three rankers
using hybrid (dense + BM25) search, which needs `repo-rag index-summaries` run there first. This is file localisation on code questions, not an architecture-answer test.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pr_search_eval import summarize

from repo_rag.config import Settings
from repo_rag.ingest.indexer import chunk_repo
from repo_rag.retrieval.retrievers import weighted_rrf
from repo_rag.store.bm25 import Bm25Index
from repo_rag.summaries.repo_map import build_summaries

DEPTH = 50


def file_ranking(index: Bm25Index, by_id: dict[str, str], query: str) -> list[str]:
    paths: list[str] = []
    for chunk_id, _ in index.search(query, DEPTH * 4):
        path = by_id[chunk_id]
        if path and path not in paths:
            paths.append(path)
    return paths[:DEPTH]


def hybrid_files(retriever, slug: str, kind: str, query: str, filters: dict) -> list[str]:
    """Hybrid (dense + BM25) search over one collection, collapsed to files, best first."""
    paths: list[str] = []
    for hit in retriever.search(slug, query, kind=kind, k=DEPTH, filters=filters):
        path = str(hit.document.metadata["path"])
        if path and path not in paths:
            paths.append(path)
    return paths


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    return next((r for r, p in enumerate(ranked, start=1) if p in gold), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="A data dir holding this repo's code and summaries indexes: adds hybrid rankers.",
    )
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.dataset.read_text(encoding="utf-8").splitlines() if x]
    slug, sha = rows[0]["repo"], rows[0]["commit_sha"]
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(data_dir=Path(tmp))
        chunks, _ = chunk_repo(slug, args.checkout, sha, settings)
        summaries, _ = build_summaries(slug, args.checkout, settings)
    code_docs = chunks["code"]
    code_index = Bm25Index.build(code_docs)
    code_paths = {str(d.metadata["chunk_id"]): str(d.metadata["path"]) for d in code_docs}
    file_docs = [d for d in summaries if d.metadata["symbol_kind"] == "file_summary"]
    sum_index = Bm25Index.build(file_docs)
    sum_paths = {str(d.metadata["chunk_id"]): str(d.metadata["path"]) for d in file_docs}

    ranks: dict[str, list[int | None]] = {
        "chunks_bm25": [],
        "summaries_bm25": [],
        "fused_bm25": [],
    }
    retriever = None
    if args.data_dir:
        from repo_rag.retrieval.retrievers import HybridRetriever
        from repo_rag.store.embeddings import get_embeddings
        from repo_rag.store.vectorstore import VectorStore

        live = Settings(data_dir=args.data_dir)
        retriever = HybridRetriever(VectorStore(live, get_embeddings(live)), live)
        ranks.update({"chunks_hybrid": [], "summaries_hybrid": [], "fused_hybrid": []})
    for row in rows:
        gold = {g["path"] for g in row["gold"]}
        by_chunks = file_ranking(code_index, code_paths, row["question"])
        by_summaries = file_ranking(sum_index, sum_paths, row["question"])
        fused = [p for p, _ in weighted_rrf({"c": by_chunks, "s": by_summaries}, {}, k=60)]
        ranks["chunks_bm25"].append(first_rank(by_chunks, gold))
        ranks["summaries_bm25"].append(first_rank(by_summaries, gold))
        ranks["fused_bm25"].append(first_rank(fused, gold))
        if retriever is not None:
            question = row["question"]
            h_chunks = hybrid_files(retriever, slug, "code", question, {})
            h_sums = hybrid_files(
                retriever, slug, "summaries", question, {"symbol_kind": "file_summary"}
            )
            h_fused = [p for p, _ in weighted_rrf({"c": h_chunks, "s": h_sums}, {}, k=60)]
            ranks["chunks_hybrid"].append(first_rank(h_chunks, gold))
            ranks["summaries_hybrid"].append(first_rank(h_sums, gold))
            ranks["fused_hybrid"].append(first_rank(h_fused, gold))

    report = {"repo": slug, "commit": sha, "questions": len(rows)}
    lines = [f"{slug}@{sha[:8]}: {len(rows)} questions, file-level gold."]
    for name, values in ranks.items():
        result = summarize(values)
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
