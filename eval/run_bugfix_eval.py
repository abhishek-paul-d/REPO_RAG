"""Execute one fixed Phase 5 method on frozen inputs; model work belongs on a GPU.

Each method runs in its own process, so the 4B and 8B are never resident together.
BM25 mode loads no model. Neural methods use the actual production Chroma/hybrid path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

from eval.prepare_bugfix_eval import read_lines, verify_corpus, verify_frozen, write  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.history.metadata_db import record_from_dict  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402

METRICS = ["hit@1", "hit@5", "hit@10", "mrr@10", "file_hit@5"]


def score(ranking: list[str], gold: list[str]) -> dict[str, float]:
    if len(set(ranking)) != len(ranking) or not gold:
        raise ValueError("ranked SHAs must be unique and gold must not be empty")
    ranks = [index for index, sha in enumerate(ranking[:10], 1) if sha in set(gold)]
    return {
        **{f"hit@{k}": float(any(rank <= k for rank in ranks)) for k in (1, 5, 10)},
        "mrr@10": 1 / min(ranks) if ranks else 0.0,
    }


def changed_paths(dataset: Path) -> dict[str, set[str]]:
    records = [record_from_dict(row) for row in read_lines(dataset / "records.jsonl")]
    return {r.sha: {p for f in r.files for p in (f.path, f.old_path) if p} for r in records}


def file_overlap(ranking: list[str], gold: list[str], paths: dict[str, set[str]]) -> float:
    """Localization proxy only; includes the exact fix and old/new rename paths."""
    gold_paths = set().union(*(paths[sha] for sha in gold))
    if not gold_paths:
        raise ValueError("File-overlap evaluation requires gold changed paths")
    return float(any(gold_paths & paths[sha] for sha in ranking[:5]))


def lexical_query(query: str, method: str, protocol: dict) -> str:
    return query[: protocol["query_budgets"]["bm25_2000"]] if method == "bm25_2000" else query


def query_bin(query: str, protocol: dict) -> str:
    low, high = protocol["query_budgets"]["length_bins_chars"]
    return (
        f"<={low}"
        if len(query) <= low
        else f"{low + 1}-{high}"
        if len(query) <= high
        else f">{high}"
    )


def settings_for(protocol: dict, data_dir: Path, method: str, device: str) -> Settings:
    neural = protocol["models"]
    retrieval = protocol["retrieval"]
    return Settings(  # type: ignore[call-arg]  # BaseSettings supports _env_file at runtime.
        _env_file=None,
        data_dir=data_dir,
        embedding_provider="huggingface",
        local_embedding_model=neural["embedding"],
        embedding_dims=neural["dims"],
        embedding_dtype=neural["dtype"],
        embedding_max_tokens=neural["max_tokens"],
        embedding_query_instruction=neural["query_instruction"],
        compute_device=cast(Literal["cpu", "cuda"], device),
        reranker_mode="on" if method.startswith("rerank") else "off",
        reranker_model=neural["rerankers"][0 if method == "rerank4b" else 1],
        retrieval_k=retrieval["retrieval_k"],
        rerank_candidate_depth=retrieval["rerank_candidate_depth"],
        rerank_instruction=retrieval["rerank_instruction"],
        rerank_max_tokens=neural["max_tokens"],
        bm25_weight=0.0 if method == "dense" else retrieval["bm25_weight"],
        dense_weight=retrieval["dense_weight"],
        rrf_k=retrieval["rrf_k"],
        rerank_first_stage_weight=0,
        rerank_instruction_mode="default",
        evidence_expansion="none",
    )


class TracedRetriever(HybridRetriever):
    pair_scores: list[dict[str, Any]]

    def _rerank(self, query, hits, kind="code"):
        candidates = [
            (hit.chunk_id, hit.document.metadata["commit_sha"], hit.score) for hit in hits
        ]
        ranked = super()._rerank(query, hits, kind)
        by_id = {hit.chunk_id: hit for hit in ranked}
        self.pair_scores = [
            {"chunk_id": cid, "sha": sha, "fused_score": fused, "p_yes": by_id[cid].rerank_score}
            for cid, sha, fused in candidates
        ]
        return ranked


class TracedStore(VectorStore):
    dense_scores: list[dict[str, Any]]

    def search(self, kind, repo, query, k, filters=None):
        result = super().search(kind, repo, query, k, filters)
        self.dense_scores = [
            {
                "chunk_id": d.metadata["chunk_id"],
                "sha": d.metadata["commit_sha"],
                "similarity": value,
            }
            for d, value in result
        ]
        return result


def synchronize(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def source_hashes() -> dict[str, str]:
    paths = [
        *ROOT.joinpath("src/repo_rag").rglob("*.py"),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "eval/prepare_bugfix_eval.py",
        ROOT / "eval/run_bugfix_eval.py",
        ROOT / "eval/analyze_bugfix_run.py",
        ROOT / "eval/export_bugfix_index.py",
        ROOT / "eval/refreeze_bugfix_v2.py",
        *ROOT.joinpath("eval/retrieval").rglob("*.py"),
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(paths)
    }


def run(dataset: Path, output: Path, method: str, *, device: str = "cuda") -> dict:
    freeze = verify_frozen(dataset)  # MUST precede any search/model initialization
    verify_corpus(dataset)
    protocol = json.loads((dataset / "protocol.json").read_text(encoding="utf-8"))
    if method not in protocol["retrieval"]["methods"]:
        raise ValueError("method outside the frozen matrix")
    if output.exists():
        raise FileExistsError("Preserve existing results; use a new output directory")
    output.mkdir(parents=True)
    docs = [
        Document(page_content=r["text"], metadata=r["meta"])
        for r in read_lines(dataset / "corpus.jsonl")
    ]
    pairs = read_lines(dataset / "pairs.jsonl")
    paths = changed_paths(dataset)
    ids = {str(d.metadata["chunk_id"]): str(d.metadata["commit_sha"]) for d in docs}
    if len(ids) != len(docs) or any(d.metadata["is_fix"] != "true" for d in docs):
        raise ValueError("Corpus must contain unique fix candidates only")
    if any(not set(p["gold"]) <= set(ids.values()) for p in pairs):
        raise ValueError("Gold SHA missing from corpus")
    settings = settings_for(protocol, output.parent / "index", method, device)
    index = Bm25Index.build(docs)
    index.save(bm25_path(settings.bm25_dir, freeze["repo"], "commits"))
    retriever = None
    model_revisions = {}
    hardware = "no neural model"
    if not method.startswith("bm25"):
        import torch

        from repo_rag.store.embeddings import get_embeddings

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        hardware = torch.cuda.get_device_name() if device == "cuda" else "cpu"
        embeddings = get_embeddings(settings)
        store = TracedStore(settings, embeddings)
        store.sync("commits", freeze["repo"], docs)
        retriever = TracedRetriever(store, settings)
    rows, traces = [], []
    pair_scores: list[dict[str, Any]]
    for i, pair in enumerate(pairs):
        synchronize(device if not method.startswith("bm25") else "cpu")
        start = time.perf_counter()
        keyword = index.search(lexical_query(pair["query"], method, protocol), settings.retrieval_k)
        if retriever is not None:
            # Keyword reconstruction is instrumentation; production search computes its own.
            start = time.perf_counter()
        if retriever is None:
            ranking = [ids[cid] for cid, _ in keyword]
            dense, pair_scores = [], []
        else:
            hits = retriever.search(
                freeze["repo"],
                pair["query"],
                kind="commits",
                k=10,
                filters={"is_fix": "true"},
                rerank=method.startswith("rerank"),
            )
            ranking = [str(hit.document.metadata["commit_sha"]) for hit in hits]
            dense = retriever.store.dense_scores  # type: ignore[attr-defined]
            pair_scores = getattr(retriever, "pair_scores", [])
        synchronize(device if not method.startswith("bm25") else "cpu")
        seconds = time.perf_counter() - start
        rows.append(
            {
                **pair,
                "method": method,
                "ranking": ranking[:10],
                **score(ranking, pair["gold"]),
                "file_hit@5": file_overlap(ranking, pair["gold"], paths),
                "candidate_pool": len(docs),
                "query_chars": len(pair["query"]),
                "query_bin": query_bin(pair["query"], protocol),
                "elapsed_seconds": seconds,
                "cold_first_query": i == 0,
            }
        )
        traces.append(
            {
                "id": pair["id"],
                "method": method,
                "dense": dense,
                "bm25": [
                    {"chunk_id": cid, "sha": ids[cid], "score": value} for cid, value in keyword
                ],
                "pairs": pair_scores,
                "ranking": ranking[:10],
            }
        )
        if i % 20 == 0:
            print(freeze["repo"], method, i + 1, "/", len(pairs), flush=True)
    if retriever:
        model = getattr(
            getattr(retriever.store.embeddings, "_model", None), "_first_module", lambda: None
        )()
        model_revisions["embedding"] = getattr(
            getattr(getattr(model, "auto_model", None), "config", None), "_commit_hash", None
        )
    if retriever and method.startswith("rerank"):
        scorer = retriever._task_rerankers.get(settings.rerank_instruction)
        model_revisions["reranker"] = getattr(
            getattr(getattr(scorer, "model", None), "config", None), "_commit_hash", None
        )
    info = {
        "repo": freeze["repo"],
        "pin": freeze["pin"],
        "method": method,
        "freeze_sha256": hashlib.sha256((dataset / "freeze.json").read_bytes()).hexdigest(),
        "protocol_sha256": freeze["sha256"]["protocol.json"],
        "source_sha256": source_hashes(),
        "settings": settings.model_dump(
            mode="json", exclude={"github_token", "anthropic_api_key", "voyage_api_key"}
        ),
        "hardware": hardware,
        "device": device,
        "model_revisions": model_revisions,
        "questions": len(rows),
        "timing": "one call per query, first cold call labelled; descriptive only, not deployment p50/p95",
    }
    write(output / "per_question.json", rows)
    write(output / "traces.json", traces)
    write(output / "run.json", info)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    run(args.dataset, args.output, args.method, device=args.device)
