"""Fixed production-path hybrid/rerank20/rerank50 stress comparison; no tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.ambiguous import read_cases, score_case, validate_cases  # noqa: E402
from eval.retrieval.comparison_protocol import protocol_settings  # noqa: E402
from eval.retrieval.dataset import write_json  # noqa: E402
from eval.retrieval.followup import production_corpus  # noqa: E402
from eval.retrieval.runner import cached_vectors, environment  # noqa: E402
from eval.retrieval.statistics import paired_intervals, summarize  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.rerank import QwenReranker  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever, weighted_rrf  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.embeddings import LocalEmbeddings  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402


class TracedRetriever(HybridRetriever):
    """Keep the production algorithm; observe exactly the candidates it passes to scoring."""

    pair_trace: list[dict]

    def _rerank(self, query, hits, kind="code"):
        self.pair_trace = [{"chunk_id": h.chunk_id} for h in hits]
        result = super()._rerank(query, hits, kind)
        scores = {h.chunk_id: h.rerank_score for h in result}
        for pair in self.pair_trace:
            pair["score"] = scores[pair["chunk_id"]]
        return result


class TracedStore(VectorStore):
    def search(self, *args, **kwargs):
        hits = super().search(*args, **kwargs)
        self.last_dense = [
            {"chunk_id": str(d.metadata["chunk_id"]), "cosine_similarity": float(s)}
            for d, s in hits
        ]
        return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "bm25", "full"), default="full")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--protocol", type=Path, help="Frozen expanded-comparison protocol")
    args = parser.parse_args()
    cases = read_cases(args.dataset)
    validate_cases(args.checkout, cases)
    if args.mode == "validate":
        print(f"Validated {len(cases)} ambiguous cases")
        return
    output = args.output
    if output.exists():
        raise FileExistsError("choose a new output to preserve completed or partial measurements")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol else None
    fixed = (
        protocol_settings(protocol)
        if protocol
        else {
            "embedding_dtype": "float32",
            "retrieval_k": 50,
            "reranker_mode": "on",
            "dense_weight": 1,
            "bm25_weight": 0.3,
            "eval_cutoffs": (1, 5, 10),
        }
    )
    if protocol and args.repetitions != protocol["timing_repetitions"]:
        raise ValueError("repetitions disagree with frozen protocol")
    settings = Settings(data_dir=output / "isolated_index", compute_device=args.device, **fixed)
    q0 = cases[0].question
    docs = production_corpus(args.checkout, q0.repo, q0.commit_sha, settings)
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    output.mkdir(parents=True)
    index = Bm25Index.build(docs)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    cold_seconds = None
    methods = ("bm25",) if args.mode == "bm25" else ("hybrid", "rerank20", "rerank50")
    retriever = scorer = store = None
    embedding = None

    def no_sync() -> None:
        pass

    sync: Callable[[], None] = no_sync
    gpu = None
    if args.mode == "full":
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use A100 notebook or explicitly choose CPU")
        sync = torch.cuda.synchronize if args.device == "cuda" else no_sync
        gpu = torch.cuda.get_device_name() if args.device == "cuda" else None
        embedding = LocalEmbeddings(settings)
        vectors = cached_vectors(
            [d.page_content for d in docs],
            settings,
            output / "vector_cache",
            embedding,
            query=False,
        )
        store = TracedStore(settings, embedding)
        store.replace_records(
            "code",
            q0.repo,
            list(by_id),
            [d.page_content for d in docs],
            [d.metadata for d in docs],
            vectors,
        )
        index.save(bm25_path(settings.bm25_dir, q0.repo, "code"))
        scorer = QwenReranker(settings, instruction=settings.code_rerank_instruction)
        sync()
        started = time.perf_counter()
        scorer.score(q0.question, [d.page_content for d in docs[:50]])
        sync()
        cold_seconds = time.perf_counter() - started
        retriever = TracedRetriever(store, settings, reranker=scorer)
        retriever.search(q0.repo, q0.question, k=10, rerank=True)
    for case in cases:
        repetitions = 1 if args.mode == "bm25" else args.repetitions
        for rep in range(repetitions):
            order = methods[rep % len(methods) :] + methods[: rep % len(methods)]
            for method in order:
                if retriever is not None:
                    retriever.pair_trace = []
                sync()
                started = time.perf_counter()
                if method == "bm25":
                    ranking = [cid for cid, _ in index.search(case.question.question, 50)]
                else:
                    assert retriever is not None
                    depth = 20 if method == "rerank20" else 50
                    retriever.settings = settings.model_copy(
                        update={"rerank_candidate_depth": depth}
                    )
                    # This invokes the actual production method, including real query embedding.
                    hits = retriever.search(
                        q0.repo, case.question.question, k=10, rerank=method != "hybrid"
                    )
                    ranking = [h.chunk_id for h in hits]
                sync()
                elapsed = (time.perf_counter() - started) * 1000
                selected = [by_id[c] for c in ranking[:10]]
                scores = score_case(case, selected, settings)
                lexical_pairs = index.search(case.question.question, settings.retrieval_k)
                dense_list = store.last_dense if store is not None else []
                fused = weighted_rrf(
                    {
                        "dense": [r["chunk_id"] for r in dense_list],
                        "bm25": [cid for cid, _ in lexical_pairs],
                    },
                    {"dense": settings.dense_weight, "bm25": settings.bm25_weight},
                    settings.rrf_k,
                )
                row = {
                    "id": case.question.id,
                    "group": case.question.group,
                    "split": "test",
                    "cohort": "ambiguous_stress",
                    "stratum": case.stratum,
                    "topic": case.topic,
                    "origin_case_id": case.origin_case_id,
                    "method": method,
                    **scores,
                    "results": [{"chunk_id": c, **by_id[c].metadata} for c in ranking[:10]],
                }
                if rep == 0:
                    rows.append(row)
                traces.append(
                    {
                        "id": case.question.id,
                        "method": method,
                        "repetition": rep,
                        "stratum": case.stratum,
                        "end_to_end_ms": elapsed,
                        "final_order": ranking[:10],
                        "scores": scores,
                        "dense": dense_list,
                        "bm25": [{"chunk_id": cid, "score": float(s)} for cid, s in lexical_pairs],
                        "fused": [{"chunk_id": cid, "score": float(s)} for cid, s in fused],
                        "pairs": retriever.pair_trace if retriever is not None else [],
                    }
                )
        write_json(output / "per_question.json", rows)
        print(f"Scored {case.question.id}", flush=True)
    summary = summarize(rows, settings)
    comparisons = (
        []
        if args.mode == "bm25"
        else [
            {
                "candidate": m,
                "baseline": "hybrid",
                "metrics": paired_intervals(
                    [r for r in rows if r["method"] == m],
                    [r for r in rows if r["method"] == "hybrid"],
                    ["mrr@10", "any_complete_proof@10", "complete_gold@10"],
                    settings,
                ),
            }
            for m in ("rerank20", "rerank50")
        ]
    )
    if args.mode == "full":
        comparisons.append(
            {
                "candidate": "rerank50",
                "baseline": "rerank20",
                "metrics": paired_intervals(
                    [r for r in rows if r["method"] == "rerank50"],
                    [r for r in rows if r["method"] == "rerank20"],
                    ["mrr@10", "any_complete_proof@10", "complete_gold@10"],
                    settings,
                ),
            }
        )
    latency = [
        {
            "method": method,
            "calls": len(values),
            "p50_ms": float(np.quantile(values, 0.5)),
            "p95_ms": float(np.quantile(values, 0.95)),
        }
        for method in methods
        if (values := [t["end_to_end_ms"] for t in traces if t["method"] == method])
    ]
    write_json(output / "summary.json", summary)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "latency.json", latency if args.mode == "full" else [])
    write_json(output / "traces.json", traces)
    write_json(
        output / "run.json",
        {
            "mode": args.mode,
            "repo": q0.repo,
            "commit_sha": q0.commit_sha,
            "environment": environment(),
            "dataset_sha256": hashlib.sha256(
                args.dataset.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "protocol_sha256": hashlib.sha256(
                args.protocol.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if args.protocol
            else None,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for p in [
                    *ROOT.joinpath("src/repo_rag").rglob("*.py"),
                    *ROOT.joinpath("eval/retrieval").rglob("*.py"),
                    Path(__file__),
                    *([ROOT / "eval/summarize_comparison.py"] if protocol else []),
                ]
            },
            "settings": settings.model_dump(
                mode="json", exclude={"anthropic_api_key", "github_token", "voyage_api_key"}
            ),
            "methods": methods,
            "quality_repetition": 0,
            "repetitions": args.repetitions,
            "gpu": gpu,
            "platform": platform.platform(),
            "cold_first_50_seconds": cold_seconds,
            "independent_review_complete": False,
            "deployment_hardware": "undecided; A100 is experiment hardware only",
            "embedding_model_revision": getattr(
                getattr(embedding.model._first_module().auto_model, "config", None),
                "_commit_hash",
                None,
            )
            if embedding
            else None,
            "reranker_model_revision": getattr(
                getattr(scorer.model, "config", None), "_commit_hash", None
            )
            if scorer
            else None,
            "interpretation": "post-hoc stress set with alternative sufficient evidence; do not pool with v2 or treat as an independent fresh deployment estimate",
        },
    )
    lines = [
        "# Ambiguous-intent production-path stress comparison",
        "",
        "Evaluation complete; rerank default pending. This stress set is source-authored on inspected behaviours, not independently reviewed or an independent holdout.",
        "",
        "Metrics accept any labelled sufficient evidence set. Supporting partial demonstrations do not count as complete proofs.",
        "",
        "| Method | Questions | MRR@10 | Any accepted proof >=90% @10 |",
        "|---|---:|---:|---:|",
    ]
    for r in summary:
        if r["split"] == "all":
            lines.append(
                f"| {r['method']} | {r['questions']} | {r['mrr@10']:.3f} | {r['any_complete_proof@10']:.3f} |"
            )
    lines += [
        "",
        "See comparisons.json for paired evidence-group intervals and latency.json for measured warm p50/p95. BM25-only mode does not measure reranking or deployment latency.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
