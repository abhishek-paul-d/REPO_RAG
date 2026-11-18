"""Evaluate production retrieval components without touching the user's index.

Exact dense search is the controlled chunking comparison. Chroma/HNSW and weighted RRF
use the same precomputed vectors, isolating ANN loss from embedding/chunking quality.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from eval.retrieval.dataset import Question, write_json
from eval.retrieval.metrics import score_question, source_spans
from eval.retrieval.statistics import summarize
from repo_rag.config import Settings
from repo_rag.retrieval.rerank import Reranker
from repo_rag.retrieval.retrievers import weighted_rrf
from repo_rag.store.bm25 import Bm25Index
from repo_rag.store.embeddings import truncate_and_normalise
from repo_rag.store.vectorstore import VectorStore


class CachedEmbeddings(Embeddings):
    """Queries already computed by the real model; no model loads during ANN experiments."""

    def __init__(self, questions: list[Question], queries: np.ndarray):
        self.queries = {
            q.question: vector.tolist() for q, vector in zip(questions, queries, strict=True)
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("evaluation imports precomputed document vectors")

    def embed_query(self, text: str) -> list[float]:
        return self.queries[text]


def cached_vectors(
    texts: list[str], settings: Settings, directory: Path, embeddings: Embeddings, *, query: bool
) -> np.ndarray:
    identity = {
        "texts": texts,
        "model": settings.local_embedding_model,
        "provider": settings.embedding_provider,
        "dims": settings.embedding_dims,
        "instruction": settings.embedding_query_instruction if query else None,
        "tokens": settings.embedding_max_tokens,
        "dtype": settings.embedding_dtype,
        "device": settings.compute_device,
        "implementation": type(embeddings).__qualname__,
        "embedding_source": hashlib.sha256(
            Path(__file__)
            .parents[2]
            .joinpath("src/repo_rag/store/embeddings.py")
            .resolve()
            .read_bytes()
        ).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.npy"
    if path.exists():
        vectors = np.load(path, allow_pickle=False)
    else:
        print(
            f"Embedding {len(texts)} {'queries' if query else 'chunks'} (cache {digest[:10]})",
            flush=True,
        )
        raw = (
            [embeddings.embed_query(t) for t in texts]
            if query
            else embeddings.embed_documents(texts)
        )
        vectors = truncate_and_normalise(np.asarray(raw, dtype=np.float32), settings.embedding_dims)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, vectors, allow_pickle=False)
        temporary.replace(path)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError(f"invalid cached vectors: {path}")
    if settings.embedding_dims and vectors.shape[1] != settings.embedding_dims:
        raise ValueError(f"wrong embedding dimension: {path}")
    return vectors


def is_test_path(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        bool(set(pure.parts[:-1]) & {"test", "tests", "__tests__"})
        or pure.name.startswith("test_")
        or pure.name.endswith(("_test.go", ".test.js", ".spec.ts"))
    )


def fuse(
    dense: list[str],
    lexical: list[str],
    docs: dict[str, Document],
    settings: Settings,
    weight: float,
    penalty: float,
) -> list[str]:
    rankings = {"dense": dense}
    if weight > 0:
        rankings["bm25"] = lexical
    scored = weighted_rrf(
        rankings, {"dense": settings.dense_weight, "bm25": weight}, settings.rrf_k
    )
    scored = [
        (cid, value * (penalty if is_test_path(str(docs[cid].metadata["path"])) else 1.0))
        for cid, value in scored
    ]
    return [cid for cid, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def evaluate_variant(
    docs: list[Document],
    questions: list[Question],
    settings: Settings,
    output: Path,
    mode: str,
    embeddings: Embeddings | None = None,
    reranker: Reranker | None = None,
) -> tuple[list[dict], list[dict], dict]:
    cutoffs = settings.eval_cutoffs
    if not cutoffs or min(cutoffs) <= 0:
        raise ValueError("evaluation cutoffs must be positive")
    n = max(settings.retrieval_k, max(cutoffs), *(settings.eval_rerank_depths if reranker else ()))
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    ids = list(by_id)
    index = Bm25Index.build(docs)
    vectors = queries = store = None
    build_start = time.perf_counter()
    if mode == "full":
        if embeddings is None:
            raise ValueError("full evaluation requires real embeddings")
        vectors = cached_vectors(
            [d.page_content for d in docs],
            settings,
            output.parent.parent / "vector_cache",
            embeddings,
            query=False,
        )
        queries = cached_vectors(
            [q.question for q in questions],
            settings,
            output.parent.parent / "vector_cache",
            embeddings,
            query=True,
        )
        store = VectorStore(
            settings.model_copy(update={"data_dir": output / "isolated_index"}),
            CachedEmbeddings(questions, queries),
        )
        store.replace_records(
            "code",
            questions[0].repo,
            ids,
            [d.page_content for d in docs],
            [d.metadata for d in docs],
            vectors,
        )
    per_question: list[dict[str, Any]] = []
    diagnostics = []
    for qi, question in enumerate(questions):
        lexical = [cid for cid, _ in index.search(question.question, n)]
        rankings: dict[str, list[str]] = {"bm25": lexical}
        if mode == "full":
            assert vectors is not None and queries is not None and store is not None
            similarity = vectors @ queries[qi]
            exact = [ids[i] for i in np.argsort(-similarity, kind="stable")[:n]]
            started = time.perf_counter()
            dense = [
                str(d.metadata["chunk_id"])
                for d, _ in store.search("code", question.repo, question.question, n)
            ]
            ann_ms = (time.perf_counter() - started) * 1000
            rankings.update({"dense_exact": exact, "dense_hnsw": dense})
            # Similarity-threshold recall gives all exact boundary ties equal credit.
            for k in cutoffs:
                size = min(k, len(ids))
                threshold = similarity[ids.index(exact[size - 1])]
                eligible = {ids[i] for i in range(len(ids)) if similarity[i] >= threshold - 1e-6}
                diagnostics.append(
                    {
                        "id": question.id,
                        "split": question.split,
                        "k": k,
                        "ann_recall": len(set(dense[:size]) & eligible) / size,
                        "ann_query_ms": ann_ms,
                    }
                )
            for weight in settings.eval_bm25_weights:
                for penalty in settings.eval_test_penalties:
                    key = f"hybrid_w{weight:g}_tests{penalty:g}"
                    rankings[key] = fuse(dense, lexical, by_id, settings, weight, penalty)
            if reranker is not None:
                # One fixed first stage for the depth experiment. Do not pick its weights
                # using test questions. Both depths share the same scored top-50 pairs.
                baseline = fuse(dense, lexical, by_id, settings, settings.bm25_weight, 1.0)
                candidates = baseline[: max(settings.eval_rerank_depths)]
                started = time.perf_counter()
                scores = reranker.score(
                    question.question, [by_id[cid].page_content for cid in candidates]
                )
                rerank_seconds = time.perf_counter() - started
                if len(scores) != len(candidates) or not all(np.isfinite(scores)):
                    raise ValueError("invalid reranker output")
                for depth in settings.eval_rerank_depths:
                    ranked = sorted(
                        zip(candidates[:depth], scores[:depth], strict=True),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                    rankings[f"rerank_{depth}"] = [cid for cid, _ in ranked]
                diagnostics.append(
                    {
                        "id": question.id,
                        "split": question.split,
                        "rerank_candidates": len(candidates),
                        "rerank_seconds": rerank_seconds,
                    }
                )
        for method, ranking in rankings.items():
            ranked_docs = [by_id[cid] for cid in ranking[: max(cutoffs)]]
            per_question.append(
                {
                    "id": question.id,
                    "split": question.split,
                    "group": question.group,
                    "category": question.category,
                    "method": method,
                    **score_question(
                        question,
                        ranked_docs,
                        cutoffs,
                        coverage_threshold=settings.eval_coverage_threshold,
                    ),
                    "results": [
                        {
                            "chunk_id": cid,
                            "path": by_id[cid].metadata["path"],
                            "start_line": by_id[cid].metadata["start_line"],
                            "end_line": by_id[cid].metadata["end_line"],
                            "source_spans": source_spans(by_id[cid]),
                        }
                        for cid in ranking[: max(cutoffs)]
                    ],
                }
            )
    summaries = summarize(per_question, settings)
    stats = {
        "chunks": len(docs),
        "avg_chars": float(np.mean([len(d.page_content) for d in docs])),
        "avg_body_chars": float(np.mean([int(d.metadata["eval_body_chars"]) for d in docs])),
        "elapsed_seconds": time.perf_counter() - build_start,
        "chunks_with_source_gaps": sum(
            sum(end - start + 1 for start, end in source_spans(d))
            < int(d.metadata["end_line"]) - int(d.metadata["start_line"]) + 1
            for d in docs
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "per_question.json", per_question)
    write_json(output / "ann_and_rerank.json", diagnostics)
    write_json(
        output / "chunks.json", [{"metadata": d.metadata, "text": d.page_content} for d in docs]
    )
    return summaries, per_question, stats


def select_on_dev(rows: list[dict], cutoffs: Sequence[int]) -> list[dict]:
    """Select per repo/strategy/header on dev only, then attach its unchanged test row."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"].startswith("hybrid_"):
            groups[(row["repo"], row["strategy"], row["headers"])].append(row)
    selected = []
    for key, group in groups.items():
        dev = [row for row in group if row["split"] == "dev"]
        if not dev:
            continue
        winner = max(
            dev, key=lambda row: (row[f"mrr@{max(cutoffs)}"], row[f"recall@{max(cutoffs)}"])
        )
        test = next(
            (row for row in group if row["split"] == "test" and row["method"] == winner["method"]),
            None,
        )
        selected.append(
            {
                "repo": key[0],
                "strategy": key[1],
                "headers": key[2],
                "selected_method": winner["method"],
                "dev": winner,
                "test": test,
            }
        )
    return selected


def environment() -> dict:
    versions = {}
    for name in ("chromadb", "sentence-transformers", "tree-sitter", "astchunk", "numpy", "torch"):
        versions[name] = importlib.metadata.version(name)
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def markdown_report(
    rows: list[dict],
    skipped: list[dict],
    mode: str,
    settings: Settings,
    *,
    comparisons: list[dict] | None = None,
    selected: list[dict] | None = None,
) -> str:
    cutoffs = settings.eval_cutoffs
    k = max(cutoffs)
    confidence = f"{settings.eval_bootstrap_confidence:.0%}"
    coverage_label = f"Gold ≥{settings.eval_coverage_threshold:.0%}@{k}"
    fixed = {
        "bm25",
        "dense_exact",
        "dense_hnsw",
        f"hybrid_w{settings.bm25_weight:g}_tests1",
        *(f"rerank_{depth}" for depth in settings.eval_rerank_depths),
    }

    def interval(row: dict, metric: str) -> str:
        ci = row["confidence_intervals"][metric]
        return f"{row[metric]:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"

    def table(sample: list[dict]) -> list[str]:
        columns = [
            "Repo",
            "Strategy",
            "Header",
            "Method",
            "n",
            "Groups",
            *(f"Recall@{c}" for c in cutoffs),
            f"MRR@{k} [{confidence} CI]",
            f"Line coverage@{k} [{confidence} CI]",
            coverage_label + f" [{confidence} CI]",
            "Chunks",
            "Avg body chars",
        ]
        lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for row in sample:
            cells = [
                row["repo"],
                row["strategy"],
                str(row["headers"]),
                row["method"],
                str(row["questions"]),
                str(row["groups"]),
                *(f"{row[f'recall@{c}']:.3f}" for c in cutoffs),
                interval(row, f"mrr@{k}"),
                interval(row, f"line_coverage@{k}"),
                interval(row, f"complete_gold@{k}"),
                str(row["chunks"]),
                f"{row['avg_body_chars']:.0f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    lines = [
        "# Phase 4 retrieval evaluation",
        "",
        f"Mode: **{mode}**. "
        + (
            "Lexical-only plumbing check; no embedding or AST retrieval advantage is established."
            if mode == "bm25"
            else "Real configured embeddings; exact dense, Chroma HNSW and production weighted RRF."
        ),
        "",
        "Recall/Hit/MRR use any overlap. Line coverage counts the union of retrieved gold source lines, including blank lines, per question.",
        f"{coverage_label} is the fraction of gold ranges reaching the coverage threshold, macro-averaged over questions. Both coverage metrics are bounded by 1.",
        "MRR is truncated at the largest cutoff. Header-on uses identical source-derived headers for all chunkers.",
        f"Intervals: {settings.eval_bootstrap_samples:,} paired evidence-group bootstrap resamples, seed {settings.eval_bootstrap_seed}; {confidence} pointwise percentile intervals.",
        "Questions sharing evidence are resampled together. Intervals are exploratory, not adjusted for multiple comparisons; they do not establish a general AST advantage.",
        "Header on/off and all/dev/test tables reuse the same questions. These are views of one experiment per repository, not independent replications; do not add their sample counts together.",
        "Pooled/all-question rows are descriptive fixed-configuration comparisons. They are not held-out estimates for tuning; Requests questions also influenced earlier model selection.",
        "Weight/penalty grid results remain in summary.json; selected settings appear only with dev and held-out test scores below, never pooled as an unbiased headline.",
        "",
    ]
    for split, label in (
        ("all", "All questions — fixed comparisons"),
        ("dev", "Development questions — diagnostic"),
        ("test", "Held-out test questions — fixed comparisons"),
    ):
        lines.extend(
            [
                f"## {label}",
                "",
                *table([r for r in rows if r["split"] == split and r["method"] in fixed]),
                "",
            ]
        )
    if selected:
        lines.extend(
            [
                "## Settings selected on development questions",
                "",
                "Selection rule: development MRR, then Recall, unchanged. Only held-out test results assess the selected setting.",
                "",
            ]
        )
        for choice in selected:
            dev, test = choice["dev"], choice["test"]
            lines.append(
                f"- {choice['repo']} / {choice['strategy']} / headers={choice['headers']}: {choice['selected_method']}; dev MRR={dev[f'mrr@{k}']:.3f}"
                + (
                    f"; test MRR={interval(test, f'mrr@{k}')}, test line coverage={interval(test, f'line_coverage@{k}')}"
                    if test
                    else "; no test sample"
                )
            )
        lines.append("")
    if comparisons:
        lines.extend(
            [
                "## Paired AST minus baseline differences",
                "",
                "Intervals concern differences on the same questions, not overlap of two separate intervals. Positive differences favour AST for this metric/sample only.",
                "",
                f"| Repo | Header | Method | Split | Baseline | ΔMRR@{k} [{confidence} CI] | ΔLine coverage@{k} [{confidence} CI] | Δ{coverage_label} [{confidence} CI] |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for comparison in comparisons:
            if comparison["method"] not in fixed or comparison["split"] == "dev":
                continue
            metrics = comparison["metrics"]
            cells = [
                comparison["repo"],
                str(comparison["headers"]),
                comparison["method"],
                comparison["split"],
                comparison["baseline"],
            ]
            for metric in (f"mrr@{k}", f"line_coverage@{k}", f"complete_gold@{k}"):
                ci = metrics[metric]
                cells.append(f"{ci['estimate']:+.3f} [{ci['low']:+.3f}, {ci['high']:+.3f}]")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if skipped:
        lines.extend(
            [
                "",
                "Unsupported comparisons:",
                *dict.fromkeys(f"- {r['repo']} / {r['strategy']}: {r['reason']}" for r in skipped),
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation limits: a one-line overlap counts as evidence found; it does not prove the retrieved chunk contains a complete answer.",
            "Coverage uses inclusive source line spans; character chunks touching only part of a line count that line. Coverage is not a semantic correctness score.",
            "Reference reconstructed-text gaps are excluded from source evidence; see source_spans in per-question results and chunks_with_source_gaps in summary.json.",
            "Character budgets count all characters; AST budgets count non-whitespace characters. This compares the planned defaults, not equal token budgets.",
            "The corpus includes tracked test/example code allowed by the production file walker. This is code retrieval only; docs and answer generation are outside this benchmark.",
            "",
        ]
    )
    return "\n".join(lines)
