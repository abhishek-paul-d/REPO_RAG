"""Fixed v2 policies, complete traces, required/supporting scoring and measured latency.

No weight selection or production-default mutation. All primary policies run independently
through query encoding and retrieval on each repetition. Tests inject offline scorers only.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from eval.retrieval.anchors import Anchor, definitions
from eval.retrieval.dataset import Question, write_json
from eval.retrieval.metrics import score_question, source_spans
from eval.retrieval.runner import CachedEmbeddings, cached_vectors, fuse
from eval.retrieval.statistics import paired_intervals, summarize
from repo_rag.chunking.ast_chunker import AstChunker
from repo_rag.chunking.common import split_header
from repo_rag.config import Settings
from repo_rag.ingest.file_walker import walk_repo
from repo_rag.retrieval.rerank import Reranker
from repo_rag.retrieval.retrievers import weighted_rrf
from repo_rag.store.bm25 import Bm25Index
from repo_rag.store.embeddings import truncate_and_normalise
from repo_rag.store.vectorstore import VectorStore

PRIMARY = ("dense_hnsw", "hybrid_w0.3_tests1", "rerank_20", "union_bm25q5_rerank20")


def production_corpus(checkout: Path, repo: str, sha: str, settings: Settings) -> list[Document]:
    """Use the actual production chunk metadata/header policy for the deployment experiment."""
    docs = []
    for file in walk_repo(checkout, max_file_bytes=settings.max_file_bytes):
        if file.kind != "code":
            continue
        chunks = AstChunker(settings).chunk(
            file.abs_path.read_text(encoding="utf-8", errors="replace"),
            repo=repo,
            commit_sha=sha,
            path=file.path,
            language=file.language,
        )
        for doc in chunks:
            doc.metadata["eval_body_chars"] = len(split_header(doc.page_content)[1])
        docs.extend(chunks)
    if not docs or len({d.metadata["chunk_id"] for d in docs}) != len(docs):
        raise ValueError("empty corpus or duplicate production chunk ids")
    return docs


def quota_union(fused: list[str], lexical: list[str], depth: int, quota: int) -> list[str]:
    if not 0 <= quota <= depth:
        raise ValueError("BM25 quota must lie between zero and candidate depth")
    return list(dict.fromkeys([*lexical[:quota], *fused]))[:depth]


def symbol_candidates(query: str, docs: list[Document], checkout: Path) -> list[str]:
    """Resolve explicit identifiers from source only, without reading any question gold."""
    names = set(re.findall(r"`([A-Za-z_][\w.]*)`", query))
    if not names:
        return []
    anchors: list[tuple[str, Anchor]] = []
    for path in sorted({str(d.metadata["path"]) for d in docs}):
        language = next(str(d.metadata["language"]) for d in docs if d.metadata["path"] == path)
        if language not in {"python", "javascript"}:
            continue
        try:
            found = definitions((checkout / path).read_text(encoding="utf-8"), language)
        except SyntaxError:
            continue
        anchors.extend(
            (path, a) for a in found if a.symbol in names or a.symbol.rsplit(".", 1)[-1] in names
        )
    return [
        str(d.metadata["chunk_id"])
        for d in docs
        if any(
            d.metadata["path"] == path
            and int(d.metadata["start_line"]) <= a.end
            and int(d.metadata["end_line"]) >= a.start
            for path, a in anchors
        )
    ]


def score_row(
    q: Question, method: str, ranking: list[str], docs: dict[str, Document], settings: Settings
) -> dict:
    selected = [docs[cid] for cid in ranking[: max(settings.eval_cutoffs)]]
    supporting = (
        score_question(
            q.model_copy(update={"gold": q.supporting_gold}),
            selected,
            settings.eval_cutoffs,
            coverage_threshold=settings.eval_coverage_threshold,
        )
        if q.supporting_gold
        else None
    )
    return {
        "id": q.id,
        "split": q.split,
        "group": q.group,
        "category": q.category,
        "cohort": q.cohort,
        "target_type": q.target_type,
        "method": method,
        **score_question(
            q, selected, settings.eval_cutoffs, coverage_threshold=settings.eval_coverage_threshold
        ),
        "supporting_scores": supporting,
        "top10_body_chars": sum(int(d.metadata["eval_body_chars"]) for d in selected),
        "results": [
            {
                "chunk_id": str(d.metadata["chunk_id"]),
                "path": d.metadata["path"],
                "start_line": d.metadata["start_line"],
                "end_line": d.metadata["end_line"],
                "source_spans": source_spans(d),
            }
            for d in selected
        ],
    }


def evaluate_followup(
    docs: list[Document],
    questions: list[Question],
    settings: Settings,
    output: Path,
    embeddings: Embeddings,
    reranker: Reranker | None,
    *,
    checkout: Path | None = None,
    synchronize: Callable[[], None] = lambda: None,
    reset_memory: Callable[[], None] = lambda: None,
    memory_snapshot: Callable[[], dict] = dict,
    legacy_questions: Sequence[Question] = (),
) -> dict:
    depth, quota = settings.eval_followup_candidate_depth, settings.eval_followup_bm25_quota
    if (
        depth != 20
        or quota != 5
        or settings.retrieval_k != 50
        or settings.bm25_weight != 0.3
        or settings.dense_weight != 1
        or settings.rrf_k != 60
    ):
        raise ValueError("follow-up settings do not match the fixed matrix")
    if any(q.schema_version != 2 for q in questions):
        raise ValueError("follow-up requires versioned v2 questions")
    output.mkdir(parents=True, exist_ok=True)
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    ids = list(by_id)
    index = Bm25Index.build(docs)
    started = time.perf_counter()
    vectors = cached_vectors(
        [d.page_content for d in docs], settings, output / "vector_cache", embeddings, query=False
    )
    queries = cached_vectors(
        [q.question for q in questions], settings, output / "vector_cache", embeddings, query=True
    )
    vector_seconds = time.perf_counter() - started
    query_cache = CachedEmbeddings(questions, queries)
    store = VectorStore(
        settings.model_copy(update={"data_dir": output / "isolated_index"}), query_cache
    )
    store.replace_records(
        "code",
        questions[0].repo,
        ids,
        [d.page_content for d in docs],
        [d.metadata for d in docs],
        vectors,
    )
    cold = None
    if reranker is not None:
        synchronize()
        started = time.perf_counter()
        warm_texts = [d.page_content for d in docs[:depth]]
        reranker.score(questions[0].question, warm_texts)
        synchronize()
        cold = time.perf_counter() - started
        reranker.score(questions[0].question, warm_texts)
    store.search("code", questions[0].repo, questions[0].question, settings.retrieval_k)
    index.search(questions[0].question, settings.retrieval_k)
    rows: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    legacy_by_id = {q.id: q for q in legacy_questions}

    def record(q: Question, method: str, ranking: list[str]) -> dict:
        row = score_row(q, method, ranking, by_id, settings)
        if q.id in legacy_by_id:
            row["v1_scores"] = score_question(
                legacy_by_id[q.id],
                [by_id[cid] for cid in ranking[: max(settings.eval_cutoffs)]],
                settings.eval_cutoffs,
                coverage_threshold=settings.eval_coverage_threshold,
            )
        return row

    trace_file = output / "traces.jsonl.gz"
    trace_file.write_bytes(b"")
    methods = PRIMARY if reranker else ("dense_hnsw", "hybrid_w0.3_tests1")
    for qi, q in enumerate(questions):
        traces = []
        lookup = symbol_candidates(q.question, docs, checkout) if checkout else []
        for repetition in range(settings.eval_followup_latency_repeats):
            # Rotate method order to reduce systematic ordering/warm-cache effects.
            order = methods[repetition % len(methods) :] + methods[: repetition % len(methods)]
            for method in order:
                synchronize()
                reset_memory()
                memory_before = memory_snapshot()
                started = time.perf_counter()
                vector = truncate_and_normalise(
                    np.asarray([embeddings.embed_query(q.question)], dtype=np.float32),
                    settings.embedding_dims,
                )[0]
                query_cache.queries[q.question] = vector.tolist()
                synchronize()
                encoded_at = time.perf_counter()
                dense_hits = store.search("code", q.repo, q.question, settings.retrieval_k)
                dense = [str(d.metadata["chunk_id"]) for d, _ in dense_hits]
                lexical_pairs = (
                    index.search(q.question, settings.retrieval_k) if method != "dense_hnsw" else []
                )
                lexical = [cid for cid, _ in lexical_pairs]
                fused = fuse(dense, lexical, by_id, settings, settings.bm25_weight, 1.0)
                candidates = []
                if method == "dense_hnsw":
                    ranking = dense
                elif method == "hybrid_w0.3_tests1":
                    ranking = fused
                else:
                    candidates = (
                        quota_union(fused, lexical, depth, quota)
                        if method == "union_bm25q5_rerank20"
                        else fused[:depth]
                    )
                    ranking = candidates
                synchronize()
                retrieved_at = time.perf_counter()
                pair_scores = []
                if candidates:
                    assert reranker is not None
                    pair_scores = reranker.score(
                        q.question, [by_id[cid].page_content for cid in candidates]
                    )
                    if len(pair_scores) != len(candidates) or not np.isfinite(pair_scores).all():
                        raise ValueError("invalid pair scores")
                    ranking = [
                        cid
                        for cid, _ in sorted(
                            zip(candidates, pair_scores, strict=True),
                            key=lambda p: p[1],
                            reverse=True,
                        )
                    ]
                synchronize()
                ended = time.perf_counter()
                token_audit = getattr(reranker, "pair_token_counts", None)
                pair_tokens = (
                    token_audit(q.question, [by_id[cid].page_content for cid in candidates])
                    if candidates and token_audit
                    else []
                )
                elapsed = {
                    "id": q.id,
                    "repo": q.repo,
                    "cohort": q.cohort,
                    "target_type": q.target_type,
                    "method": method,
                    "repetition": repetition,
                    "query_ms": (encoded_at - started) * 1000,
                    "retrieval_ms": (retrieved_at - encoded_at) * 1000,
                    "rerank_ms": (ended - retrieved_at) * 1000,
                    "end_to_end_ms": (ended - started) * 1000,
                }
                timing.append(elapsed)
                trace = {
                    **elapsed,
                    "cuda_memory_before": memory_before,
                    "cuda_memory_after": memory_snapshot(),
                    "dense": [
                        {
                            "chunk_id": str(d.metadata["chunk_id"]),
                            "rank": r,
                            "cosine_similarity": float(s),
                        }
                        for r, (d, s) in enumerate(dense_hits, 1)
                    ],
                    "bm25": [
                        {"chunk_id": cid, "rank": r, "score": float(s)}
                        for r, (cid, s) in enumerate(lexical_pairs, 1)
                    ],
                    "fused_order": fused,
                    "fusion_scores": [
                        {"chunk_id": cid, "score": float(score)}
                        for cid, score in weighted_rrf(
                            {"dense": dense, "bm25": lexical},
                            {"dense": settings.dense_weight, "bm25": settings.bm25_weight},
                            settings.rrf_k,
                        )
                    ],
                    "candidates": candidates,
                    "final_order": ranking,
                    "pairs": [
                        {
                            "chunk_id": cid,
                            "score": float(s),
                            **(pair_tokens[i] if pair_tokens else {}),
                        }
                        for i, (cid, s) in enumerate(zip(candidates, pair_scores, strict=True))
                    ],
                }
                if repetition == 0:
                    rows.append(record(q, method, ranking))
                if method == "dense_hnsw" and repetition == 0:
                    similarities = vectors @ vector
                    exact = [
                        ids[i]
                        for i in np.argsort(-similarities, kind="stable")[: settings.retrieval_k]
                    ]
                    trace["exact_dense"] = [
                        {
                            "chunk_id": cid,
                            "rank": r,
                            "cosine_similarity": float(similarities[ids.index(cid)]),
                        }
                        for r, cid in enumerate(exact, 1)
                    ]
                    trace["ann_recall"] = {
                        str(k): len(
                            set(dense[:k])
                            & {
                                ids[i]
                                for i in range(len(ids))
                                if similarities[i]
                                >= similarities[ids.index(exact[min(k, len(exact)) - 1])] - 1e-6
                            }
                        )
                        / min(k, len(ids))
                        for k in (*settings.eval_cutoffs, settings.retrieval_k)
                    }
                    lex = [cid for cid, _ in index.search(q.question, settings.retrieval_k)]
                    diagnostic = {
                        "dense_exact": exact,
                        "bm25": lex,
                        "hybrid_tests0.8_diagnostic": fuse(
                            dense, lex, by_id, settings, settings.bm25_weight, 0.8
                        ),
                        "symbol_route_dense": list(dict.fromkeys([*lookup, *dense])),
                    }
                    rows.extend(record(q, name, ranks) for name, ranks in diagnostic.items())
                    if q.id in settings.eval_followup_trace_ids and reranker is not None:
                        pool = fuse(dense, lex, by_id, settings, settings.bm25_weight, 1.0)[
                            : settings.retrieval_k
                        ]
                        trace_scores = reranker.score(
                            q.question, [by_id[cid].page_content for cid in pool]
                        )
                        if len(trace_scores) != len(pool) or not np.isfinite(trace_scores).all():
                            raise ValueError("invalid top50 trace scores")
                        trace["top50_diagnostic_pairs"] = [
                            {"chunk_id": cid, "score": float(s)}
                            for cid, s in zip(pool, trace_scores, strict=True)
                        ]
                traces.append(trace)
        with trace_file.open("ab") as stream:
            stream.write(
                gzip.compress(
                    "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces).encode(),
                    mtime=0,
                )
            )
        write_json(output / "per_question.json", rows)  # Retain completed questions if interrupted.
        print(f"Scored {qi + 1}/{len(questions)} {q.id}", flush=True)
    summaries, comparisons = [], []
    for cohort in ("fresh", "legacy_regression"):
        for stratum in ("all", "implementation", "test", "example", "symbol_lookup"):
            subset = [
                r
                for r in rows
                if r["cohort"] == cohort
                and (
                    stratum == "all"
                    or r["target_type"] == stratum
                    or (stratum == "symbol_lookup" and r["category"] == stratum)
                )
            ]
            if not subset:
                continue
            for summary in summarize(subset, settings):
                if summary["split"] == "all":
                    summaries.append({"cohort": cohort, "stratum": stratum, **summary})
            base = [r for r in subset if r["method"] == "hybrid_w0.3_tests1"]
            for method in (
                "dense_hnsw",
                "rerank_20",
                "union_bm25q5_rerank20",
                "hybrid_tests0.8_diagnostic",
                "symbol_route_dense",
            ):
                candidate = [r for r in subset if r["method"] == method]
                if candidate:
                    comparisons.append(
                        {
                            "cohort": cohort,
                            "stratum": stratum,
                            "candidate": method,
                            "baseline": "hybrid_w0.3_tests1",
                            "metrics": paired_intervals(
                                candidate,
                                base,
                                ["mrr@10", "complete_gold@10", "recall@10"],
                                settings,
                            ),
                        }
                    )
    latencies = []
    for method in methods:
        for cohort in ("fresh", "legacy_regression"):
            for stratum in ("all", "implementation", "test", "example"):
                values = [
                    r["end_to_end_ms"]
                    for r in timing
                    if r["method"] == method
                    and r["cohort"] == cohort
                    and (stratum == "all" or r["target_type"] == stratum)
                ]
                if values:
                    latencies.append(
                        {
                            "method": method,
                            "cohort": cohort,
                            "stratum": stratum,
                            "observations": len(values),
                            "p50_ms": float(np.quantile(values, 0.5)),
                            "p95_ms": float(np.quantile(values, 0.95)),
                            "repetitions": settings.eval_followup_latency_repeats,
                        }
                    )
    write_json(output / "per_question.json", rows)
    write_json(output / "summary.json", summaries)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "latency.json", latencies)
    supporting_rows = [
        {
            "id": r["id"],
            "split": "test",
            "group": r["group"],
            "method": r["method"],
            **r["supporting_scores"],
        }
        for r in rows
        if r["supporting_scores"] is not None
    ]
    supporting_summary = summarize(supporting_rows, settings) if supporting_rows else []
    write_json(output / "supporting_summary.json", supporting_summary)
    v1 = [
        {
            "id": r["id"],
            "split": "test",
            "group": legacy_by_id[r["id"]].group,
            "method": r["method"],
            **r["v1_scores"],
        }
        for r in rows
        if "v1_scores" in r
    ]
    write_json(output / "v1_regression.json", summarize(v1, settings) if v1 else [])
    write_json(
        output / "chunks.json", [{"metadata": d.metadata, "text": d.page_content} for d in docs]
    )
    tokenizer = getattr(getattr(embeddings, "_model", None), "tokenizer", None)
    tokens: dict[str, Any] | None = None
    if tokenizer is not None:
        tokens = {
            "documents": [
                len(ids)
                for ids in tokenizer([d.page_content for d in docs], truncation=False)["input_ids"]
            ],
            "queries": [
                len(ids)
                for ids in tokenizer(
                    [settings.embedding_query_instruction + q.question for q in questions],
                    truncation=False,
                )["input_ids"]
            ],
            "embedding_cap": settings.embedding_max_tokens,
            "rerank_cap": settings.rerank_max_tokens,
        }
        token_by_id = dict(zip(ids, tokens["documents"], strict=True))
        for row in rows:
            row["top10_embedding_tokens"] = sum(token_by_id[r["chunk_id"]] for r in row["results"])
        write_json(output / "per_question.json", rows)
    write_json(output / "token_counts.json", tokens)
    write_json(
        output / "preparation.json",
        {
            "vector_preparation_seconds": vector_seconds,
            "reranker_cold_first_pool_seconds": cold,
            "timing": "independent synchronized primary policy calls; excludes tokenizer diagnostics and tracing writes",
            "quality_repetition": 0,
            "reranker_model_revision": getattr(
                getattr(getattr(reranker, "model", None), "config", None), "_commit_hash", None
            ),
            "embedding_model_revision": getattr(
                getattr(
                    getattr(cast(Any, embeddings)._model._first_module(), "auto_model", None),
                    "config",
                    None,
                ),
                "_commit_hash",
                None,
            )
            if getattr(embeddings, "_model", None) is not None
            else None,
        },
    )
    return {
        "summary": summaries,
        "comparisons": comparisons,
        "latency": latencies,
        "v1_regression": summarize(v1, settings) if v1 else [],
        "supporting_summary": supporting_summary,
    }


def render_report(result: dict, repo: str) -> str:
    lines = [
        f"# Phase 4 v2: {repo}",
        "",
        "Fixed matrix; no weight tuning. Gold means required evidence. Supporting evidence is recorded separately per question.",
        "",
        "| Cohort | Target stratum | Method | Questions | MRR@10 [95% CI] | Required ≥90%@10 [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["summary"]:
        cells = []
        for metric in ("mrr@10", "complete_gold@10"):
            ci = row["confidence_intervals"][metric]
            cells.append(f"{ci['estimate']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]")
        lines.append(
            f"| {row['cohort']} | {row['stratum']} | {row['method']} | {row['questions']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Warm measured latency",
        "",
        "Includes query embedding, retrieval/fusion and actual pair scoring in each policy call. Excludes cold model load, optional symbol diagnostics and report writes. Repetitions are not independent questions.",
        "",
        "| Cohort | Stratum | Method | Calls | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["latency"]:
        lines.append(
            f"| {row['cohort']} | {row['stratum']} | {row['method']} | {row['observations']} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} |"
        )
    lines += [
        "",
        "## Frozen v1 gold regression",
        "",
        "Current rankings rescored against unchanged v1 evidence, separate from the reviewed v2 required/supporting labels.",
        "",
        "| Method | Questions | MRR@10 | Whole-gold ≥90%@10 |",
        "|---|---|---|---|",
    ]
    for row in result.get("v1_regression", []):
        if row["split"] == "all":
            lines.append(
                f"| {row['method']} | {row['questions']} | {row['mrr@10']:.3f} | {row['complete_gold@10']:.3f} |"
            )
    lines += [
        "",
        "## Optional supporting evidence",
        "",
        "Only questions with explicit supporting labels enter this table. Supporting coverage is not required-answer correctness.",
        "",
        "| Method | Labelled questions | Supporting ≥90%@10 |",
        "|---|---|---|",
    ]
    for row in result.get("supporting_summary", []):
        if row["split"] == "all":
            lines.append(
                f"| {row['method']} | {row['questions']} | {row['complete_gold@10']:.3f} |"
            )
    lines += [
        "",
        "Legacy regression rows include previously inspected questions and changed v2 required/supporting labels; do not compare their scores directly with v1 gold scores.",
        "Fresh means newly authored questions, not guaranteed unseen training data or independent authorship. Some new symbol lookups target previously evaluated implementations.",
        "Intervals resample evidence groups, are pointwise/exploratory and are not multiplicity-adjusted. No production setting is changed by this report.",
    ]
    return "\n".join(lines) + "\n"
