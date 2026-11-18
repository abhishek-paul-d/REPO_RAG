"""Audit Phase 4 failures from exported chunks/vectors, without loading neural models.

Run from the project root with uv run python eval/analyze_retrieval_failures.py
--artifacts data/phase4_colab --output eval/retrieval_results/failure_analysis.
The original results ZIP supplies chunks.json and vector_cache/*.npy. Optional --tokenizer
points to a local Qwen tokenizer.json; token lengths are a local diagnostic, not GPU telemetry.
No retrieval defaults, gold labels or original result files are modified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.dataset import Question, read_questions  # noqa: E402
from eval.retrieval.metrics import score_question, source_spans  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.retrievers import weighted_rrf  # noqa: E402
from repo_rag.store.bm25 import Bm25Index  # noqa: E402


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def target_type(path: str) -> str:
    parts = Path(path).parts
    if set(parts[:-1]) & {"test", "tests", "__tests__"}:
        return "test"
    if set(parts[:-1]) & {"example", "examples"}:
        return "example"
    return "implementation_or_other"


def vectors_for(texts: list[str], settings: Settings, run: dict, folder: Path, query=False):
    identity = {
        "texts": texts,
        "model": settings.local_embedding_model,
        "provider": settings.embedding_provider,
        "dims": settings.embedding_dims,
        "instruction": settings.embedding_query_instruction if query else None,
        "tokens": settings.embedding_max_tokens,
        "dtype": settings.embedding_dtype,
        "device": settings.compute_device,
        "implementation": "LocalEmbeddings",
        "embedding_source": run["source_sha256"]["src/repo_rag/store/embeddings.py"],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    vectors = np.load(folder / f"{digest}.npy", allow_pickle=False)
    assert vectors.shape == (len(texts), settings.embedding_dims)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    return vectors


def overlap(question: Question, doc: Document) -> bool:
    return any(
        doc.metadata["path"] == gold.path and a <= gold.end_line and b >= gold.start_line
        for gold in question.gold
        for a, b in source_spans(doc)
    )


def aggregate(rows: list[dict]) -> dict:
    return {
        "questions": len(rows),
        **{key: float(np.mean([row[key] for row in rows])) for key in rows[0] if "@" in key},
        "all_gold_90_complete_at_10_questions": sum(row["complete_gold@10"] == 1 for row in rows),
        "zero_gold_hit_at_10_questions": sum(row["hit@10"] == 0 for row in rows),
        "top1_type": dict(Counter(row["top1_type"] for row in rows)),
        "mean_top10_body_chars": float(np.mean([row["top10_body_chars"] for row in rows])),
        "mean_top10_test_chunks": float(np.mean([row["top10_test_chunks"] for row in rows])),
        "mean_top10_example_chunks": float(np.mean([row["top10_example_chunks"] for row in rows])),
    }


def location(doc: Document) -> str:
    meta = doc.metadata
    return f"{meta['path']}:{meta['start_line']}-{meta['end_line']}"


def audit_express_context(artifacts: Path, checkout: Path, settings: Settings) -> dict:
    from tree_sitter_language_pack import get_parser

    from repo_rag.chunking.ast_chunker import AstChunker
    from repo_rag.chunking.common import split_header

    get_parser("javascript")  # Fail first if the grammar is unavailable; do not audit fallback.
    exported = read(artifacts / "expressjs__express/full/ast_headers_True/chunks.json")
    production = []
    for path in sorted({row["metadata"]["path"] for row in exported}):
        docs = AstChunker(settings).chunk(
            (checkout / path).read_text(encoding="utf-8"),
            repo="expressjs/express",
            commit_sha=exported[0]["metadata"]["commit_sha"],
            path=path,
            language="javascript",
        )
        assert all(d.metadata["chunker"] == "ast" for d in docs), path
        production.extend(docs)
    by_span = {
        (d.metadata["path"], d.metadata["start_line"], d.metadata["end_line"]): d
        for d in production
    }
    assert len(by_span) == len(exported)
    for row in exported:
        meta = row["metadata"]
        doc = by_span[(meta["path"], meta["start_line"], meta["end_line"])]
        assert split_header(doc.page_content)[1] == split_header(row["text"])[1]
    return {
        "repo": "expressjs/express",
        "chunks": len(production),
        "body_and_span_matches": len(exported),
        "production_signature_chunks": sum(bool(d.metadata["signature"]) for d in production),
        "eval_signature_chunks": sum(bool(r["metadata"]["signature"]) for r in exported),
        "production_lib_signature_chunks": sum(
            bool(d.metadata["signature"])
            for d in production
            if str(d.metadata["path"]).startswith("lib/")
        ),
        "eval_lib_signature_chunks": sum(
            bool(r["metadata"]["signature"])
            for r in exported
            if r["metadata"]["path"].startswith("lib/")
        ),
        "lib_chunks": sum(str(d.metadata["path"]).startswith("lib/") for d in production),
        "limitation": "Header text differs; no new embeddings or retrieval scores calculated for production headers.",
    }


def run_analysis(artifacts: Path, output: Path, tokenizer_path: Path | None) -> dict:
    tokenizer = None
    if tokenizer_path:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_truncation()
        tokenizer.no_padding()

    report: dict[str, Any] = {
        "scope": "Post-hoc diagnostics of frozen original questions; not fresh held-out evidence",
        "inference_repeated": False,
        "candidate_pool_limit": "Deeper ranks and pools use reconstructed exact dense, not logged ANN lists; original ANN@50 and reranker scores were not exported",
        "tokenizer": {
            "file": str(tokenizer_path),
            "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "limitation": "Local cached tokenizer; run metadata did not pin its model revision",
        }
        if tokenizer_path
        else None,
        "variants": [],
        "inputs": [],
    }
    for name, slug in (("requests", "psf__requests"), ("express", "expressjs__express")):
        directory = artifacts / slug / "full"
        run = read(directory / "run.json")
        frozen_path = ROOT / "eval/retrieval_results/colab_full/phase4_v1_source.zip"
        for source_path, expected_hash in run["source_sha256"].items():
            if frozen_path.exists():
                with zipfile.ZipFile(frozen_path) as frozen:
                    source = frozen.read(source_path)
            else:
                source = (ROOT / source_path).read_bytes()
            actual = hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()
            assert actual == expected_hash, source_path
        report["inputs"].append(
            {
                "run": str((directory / "run.json").relative_to(artifacts)),
                "run_sha256": hashlib.sha256((directory / "run.json").read_bytes()).hexdigest(),
                "source_hashes_verified": len(run["source_sha256"]),
                "source_reference": "frozen v1 source ZIP"
                if frozen_path.exists()
                else "working tree",
                "dataset_sha256": run["dataset_sha256"],
            }
        )
        settings = Settings.model_validate(run["settings"])
        questions_path = ROOT / "eval/datasets" / f"{name}_qa.jsonl"
        assert (
            hashlib.sha256(questions_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == run["dataset_sha256"]
        )
        questions = read_questions(questions_path)
        for variant in sorted(directory.glob("*_headers_*")):
            if not variant.is_dir():
                continue
            exported = read(variant / "chunks.json")
            docs = [
                Document(page_content=row["text"], metadata=row["metadata"]) for row in exported
            ]
            by_id = {str(doc.metadata["chunk_id"]): doc for doc in docs}
            ids = list(by_id)
            vectors = vectors_for(
                [d.page_content for d in docs], settings, run, directory.parent / "vector_cache"
            )
            queries = vectors_for(
                [q.question for q in questions],
                settings,
                run,
                directory.parent / "vector_cache",
                query=True,
            )
            bm25 = Bm25Index.build(docs)
            stored = {(r["id"], r["method"]): r for r in read(variant / "per_question.json")}
            rerank_path = (
                artifacts / "rerank_experiment" / slug / "full" / variant.name / "per_question.json"
            )
            rerank_rows = read(rerank_path) if rerank_path.exists() else []
            reranks = {
                (r["id"], r["method"]): r for r in rerank_rows if r["method"].startswith("rerank_")
            }
            cases = []
            for qi, question in enumerate(questions):
                similarity = vectors @ queries[qi]
                dense = [ids[i] for i in np.argsort(-similarity, kind="stable")]
                lexical = [cid for cid, _ in bm25.search(question.question, len(docs))]
                hybrid = [
                    cid
                    for cid, _ in weighted_rrf(
                        {
                            "dense": dense[: settings.retrieval_k],
                            "bm25": lexical[: settings.retrieval_k],
                        },
                        {"dense": settings.dense_weight, "bm25": settings.bm25_weight},
                        settings.rrf_k,
                    )
                ]
                rankings = {
                    "dense_exact": dense,
                    "bm25": lexical,
                    "hybrid_exact_reconstructed": hybrid,
                    "candidate_union": list(dict.fromkeys(dense[:50] + lexical[:50])),
                    "entire_corpus": ids,
                }
                case: dict[str, Any] = {
                    "id": question.id,
                    "question": question.question,
                    "split": question.split,
                    "group": question.group,
                    "category": question.category,
                    "gold": [g.model_dump() for g in question.gold],
                    "gold_symbol_leaf_literal_in_query": any(
                        re.search(
                            r"(?<!\w)" + re.escape(g.symbol.rsplit(".", 1)[-1]) + r"(?!\w)",
                            question.question,
                            re.I,
                        )
                        for g in question.gold
                    ),
                    "methods": {},
                }
                for method, ranking in rankings.items():
                    ranked_docs = [by_id[cid] for cid in ranking]
                    if method in {"candidate_union", "entire_corpus"}:
                        # This is a pool-coverage bound, not an ordered retrieval result.
                        case[method] = {
                            "chunks": len(ranking),
                            **score_question(question, ranked_docs, [max(1, len(ranking))]),
                        }
                        continue
                    scores = score_question(question, ranked_docs, [1, 5, 10, 20, 50])
                    scores["mrr@10"] = score_question(question, ranked_docs, [10])["mrr@10"]
                    expected = stored[
                        (
                            question.id,
                            "hybrid_w0.3_tests1"
                            if method == "hybrid_exact_reconstructed"
                            else method,
                        )
                    ]
                    matches_saved = ranking[:10] == [r["chunk_id"] for r in expected["results"]]
                    if method != "hybrid_exact_reconstructed":
                        assert matches_saved, (name, variant.name, question.id, method)
                        assert all(
                            abs(scores[key] - expected[key]) < 1e-12
                            for key in expected
                            if "@" in key
                        )
                    first = next(
                        (i for i, d in enumerate(ranked_docs, 1) if overlap(question, d)), None
                    )
                    gold_ranks = []
                    for gold in question.gold:
                        single = question.model_copy(update={"gold": [gold]})
                        gold_ranks.append(
                            {
                                "symbol": gold.symbol,
                                "first_overlap_rank": next(
                                    (i for i, d in enumerate(ranked_docs, 1) if overlap(single, d)),
                                    None,
                                ),
                            }
                        )
                    case["methods"][method] = {
                        **scores,
                        "matches_saved_top10": matches_saved,
                        "first_overlap_rank_full_list": first,
                        "per_gold": gold_ranks,
                        "top1_type": target_type(ranked_docs[0].metadata["path"])
                        if ranked_docs
                        else "empty",
                        "top3": [location(d) for d in ranked_docs[:3]],
                        "top10_body_chars": sum(
                            int(d.metadata["eval_body_chars"]) for d in ranked_docs[:10]
                        ),
                        "top10_test_chunks": sum(
                            target_type(d.metadata["path"]) == "test" for d in ranked_docs[:10]
                        ),
                        "top10_example_chunks": sum(
                            target_type(d.metadata["path"]) == "example" for d in ranked_docs[:10]
                        ),
                    }
                for method in ("hybrid_w0.3_tests1", "rerank_20", "rerank_50"):
                    row = (
                        stored[(question.id, method)]
                        if method == "hybrid_w0.3_tests1"
                        else reranks.get((question.id, method))
                    )
                    if row:
                        ranked_docs = [by_id[r["chunk_id"]] for r in row["results"]]
                        scores = score_question(question, ranked_docs, [1, 5, 10])
                        assert all(abs(scores[key] - row[key]) < 1e-12 for key in scores)
                        case["methods"][method] = {
                            **scores,
                            "top1_type": target_type(ranked_docs[0].metadata["path"]),
                            "top3": [location(d) for d in ranked_docs[:3]],
                            "top10_body_chars": sum(
                                int(d.metadata["eval_body_chars"]) for d in ranked_docs[:10]
                            ),
                            "top10_test_chunks": sum(
                                target_type(d.metadata["path"]) == "test" for d in ranked_docs
                            ),
                            "top10_example_chunks": sum(
                                target_type(d.metadata["path"]) == "example" for d in ranked_docs
                            ),
                        }
                # Provenance of the first hybrid result makes rank-agreement effects inspectable.
                top = stored[(question.id, "hybrid_w0.3_tests1")]["results"][0]["chunk_id"]
                case["hybrid_top1_inputs"] = {
                    "location": location(by_id[top]),
                    "dense_rank": dense.index(top) + 1,
                    "bm25_rank": lexical.index(top) + 1 if top in lexical else None,
                    "overlaps_gold": overlap(question, by_id[top]),
                }
                cases.append(case)
            summaries = []
            for method in cases[0]["methods"]:
                for split in ("all", "dev", "test"):
                    chosen = [c for c in cases if split == "all" or c["split"] == split]
                    summaries.append(
                        {
                            "method": method,
                            "split": split,
                            **aggregate([c["methods"][method] for c in chosen]),
                        }
                    )
            token_lengths = (
                [len(tokenizer.encode(d.page_content).ids) for d in docs] if tokenizer else []
            )
            vreport = {
                "repo": run["repo"],
                "variant": variant.name,
                "chunk_count": len(docs),
                "chunk_types": dict(Counter(target_type(d.metadata["path"]) for d in docs)),
                "signature_chunks": sum(bool(d.metadata["signature"]) for d in docs),
                "parent_chunks": sum(bool(d.metadata["parent_symbol"]) for d in docs),
                "tokens": {
                    "max": max(token_lengths),
                    "median": float(np.median(token_lengths)),
                    "over_cap": sum(n > settings.embedding_max_tokens for n in token_lengths),
                    "cap": settings.embedding_max_tokens,
                }
                if token_lengths
                else None,
                "summary": summaries,
                "all_gold_90_complete_questions_in_pools": {
                    key: sum(
                        next(
                            value
                            for metric, value in c[key].items()
                            if metric.startswith("complete_gold@")
                        )
                        == 1
                        for c in cases
                    )
                    for key in ("candidate_union", "entire_corpus")
                },
                "questions": cases,
            }
            report["variants"].append(vreport)
            print(
                name,
                variant.name,
                len(docs),
                "chunks; exact/BM25 rankings and stored metrics reproduced; reconstructed hybrid top10 mismatches:",
                sum(
                    not c["methods"]["hybrid_exact_reconstructed"]["matches_saved_top10"]
                    for c in cases
                ),
                flush=True,
            )
    output.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (output / "failure_analysis.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    summary = {key: value for key, value in report.items() if key != "variants"}
    summary["variants"] = [
        {key: value for key, value in v.items() if key != "questions"} for v in report["variants"]
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    focus = [v for v in report["variants"] if v["variant"] == "ast_headers_True"]
    lines = [
        "# Question-level failure audit",
        "",
        "Frozen original gold; post-hoc diagnosis, not a new held-out evaluation.",
        "",
        "Ranks below use any overlap, not completeness. A dash means no overlap in the stored hybrid/reranked top 10.",
        "",
        "Dense/BM25 ranks beyond 10 are reconstructed from cached vectors and exported chunks. Original ANN@50 lists were not logged.",
        "",
    ]
    for variant in focus:
        lines += [
            f"## {variant['repo']}",
            "",
            "| Question | Split | Dense first rank | BM25 first rank | Hybrid first rank | Rerank20 first rank | Hybrid ≥90% gold@10 |",
            "|---|---|---|---|---|---|---|",
        ]
        for case in variant["questions"]:
            methods = case["methods"]
            rr = methods["rerank_20"]["mrr@10"]
            ranks = [methods[m]["first_overlap_rank_full_list"] for m in ("dense_exact", "bm25")]
            hybrid_mrr = methods["hybrid_w0.3_tests1"]["mrr@10"]
            ranks.append(round(1 / hybrid_mrr) if hybrid_mrr else None)
            lines.append(
                f"| {case['id']}: {case['question'].replace('|', '/')} | {case['split']} | "
                + " | ".join(str(r) if r else "—" for r in ranks)
                + f" | {round(1 / rr) if rr else '—'} | {methods['hybrid_w0.3_tests1']['complete_gold@10']:.3f} |"
            )
        lines.append("")
    (output / "questions.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "data/phase4_colab")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "eval/retrieval_results/failure_analysis"
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--express-checkout",
        type=Path,
        help="Optional source/header parity audit; requires cached JavaScript grammar",
    )
    args = parser.parse_args()
    run_analysis(args.artifacts, args.output, args.tokenizer)
    if args.express_checkout:
        settings = Settings.model_validate(
            read(args.artifacts / "expressjs__express/full/run.json")["settings"]
        )
        context = audit_express_context(args.artifacts, args.express_checkout, settings)
        (args.output / "context_audit.json").write_text(
            json.dumps(context, indent=2) + "\n", encoding="utf-8"
        )
        print("Context parity:", context, flush=True)
