"""Run the pre-chosen Phase 4 v2 matrix; full model inference requires CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.dataset import read_questions, validate_checkout, write_json  # noqa: E402
from eval.retrieval.followup import evaluate_followup, production_corpus, render_report, score_row  # noqa: E402
from eval.retrieval.promotion import promotion_check  # noqa: E402
from eval.retrieval.runner import environment  # noqa: E402
from eval.retrieval.statistics import summarize  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.store.bm25 import Bm25Index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/phase4_followup")
    parser.add_argument("--mode", choices=("validate", "bm25", "full"), default="full")
    args = parser.parse_args()
    questions = read_questions(args.dataset)
    if questions[0].repo != args.repo or any(q.schema_version != 2 for q in questions):
        raise ValueError("repo/dataset/version mismatch")
    validate_checkout(args.checkout, questions)
    legacy_path = ROOT / "eval/datasets" / f"{args.repo.split('/')[-1]}_qa.jsonl"
    legacy_questions = read_questions(legacy_path) if legacy_path.exists() else []
    if args.mode == "validate":
        print(f"Validated {len(questions)} v2 questions for {args.repo}")
        return
    matrix_path = ROOT / "eval/datasets/phase4_v2/matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    settings = Settings()
    fixed = {
        "local_embedding_model": matrix["embedding_model"],
        "embedding_dims": matrix["embedding_dims"],
        "embedding_dtype": matrix["embedding_dtype"],
        "reranker_model": matrix["reranker_model"],
        "rerank_max_tokens": matrix["rerank_max_tokens"],
        "code_rerank_instruction": matrix["code_rerank_instruction"],
        "retrieval_k": matrix["first_stage_depth_per_signal"],
        "dense_weight": matrix["dense_weight"],
        "bm25_weight": matrix["bm25_weight"],
        "rrf_k": matrix["rrf_k"],
        "eval_followup_candidate_depth": matrix["rerank_depth"],
        "eval_followup_bm25_quota": matrix["bm25_quota"],
        "eval_followup_latency_repeats": matrix["latency_repetitions"],
        "eval_cutoffs": (1, 5, 10),
    }
    settings = settings.model_copy(update=fixed)
    docs = production_corpus(args.checkout, args.repo, questions[0].commit_sha, settings)
    output = args.output / args.repo.replace("/", "__")
    if (output / "run.json").exists():
        raise FileExistsError(
            "completed output already exists; choose a new --output to preserve it"
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "bm25":
        index = Bm25Index.build(docs)
        by_id = {str(d.metadata["chunk_id"]): d for d in docs}
        rows = [
            score_row(
                q,
                "bm25",
                [cid for cid, _ in index.search(q.question, settings.retrieval_k)],
                by_id,
                settings,
            )
            for q in questions
        ]
        summaries: list[dict[str, Any]] = []
        for cohort in ("fresh", "legacy_regression"):
            for target in ("all", "implementation", "test", "example", "symbol_lookup"):
                chosen = [
                    r
                    for r in rows
                    if r["cohort"] == cohort
                    and (
                        target == "all"
                        or r["target_type"] == target
                        or (target == "symbol_lookup" and r["category"] == target)
                    )
                ]
                if chosen:
                    summaries.extend(
                        {"cohort": cohort, "stratum": target, **s}
                        for s in summarize(chosen, settings)
                        if s["split"] == "all"
                    )
        result = {"summary": summaries, "latency": [], "comparisons": []}
        write_json(output / "per_question.json", rows)
        write_json(output / "summary.json", summaries)
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("full v2 matrix requires a CUDA GPU; use the compact Colab notebook")
        from repo_rag.retrieval.rerank import QwenReranker
        from repo_rag.store.embeddings import LocalEmbeddings

        settings = settings.model_copy(update={"compute_device": "cuda"})
        embeddings = LocalEmbeddings(settings)
        reranker = QwenReranker(settings, instruction=settings.code_rerank_instruction)
        result = evaluate_followup(
            docs,
            questions,
            settings,
            output,
            embeddings,
            reranker,
            checkout=args.checkout,
            synchronize=torch.cuda.synchronize,
            reset_memory=torch.cuda.reset_peak_memory_stats,
            memory_snapshot=lambda: {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
            legacy_questions=legacy_questions,
        )
    sources = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for folder in (ROOT / "src/repo_rag", ROOT / "eval/retrieval")
        for p in sorted(folder.rglob("*.py"))
    }
    for path in (Path(__file__).resolve(), matrix_path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
    write_json(
        output / "run.json",
        {
            "repo": args.repo,
            "commit_sha": questions[0].commit_sha,
            "dataset_sha256": hashlib.sha256(
                args.dataset.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "matrix": matrix,
            "mode": args.mode,
            "environment": environment(),
            "source_sha256": sources,
            "settings": settings.model_dump(
                mode="json", exclude={"anthropic_api_key", "github_token", "voyage_api_key"}
            ),
            "independent_review_complete": all(
                q.review_status == "independently_reviewed"
                for q in questions
                if q.cohort == "fresh"
            ),
            "production_default_changed": False,
        },
    )
    mode_note = (
        "**BM25-only preflight: GPU ranking and latency are unmeasured.**\n\n"
        if args.mode == "bm25"
        else ""
    )
    (output / "summary.md").write_text(
        mode_note + render_report(result, args.repo), encoding="utf-8"
    )
    if args.mode == "full":
        write_json(
            output / "decision.json",
            promotion_check(
                result["comparisons"],
                all(
                    q.review_status == "independently_reviewed"
                    for q in questions
                    if q.cohort == "fresh"
                ),
            ),
        )
    print(f"Saved {output / 'summary.md'}")


if __name__ == "__main__":
    main()
