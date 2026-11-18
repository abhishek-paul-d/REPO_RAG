"""Phase 4 CLI. Heavy full runs belong on Colab; validation and BM25 run offline."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.chunkers import STRATEGIES, UnsupportedStrategy, chunk_corpus  # noqa: E402
from eval.retrieval.dataset import read_questions, validate_checkout, write_json  # noqa: E402
from eval.retrieval.runner import environment, evaluate_variant, markdown_report, select_on_dev  # noqa: E402
from eval.retrieval.statistics import compare_strategies  # noqa: E402
from repo_rag.config import Settings  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--checkout", type=Path, help="clean checkout of dataset's pinned commit")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--strategies", default=",".join(STRATEGIES))
    p.add_argument("--headers", choices=("both", "on", "off"), default="both")
    p.add_argument("--mode", choices=("validate", "bm25", "full"), default="full")
    p.add_argument("--output", type=Path, default=ROOT / "data/retrieval_eval")
    p.add_argument("--rerank", action="store_true", help="GPU-only top-20/top-50 code experiment")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    settings = Settings()
    strategies = args.strategies.split(",")
    if not strategies or set(strategies) - set(STRATEGIES):
        raise ValueError(f"strategies must come from {STRATEGIES}")
    if any(w < 0 for w in settings.eval_bm25_weights) or any(
        not 0 < p <= 1 for p in settings.eval_test_penalties
    ):
        raise ValueError("invalid evaluation weights/penalties")
    dataset = args.dataset or ROOT / "eval/datasets" / f"{args.repo.split('/')[-1]}_qa.jsonl"
    questions = read_questions(dataset)
    if questions[0].repo != args.repo:
        raise ValueError("dataset repo does not match --repo")
    checkout = args.checkout or settings.clones_dir / args.repo.replace("/", "__")
    validate_checkout(checkout, questions)
    if args.mode == "validate":
        print(f"Validated {len(questions)} questions at {questions[0].commit_sha}")
        return
    output = args.output.resolve() / args.repo.replace("/", "__") / args.mode
    # An explicit output must never overlap the production data stores or the source checkout.
    for protected in (settings.chroma_dir, settings.bm25_dir, checkout):
        protected = protected.resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"evaluation output overlaps protected data: {protected}")
    embeddings = reranker = None
    if args.mode == "full":
        from repo_rag.store.embeddings import get_embeddings

        embeddings = get_embeddings(settings)
    if args.rerank:
        from repo_rag.retrieval.rerank import QwenReranker
        from repo_rag.store.embeddings import resolve_device

        if args.mode != "full" or resolve_device(settings.compute_device) != "cuda":
            raise ValueError("--rerank requires full mode on a CUDA GPU")
        reranker = QwenReranker(settings, instruction=settings.eval_code_instruction)
    all_rows: list[dict[str, Any]] = []
    all_questions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for strategy in strategies:
        for headers in [False, True] if args.headers == "both" else [args.headers == "on"]:
            print(f"{args.repo}: {strategy}, headers={headers}, mode={args.mode}", flush=True)
            try:
                docs = chunk_corpus(
                    checkout, args.repo, questions[0].commit_sha, strategy, headers, settings
                )
            except UnsupportedStrategy as exc:
                skipped.append(
                    {
                        "repo": args.repo,
                        "strategy": strategy,
                        "headers": headers,
                        "reason": str(exc),
                    }
                )
                print(f"N/A: {exc}", flush=True)
                continue
            variant = output / f"{strategy}_headers_{headers}"
            rows, per_question, stats = evaluate_variant(
                docs, questions, settings, variant, args.mode, embeddings, reranker
            )
            all_rows.extend(
                {"repo": args.repo, "strategy": strategy, "headers": headers, **stats, **row}
                for row in rows
            )
            all_questions.extend(
                {"repo": args.repo, "strategy": strategy, "headers": headers, **q}
                for q in per_question
            )
            write_json(
                output / "summary.json", all_rows
            )  # retain completed variants if a later one fails
    if not all_rows:
        raise ValueError("no supported comparisons ran")
    selected = select_on_dev(all_rows, settings.eval_cutoffs)
    comparisons = compare_strategies(all_questions, settings)
    write_json(output / "selected_on_dev.json", selected)
    write_json(output / "comparisons.json", comparisons)
    package_hashes = {
        str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in (ROOT / "src/repo_rag", ROOT / "eval/retrieval")
        for p in sorted(directory.rglob("*.py"))
    }
    write_json(
        output / "run.json",
        {
            "repo": args.repo,
            "commit_sha": questions[0].commit_sha,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "settings": settings.model_dump(
                mode="json", exclude={"anthropic_api_key", "github_token", "voyage_api_key"}
            ),
            "mode": args.mode,
            "rerank": args.rerank,
            "source_sha256": package_hashes,
            "environment": environment(),
            "skipped": skipped,
            "statistics": {
                "bootstrap_unit": "evidence_group",
                "interval": "pointwise percentile",
                "multiplicity_adjusted": False,
                "pooled_fixed_comparisons": "descriptive; not held out from earlier model selection or future tuning",
                "coverage_threshold": settings.eval_coverage_threshold,
            },
        },
    )
    report = output / "summary.md"
    report.write_text(
        markdown_report(
            all_rows, skipped, args.mode, settings, comparisons=comparisons, selected=selected
        ),
        encoding="utf-8",
    )
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
