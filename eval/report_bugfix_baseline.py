"""Audit and report the real BM25-only Phase 5 preflight, without inventing neural rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analyze_bugfix_run import audit_method  # noqa: E402
from eval.prepare_bugfix_eval import verify_corpus, write  # noqa: E402
from eval.retrieval.statistics import bootstrap_intervals, paired_intervals  # noqa: E402
from eval.run_bugfix_eval import METRICS, changed_paths, file_overlap, score  # noqa: E402
from repo_rag.config import Settings  # noqa: E402


def report(
    datasets: Path, runs: Path, output: Path, previous_runs: Path | None = None
) -> list[dict]:
    if output.exists():
        raise FileExistsError("Preserve earlier reports; choose another output")
    rows = []
    for dataset in sorted(datasets.iterdir()):
        verify_corpus(dataset)
        for method in ("bm25", "bm25_2000"):
            rows.extend(audit_method(dataset, runs / dataset.name / method, method))
    summaries = []
    for method in ("bm25", "bm25_2000"):
        for repo in ["all", *sorted({row["repo"] for row in rows})]:
            for split in ["all", "dev", "test"]:
                sample = [
                    r
                    for r in rows
                    if r["method"] == method
                    and (repo == "all" or r["repo"] == repo)
                    and (split == "all" or r["split"] == split)
                ]
                intervals = bootstrap_intervals(sample, METRICS, Settings())
                summaries.append(
                    {
                        "method": method,
                        "repo": repo,
                        "split": split,
                        "questions": len(sample),
                        "candidate_pools": {r["repo"]: r["candidate_pool"] for r in sample},
                        "groups": len({r["group"] for r in sample}),
                        **{metric: ci["estimate"] for metric, ci in intervals.items()},
                        "intervals": intervals,
                    }
                )
    output.mkdir(parents=True)
    write(output / "summary.json", summaries)
    length_rows = []
    for method in ("bm25", "bm25_2000"):
        for repo in sorted({r["repo"] for r in rows}):
            for length in sorted({r["query_bin"] for r in rows}):
                sample = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["repo"] == repo
                    and r["split"] == "test"
                    and r["query_bin"] == length
                ]
                if sample:
                    length_rows.append(
                        {
                            "method": method,
                            "repo": repo,
                            "query_bin": length,
                            "questions": len(sample),
                            "intervals": bootstrap_intervals(sample, METRICS, Settings()),
                        }
                    )
    write(output / "query_length_summary.json", length_rows)
    if previous_runs is not None:
        comparisons = []
        old_rows = []
        for dataset in sorted(datasets.iterdir()):
            paths = changed_paths(dataset)
            old = json.loads(
                (previous_runs / dataset.name / "bm25/per_question.json").read_text(
                    encoding="utf-8"
                )
            )
            for row in old:
                if any(row[m] != value for m, value in score(row["ranking"], row["gold"]).items()):
                    raise ValueError("Historical metric does not reproduce")
                row["file_hit@5"] = file_overlap(row["ranking"], row["gold"], paths)
            old_rows.extend(old)
        for repo in ["all", *sorted({r["repo"] for r in rows})]:
            for split in ("all", "dev", "test"):
                current = [
                    r
                    for r in rows
                    if r["method"] == "bm25"
                    and (repo == "all" or r["repo"] == repo)
                    and (split == "all" or r["split"] == split)
                ]
                previous = [
                    r
                    for r in old_rows
                    if (repo == "all" or r["repo"] == repo)
                    and (split == "all" or r["split"] == split)
                ]
                if {r["id"]: (r["query"], r["gold"]) for r in current} != {
                    r["id"]: (r["query"], r["gold"]) for r in previous
                }:
                    raise ValueError("Formatter comparison requires identical queries and gold")
                comparisons.append(
                    {
                        "repo": repo,
                        "split": split,
                        "candidate": "v2 bm25",
                        "baseline": "v1 bm25",
                        "intervals": paired_intervals(current, previous, METRICS, Settings()),
                    }
                )
        write(output / "formatter_comparison.json", comparisons)
    lines = [
        "# Phase 5 BM25-only baseline",
        "",
        "Frozen v2 issue queries; no dense or reranker inference in this preflight.",
        "Measures own-fix lookup, not useful similar historical fixes for a new bug.",
        "File Hit@5 is a path-localization proxy (exact gold itself qualifies), not semantic relevance.",
        "BM25 uses full queries; BM25_2000 uses the first 2000 characters as a sensitivity check.",
        "Link-derived proxy gold, with incomplete alternative-fix labels. Requests may overlap earlier T3.",
        "Pooled all-question rows are descriptive. Test was not used for tuning.",
        "Pointwise group-bootstrap intervals in summary.json are exploratory, without multiplicity adjustment.",
        "",
        "| Method | Repo | Split | n | Groups | Pool(s) | Hit@1 | Hit@5 | Hit@10 | MRR@10 | File Hit@5 |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['repo']} | {row['split']} | {row['questions']} | {row['groups']} | {row['candidate_pools']} | "
            + " | ".join(f"{row[m]:.4f}" for m in METRICS)
            + " |"
        )
    lines += [
        "",
        "FastAPI has only 18 queries (14 test); its estimates are particularly uncertain.",
        "4B/8B comparison, generated answer quality, manual scale run and deployment latency remain unmeasured.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", type=Path, default=ROOT / "eval/datasets/phase5_bugfix_v2/repos"
    )
    parser.add_argument(
        "--runs", type=Path, default=ROOT / "eval/retrieval_results/phase5_preflight_v2/bugfix"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-runs", type=Path)
    args = parser.parse_args()
    report(args.datasets, args.runs, args.output, args.previous_runs)
    print(args.output / "summary.md")
