"""Re-run the existing 120 code questions with BM25 after Phase 5 integration.

Compare every top-10 chunk ID and every retrieval metric against the immutable Phase 4
preflight, rather than claiming new GPU results. Output is reviewable JSON plus a table.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.prepare_bugfix_eval import write  # noqa: E402
from eval.retrieval.ambiguous import read_cases, score_case  # noqa: E402
from eval.retrieval.followup import production_corpus  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.store.bm25 import Bm25Index  # noqa: E402


def run(output: Path) -> list[dict]:
    if output.exists():
        raise FileExistsError("Preserve prior results; use another output directory")
    settings = Settings(header_defines=False)
    repos = [
        ("requests", ROOT / "data/repos/psf__requests"),
        ("express", ROOT / "data/eval_repos/expressjs__express"),
        ("flask", ROOT / "data/eval_repos/pallets__flask"),
    ]
    rows: list[dict] = []
    summary: list[dict] = []
    for name, checkout in repos:
        cases = read_cases(ROOT / f"eval/datasets/phase4_intent_v1/{name}_qa.jsonl")
        first = cases[0].question
        old_path = (
            ROOT
            / "eval/retrieval_results/phase4_expansion_preflight"
            / first.repo.replace("/", "__")
            / "per_question.json.gz"
        )
        before = {
            r["id"]: r
            for r in json.loads(gzip.decompress(old_path.read_bytes()))
            if r["cohort"] == "regression"
        }
        docs = production_corpus(checkout, first.repo, first.commit_sha, settings)
        by_id = {str(d.metadata["chunk_id"]): d for d in docs}
        index = Bm25Index.build(docs)
        changed_rankings = changed_metrics = 0
        sample: list[dict] = []
        for case in cases:
            ranking = [cid for cid, _ in index.search(case.question.question, 50)]
            metrics = score_case(case, [by_id[cid] for cid in ranking], settings)
            old = before[case.question.id]
            changed_rankings += ranking[:10] != [r["chunk_id"] for r in old["results"]]
            changed_metrics += any(metrics[key] != old[key] for key in metrics)
            row = {
                "id": case.question.id,
                "repo": first.repo,
                "ranking": ranking[:10],
                "before": {key: old[key] for key in metrics},
                "after": metrics,
            }
            sample.append(row)
            rows.append(row)
        summary.append(
            {
                "repo": first.repo,
                "questions": len(sample),
                "chunks": len(docs),
                "changed_rankings": changed_rankings,
                "changed_metrics": changed_metrics,
                "before_mrr": sum(r["before"]["mrr@10"] for r in sample) / len(sample),
                "after_mrr": sum(r["after"]["mrr@10"] for r in sample) / len(sample),
                "before_proof": sum(r["before"]["any_complete_proof@10"] for r in sample)
                / len(sample),
                "after_proof": sum(r["after"]["any_complete_proof@10"] for r in sample)
                / len(sample),
                "baseline_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            }
        )
        if changed_metrics or changed_rankings:
            raise ValueError(f"Code regression for {first.repo}")
    output.mkdir(parents=True)
    write(output / "code_regression_per_question.json", rows)
    write(output / "code_regression_summary.json", summary)
    lines = [
        "# Phase 5 code regression: BM25 only",
        "",
        "No new neural inference, ANN accuracy or latency is measured here.",
        "",
        "| Repo | n | Before MRR | After MRR | Before proof | After proof | Changed rankings / metrics |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['repo']} | {row['questions']} | {row['before_mrr']:.4f} | {row['after_mrr']:.4f} | {row['before_proof']:.4f} | {row['after_proof']:.4f} | {row['changed_rankings']} / {row['changed_metrics']} |"
        )
    (output / "code_regression.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
