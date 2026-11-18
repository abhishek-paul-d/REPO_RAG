"""Combine the three completed v2 GPU runs without selecting or tuning a policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.dataset import write_json  # noqa: E402
from eval.retrieval.followup import PRIMARY, render_report  # noqa: E402
from eval.retrieval.promotion import promotion_check  # noqa: E402
from eval.retrieval.statistics import paired_intervals, summarize  # noqa: E402
from repo_rag.config import Settings  # noqa: E402


def aggregate(results: Path) -> dict:
    rows: list[dict] = []
    timings: list[dict] = []
    runs: list[dict] = []
    for slug in ("psf__requests", "expressjs__express", "pallets__flask"):
        folder = results / slug
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if run["mode"] != "full":
            raise ValueError("aggregate requires all three completed real full runs")
        runs.append(run)
        for row in json.loads((folder / "per_question.json").read_text(encoding="utf-8")):
            rows.append({**row, "group": f"{run['repo']}:{row['group']}"})
        timings.extend(
            {"repo": run["repo"], **r}
            for r in json.loads((folder / "latency.json").read_text(encoding="utf-8"))
        )
    if any(run["matrix"] != runs[0]["matrix"] for run in runs):
        raise ValueError("matrix mismatch across repositories")
    settings = Settings.model_validate(runs[0]["settings"])
    summary: list[dict] = []
    comparisons: list[dict] = []
    for target in ("all", "implementation", "test", "example", "symbol_lookup"):
        sample = [
            r
            for r in rows
            if r["cohort"] == "fresh"
            and (
                target == "all"
                or r["target_type"] == target
                or (target == "symbol_lookup" and r["category"] == target)
            )
        ]
        summary.extend(
            {"cohort": "fresh", "stratum": target, **s}
            for s in summarize(sample, settings)
            if s["split"] == "all"
        )
        base = [r for r in sample if r["method"] == "hybrid_w0.3_tests1"]
        for method in PRIMARY:
            if method == "hybrid_w0.3_tests1":
                continue
            candidate = [r for r in sample if r["method"] == method]
            comparisons.append(
                {
                    "cohort": "fresh",
                    "stratum": target,
                    "candidate": method,
                    "baseline": "hybrid_w0.3_tests1",
                    "metrics": paired_intervals(
                        candidate, base, ["mrr@10", "complete_gold@10", "recall@10"], settings
                    ),
                }
            )
    output = results / "aggregate"
    output.mkdir(exist_ok=True)
    decision = promotion_check(comparisons, all(run["independent_review_complete"] for run in runs))
    decision["per_repo_checks"] = {
        run["repo"]: json.loads(
            (results / run["repo"].replace("/", "__") / "decision.json").read_text()
        )
        for run in runs
    }
    write_json(output / "summary.json", summary)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "decision.json", decision)
    write_json(output / "per_repo_latency.json", timings)
    (output / "summary.md").write_text(
        render_report(
            {"summary": summary, "latency": []}, "pooled fresh questions across three repos"
        )
        + "\nLatency remains repo-specific: inspect each latency.json; do not average percentiles. Pooled estimates are question-weighted; repo reports remain required. No defaults change automatically.\n",
        encoding="utf-8",
    )
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    aggregate(parser.parse_args().results)
