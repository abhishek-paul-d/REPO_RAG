"""Conservative evidence checklist; never enables production reranking automatically."""

from __future__ import annotations


def promotion_check(comparisons: list[dict], independent_review: bool) -> dict:
    candidates = []
    for method in ("rerank_20", "union_bm25q5_rerank20"):
        rows = {
            r["stratum"]: r
            for r in comparisons
            if r["cohort"] == "fresh" and r["candidate"] == method
        }
        overall = rows.get("all")
        checks = {
            "fresh_mrr_win": bool(overall and overall["metrics"]["mrr@10"]["low"] > 0),
            "required_coverage_no_regression": bool(
                overall and overall["metrics"]["complete_gold@10"]["low"] >= 0
            ),
        }
        for target in ("test", "example"):
            row = rows.get(target)
            checks[f"{target}_stratum_assessed"] = row is not None
            checks[f"{target}_no_regression"] = bool(
                row
                and all(
                    row["metrics"][metric]["low"] >= 0 for metric in ("mrr@10", "complete_gold@10")
                )
            )
        candidates.append(
            {"method": method, "checks": checks, "statistical_checks_pass": all(checks.values())}
        )
    return {
        "production_default_changed": False,
        "independent_review_complete": independent_review,
        "latency_review": "required: inspect measured warm p50/p95; no latency decision is automated",
        "candidates": candidates,
        "status": "review measured results; keep opt-in until all evidence and latency checks are accepted",
        "limits": "pointwise exploratory intervals; absent or inconclusive strata are not safety passes",
    }
