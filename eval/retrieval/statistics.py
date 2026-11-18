"""Paired evidence-group bootstrap. Related questions are resampled together.

Point estimates are question-macro averages. Bootstrap samples replicate entire groups,
then divide their total scores by their total question counts (groups may differ in size).
Intervals are pointwise percentile intervals, not multiplicity-adjusted discovery tests.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from repo_rag.config import Settings

Row = dict[str, Any]


def bootstrap_intervals(rows: list[Row], metrics: list[str], settings: Settings) -> dict[str, Row]:
    if not rows:
        raise ValueError("cannot bootstrap an empty sample")
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: str(r["id"])):
        grouped[str(row["group"])].append(row)
    clusters = [grouped[name] for name in sorted(grouped)]
    totals = np.array(
        [[sum(float(row[m]) for row in group) for m in metrics] for group in clusters]
    )
    sizes = np.array([len(group) for group in clusters])
    rng = np.random.default_rng(settings.eval_bootstrap_seed)
    indices = rng.integers(len(clusters), size=(settings.eval_bootstrap_samples, len(clusters)))
    means = totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None]
    alpha = (1 - settings.eval_bootstrap_confidence) / 2
    bounds = np.quantile(means, [alpha, 1 - alpha], axis=0)
    return {
        metric: {
            "estimate": float(totals[:, i].sum()) / float(sizes.sum()),
            "low": float(bounds[0, i]),
            "high": float(bounds[1, i]),
            "questions": len(rows),
            "groups": len(clusters),
            "confidence": settings.eval_bootstrap_confidence,
            "samples": settings.eval_bootstrap_samples,
            "seed": settings.eval_bootstrap_seed,
            "unit": "evidence_group",
        }
        for i, metric in enumerate(metrics)
    }


def summarize(rows: list[Row], settings: Settings) -> list[Row]:
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["method"]))].append(row)
        grouped[("all", str(row["method"]))].append(row)
    summaries = []
    for (split, method), sample in sorted(grouped.items()):
        metrics = sorted(key for key in sample[0] if "@" in key)
        intervals = bootstrap_intervals(sample, metrics, settings)
        summaries.append(
            {
                "split": split,
                "method": method,
                "questions": len(sample),
                "groups": len({row["group"] for row in sample}),
                **{m: ci["estimate"] for m, ci in intervals.items()},
                "confidence_intervals": intervals,
            }
        )
    return summaries


def paired_intervals(
    left: list[Row], right: list[Row], metrics: list[str], settings: Settings
) -> dict[str, Row]:
    a = {row["id"]: row for row in left}
    b = {row["id"]: row for row in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("paired comparisons require identical, unique question ids")
    differences = []
    for qid in sorted(a):
        if a[qid]["group"] != b[qid]["group"] or a[qid]["split"] != b[qid]["split"]:
            raise ValueError("paired question metadata does not match")
        differences.append(
            {
                "id": qid,
                "group": a[qid]["group"],
                **{m: float(a[qid][m]) - float(b[qid][m]) for m in metrics},
            }
        )
    return bootstrap_intervals(differences, metrics, settings)


def compare_strategies(rows: list[Row], settings: Settings) -> list[Row]:
    grouped: dict[tuple[str, bool, str], dict[str, list[Row]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(str(row["repo"]), bool(row["headers"]), str(row["method"]))][
            str(row["strategy"])
        ].append(row)
    k = max(settings.eval_cutoffs)
    metrics = [f"mrr@{k}", f"recall@{k}", f"line_coverage@{k}", f"complete_gold@{k}"]
    comparisons = []
    for (repo, headers, method), variants in sorted(grouped.items()):
        if "ast" not in variants:
            continue
        for baseline in ("char", "lc_lang", "astchunk"):
            if baseline not in variants:
                continue
            for split in ("all", "dev", "test"):
                left = [r for r in variants["ast"] if split == "all" or r["split"] == split]
                right = [r for r in variants[baseline] if split == "all" or r["split"] == split]
                if not left and not right:
                    continue
                comparisons.append(
                    {
                        "repo": repo,
                        "headers": headers,
                        "method": method,
                        "strategy": "ast",
                        "baseline": baseline,
                        "split": split,
                        "metrics": paired_intervals(left, right, metrics, settings),
                    }
                )
    return comparisons
