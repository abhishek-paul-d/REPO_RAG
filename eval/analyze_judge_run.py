"""Apply the frozen judge protocol: Qwen3-Reranker-8B vs 4B, and hybrid for reference.

Usage: PYTHONPATH=. uv run python eval/analyze_judge_run.py <results dir> [protocol.json]
Writes aggregate/summary.md and aggregate/decision.json into the results directory.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.statistics import paired_intervals  # noqa: E402
from repo_rag.config import Settings  # noqa: E402

RESULTS = Path(sys.argv[1])
PROTOCOL = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "eval/datasets/phase4_judge_v1/protocol.json"
)
METRICS = ["mrr@10", "any_complete_proof@10"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    baseline, candidate = protocol["baseline"], protocol["candidate"]
    settings = Settings(
        eval_bootstrap_samples=protocol["bootstrap"]["samples"],
        eval_bootstrap_seed=protocol["bootstrap"]["seed"],
        eval_bootstrap_confidence=protocol["bootstrap"]["confidence"],
    )
    rows: list[dict] = []
    latency: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in sorted(RESULTS.glob("*/run.json")):
        meta = json.loads(run.read_text(encoding="utf-8"))
        if meta["judge_protocol_sha256"] != digest(PROTOCOL):
            raise ValueError(f"{run.parent.name}: protocol hash differs from the frozen protocol")
        per_question = json.loads((run.parent / "per_question.json").read_text(encoding="utf-8"))
        rows += per_question
        for r in per_question:
            latency[meta["repo"]][r["method"]].append(r["elapsed_ms"])
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["method"]] = r
    ids = sorted(by)
    if any(set(by[i]) != {"hybrid", baseline, candidate} for i in ids):
        raise ValueError("every question needs hybrid, baseline and candidate rows")

    def pick(method: str, keep) -> list[dict]:
        return [by[i][method] for i in ids if keep(by[i][method])]

    views = {"pooled150": lambda r: True, "fresh30": lambda r: r["cohort"] == "fresh"}
    views |= {
        f"regression:{s}": (lambda r, s=s: r["cohort"] == "regression" and r["stratum"] == s)
        for s in ("ambiguous", "multi_block", "implementation", "test", "example")
    }
    summary: dict[str, dict] = {}
    for name, keep in views.items():
        if not pick(baseline, keep):
            continue  # a stratum absent from this run (never the case for the frozen set)
        entry = {"questions": len(pick(baseline, keep))}
        for method in ("hybrid", baseline, candidate):
            sel = pick(method, keep)
            entry[method] = {m: st.mean(r[m] for r in sel) for m in METRICS}
        entry["difference"] = paired_intervals(
            pick(candidate, keep), pick(baseline, keep), METRICS, settings
        )
        summary[name] = entry

    rule = protocol["adopt_8b_rule"]
    pooled = summary["pooled150"]["difference"]
    losses = {
        name: {m: entry["difference"][m]["estimate"] for m in METRICS}
        for name, entry in summary.items()
        if name != "pooled150"
    }
    checks = {
        "pooled_mrr_gain": pooled["mrr@10"]["estimate"] >= rule["pooled150_mrr_gain_at_least"],
        "pooled_mrr_lower_positive": pooled["mrr@10"]["low"]
        > rule["pooled150_mrr_lower_bound_above"],
        "pooled_proof_not_lower": pooled["any_complete_proof@10"]["estimate"]
        >= rule["pooled150_proof_point_estimate_at_least"],
        "pooled_proof_lower_bound": pooled["any_complete_proof@10"]["low"]
        >= rule["pooled150_proof_lower_bound_at_least"],
        "no_view_loses_more_than_limit": all(
            v >= -rule["no_stratum_or_cohort_loses_more_than"]
            for per_view in losses.values()
            for v in per_view.values()
        ),
    }
    adopt = all(checks.values())
    timing = {
        repo: {
            method: {
                "p50_ms": st.median(values),
                "calls": len(values),
            }
            for method, values in methods.items()
        }
        for repo, methods in latency.items()
    }
    decision = {
        "adopt_8b": adopt,
        "checks": checks,
        "pooled_difference": pooled,
        "views_point_estimates": losses,
        "timing_indicative_only": timing,
        "judge_protocol_sha256": digest(PROTOCOL),
        "answer_quality_measured": False,
        "next_step": (
            "8B is eligible as the experiment judge; deployment cost still needs a decision"
            if adopt
            else protocol["if_not_adopted"]
        ),
    }
    out = RESULTS / "aggregate"
    out.mkdir(exist_ok=True)
    (out / "decision.json").write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")
    lines = [
        "# Judge comparison: Qwen3-Reranker 8B vs 4B",
        "",
        "Quality (retrieval completeness), not answer accuracy. Labels author-checked;"
        " independent review pending. Difference = 8B minus 4B, paired evidence-group bootstrap.",
        "",
        "| View | n | hybrid MRR | 4B MRR | 8B MRR | MRR diff [95% CI] | 4B proof | 8B proof |"
        " proof diff [95% CI] |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]

    def fmt(m: dict) -> str:
        return f"{m['estimate']:+.3f} [{m['low']:+.3f}, {m['high']:+.3f}]"

    for name, e in summary.items():
        lines.append(
            f"| {name} | {e['questions']} | {e['hybrid']['mrr@10']:.3f} |"
            f" {e[baseline]['mrr@10']:.3f} | {e[candidate]['mrr@10']:.3f} |"
            f" {fmt(e['difference']['mrr@10'])} | {e[baseline]['any_complete_proof@10']:.3f} |"
            f" {e[candidate]['any_complete_proof@10']:.3f} |"
            f" {fmt(e['difference']['any_complete_proof@10'])} |"
        )
    lines += ["", f"Checks: {checks}", "", decision["next_step"], ""]
    lines += ["Warm p50 per repo (one call per question, indicative only):", ""]
    for repo, methods in timing.items():
        parts = ", ".join(f"{m} {v['p50_ms']:.0f} ms" for m, v in methods.items())
        lines.append(f"* {repo}: {parts}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
