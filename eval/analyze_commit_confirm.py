"""Audit and decide the pre-registered retrieval_k confirmation from raw rows and traces."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.confirm_lib import (  # noqa: E402
    METRICS,
    audit_run,
    load_pairs,
    read,
    stem_of,
    verify_experiment,
)
from eval.prepare_bugfix_eval import digest, write  # noqa: E402
from eval.retrieval.statistics import bootstrap_intervals, paired_intervals  # noqa: E402
from eval.run_commit_confirm_eval import dataset_folder  # noqa: E402
from repo_rag.config import Settings  # noqa: E402

COMPARISONS = [
    ("B", "A"),
    ("C", "A"),
    ("B_first", "A_first"),
    ("C_first", "A_first"),
    ("A_rerun", "A"),
]


def provenance(info: dict, protocol: dict, experiment: Path, reference: dict | None) -> dict:
    confirm = protocol["confirm"]
    if info["experiment_freeze_sha256"] != digest(experiment / "freeze.json") or info[
        "protocol_sha256"
    ] != digest(experiment / "protocol.json"):
        raise ValueError("Experiment provenance mismatch")
    if reference is not None and info["source_sha256"] != reference:
        raise ValueError("Mixed source snapshots")
    s = info["settings"]
    retrieval = protocol["retrieval"]
    models = protocol["models"]
    expected = {
        "local_embedding_model": models["embedding"],
        "embedding_dims": models["dims"],
        "embedding_dtype": models["dtype"],
        "reranker_model": models["rerankers"][1],
        "reranker_mode": "on",
        "rerank_candidate_depth": confirm["rerank_depth"],
        "rerank_instruction": retrieval["rerank_instruction"],
        "rerank_first_stage_weight": 0,
        "rerank_instruction_mode": "default",
        "evidence_expansion": "none",
        "bm25_weight": confirm["bm25_weight"],
        "dense_weight": retrieval["dense_weight"],
        "rrf_k": retrieval["rrf_k"],
    }
    for key, value in expected.items():
        if s[key] != value:
            raise ValueError(f"Active setting differs from the protocol: {key}")
    revisions = info["model_revisions"]
    wanted = protocol["expected_revisions"]
    if revisions["embedding"] != wanted["embedding"] or revisions["reranker"] != wanted["rerank8b"]:
        raise ValueError("Model revision differs from the preregistration")
    return info["source_sha256"]


def timing_rows(
    run_dir: Path, folder: Path, protocol: dict, freeze: dict, quality: dict
) -> tuple[list[dict], list[dict]]:
    _, pairs = load_pairs(folder, protocol, None, freeze["excluded"])
    wanted = [qid for qid in freeze["timing_ids"] if qid in {p["id"] for p in pairs}]
    rows = read(run_dir / "per_question.json")
    info = read(run_dir / "run.json")
    if info["stage"] != "timing" or info["dataset_freeze_sha256"] != digest(folder / "freeze.json"):
        raise ValueError("Timing provenance mismatch")
    arms = list(protocol["confirm"]["arms"])
    reps = protocol["confirm"]["timing"]["repetitions"]
    keys = {(q, a, r) for q in wanted for a in arms for r in range(reps)}
    if len(rows) != len(keys) or {(r["id"], r["arm"], r["repetition"]) for r in rows} != keys:
        raise ValueError("Timing matrix incomplete")
    for r in rows:
        offset = (freeze["timing_ids"].index(r["id"]) + r["repetition"]) % len(arms)
        if r["position"] != (arms[offset:] + arms[:offset]).index(r["arm"]) or not (
            math.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > 0
        ):
            raise ValueError("Invalid timing order or value")
    repo = rows[0]["repo"]
    summary = []
    for arm in arms:
        sample = [r["elapsed_seconds"] for r in rows if r["arm"] == arm]
        same = [r for r in rows if r["arm"] == arm]
        summary.append(
            {
                "repo": repo,
                "arm": arm,
                "questions": len(wanted),
                "calls": len(sample),
                "p50_seconds": float(np.quantile(sample, 0.5)),
                "p95_seconds": float(np.quantile(sample, 0.95)),
                "gpu": info["gpu"],
                "identical_top10_vs_quality_run": float(
                    np.mean([r["ranking"] == quality[(r["id"], arm)] for r in same])
                ),
            }
        )
    base = summary[0]["p50_seconds"]
    for item in summary:
        item["p50_ratio_to_A"] = item["p50_seconds"] / base
    return rows, summary


def contexts(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        for cohort in ["all", "source"] if r["source_path"] else ["all"]:
            for repo in ("all", r["repo"]):
                grouped[(cohort, repo)].append(r)
    return grouped


def decide(comparisons: list[dict], questions: int, protocol: dict) -> dict:
    rule = protocol["confirm"]["rule"]

    def find(candidate, baseline, cohort="all", repo="all"):
        return next(
            c
            for c in comparisons
            if c["group"] == "fresh"
            and c["cohort"] == cohort
            and c["repo"] == repo
            and c["candidate"] == candidate
            and c["baseline"] == baseline
        )

    b = find("B", "A")["intervals"]
    hit, mrr = b["hit@5"], b["mrr@10"]
    guards = [
        c
        for c in comparisons
        if c["group"] == "fresh"
        and c["cohort"] == "all"
        and c["repo"] != "all"
        and c["candidate"] == "B"
        and c["baseline"] == "A"
    ]
    eligible = [
        c for c in guards if c["intervals"]["hit@5"]["groups"] >= rule["repo_guard_min_groups"]
    ]
    checks = {
        "hit5_gain": hit["estimate"] + 1e-12 >= rule["min_hit5_gain"],
        "mrr_not_negative": mrr["estimate"] + 1e-12 >= rule["min_mrr_delta"],
        "hit5_lower_bound": hit["low"] + 1e-12 >= rule["lower_bound_floor"],
        "mrr_lower_bound": mrr["low"] + 1e-12 >= rule["lower_bound_floor"],
        "repo_guards": all(
            c["intervals"]["hit@5"]["estimate"] >= -rule["max_repo_hit5_loss"] - 1e-12
            for c in eligible
        ),
    }
    noise = find("A_rerun", "A")["intervals"]["hit@5"]["estimate"]
    width = hit["high"] - hit["low"]
    flags = {
        "noise_ok": abs(noise) <= max(hit["estimate"], 0.0) + 1e-12,
        "interval_narrow_enough": width <= rule["max_ci_width_hit5"] + 1e-12,
    }
    if not all(checks.values()):
        status = "keep_50"
    elif not all(flags.values()):
        status = "inconclusive"
    else:
        status = "eligible_pending_review"
    return {
        "status": status,
        "checks": checks,
        "flags": flags,
        "b_minus_a": b,
        "hit5_interval_width": width,
        "rerun_hit5_difference": noise,
        "eligible_repo_guards": eligible,
        "descriptive_small_repos": [c for c in guards if c not in eligible],
        "fresh_questions": questions,
        "tentative": questions < rule["tentative_below_questions"],
        "recommended_commit_retrieval_k": 200 if status == "eligible_pending_review" else 50,
        "default_changed": False,
        "if_adopted": rule["if_adopted"],
    }


def analyze(root: Path, output: Path | None = None) -> dict:
    output = output or root / "aggregate"
    if output.exists():
        raise FileExistsError("Preserve earlier analysis")
    experiment = root / "experiment"
    protocol = verify_experiment(experiment, root / "datasets")
    freeze = read(experiment / "freeze.json")
    layout = {
        "fresh": list(freeze["counts"]),
        "v2dev": [stem_of(r) for r in protocol["supporting"]["repos"]],
    }
    rows: list[dict] = []
    diagnostics, raw_timing, latency = {}, [], []
    untimed: list[str] = []
    reference = None
    for group, stems in layout.items():
        for stem in stems:
            folder = dataset_folder(root, group, stem)
            run_dir = root / "runs" / group / stem / "confirm"
            part, diag = audit_run(
                folder,
                run_dir,
                protocol,
                split="dev" if group == "v2dev" else None,
                exclude=freeze["excluded"] if group == "fresh" else (),
            )
            reference = provenance(read(run_dir / "run.json"), protocol, experiment, reference)
            rows.extend({**r, "run_group": group} for r in part)  # "group" stays the evidence group
            diagnostics[f"{group}/{stem}"] = diag
            if group == "fresh":
                quality = {(r["id"], r["method"]): r["ranking"] for r in part}
                timing_dir = root / "runs" / group / stem / "timing"
                if not (timing_dir / "run.json").exists():
                    untimed.append(stem)  # timing stopped early; quality is unaffected
                    continue
                reference = provenance(
                    read(timing_dir / "run.json"), protocol, experiment, reference
                )
                timed, summary = timing_rows(timing_dir, folder, protocol, freeze, quality)
                raw_timing.extend(timed)
                latency.extend(summary)
    with zipfile.ZipFile(root / "source.zip") as archive:
        for name, expected in (reference or {}).items():
            if hashlib.sha256(archive.read(name).replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError("Measured source differs from source.zip")
    stats = Settings(
        eval_bootstrap_samples=protocol["statistics"]["samples"],
        eval_bootstrap_seed=protocol["statistics"]["seed"],
        eval_bootstrap_confidence=protocol["statistics"]["confidence"],
    )
    summaries: list[dict] = []
    comparisons: list[dict] = []
    for group in layout:
        part = [r for r in rows if r["run_group"] == group]
        for (cohort, repo), sample in sorted(contexts(part).items()):
            methods = {
                m: [r for r in sample if r["method"] == m]
                for m in sorted({r["method"] for r in sample})
            }
            context = {"group": group, "cohort": cohort, "repo": repo}
            for method, items in methods.items():
                summaries.append(
                    {
                        **context,
                        "method": method,
                        "questions": len(items),
                        "intervals": bootstrap_intervals(items, METRICS, stats),
                    }
                )
            for left, right in COMPARISONS:
                comparisons.append(
                    {
                        **context,
                        "candidate": left,
                        "baseline": right,
                        "intervals": paired_intervals(
                            methods[left], methods[right], METRICS, stats
                        ),
                    }
                )
    fresh_questions = len({r["id"] for r in rows if r["run_group"] == "fresh"})
    decision = decide(comparisons, fresh_questions, protocol)
    decision["latency_p50_ratio_to_A"] = {
        f"{r['repo']}/{r['arm']}": r["p50_ratio_to_A"] for r in latency if r["arm"] != "A"
    }
    decision["not_measured"] = protocol["not_measured"]
    decision["timing_not_run_for"] = untimed
    output.mkdir(parents=True)
    for name, value in [
        ("per_question.json", rows),
        ("summary.json", summaries),
        ("comparisons.json", comparisons),
        ("decision.json", decision),
        ("timing_raw.json", raw_timing),
        ("latency.json", latency),
        ("diagnostics.json", diagnostics),
    ]:
        write(output / name, value)
    text = [
        "# retrieval_k confirmation for commit search",
        "",
        f"Status: **{decision['status']}** (nothing changes automatically; default_changed=false).",
        f"Fresh questions: {fresh_questions}"
        + (
            " - TENTATIVE: fewer than the preregistered power floor."
            if decision["tentative"]
            else "."
        ),
        f"Checks: {decision['checks']}. Flags: {decision['flags']}.",
        "",
        "Arms: A retrieval_k=50 (control), B 200 (hypothesis), C 100 (descriptive); rerank depth 50,",
        "8B, shared per-question scores, so arms differ only by which candidates enter the top 50.",
        "Labels are link-derived with no independent review. This measures finding an issue's own",
        "fix, not useful similar past fixes. Intervals are pointwise paired group bootstraps.",
        "",
        "| Group | Cohort | Repo | Method | n | Hit@5 | MRR@10 | Pool recall |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        ci = r["intervals"]
        text.append(
            f"| {r['group']} | {r['cohort']} | {r['repo']} | {r['method']} | {r['questions']} "
            f"| {ci['hit@5']['estimate']:.4f} | {ci['mrr@10']['estimate']:.4f} "
            f"| {ci['pool_recall']['estimate']:.4f} |"
        )
    text += [
        "",
        "| Group | Cohort | Repo | Comparison | Hit@5 diff [95% CI] | MRR@10 diff [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        h, m = c["intervals"]["hit@5"], c["intervals"]["mrr@10"]
        text.append(
            f"| {c['group']} | {c['cohort']} | {c['repo']} | {c['candidate']} - {c['baseline']} "
            f"| {h['estimate']:+.4f} [{h['low']:+.4f}, {h['high']:+.4f}] "
            f"| {m['estimate']:+.4f} [{m['low']:+.4f}, {m['high']:+.4f}] |"
        )
    text += [
        "",
        "Warm A100 experiment timing; per-repo percentiles, never averaged across repositories.",
        "",
        "| Repo | Arm | Calls | p50 s | p95 s | p50 ratio to A | Top-10 identical to quality run |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    text += [
        f"| {r['repo']} | {r['arm']} | {r['calls']} | {r['p50_seconds']:.3f} "
        f"| {r['p95_seconds']:.3f} | {r['p50_ratio_to_A']:.2f} "
        f"| {r['identical_top10_vs_quality_run']:.3f} |"
        for r in latency
    ]
    if untimed:
        text += ["", "Timing was not run for: " + ", ".join(untimed) + " (stopped early)."]
    text += ["", "Not measured: " + ", ".join(protocol["not_measured"]), ""]
    (output / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(analyze(args.root, args.output))
