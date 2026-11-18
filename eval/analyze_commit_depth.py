"""Audit every prefix/weight ranking, paired interval and pre-registered depth decision."""

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

from eval.analyze_bugfix_run import audit_method  # noqa: E402
from eval.prepare_bugfix_eval import digest, write  # noqa: E402
from eval.retrieval.commit_depth import (  # noqa: E402
    METRICS,
    audit_run,
    labels,
    read,
    select_weight,
    settings,
    verify_protocol,
)
from eval.retrieval.statistics import bootstrap_intervals, paired_intervals  # noqa: E402
from repo_rag.config import Settings  # noqa: E402


def choose_depth(comparisons: list[dict], protocol: dict) -> dict:
    rule = protocol["rule"]
    primary = protocol["depth"]["primary_judge"]
    base = min(protocol["depth"]["depths"])
    checks = []

    def find(candidate, baseline, cohort="all", repo="all"):
        return next(
            c
            for c in comparisons
            if c["split"] == "test"
            and c["cohort"] == cohort
            and c["repo"] == repo
            and c["link_path"] == "all"
            and c["candidate"] == f"{primary}_d{candidate}"
            and c["baseline"] == f"{primary}_d{baseline}"
        )

    passed = []
    for depth in protocol["depth"]["depths"][1:]:
        all_ci = find(depth, base)["intervals"]
        source_ci = find(depth, base, "source")["intervals"]
        guards = [find(depth, base, repo=repo) for repo in protocol["repos"]]
        eligible = [
            c for c in guards if c["intervals"]["hit@5"]["groups"] >= rule["repo_guard_min_groups"]
        ]
        verdict = {
            "hit5_gain": all_ci["hit@5"]["estimate"] + 1e-12 >= rule["min_hit5_gain"],
            "hit5_lower": all_ci["hit@5"]["low"] > rule["hit5_ci_lower_above"],
            "mrr_guard": all_ci["mrr@10"]["estimate"] >= rule["min_mrr_delta"] - 1e-12,
            "source_guard": source_ci["hit@5"]["estimate"] >= rule["source_min_hit5_delta"] - 1e-12,
            "repo_guards": all(
                c["intervals"]["hit@5"]["estimate"] >= -rule["max_repo_hit5_loss"] - 1e-12
                for c in eligible
            ),
        }
        checks.append(
            {
                "depth": depth,
                "checks": verdict,
                "pooled": all_ci,
                "source": source_ci,
                "eligible_repo_guards": eligible,
                "descriptive_small_repos": [c for c in guards if c not in eligible],
            }
        )
        if all(verdict.values()):
            passed.append(depth)
    chosen = min(passed) if passed else base
    if len(passed) == 2:
        ci = find(max(passed), min(passed))["intervals"]["hit@5"]
        if (
            ci["estimate"] + 1e-12 >= rule["prefer_200_over_100_gain"]
            and ci["low"] > rule["prefer_200_ci_lower_above"]
        ):
            chosen = max(passed)
    return {
        "recommended_depth": chosen,
        "checks": checks,
        "default_changed": False,
        "status": "keep_50" if not passed else "eligible_pending_review",
    }


def provenance(info: dict, protocol: dict, experiment: Path, reference: dict | None) -> dict:
    if info["experiment_freeze_sha256"] != digest(experiment / "freeze.json") or info[
        "protocol_sha256"
    ] != digest(experiment / "protocol.json"):
        raise ValueError("Experiment provenance mismatch")
    if reference is not None and info["source_sha256"] != reference:
        raise ValueError("Mixed source snapshots")
    actual = info["settings"]
    stage = info["stage"]
    expected = settings(
        protocol,
        Path("unused"),
        "dense" if stage == "weights" else info["judge"],
        info["device"],
        depth=actual["rerank_candidate_depth"],
        retrieval_k=actual["retrieval_k"],
        weight=actual["bm25_weight"],
    ).model_dump(mode="json")
    for key in (
        "embedding_provider",
        "local_embedding_model",
        "embedding_dims",
        "embedding_dtype",
        "embedding_max_tokens",
        "embedding_batch_size",
        "embedding_query_instruction",
        "compute_device",
        "reranker_model",
        "reranker_mode",
        "rerank_instruction",
        "rerank_max_tokens",
        "rerank_first_stage_weight",
        "rerank_instruction_mode",
        "evidence_expansion",
        "dense_weight",
        "rrf_k",
    ):
        if actual[key] != expected[key]:
            raise ValueError(f"Active setting differs: {key}")
    wanted_k = (
        protocol["depth"]["retrieval_k"]
        if stage in {"depth", "timing"}
        else protocol["weight"]["retrieval_k"]
    )
    wanted_depth = (
        max(protocol["depth"]["depths"]) if stage == "depth" else min(protocol["depth"]["depths"])
    )
    if actual["retrieval_k"] != wanted_k or actual["rerank_candidate_depth"] != wanted_depth:
        raise ValueError("Depth/retrieval settings differ from protocol")
    if stage != "selected" and actual["bm25_weight"] != protocol["depth"]["bm25_weight"]:
        raise ValueError("Weight differs from protocol")
    revisions = info["model_revisions"]
    if revisions["embedding"] != protocol["expected_revisions"]["embedding"]:
        raise ValueError("Embedding revision differs from control")
    if (
        stage != "weights"
        and revisions["reranker"] != protocol["expected_revisions"][info["judge"]]
    ):
        raise ValueError("Judge revision differs from preregistration")
    return info["source_sha256"]


def historical(experiment: Path, dataset: Path, protocol: dict, cohorts: dict) -> list[dict]:
    _, pairs = labels(dataset, protocol, cohorts)
    by_id = {p["id"]: p for p in pairs}
    result = []
    with zipfile.ZipFile(experiment / "historical/source.zip") as archive:
        for judge in ("rerank8b", "rerank4b"):
            folder = experiment / "historical" / dataset.name / judge
            rows = audit_method(dataset, folder, judge)
            info = read(folder / "run.json")
            for name, expected in info["source_sha256"].items():
                if (
                    hashlib.sha256(archive.read(name).replace(b"\r\n", b"\n")).hexdigest()
                    != expected
                ):
                    raise ValueError("Historical source mismatch")
            if info["model_revisions"] != {
                "embedding": protocol["expected_revisions"]["embedding"],
                "reranker": protocol["expected_revisions"][judge],
            }:
                raise ValueError("Historical model revisions differ")
            traces = {t["id"]: t for t in read(folder / "traces.json")}
            for row in rows:
                pair = by_id[row["id"]]
                result.append(
                    {
                        **pair,
                        "method": f"historical_{judge}",
                        "ranking": row["ranking"],
                        **{m: row[m] for m in METRICS if m != "pool_recall"},
                        "pool_recall": float(
                            any(p["sha"] in pair["gold"] for p in traces[row["id"]]["pairs"])
                        ),
                    }
                )
    return result


def timing_rows(
    folder: Path, dataset: Path, protocol: dict, cohorts: dict
) -> tuple[list[dict], list[dict]]:
    _, pairs = labels(dataset, protocol, cohorts)
    wanted = {p["id"] for p in pairs} & set(cohorts["timing_ids"])
    rows = read(folder / "per_question.json")
    info = read(folder / "run.json")
    if (
        info["dataset_freeze_sha256"] != digest(dataset / "freeze.json")
        or info["stage"] != "timing"
        or info["judge"] != protocol["timing"]["judge"]
    ):
        raise ValueError("Timing provenance mismatch")
    keys = {
        (qid, d, rep)
        for qid in wanted
        for d in protocol["depth"]["depths"]
        for rep in range(protocol["timing"]["repetitions"])
    }
    if len(rows) != len(keys) or {(r["id"], r["depth"], r["repetition"]) for r in rows} != keys:
        raise ValueError("Timing subset/repetition matrix incomplete")
    repo = pairs[0]["repo"]
    for r in rows:
        depths = protocol["depth"]["depths"]
        offset = (cohorts["timing_ids"].index(r["id"]) + r["repetition"]) % len(depths)
        if (
            r["repo"] != repo
            or r["position"] != (depths[offset:] + depths[:offset]).index(r["depth"])
            or not math.isfinite(r["elapsed_seconds"])
            or r["elapsed_seconds"] <= 0
        ):
            raise ValueError("Invalid timing/order")
    summaries = []
    for d in protocol["depth"]["depths"]:
        sample = [r["elapsed_seconds"] for r in rows if r["depth"] == d]
        summaries.append(
            {
                "repo": repo,
                "depth": d,
                "questions": len(wanted),
                "calls": len(sample),
                "p50_seconds": float(np.quantile(sample, 0.5)),
                "p95_seconds": float(np.quantile(sample, 0.95)),
                "gpu": info["gpu"],
                "label": protocol["timing"]["label"],
            }
        )
    baseline = summaries[0]["p50_seconds"]
    for item in summaries:
        item["p50_ratio_to_50"] = item["p50_seconds"] / baseline
    return rows, summaries


def analyze(root: Path, output: Path | None = None) -> dict:
    output = output or root / "aggregate"
    if output.exists():
        raise FileExistsError("Preserve earlier analysis")
    experiment = root / "experiment"
    protocol = verify_protocol(experiment, root / "datasets")
    cohorts = read(experiment / "cohorts.json")
    rows, latency, raw_timing = [], [], []
    reference = None
    for stem in protocol["input_freeze_sha256"]:
        dataset = root / "datasets" / stem
        reference_dense = None
        for stage, folder_name in (
            ("weights", "weights"),
            ("depth", "rerank8b"),
            ("depth", "rerank4b"),
        ):
            folder = root / "runs" / stem / folder_name
            rows.extend(audit_run(dataset, folder, protocol, cohorts, stage))
            reference = provenance(read(folder / "run.json"), protocol, experiment, reference)
            if stage == "depth":
                dense = {
                    t["id"]: [(d["chunk_id"], d["sha"]) for d in t["dense"]]
                    for t in read(folder / "traces.json")
                }
                if reference_dense is not None and dense != reference_dense:
                    raise ValueError("Judges did not use the same 200-deep first stage")
                reference_dense = dense
        rows.extend(historical(experiment, dataset, protocol, cohorts))
        folder = root / "runs" / stem / "timing"
        reference = provenance(read(folder / "run.json"), protocol, experiment, reference)
        timed, summary = timing_rows(folder, dataset, protocol, cohorts)
        raw_timing.extend(timed)
        latency.extend(summary)
    choice = select_weight([r for r in rows if r["method"].startswith("weight_")], protocol)
    if choice != read(root / "selection.json"):
        raise ValueError("Selection does not reconstruct from dev")
    if choice["selected"] != choice["baseline"]:
        for stem in protocol["input_freeze_sha256"]:
            dense_reference = {
                t["id"]: [(d["chunk_id"], d["sha"]) for d in t["dense"]]
                for t in read(root / "runs" / stem / "weights/traces.json")
            }
            for weight in (choice["baseline"], choice["selected"]):
                folder = root / "runs" / stem / f"selected_{weight:g}"
                info = read(folder / "run.json")
                if info["settings"]["bm25_weight"] != weight or info["judge"] != "rerank8b":
                    raise ValueError("Conditional weight setting mismatch")
                reference = provenance(info, protocol, experiment, reference)
                actual_dense = {
                    t["id"]: [(d["chunk_id"], d["sha"]) for d in t["dense"]]
                    for t in read(folder / "traces.json")
                }
                if actual_dense != dense_reference:
                    raise ValueError("Conditional weight arms changed the dense first stage")
                rows.extend(
                    audit_run(root / "datasets" / stem, folder, protocol, cohorts, "selected")
                )
    with zipfile.ZipFile(root / "source.zip") as archive:
        for name, expected in (reference or {}).items():
            if hashlib.sha256(archive.read(name).replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError("Measured source differs from source.zip")
    test = {r["id"]: r for r in rows if r["split"] == "test"}
    if (
        len(test) != protocol["cohorts"]["expected_test"]
        or sum(r["source_path"] for r in test.values())
        != protocol["cohorts"]["expected_source_test"]
    ):
        raise ValueError("Test/source cohort count mismatch")
    stats = Settings(
        eval_bootstrap_samples=protocol["statistics"]["samples"],
        eval_bootstrap_seed=protocol["statistics"]["seed"],
        eval_bootstrap_confidence=protocol["statistics"]["confidence"],
    )
    grouped = defaultdict(list)
    for r in rows:
        for split in ("all", r["split"]):
            for cohort in ["all", "source"] if r["source_path"] else ["all"]:
                for repo, link in [
                    ("all", "all"),
                    (r["repo"], "all"),
                    *[(r["repo"], p) for p in r["link_paths"]],
                ]:
                    grouped[(split, cohort, repo, link)].append(r)
    summaries, comparisons = [], []
    pairs = [
        (f"{j}_d{d}", f"{j}_d{b}")
        for j in ("rerank8b", "rerank4b")
        for d, b in ((100, 50), (200, 50), (200, 100))
    ]
    pairs += [(f"{j}_d50", f"historical_{j}") for j in ("rerank8b", "rerank4b")]
    if choice["selected"] != choice["baseline"]:
        pairs.append((f"selected_{choice['selected']:g}", f"selected_{choice['baseline']:g}"))
    for (split, cohort, repo, link), sample in sorted(grouped.items()):
        methods = {
            m: [r for r in sample if r["method"] == m]
            for m in sorted({r["method"] for r in sample})
        }
        context = {"split": split, "cohort": cohort, "repo": repo, "link_path": link}
        for method, part in methods.items():
            summaries.append(
                {
                    **context,
                    "method": method,
                    "questions": len(part),
                    "candidate_pools": {r["repo"]: r["candidate_pool"] for r in part},
                    "intervals": bootstrap_intervals(part, METRICS, stats),
                }
            )
        for left, right in pairs:
            comparisons.append(
                {
                    **context,
                    "candidate": left,
                    "baseline": right,
                    "intervals": paired_intervals(methods[left], methods[right], METRICS, stats),
                }
            )
    decision = choose_depth(comparisons, protocol)
    decision["latency_p50_ratios"] = {
        r["repo"]: r["p50_ratio_to_50"]
        for r in latency
        if r["depth"] == decision["recommended_depth"]
    }
    decision["weight_selection"] = choice
    decision["not_measured"] = protocol["not_measured"]
    output.mkdir(parents=True)
    for name, value in [
        ("per_question.json", rows),
        ("summary.json", summaries),
        ("comparisons.json", comparisons),
        ("decision.json", decision),
        ("timing_raw.json", raw_timing),
        ("latency.json", latency),
    ]:
        write(output / name, value)
    length_rows = []
    for method in sorted({r["method"] for r in rows}):
        for length in sorted({r["query_bin"] for r in rows}):
            sample = [
                r
                for r in rows
                if r["method"] == method and r["query_bin"] == length and r["split"] == "test"
            ]
            if sample:
                length_rows.append(
                    {
                        "method": method,
                        "query_bin": length,
                        "questions": len(sample),
                        "intervals": bootstrap_intervals(sample, METRICS, stats),
                    }
                )
    write(output / "query_length_summary.json", length_rows)
    text = [
        "# Commit depth/weight comparison",
        "",
        f"Depth recommendation: {decision['recommended_depth']}, pending review. No defaults changed.",
        f"Per-repo p50 ratios to prefix50: {decision['latency_p50_ratios']}.",
        f"Dev-selected weight: {choice['selected']}; test rows never select weights.",
        "",
        "Source-path subset first; this proxy includes tests/configs. Gold is link-derived and reused, not independent new-bug labels.",
        "Finds an issue's own fix, not demonstrated useful similar fixes for a new bug. Small strata (<30 groups) are descriptive.",
        "Shared probabilities define depth comparisons. Historical depth50 used shorter first-stage lists and is a separate control.",
        "4B is secondary. Intervals are pointwise exploratory group bootstraps, not adjusted for multiple comparisons.",
        "",
        "| Cohort | Repo | Path | Method | n | Pool(s) | Hit@5 | MRR@10 | Pool recall |",
        "|---|---|---|---|---:|---|---:|---:|---:|",
    ]
    for cohort in ("source", "all"):
        for r in summaries:
            if r["split"] == "test" and r["cohort"] == cohort:
                ci = r["intervals"]
                text.append(
                    f"| {cohort} | {r['repo']} | {r['link_path']} | {r['method']} | {r['questions']} | {r['candidate_pools']} | {ci['hit@5']['estimate']:.4f} | {ci['mrr@10']['estimate']:.4f} | {ci['pool_recall']['estimate']:.4f} |"
                )
    text += [
        "",
        "Warm A100 experiment timing; per-repo percentiles, never averaged across repos.",
        "",
        "| Repo | Depth | Calls | p50 s | p95 s | p50 ratio to prefix50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    text += [
        f"| {r['repo']} | {r['depth']} | {r['calls']} | {r['p50_seconds']:.3f} | {r['p95_seconds']:.3f} | {r['p50_ratio_to_50']:.2f} |"
        for r in latency
    ]
    text += [
        "",
        "Timing uses production search with retrieval_k=200 for every depth, matching the quality prefixes; it is not laptop latency.",
        "Not measured: " + ", ".join(protocol["not_measured"]),
        "",
    ]
    (output / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(analyze(args.root, args.output))
