"""Report all/added/retained questions and target strata without tuning the frozen matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.ambiguous import read_cases  # noqa: E402
from eval.retrieval.comparison_protocol import protocol_settings  # noqa: E402
from eval.retrieval.dataset import write_json  # noqa: E402
from eval.retrieval.statistics import paired_intervals, summarize  # noqa: E402
from repo_rag.config import Settings  # noqa: E402

METHODS = ("hybrid", "rerank20", "rerank50")
METRICS = ("mrr@10", "any_complete_proof@10", "complete_gold@10")
STRATA = ("all", "ambiguous", "multi_block", "implementation", "test", "example")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(results: Path, manifest_path: Path) -> dict:
    manifest = read(manifest_path)
    protocol_path = manifest_path.with_name("protocol.json")
    protocol = read(protocol_path)
    fixed = protocol_settings(protocol)
    protocol_hash = hashlib.sha256(protocol_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    rows: list[dict] = []
    runs: list[dict] = []
    latencies: list[dict] = []
    stability: list[dict] = []
    settings = None
    for item in manifest["datasets"]:
        repo = item["repo"]
        folder = results / repo.replace("/", "__")
        run = read(folder / "run.json")
        if run["mode"] != "full" or run["dataset_sha256"] != item["sha256"]:
            raise ValueError("aggregate requires matching completed full runs for every repo")
        if run["repo"] != repo or run["commit_sha"] != item["commit_sha"]:
            raise ValueError("pinned repository mismatch")
        if run["methods"] != list(METHODS):
            raise ValueError("unexpected method matrix")
        if run["repetitions"] != 3:
            raise ValueError("matrix requires three independent calls per method/question")
        if runs and run["source_sha256"] != runs[0]["source_sha256"]:
            raise ValueError("source mismatch between repo runs")
        chosen_settings = Settings.model_validate(run["settings"])
        if run.get("protocol_sha256") != protocol_hash or any(
            getattr(chosen_settings, key) != value for key, value in fixed.items()
        ):
            raise ValueError("run does not match frozen protocol settings")
        if settings and chosen_settings.model_dump(exclude={"data_dir"}) != settings.model_dump(
            exclude={"data_dir"}
        ):
            raise ValueError("settings differ between repo runs")
        settings = chosen_settings
        current = read(folder / "per_question.json")
        dataset_path = manifest_path.parent / item["file"]
        if (
            hashlib.sha256(dataset_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("label snapshot checksum mismatch")
        cases = {c.question.id: c for c in read_cases(dataset_path)}
        if len(current) != item["questions"] * len(METHODS):
            raise ValueError("incomplete quality rows")
        if len({(r["id"], r["method"]) for r in current}) != len(current):
            raise ValueError("duplicate quality rows")
        if {(r["id"], r["method"]) for r in current} != {
            (cid, method) for cid in cases for method in METHODS
        }:
            raise ValueError("quality rows do not match frozen question/method matrix")
        for row in current:
            case = cases[row["id"]]
            if (row["group"], row["stratum"], row.get("origin_case_id")) != (
                case.question.group,
                case.stratum,
                case.origin_case_id,
            ):
                raise ValueError("row metadata disagrees with frozen labels")
        rows.extend({**r, "repo": repo, "group": f"{repo}:{r['group']}"} for r in current)
        traces = read(folder / "traces.json")
        if len(traces) != item["questions"] * 9:
            raise ValueError("incomplete repetition traces")
        keyed = {(r["id"], r["method"], r["repetition"]): r for r in traces}
        if len(keyed) != len(traces):
            raise ValueError("duplicate timed calls")
        for trace in traces:
            expected = {"hybrid": 0, "rerank20": 20, "rerank50": 50}[trace["method"]]
            if len(trace["pairs"]) != expected:
                raise ValueError("recorded pair count does not match the requested depth")
        for row in current:
            base = keyed[row["id"], row["method"], 0]
            if [r["chunk_id"] for r in row["results"]] != base["final_order"] or any(
                row[m] != base["scores"][m] for m in METRICS
            ):
                raise ValueError("quality does not match the pre-chosen repetition zero")
            for rep in (1, 2):
                other = keyed[row["id"], row["method"], rep]
                if base["final_order"] != other["final_order"] or any(
                    base["scores"][m] != other["scores"][m] for m in METRICS
                ):
                    stability.append(
                        {
                            "repo": repo,
                            "id": row["id"],
                            "method": row["method"],
                            "repetition": rep,
                            "order_changed": base["final_order"] != other["final_order"],
                            "metric_changed": any(
                                base["scores"][m] != other["scores"][m] for m in METRICS
                            ),
                        }
                    )
        latencies.extend({"repo": repo, **r} for r in read(folder / "latency.json"))
        runs.append(run)
    assert settings is not None
    summaries: list[dict] = []
    comparisons: list[dict] = []
    for repo in ("pooled", *[r["repo"] for r in runs]):
        for cohort in ("all90", "added75", "retained15"):
            for stratum in STRATA:
                sample = [
                    r
                    for r in rows
                    if (repo == "pooled" or r["repo"] == repo)
                    and (
                        cohort == "all90"
                        or bool(r.get("origin_case_id")) == (cohort == "retained15")
                    )
                    and (stratum == "all" or r["stratum"] == stratum)
                ]
                if not sample:
                    continue
                meta = {"repo": repo, "cohort": cohort, "stratum": stratum}
                summaries.extend(
                    {**meta, **s} for s in summarize(sample, settings) if s["split"] == "all"
                )
                for left, right in (
                    ("rerank20", "hybrid"),
                    ("rerank50", "hybrid"),
                    ("rerank50", "rerank20"),
                ):
                    comparisons.append(
                        {
                            **meta,
                            "candidate": left,
                            "baseline": right,
                            "metrics": paired_intervals(
                                [r for r in sample if r["method"] == left],
                                [r for r in sample if r["method"] == right],
                                list(METRICS),
                                settings,
                            ),
                        }
                    )
    output = results / "aggregate"
    output.mkdir(exist_ok=False)
    write_json(output / "summary.json", summaries)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "per_repo_latency.json", latencies)
    write_json(output / "repetition_stability.json", stability)
    write_json(
        output / "provenance.json",
        {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "protocol_sha256": protocol_hash,
            "mode": "full",
            "quality_rows": len(rows),
            "quality_repetition": 0,
            "primary_comparison": "rerank50 minus rerank20 on all90; hybrid controls and target guards retained",
            "production_default_changed": False,
            "independent_review_complete": False,
            "limits": "Stratified source-authored diagnostics, not random/independent review. Added75 is newly authored text with past source overlap recorded. Groups, strata, retained/added views and repetitions are not independent replications. Intervals are pointwise exploratory. Latency stays repo-specific on experimental hardware.",
        },
    )
    lines = [
        "# Expanded depth comparison",
        "",
        "Evaluation complete; rerank default pending. Three independently executed policies; no tuning. Original15 and added75 remain separate views of one source-authored diagnostic sample.",
        "",
        "| Repo | Cohort | Stratum | Method | Questions | Groups | MRR@10 | Complete proof@10 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        if r["repo"] == "pooled" or (r["cohort"] == "all90" and r["stratum"] == "all"):
            lines.append(
                f"| {r['repo']} | {r['cohort']} | {r['stratum']} | {r['method']} | {r['questions']} | {r['groups']} | {r['mrr@10']:.3f} | {r['any_complete_proof@10']:.3f} |"
            )
    lines.extend(
        [
            "",
            "See comparisons.json for paired group-bootstrap intervals including direct 50-20 differences. Pooled point estimates weight each repo equally because all90 contains 30 per repo; inspect per-repo and strata results. Do not equate a CI spanning zero with proven equivalence. Per-repo p50/p95 are in per_repo_latency.json; no averaging percentiles across repos. No production policy is selected automatically.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"quality_rows": len(rows), "repos": len(runs), "repetition_changes": len(stability)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(aggregate(args.results, args.manifest))
