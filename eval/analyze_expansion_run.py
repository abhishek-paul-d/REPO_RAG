"""Audit the real expansion matrix, rescore proofs, and apply the frozen success rule."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import cast

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.ambiguous import score_case  # noqa: E402
from eval.retrieval.dataset import write_json  # noqa: E402
from eval.retrieval.expansion_eval import (  # noqa: E402
    GUARD_STRATA,
    METHODS,
    METRICS,
    fixed_settings,
    success_rule,
)
from eval.retrieval.statistics import paired_intervals, summarize  # noqa: E402
from eval.run_expansion_eval import digest, load_labels  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.expand import ChunkIndex, Mode, expand  # noqa: E402
from repo_rag.retrieval.retrievers import weighted_rrf  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_repo(folder: Path, labels: Path, repo: str, protocol: dict) -> list[dict]:
    _, entries, labelled = load_labels(labels, repo)
    cases = {c.question.id: (c, role) for c, role in labelled}
    run = read(folder / "run.json")
    if run["mode"] != "full" or run["methods"] != list(METHODS) or run["repo"] != repo:
        raise ValueError("only the complete actual production/GPU matrix can support a decision")
    if run["commit_sha"] != entries[0]["commit_sha"] or run["protocol_sha256"] != digest(
        labels / "protocol.json"
    ):
        raise ValueError("repository/protocol mismatch")
    if run["manifest_sha256"] != digest(labels / "manifest.json") or run["dataset_sha256"] != {
        e["file"]: e["sha256"] for e in entries
    }:
        raise ValueError("label snapshot mismatch")
    settings = Settings.model_validate(run["settings"])
    if settings.compute_device != "cuda" or "A100" not in str(run["gpu"]):
        raise ValueError("the frozen primary comparison requires the experimental A100 run")
    if any(getattr(settings, k) != v for k, v in fixed_settings(protocol).items()):
        raise ValueError("run settings disagree with frozen matrix")
    chunks = folder / "chunks.jsonl"
    if run["chunks_sha256"] != digest(chunks):
        raise ValueError("exported chunk snapshot changed")
    docs = [
        Document(**json.loads(line)) for line in chunks.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    if len(by_id) != len(docs) or any(d.metadata["repo"] != repo for d in docs):
        raise ValueError("invalid corpus identity")
    index = ChunkIndex(
        sorted(
            docs,
            key=lambda d: (
                str(d.metadata["path"]),
                int(d.metadata["start_line"]),
                str(d.metadata["chunk_id"]),
            ),
        ),
        settings.expansion_max_definers,
    )
    rows, traces = read(folder / "per_question.json"), read(folder / "traces.json")
    expected = {(qid, method) for qid in cases for method in METHODS}
    keyed = {(t["id"], t["method"]): t for t in traces}
    if len(rows) != len(expected) or {(r["id"], r["method"]) for r in rows} != expected:
        raise ValueError("missing/duplicate/unexpected quality rows")
    if len(traces) != len(expected) or keyed.keys() != expected:
        raise ValueError("missing/duplicate/unexpected traces")
    for row in rows:
        case, role = cases[row["id"]]
        if (row["group"], row["stratum"], row["cohort"], row["split"]) != (
            case.question.group,
            case.stratum,
            role,
            "test",
        ):
            raise ValueError("question metadata mismatch")
        trace = keyed[row["id"], row["method"]]
        pairs = trace["pairs"]
        if (
            len(pairs) != 20
            or len({p["chunk_id"] for p in pairs}) != 20
            or any(
                p["score"] is None or not math.isfinite(p["score"]) or not 0 <= p["score"] <= 1
                for p in pairs
            )
        ):
            raise ValueError("expected twenty unique, finite probability-scored candidates")
        baseline = keyed[row["id"], "plain"]
        if (
            pairs != baseline["pairs"]
            or trace["dense"] != baseline["dense"]
            or trace["bm25"] != baseline["bm25"]
        ):
            raise ValueError("methods did not share identical first-stage candidates/probabilities")
        fused = weighted_rrf(
            {
                "dense": [d["chunk_id"] for d in trace["dense"]],
                "bm25": [d["chunk_id"] for d in trace["bm25"]],
            },
            {"dense": settings.dense_weight, "bm25": settings.bm25_weight},
            settings.rrf_k,
        )[:20]
        if [(p["chunk_id"], p["fused_score"]) for p in pairs] != fused:
            raise ValueError("rerank input does not reproduce from logged first-stage fusion")
        ranked = [p["chunk_id"] for p in sorted(pairs, key=lambda p: p["score"], reverse=True)[:10]]
        if trace["ranked_top10"] != ranked:
            raise ValueError("saved pre-expansion order is not plain probability ordering")
        mode = "none" if row["method"] == "plain" else row["method"]
        rebuilt = expand(
            [by_id[c] for c in ranked],
            index,
            mode=cast(Mode, mode),
            keep=settings.expansion_keep,
            seeds=settings.expansion_seeds,
            per_seed=settings.expansion_per_seed,
        )
        ids = [str(d.metadata["chunk_id"]) for d in rebuilt]
        if trace["final_order"] != ids or [r["chunk_id"] for r in row["results"]] != ids:
            raise ValueError(
                "final order cannot be reproduced from the stored corpus and ranked hits"
            )
        rescored = score_case(case, rebuilt, settings)
        if any(row[k] != value for k, value in rescored.items()):
            raise ValueError("saved retrieval metrics do not reproduce")
        measured = {
            "returned_chunks": len(ids),
            "returned_chars": sum(len(d.page_content) for d in rebuilt),
            "expanded_ids": [cid for cid in ids if cid not in ranked],
            "displaced_ids": [cid for cid in ranked if cid not in ids],
        }
        if any(row[k] != v for k, v in measured.items()):
            raise ValueError("saved text budget or displacement does not reproduce")
    return [{**r, "repo": repo, "group": f"{repo}:{r['group']}"} for r in rows]


def analyze(results: Path) -> dict:
    labels = results / "labels"
    protocol = read(labels / "protocol.json")
    manifest = read(labels / "manifest.json")
    repos = sorted({e["repo"] for e in manifest["datasets"]})
    rows: list[dict] = []
    runs: list[dict] = []
    with zipfile.ZipFile(results / "phase4_expansion_source.zip") as package:
        import hashlib

        for repo in repos:
            folder = results / repo.replace("/", "__")
            rows.extend(audit_repo(folder, labels, repo, protocol))
            run = read(folder / "run.json")
            for path, sha in run["source_sha256"].items():
                if hashlib.sha256(package.read(path).replace(b"\r\n", b"\n")).hexdigest() != sha:
                    raise ValueError(f"source snapshot mismatch: {path}")
            if runs and (
                run["source_sha256"] != runs[0]["source_sha256"]
                or any(
                    run[k] != runs[0][k]
                    for k in ("embedding_model_revision", "reranker_model_revision")
                )
            ):
                raise ValueError("mixed source/model revisions across repositories")
            runs.append(run)
    settings = Settings(**fixed_settings(protocol))
    for cohort, count in protocol["expected_questions"].items():
        if len([r for r in rows if r["cohort"] == cohort and r["method"] == "plain"]) != count:
            raise ValueError("incomplete primary or regression cohort")
    summaries, comparisons = [], []
    for repo in ("pooled", *repos):
        for cohort in ("fresh", "regression"):
            for stratum in ("all", *GUARD_STRATA):
                sample = [
                    r
                    for r in rows
                    if r["cohort"] == cohort
                    and (repo == "pooled" or r["repo"] == repo)
                    and (stratum == "all" or r["stratum"] == stratum)
                ]
                if not sample:
                    continue
                meta = {"repo": repo, "cohort": cohort, "stratum": stratum}
                for s in summarize(sample, settings):
                    if s["split"] != "all":
                        continue
                    selected = [r for r in sample if r["method"] == s["method"]]
                    summaries.append(
                        {
                            **meta,
                            **s,
                            "mean_returned_chars": sum(r["returned_chars"] for r in selected)
                            / len(selected),
                            "mean_returned_chunks": sum(r["returned_chunks"] for r in selected)
                            / len(selected),
                            "mean_displaced": sum(len(r["displaced_ids"]) for r in selected)
                            / len(selected),
                        }
                    )
                for method in METHODS[1:]:
                    a = [r for r in sample if r["method"] == method]
                    b = [r for r in sample if r["method"] == "plain"]
                    chars = sum(r["returned_chars"] for r in b)
                    if chars <= 0:
                        raise ValueError("empty returned-text baseline")
                    comparisons.append(
                        {
                            **meta,
                            "candidate": method,
                            "baseline": "plain",
                            "metrics": paired_intervals(a, b, list(METRICS), settings),
                            "returned_text_growth": sum(r["returned_chars"] for r in a) / chars - 1,
                            "proof_better": sum(
                                x["any_complete_proof@10"] > y["any_complete_proof@10"]
                                for x, y in zip(a, b, strict=True)
                            ),
                            "proof_worse": sum(
                                x["any_complete_proof@10"] < y["any_complete_proof@10"]
                                for x, y in zip(a, b, strict=True)
                            ),
                        }
                    )
    primary = [r for r in comparisons if r["repo"] == "pooled" and r["candidate"] == "callee"]
    fresh = next(r for r in primary if r["cohort"] == "fresh" and r["stratum"] == "all")
    guards = {
        r["stratum"]: r["metrics"]
        for r in primary
        if r["cohort"] == "regression" and r["stratum"] in GUARD_STRATA
    }
    growth = {r["cohort"]: r["returned_text_growth"] for r in primary if r["stratum"] == "all"}
    decision = {
        **success_rule(fresh["metrics"], guards, growth, protocol),
        "fresh_metrics": fresh["metrics"],
        "regression_guards": guards,
        "returned_text_growth": growth,
        "protocol_sha256": digest(labels / "protocol.json"),
        "quality_rows": len(rows),
        "source_audit_passed": True,
        "answer_quality_measured": False,
        "latency_measured": False,
    }
    output = results / "aggregate"
    output.mkdir(exist_ok=False)
    write_json(output / "summary.json", summaries)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "decision.json", decision)
    write_json(output / "per_question.json", rows)
    lines = [
        "# Frozen evidence-expansion comparison",
        "",
        "Quality only: shared exact pair scores, no independent timing. Source/corpus/labels/metrics audited. Independent label review pending.",
        "",
        "| Cohort | Stratum | Method | Questions | Groups | MRR@10 | Complete proof@10 | Mean chars | Mean displaced |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        if r["repo"] == "pooled":
            lines.append(
                f"| {r['cohort']} | {r['stratum']} | {r['method']} | {r['questions']} | {r['groups']} | {r['mrr@10']:.3f} | {r['any_complete_proof@10']:.3f} | {r['mean_returned_chars']:.0f} | {r['mean_displaced']:.2f} |"
            )
    lines.extend(
        [
            "",
            f"Callee eligibility: **{decision['eligible_callee']}**. Checks: {decision['checks']}.",
            "",
            decision["next_step"],
            "",
            "Both is a secondary diagnostic, never a substitute primary. No default was changed. Pointwise intervals are exploratory; a zero-spanning interval is inconclusive. Fresh30 (including 12 Click) and prior120 are distinct views, not independent replications or generated-answer accuracy.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print(json.dumps(analyze(parser.parse_args().results), indent=2))
