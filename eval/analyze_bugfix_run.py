"""Rebuild Phase 5 metrics, paired intervals and the fixed 8B evidence decision.

Audit frozen inputs, raw rankings, BM25 scores, RRF pools and rerank probabilities.
This verifies reported computations; it cannot independently reproduce GPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

from eval.prepare_bugfix_eval import digest, read_lines, verify_corpus, verify_frozen, write  # noqa: E402
from eval.retrieval.statistics import bootstrap_intervals, paired_intervals  # noqa: E402
from eval.run_bugfix_eval import (  # noqa: E402
    METRICS,
    changed_paths,
    file_overlap,
    lexical_query,
    query_bin,
    score,
    settings_for,
)
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.retrievers import weighted_rrf  # noqa: E402
from repo_rag.store.bm25 import Bm25Index  # noqa: E402


def audit_method(dataset: Path, folder: Path, method: str) -> list[dict]:
    freeze = verify_frozen(dataset)
    protocol = json.loads((dataset / "protocol.json").read_text(encoding="utf-8"))
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if (
        info["repo"] != freeze["repo"]
        or info["pin"] != freeze["pin"]
        or info["freeze_sha256"] != digest(dataset / "freeze.json")
        or info["method"] != method
        or info["protocol_sha256"] != freeze["sha256"]["protocol.json"]
    ):
        raise ValueError("Run provenance mismatch")
    docs = [
        Document(page_content=row["text"], metadata=row["meta"])
        for row in read_lines(dataset / "corpus.jsonl")
    ]
    doc_ids = {str(d.metadata["chunk_id"]): str(d.metadata["commit_sha"]) for d in docs}
    if len(doc_ids) != len(docs):
        raise ValueError("duplicate corpus chunk IDs")
    index = Bm25Index.build(docs)
    pairs = {row["id"]: row for row in read_lines(dataset / "pairs.jsonl")}
    paths = changed_paths(dataset)
    rows = json.loads((folder / "per_question.json").read_text(encoding="utf-8"))
    traces = json.loads((folder / "traces.json").read_text(encoding="utf-8"))
    if (
        len(rows) != len(pairs)
        or {r["id"] for r in rows} != pairs.keys()
        or len(traces) != len(pairs)
    ):
        raise ValueError("Missing or duplicated questions")
    by_trace = {t["id"]: t for t in traces}
    if len(by_trace) != len(traces) or by_trace.keys() != pairs.keys():
        raise ValueError("Trace IDs differ")
    retrieval = protocol["retrieval"]
    for row in rows:
        pair = pairs[row["id"]]
        trace = by_trace[row["id"]]
        if any(row[key] != value for key, value in pair.items()) or row["method"] != method:
            raise ValueError("Question labels changed")
        bm25 = index.search(
            lexical_query(pair["query"], method, protocol), retrieval["retrieval_k"]
        )
        expected: list[dict] = [
            {"chunk_id": cid, "sha": doc_ids[cid], "score": value} for cid, value in bm25
        ]
        if len(trace["bm25"]) != len(expected) or any(
            actual["chunk_id"] != reference["chunk_id"]
            or actual["sha"] != reference["sha"]
            or not math.isclose(actual["score"], reference["score"], rel_tol=1e-10, abs_tol=1e-12)
            for actual, reference in zip(trace["bm25"], expected, strict=True)
        ):
            raise ValueError("BM25 trace does not reproduce")
        dense = trace["dense"]
        if (
            len({d["chunk_id"] for d in dense}) != len(dense)
            or len(dense) > retrieval["retrieval_k"]
        ):
            raise ValueError("Dense candidate list is invalid")
        for item in dense:
            if doc_ids.get(item["chunk_id"]) != item["sha"] or not math.isfinite(
                item["similarity"]
            ):
                raise ValueError("Invalid dense score/identity")
        fused = weighted_rrf(
            {
                "dense": [d["chunk_id"] for d in dense],
                "bm25": [] if method == "dense" else [cid for cid, _ in bm25],
            },
            {"dense": retrieval["dense_weight"], "bm25": retrieval["bm25_weight"]},
            retrieval["rrf_k"],
        )
        if method.startswith("bm25"):
            expected_ranking = [doc_ids[cid] for cid, _ in bm25][:10]
            if dense or trace["pairs"]:
                raise ValueError("BM25 run unexpectedly contains neural scores")
        elif method.startswith("rerank"):
            candidates = fused[: retrieval["rerank_candidate_depth"]]
            if [(p["chunk_id"], p["fused_score"]) for p in trace["pairs"]] != candidates:
                raise ValueError("Rerank pool differs from reconstructed hybrid")
            for item in trace["pairs"]:
                if (
                    doc_ids[item["chunk_id"]] != item["sha"]
                    or not isinstance(item["p_yes"], (int, float))
                    or not math.isfinite(item["p_yes"])
                    or not 0 <= item["p_yes"] <= 1
                ):
                    raise ValueError("Invalid rerank pair score")
            ordered = sorted(trace["pairs"], key=lambda p: p["p_yes"], reverse=True)
            expected_ranking = [p["sha"] for p in ordered[:10]]
        else:
            if trace["pairs"]:
                raise ValueError("Unreranked method contains pair scores")
            expected_ranking = [doc_ids[cid] for cid, _ in fused[:10]]
        if row["ranking"] != expected_ranking or trace["ranking"] != expected_ranking:
            raise ValueError("Returned ranking differs from trace")
        metrics = {
            **score(expected_ranking, pair["gold"]),
            "file_hit@5": file_overlap(expected_ranking, pair["gold"], paths),
        }
        for metric, value in metrics.items():
            if row[metric] != value:
                raise ValueError(f"Metric does not reproduce: {metric}")
        if (
            row["candidate_pool"] != len(docs)
            or row["query_chars"] != len(pair["query"])
            or row["query_bin"] != query_bin(pair["query"], protocol)
        ):
            raise ValueError("Pool/query metadata does not reproduce")
        if not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0:
            raise ValueError("Invalid timing")
    return rows


def analyze(root: Path, *, output: Path | None = None) -> dict:
    output = output or root / "aggregate"
    if output.exists():
        raise FileExistsError("Preserve earlier analysis; use another --output")
    folders = sorted(p for p in (root / "datasets").iterdir() if (p / "freeze.json").exists())
    if not folders:
        raise ValueError("No frozen datasets")
    protocol = json.loads((folders[0] / "protocol.json").read_text(encoding="utf-8"))
    protocol_hash = digest(folders[0] / "protocol.json")
    all_rows: list[dict] = []
    seen_repos = set()
    source_reference = None
    for dataset in folders:
        freeze = verify_frozen(dataset)
        verify_corpus(dataset)
        seen_repos.add(freeze["repo"])
        if digest(dataset / "protocol.json") != protocol_hash:
            raise ValueError("Runs use different protocols")
        reference_dense = None
        for method in protocol["retrieval"]["methods"]:
            folder = root / "runs" / dataset.name / method
            rows = audit_method(dataset, folder, method)
            info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            hashes = info["source_sha256"]
            if source_reference is None:
                source_reference = hashes
            if hashes != source_reference:
                raise ValueError("Methods used different source snapshots")
            if not method.startswith("bm25"):
                config = info["settings"]
                expected = settings_for(
                    protocol, Path("unused"), method, info["device"]
                ).model_dump()
                active = [
                    "embedding_provider",
                    "local_embedding_model",
                    "embedding_dims",
                    "embedding_dtype",
                    "embedding_max_tokens",
                    "embedding_query_instruction",
                    "compute_device",
                    "retrieval_k",
                    "dense_weight",
                    "bm25_weight",
                    "rrf_k",
                    "reranker_mode",
                    "rerank_candidate_depth",
                    "rerank_instruction",
                    "rerank_max_tokens",
                    "rerank_first_stage_weight",
                    "rerank_instruction_mode",
                    "evidence_expansion",
                ]
                if any(config[key] != expected[key] for key in active):
                    raise ValueError("Active retrieval settings differ from protocol")
                model = protocol["models"]
                if (
                    config["local_embedding_model"] != model["embedding"]
                    or config["embedding_dims"] != model["dims"]
                    or config["embedding_dtype"] != model["dtype"]
                    or config["rerank_instruction"] != protocol["retrieval"]["rerank_instruction"]
                    or config["rerank_candidate_depth"]
                    != protocol["retrieval"]["rerank_candidate_depth"]
                ):
                    raise ValueError("Model/settings do not match protocol")
                if not info["model_revisions"].get("embedding"):
                    raise ValueError("Embedding revision missing")
                if method.startswith("rerank"):
                    expected_model = model["rerankers"][0 if method == "rerank4b" else 1]
                    if config["reranker_model"] != expected_model or not info[
                        "model_revisions"
                    ].get("reranker"):
                        raise ValueError("Reranker provenance missing")
                traces = json.loads((folder / "traces.json").read_text(encoding="utf-8"))
                dense = {t["id"]: [(d["chunk_id"], d["sha"]) for d in t["dense"]] for t in traces}
                if reference_dense is None:
                    reference_dense = dense
                if dense != reference_dense:
                    raise ValueError(
                        "First-stage lists differ across neural methods; comparison is confounded"
                    )
            all_rows.extend(rows)
    if seen_repos != set(protocol["repos"]):
        raise ValueError("Incomplete repository matrix")
    archive_path = root / "source.zip"
    if not archive_path.exists():
        raise ValueError("Frozen source.zip is required")
    import zipfile

    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in (source_reference or {}).items():
            if hashlib.sha256(archive.read(name).replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError(f"Frozen source does not match run: {name}")
    statistics = protocol["statistics"]
    settings = Settings(  # type: ignore[call-arg]  # BaseSettings supports _env_file at runtime.
        _env_file=None,
        eval_bootstrap_samples=statistics["samples"],
        eval_bootstrap_seed=statistics["seed"],
        eval_bootstrap_confidence=statistics["confidence"],
    )
    summaries: list[dict] = []
    comparisons: list[dict] = []
    cohorts: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        for split in ("all", row["split"]):
            for repo in ("all", row["repo"]):
                cohorts[(split, repo, "all")].append(row)
                for path in row["link_paths"]:
                    cohorts[(split, repo, path)].append(row)
    for (split, repo, link_path), sample in sorted(cohorts.items()):
        methods = {
            method: [r for r in sample if r["method"] == method]
            for method in protocol["retrieval"]["methods"]
        }
        for method, rows in methods.items():
            ci = bootstrap_intervals(rows, METRICS, settings)
            summaries.append(
                {
                    "split": split,
                    "repo": repo,
                    "link_path": link_path,
                    "method": method,
                    "questions": len(rows),
                    "groups": len({r["group"] for r in rows}),
                    "candidate_pools": {r["repo"]: r["candidate_pool"] for r in rows},
                    **{m: value["estimate"] for m, value in ci.items()},
                    "intervals": ci,
                }
            )
        for left, right in [
            ("hybrid", "bm25"),
            ("hybrid", "dense"),
            ("rerank4b", "hybrid"),
            ("rerank8b", "hybrid"),
            ("rerank8b", "rerank4b"),
            ("bm25_2000", "bm25"),
        ]:
            comparisons.append(
                {
                    "split": split,
                    "repo": repo,
                    "link_path": link_path,
                    "candidate": left,
                    "baseline": right,
                    "intervals": paired_intervals(methods[left], methods[right], METRICS, settings),
                }
            )
    primary = next(
        c
        for c in comparisons
        if c["split"] == "test"
        and c["repo"] == "all"
        and c["link_path"] == "all"
        and c["baseline"] == "rerank4b"
        and c["candidate"] == "rerank8b"
    )
    ci = primary["intervals"]["mrr@10"]
    rule = protocol["evidence_for_8b_on_commits"]
    guard_rows = [
        c
        for c in comparisons
        if c["split"] == "test"
        and c["repo"] != "all"
        and c["link_path"] != "all"
        and c["baseline"] == "rerank4b"
        and c["candidate"] == "rerank8b"
    ]
    descriptive_guards = [
        c for c in guard_rows if c["intervals"]["hit@5"]["groups"] < rule["guard_min_test_groups"]
    ]
    eligible_guards = [c for c in guard_rows if c not in descriptive_guards]
    enough = ci["groups"] >= rule["min_test_groups"]
    passed = (
        enough
        and ci["estimate"] >= rule["min_mrr_gain"]
        and ci["low"] > rule["mrr_ci_lower_above"]
        and all(
            c["intervals"]["hit@5"]["estimate"] >= -rule["max_hit5_loss_per_repo_and_link_path"]
            for c in eligible_guards
        )
    )
    decision = {
        "status": "inconclusive" if not enough else "passes" if passed else "does_not_pass",
        "primary": primary,
        "guards": eligible_guards,
        "descriptive_small_strata": descriptive_guards,
        "rule": rule,
        "default_changed": False,
        "not_measured": protocol["not_measured"],
    }
    output.mkdir(parents=True)
    write(output / "per_question.json", all_rows)
    write(output / "summary.json", summaries)
    write(output / "comparisons.json", comparisons)
    write(output / "decision.json", decision)
    lines = [
        "# Phase 5 bug-to-fix comparison",
        "",
        f"Fixed 8B rule: **{decision['status']}**.",
        "",
        "Link-derived labels; no independent label review. Pooled rows are descriptive, not held-out estimates.",
        "This scores finding the issue's own fix, not useful similar past fixes for a new bug.",
        "File Hit@5 is a changed-path localization proxy, including the exact fix itself; it does not establish semantic fix relevance.",
        "Small repo/link-path strata (<30 connected test groups) are descriptive, not safety evidence or vetoes.",
        "BM25 uses full queries; BM25_2000 is a separate 2000-character sensitivity check. Neural token truncation still differs.",
        "Pointwise group-bootstrap intervals are exploratory and unadjusted for multiple comparisons.",
        "",
        "| Split | Repo | Link path | Method | n | Pool(s) | Hit@1 | Hit@5 | Hit@10 | MRR@10 | File Hit@5 |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['split']} | {row['repo']} | {row['link_path']} | {row['method']} | {row['questions']} | {row['candidate_pools']} | "
            + " | ".join(f"{row[m]:.3f}" for m in METRICS)
            + " |"
        )
    lines.extend(
        [
            "",
            f"Test 8B-4B MRR: {ci['estimate']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}].",
            "",
            "Timing is a single call/query with a labelled cold first call, not a deployment latency benchmark.",
            "Not measured: " + ", ".join(protocol["not_measured"]) + ".",
            "",
        ]
    )
    lengths = []
    for method in protocol["retrieval"]["methods"]:
        for length in sorted({r["query_bin"] for r in all_rows}):
            rows = [
                r
                for r in all_rows
                if r["method"] == method and r["query_bin"] == length and r["split"] == "test"
            ]
            if rows:
                lengths.append(
                    {
                        "method": method,
                        "query_bin": length,
                        "questions": len(rows),
                        "intervals": bootstrap_intervals(rows, METRICS, settings),
                    }
                )
    write(output / "query_length_summary.json", lengths)
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.root, output=args.output)
