"""Frozen commit-depth experiment: shared scores, dev-only weight selection and audits."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path, PurePosixPath

from langchain_core.documents import Document

from eval.prepare_bugfix_eval import digest, read_lines, verify_corpus, verify_frozen
from eval.run_bugfix_eval import changed_paths, query_bin, score, settings_for, source_hashes
from repo_rag.config import Settings
from repo_rag.retrieval.retrievers import weighted_rrf
from repo_rag.store.bm25 import Bm25Index

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "eval/datasets/phase5_commit_depth_v1"
METRICS = ["hit@1", "hit@5", "hit@10", "mrr@10", "pool_recall"]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protocol(experiment: Path, datasets: Path) -> dict:
    freeze = read(experiment / "freeze.json")
    for name, expected in freeze["sha256"].items():
        path = (experiment / name).resolve()
        if not path.is_relative_to(experiment.resolve()) or digest(path) != expected:
            raise ValueError(f"Experiment input changed: {name}")
    protocol = read(experiment / "protocol.json")
    all_pairs = []
    queues = []
    for stem, expected in sorted(protocol["input_freeze_sha256"].items()):
        folder = datasets / stem
        if digest(folder / "freeze.json") != expected:
            raise ValueError("Input dataset freeze differs from preregistration")
        data_freeze = verify_frozen(folder)
        if data_freeze["sha256"]["protocol.json"] != protocol["input_protocol_sha256"]:
            raise ValueError("Dataset protocol differs from preregistration")
        pairs = read_lines(folder / "pairs.jsonl")
        all_pairs.extend(pairs)
        queues.append(sorted(p["id"] for p in pairs if p["split"] == "test"))
    cohorts = read(experiment / "cohorts.json")
    if set(cohorts["source_path"]) != {p["id"] for p in all_pairs}:
        raise ValueError("Source cohort IDs differ from dataset")
    ordered = [qid for row in zip_longest(*queues) for qid in row if qid is not None]
    if cohorts["timing_ids"] != ordered[: protocol["timing"]["questions"]]:
        raise ValueError("Timing questions do not match frozen round-robin test order")
    return protocol


def source_path(path: str) -> bool:
    source = PurePosixPath(path.lower())
    return not (
        set(source.parts) & {"docs", "doc", "documentation"}
        or source.suffix in {".rst", ".md", ".txt"}
        or source.name.startswith(("changes", "changelog", "history", "news"))
    )


def labels(dataset: Path, protocol: dict, cohorts: dict) -> tuple[list[Document], list[dict]]:
    verify_corpus(dataset)
    docs = [
        Document(page_content=r["text"], metadata=r["meta"])
        for r in read_lines(dataset / "corpus.jsonl")
    ]
    paths = changed_paths(dataset)
    pairs = read_lines(dataset / "pairs.jsonl")
    for pair in pairs:
        source = any(source_path(p) for sha in pair["gold"] for p in paths[sha])
        if cohorts["source_path"][pair["id"]] != source:
            raise ValueError("Frozen source-path cohort does not rebuild")
        pair.update(
            source_path=source,
            query_bin=query_bin(pair["query"], protocol),
            candidate_pool=len(docs),
        )
    return docs, pairs


def settings(
    protocol: dict,
    data_dir: Path,
    judge: str,
    device: str,
    *,
    depth: int = 200,
    weight: float = 0.3,
    retrieval_k: int = 200,
) -> Settings:
    base = settings_for(protocol, data_dir, judge, device)
    return base.model_copy(
        update={"retrieval_k": retrieval_k, "rerank_candidate_depth": depth, "bm25_weight": weight}
    )


def hashes() -> dict[str, str]:
    value = source_hashes()
    for name in ("eval/run_commit_depth_eval.py", "eval/analyze_commit_depth.py"):
        value[name] = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return dict(sorted(value.items()))


def fused(
    dense: list[dict], keyword: list[dict], weight: float, protocol: dict
) -> list[tuple[str, float]]:
    return weighted_rrf(
        {
            "dense": [d["chunk_id"] for d in dense],
            "bm25": [d["chunk_id"] for d in keyword] if weight else [],
        },
        {"dense": protocol["depth"]["dense_weight"], "bm25": weight},
        protocol["depth"]["rrf_k"],
    )


def prefix_ranking(pairs: list[dict], depth: int) -> list[str]:
    if depth < 1 or len({p["sha"] for p in pairs}) != len(pairs):
        raise ValueError("Invalid depth or duplicated pair identities")
    for item in pairs:
        probability = item["p_yes"]
        if (
            not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Invalid pair probability")
    # Stable sort explicitly preserves original fused order for equal probabilities.
    ordered = sorted(enumerate(pairs[:depth]), key=lambda item: (-item[1]["p_yes"], item[0]))
    return [p["sha"] for _, p in ordered[:10]]


def depth_rows(pair: dict, candidates: list[dict], judge: str, protocol: dict) -> list[dict]:
    rows = []
    for depth in protocol["depth"]["depths"]:
        ranking = prefix_ranking(candidates, depth)
        rows.append(
            {
                **pair,
                "method": f"{judge}_d{depth}",
                "judge": judge,
                "depth": depth,
                "ranking": ranking,
                **score(ranking, pair["gold"]),
                "pool_recall": float(any(p["sha"] in pair["gold"] for p in candidates[:depth])),
            }
        )
    return rows


def weight_rows(
    pair: dict, dense: list[dict], keyword: list[dict], ids: dict, protocol: dict
) -> list[dict]:
    rows = []
    for weight in protocol["weight"]["weights"]:
        pool = [
            ids[cid]
            for cid, _ in fused(dense, keyword, weight, protocol)[
                : protocol["weight"]["retrieval_k"]
            ]
        ]
        ranking = pool[:10]
        rows.append(
            {
                **pair,
                "method": f"weight_{weight:g}",
                "weight": weight,
                "ranking": ranking,
                **score(ranking, pair["gold"]),
                "pool_recall": float(bool(set(pool) & set(pair["gold"]))),
            }
        )
    return rows


def select_weight(rows: list[dict], protocol: dict) -> dict:
    rule = protocol["weight"]
    dev = [r for r in rows if r["split"] == "dev"]
    by_weight = {w: [r for r in dev if r["weight"] == w] for w in rule["weights"]}
    reference = {(r["id"], r["group"]) for r in by_weight[rule["baseline"]]}
    if not reference or any(
        len(sample) != len(reference) or {(r["id"], r["group"]) for r in sample} != reference
        for sample in by_weight.values()
    ):
        raise ValueError("Dev weight matrix is incomplete or duplicated")
    means = {w: sum(r["mrr@10"] for r in sample) / len(sample) for w, sample in by_weight.items()}
    best = min(means, key=lambda w: (-means[w], abs(w - rule["baseline"]), w))
    gain = means[best] - means[rule["baseline"]]
    selected = best if gain + 1e-12 >= rule["min_dev_mrr_gain"] else rule["baseline"]
    return {
        "selected": selected,
        "baseline": rule["baseline"],
        "dev_mrr": {f"{w:g}": m for w, m in means.items()},
        "dev_questions": len(reference),
        "dev_gain": gain,
        "selection_split": "dev",
        "default_changed": False,
    }


def check_trace(
    trace: dict,
    docs: list[Document],
    pair: dict,
    protocol: dict,
    *,
    retrieval_k: int,
    weight: float,
    index: Bm25Index,
) -> list[tuple[str, float]]:
    ids = {str(d.metadata["chunk_id"]): str(d.metadata["commit_sha"]) for d in docs}
    expected = index.search(pair["query"], retrieval_k)
    keyword = trace["bm25"]
    if len(keyword) != len(expected) or any(
        a["chunk_id"] != cid
        or a["sha"] != ids[cid]
        or not math.isclose(a["score"], value, rel_tol=1e-10, abs_tol=1e-12)
        for a, (cid, value) in zip(keyword, expected, strict=True)
    ):
        raise ValueError("Lexical trace does not reproduce")
    dense = trace["dense"]
    if len(dense) != min(len(docs), retrieval_k) or len({d["chunk_id"] for d in dense}) != len(
        dense
    ):
        raise ValueError("Incomplete/duplicate dense candidates")
    if any(ids.get(d["chunk_id"]) != d["sha"] or not math.isfinite(d["similarity"]) for d in dense):
        raise ValueError("Invalid dense identity/score")
    return fused(dense, keyword, weight, protocol)


def audit_run(dataset: Path, folder: Path, protocol: dict, cohorts: dict, stage: str) -> list[dict]:
    docs, pairs = labels(dataset, protocol, cohorts)
    info, traces, rows = (
        read(folder / name) for name in ("run.json", "traces.json", "per_question.json")
    )
    if info["rows"] != len(rows):
        raise ValueError("Run row count differs")
    if info["dataset_freeze_sha256"] != digest(dataset / "freeze.json") or info["stage"] != stage:
        raise ValueError("Run provenance mismatch")
    if len(traces) != len(pairs) or len({t["id"] for t in traces}) != len(pairs):
        raise ValueError("Incomplete/duplicate query traces")
    by_id = {p["id"]: p for p in pairs}
    expected_rows = []
    ids = {str(d.metadata["chunk_id"]): str(d.metadata["commit_sha"]) for d in docs}
    index = Bm25Index.build(docs)
    for trace in traces:
        pair = by_id[trace["id"]]
        weight = info["settings"]["bm25_weight"]
        k = (
            protocol["weight"]["retrieval_k"]
            if stage in {"weights", "selected"}
            else protocol["depth"]["retrieval_k"]
        )
        pool = check_trace(trace, docs, pair, protocol, retrieval_k=k, weight=weight, index=index)
        if stage == "weights":
            expected_rows.extend(weight_rows(pair, trace["dense"], trace["bm25"], ids, protocol))
        else:
            depth = info["settings"]["rerank_candidate_depth"]
            scored = trace["pairs"]
            if [(p["chunk_id"], p["fused_score"]) for p in scored] != pool[:depth]:
                raise ValueError("Scored pool is not the fused prefix")
            if any(p["sha"] != ids[p["chunk_id"]] for p in scored):
                raise ValueError("Pair identity mismatch")
            if stage == "depth":
                expected_rows.extend(depth_rows(pair, scored, info["judge"], protocol))
            else:
                ranking = prefix_ranking(scored, depth)
                expected_rows.append(
                    {
                        **pair,
                        "method": f"selected_{weight:g}",
                        "weight": weight,
                        "ranking": ranking,
                        **score(ranking, pair["gold"]),
                        "pool_recall": float(any(p["sha"] in pair["gold"] for p in scored)),
                    }
                )
    if rows != expected_rows:
        raise ValueError("Rows/prefix rankings or metrics do not reconstruct from stored pairs")
    return rows
