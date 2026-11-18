"""Pre-registered confirmation that retrieval_k=200 helps commit search at rerank depth 50.

Kept outside eval/retrieval/ on purpose: other frozen notebooks hash that folder, and adding a
file there would change their source ZIPs and break checkpoint resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection
from itertools import zip_longest
from pathlib import Path

from langchain_core.documents import Document

from eval.prepare_bugfix_eval import digest, read_lines, verify_corpus, verify_frozen
from eval.retrieval.commit_depth import prefix_ranking, source_path
from eval.run_bugfix_eval import changed_paths, query_bin, score, settings_for, source_hashes
from repo_rag.config import Settings
from repo_rag.history.commit_indexer import is_fix_message
from repo_rag.retrieval.retrievers import weighted_rrf
from repo_rag.store.bm25 import Bm25Index

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "eval/datasets/phase5_commit_confirm_v1"
V2 = ROOT / "eval/datasets/phase5_bugfix_v2"
METRICS = ["hit@1", "hit@5", "hit@10", "mrr@10", "pool_recall"]
GROUPS = ("fresh", "v2dev")
# A real address needs a letters-only top-level label: this skips `actions/cache@v3.0.9`.
EMAIL = re.compile(r"(?<![\w/.-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stem_of(repo: str) -> str:
    return repo.replace("/", "__")


def hashes() -> dict[str, str]:
    value = source_hashes()
    for name in (
        "eval/confirm_lib.py",
        "eval/prepare_commit_confirm.py",
        "eval/run_commit_confirm_eval.py",
        "eval/analyze_commit_confirm.py",
    ):
        value[name] = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return dict(sorted(value.items()))


# ------------------------------------------------------------------ dataset hygiene


def leakage_report(folder: Path) -> dict:
    """The v2 leakage checks, re-run on any frozen dataset.

    Question-level violations list the affected question IDs (`flagged_questions`); the experiment
    freeze must exclude exactly those. Document-level email text can not be excluded: it must be 0.
    """
    freeze = verify_frozen(folder)
    verify_corpus(folder)
    records = {r["sha"]: r for r in read_lines(folder / "records.jsonl")}
    corpus = {r["meta"]["commit_sha"]: r["text"] for r in read_lines(folder / "corpus.jsonl")}
    pairs = read_lines(folder / "pairs.jsonl")
    flagged: set[str] = set()
    violations: dict[str, list[str]] = {
        "gold_sha_in_query": [],
        "gold_message_empty": [],
        "own_issue_reference_in_gold_document": [],
        "email_like_text_in_documents": [],
        "gold_not_candidate": [],
        "empty_query_part": [],
    }
    for pair in pairs:
        number = pair["issue_number"]
        own = re.compile(
            rf"(?:#\s*{number}(?!\d)|/(?:issues|pull)/{number}(?!\d)|\b[Ii]ssue\s+{number}(?!\d))"
        )
        title, _, body = pair["query"].partition("\n\n")
        if not title.strip() or not body.strip():
            violations["empty_query_part"].append(pair["id"])
            flagged.add(pair["id"])
        for sha in pair["gold"]:
            if len(sha) < 7 or sha not in corpus:
                violations["gold_not_candidate"].append(f"{pair['id']}:{sha[:7]}")
                flagged.add(pair["id"])
                continue
            if re.search(rf"\b{sha[:7]}[0-9a-f]*\b", pair["query"]):
                violations["gold_sha_in_query"].append(f"{pair['id']}:{sha[:7]}")
                flagged.add(pair["id"])
            if not str(records[sha]["message"]).strip():
                violations["gold_message_empty"].append(f"{pair['id']}:{sha[:7]}")
                flagged.add(pair["id"])
            if own.search(corpus[sha]):
                violations["own_issue_reference_in_gold_document"].append(f"{pair['id']}:{sha[:7]}")
                flagged.add(pair["id"])
            if not is_fix_message(str(records[sha]["message"])):
                violations["gold_not_candidate"].append(f"{pair['id']}:{sha[:7]}:heuristic")
                flagged.add(pair["id"])
    for sha, text in corpus.items():
        if EMAIL.search(text):
            violations["email_like_text_in_documents"].append(sha[:7])
    return {
        "repo": freeze["repo"],
        "questions": len(pairs),
        "groups": len({p["group"] for p in pairs}),
        "documents": len(corpus),
        "violations": {k: len(v) for k, v in violations.items()},
        "flagged_questions": sorted(flagged),
        "examples": {k: v[:5] for k, v in violations.items() if v},
    }


def timing_ids(protocol: dict, pairs_by_stem: dict[str, list[dict]]) -> list[str]:
    """Deterministic: sorted IDs per sorted repo stem, round-robin, first N."""
    queues = [sorted(p["id"] for p in pairs_by_stem[stem]) for stem in sorted(pairs_by_stem)]
    ordered = [qid for row in zip_longest(*queues) for qid in row if qid is not None]
    return ordered[: protocol["confirm"]["timing"]["questions"]]


# ------------------------------------------------------------------ verification


def verify_experiment(experiment: Path, data: Path) -> dict:
    """data/fresh/<stem> and data/v2/<stem> hold frozen datasets. Fails on any difference."""
    freeze = read(experiment / "freeze.json")
    for name, expected in freeze["sha256"].items():
        if digest(experiment / name) != expected:
            raise ValueError(f"Experiment input changed: {name}")
    protocol = read(experiment / "protocol.json")
    lock = read(experiment / "protocol.freeze.json")
    if lock["protocol_sha256"] != digest(experiment / "protocol.json"):
        raise ValueError("Protocol changed after its pre-sampling freeze")
    fresh: dict[str, list[dict]] = {}
    for repo in protocol["repos"]:
        stem = stem_of(repo)
        folder = data / "fresh" / stem
        if not (folder / "freeze.json").exists():
            continue  # a repo that produced no questions is reported, never silently dropped
        if digest(folder / "freeze.json") != freeze["dataset_freeze_sha256"][stem]:
            raise ValueError("Fresh dataset freeze differs from the experiment freeze")
        if (folder / "protocol.json").read_bytes() != (experiment / "protocol.json").read_bytes():
            raise ValueError("Dataset was prepared under a different protocol")
        verify_corpus(folder)
        report = leakage_report(folder)
        if report["violations"]["email_like_text_in_documents"]:
            raise ValueError(f"Email-like text in {repo} documents")
        if set(report["flagged_questions"]) != set(freeze["excluded"]).intersection(
            p["id"] for p in read_lines(folder / "pairs.jsonl")
        ):
            raise ValueError(f"Leakage-flagged questions differ from the frozen exclusions: {repo}")
        fresh[stem] = [
            p for p in read_lines(folder / "pairs.jsonl") if p["id"] not in freeze["excluded"]
        ]
    if set(fresh) != set(freeze["dataset_freeze_sha256"]):
        raise ValueError("Fresh repository set differs from the experiment freeze")
    if {s: len(p) for s, p in fresh.items()} != freeze["counts"]:
        raise ValueError("Question counts differ from the experiment freeze")
    if timing_ids(protocol, fresh) != freeze["timing_ids"]:
        raise ValueError("Timing IDs do not rebuild")
    support = protocol["supporting"]
    for repo in support["repos"]:
        stem = stem_of(repo)
        folder = data / "v2" / stem
        if digest(folder / "freeze.json") != support["freeze_sha256"][stem]:
            raise ValueError("v2 dataset freeze differs from preregistration")
        verify_corpus(folder)
    return protocol


# ------------------------------------------------------------------ labels and settings


def load_pairs(
    folder: Path, protocol: dict, split: str | None = None, exclude: Collection[str] = ()
) -> tuple[list[Document], list[dict]]:
    verify_corpus(folder)
    docs = [
        Document(page_content=r["text"], metadata=r["meta"])
        for r in read_lines(folder / "corpus.jsonl")
    ]
    paths = changed_paths(folder)
    pairs = [
        p
        for p in read_lines(folder / "pairs.jsonl")
        if split in (None, p["split"]) and p["id"] not in exclude
    ]
    for pair in pairs:
        pair.update(
            source_path=any(source_path(p) for sha in pair["gold"] for p in paths[sha]),
            query_bin=query_bin(pair["query"], protocol),
            candidate_pool=len(docs),
        )
    return docs, pairs


def arm_settings(protocol: dict, data_dir: Path, device: str, retrieval_k: int) -> Settings:
    confirm = protocol["confirm"]
    base = settings_for(protocol, data_dir, confirm["judge"], device)
    return base.model_copy(
        update={
            "retrieval_k": retrieval_k,
            "rerank_candidate_depth": confirm["rerank_depth"],
            "bm25_weight": confirm["bm25_weight"],
        }
    )


# ------------------------------------------------------------------ rows and audit


def make_row(pair: dict, method: str, ranking: list[str], scored: list[dict]) -> dict:
    return {
        **pair,
        "method": method,
        "ranking": ranking,
        **score(ranking, pair["gold"]),
        "pool_recall": float(any(p["sha"] in pair["gold"] for p in scored)),
    }


def question_rows(pair: dict, trace: dict, protocol: dict) -> list[dict]:
    rows = []
    for arm in protocol["confirm"]["arms"]:
        scored = trace["arms"][arm]["pairs"]
        rows.append(make_row(pair, f"{arm}_first", [p["sha"] for p in scored[:10]], scored))
        rows.append(make_row(pair, arm, prefix_ranking(scored, len(scored)), scored))
    rerun = trace["rerun"]["pairs"]
    rows.append(make_row(pair, "A_rerun", prefix_ranking(rerun, len(rerun)), rerun))
    return rows


def audit_run(
    folder: Path,
    run_dir: Path,
    protocol: dict,
    *,
    split: str | None,
    exclude: Collection[str] = (),
) -> tuple[list[dict], dict]:
    """Rebuild every candidate list, fusion, shared score and metric from stored lists."""
    docs, pairs = load_pairs(folder, protocol, split, exclude)
    info, traces, rows = (
        read(run_dir / name) for name in ("run.json", "traces.json", "per_question.json")
    )
    if info["rows"] != len(rows) or info["stage"] != "confirm":
        raise ValueError("Run row count or stage differs")
    if info["dataset_freeze_sha256"] != digest(folder / "freeze.json"):
        raise ValueError("Run provenance mismatch")
    if len(traces) != len(pairs) or {t["id"] for t in traces} != {p["id"] for p in pairs}:
        raise ValueError("Incomplete or duplicate query traces")
    by_id = {p["id"]: p for p in pairs}
    ids = {str(d.metadata["chunk_id"]): str(d.metadata["commit_sha"]) for d in docs}
    index = Bm25Index.build(docs)
    confirm = protocol["confirm"]
    expected_rows: list[dict] = []
    max_rerun_diff = 0.0
    changed_rerank = 0
    for trace in traces:
        pair = by_id[trace["id"]]
        shared: dict[str, float] = {}
        for arm, k in confirm["arms"].items():
            t = trace["arms"][arm]
            if t["retrieval_k"] != k:
                raise ValueError("Arm retrieval_k differs from protocol")
            expected = index.search(pair["query"], k)
            keyword = t["bm25"]
            if len(keyword) != len(expected) or any(
                a["chunk_id"] != cid
                or a["sha"] != ids[cid]
                or not math.isclose(a["score"], value, rel_tol=1e-10, abs_tol=1e-12)
                for a, (cid, value) in zip(keyword, expected, strict=True)
            ):
                raise ValueError("Lexical trace does not reproduce")
            dense = t["dense"]
            if len(dense) != min(len(docs), k) or len({d["chunk_id"] for d in dense}) != len(dense):
                raise ValueError("Incomplete or duplicate dense candidates")
            if any(ids.get(d["chunk_id"]) != d["sha"] for d in dense):
                raise ValueError("Dense identity mismatch")
            pool = weighted_rrf(
                {
                    "dense": [d["chunk_id"] for d in dense],
                    "bm25": [d["chunk_id"] for d in keyword],
                },
                {"dense": protocol["retrieval"]["dense_weight"], "bm25": confirm["bm25_weight"]},
                protocol["retrieval"]["rrf_k"],
            )[: confirm["rerank_depth"]]
            scored = t["pairs"]
            if [(p["chunk_id"], p["fused_score"]) for p in scored] != pool:
                raise ValueError("Scored pool is not the fused top of this arm's lists")
            for p in scored:
                if p["sha"] != ids[p["chunk_id"]]:
                    raise ValueError("Pair identity mismatch")
                if shared.setdefault(p["chunk_id"], p["p_yes"]) != p["p_yes"]:
                    raise ValueError("A pair received different scores in different arms")
        rerun = trace["rerun"]["pairs"]
        first = trace["arms"]["A"]["pairs"]
        if [p["chunk_id"] for p in rerun] != [p["chunk_id"] for p in first]:
            raise ValueError("Rerun scored a different pool than arm A")
        max_rerun_diff = max(
            max_rerun_diff,
            *(abs(a["p_yes"] - b["p_yes"]) for a, b in zip(rerun, first, strict=True)),
        )
        changed_rerank += int(
            prefix_ranking(rerun, len(rerun)) != prefix_ranking(first, len(first))
        )
        expected_rows.extend(question_rows(pair, trace, protocol))
    if rows != expected_rows:
        raise ValueError("Rows or rankings do not reconstruct from stored candidate scores")
    return rows, {
        "questions": len(pairs),
        "max_rerun_probability_difference": max_rerun_diff,
        "rerun_changed_top10_questions": changed_rerank,
    }
