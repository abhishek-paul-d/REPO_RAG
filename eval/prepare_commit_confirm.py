"""Build the fresh holdout for the retrieval_k=200 confirmation, in three separate steps.

1. `protocol`: record repository pins (from local full clones) and freeze the protocol
   BEFORE any issue is sampled.
2. `prepare`: sample, scrub and freeze each repository's dataset (needs GITHUB_TOKEN; reads
   the GitHub API only). A repository with no valid questions is reported, never dropped quietly.
3. `freeze`: leakage-check every dataset and write the experiment freeze before retrieval.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.confirm_lib import EXPERIMENT, V2, leakage_report, read, stem_of, timing_ids  # noqa: E402
from eval.prepare_bugfix_eval import digest, prepare, read_lines, write  # noqa: E402
from repo_rag.history.commit_indexer import is_fix_message  # noqa: E402

REPOS = ["pallets/click", "encode/httpx", "pallets/werkzeug", "pallets/jinja"]
DEPTH_PROTOCOL = ROOT / "eval/datasets/phase5_commit_depth_v1/protocol.json"


def build_protocol(pins: dict[str, str]) -> dict:
    v2 = read(V2 / "protocol.json")
    depth = read(DEPTH_PROTOCOL)
    if sorted(pins) != sorted(REPOS):
        raise ValueError("Pins must cover exactly the preregistered repositories")
    selection = copy.deepcopy(v2["selection"])
    selection.update(
        {
            "direct_repos": REPOS,
            "via_pr_repos": [],
            "order": "most recent 5000 non-merge commits at the recorded pin; closed GitHub Issues "
            "with nonempty title and body reached by verified direct closing references",
            "historical_overlap": "none of these repositories appears in any earlier evaluation; "
            "Werkzeug and Jinja share maintainers and style with Flask, which stays a caveat",
        }
    )
    return {
        "version": 2,
        "written": "2026-10-27",
        "task": "confirm the retrieval_k=200 candidate-pool effect for commit bug-to-fix search "
        "on repositories not used before; issue to its own linked fix, not similar past fixes",
        "repos": REPOS,
        "pins": pins,
        "selection": selection,
        "documents": v2["documents"],
        "split": v2["split"],
        "models": v2["models"],
        "retrieval": v2["retrieval"],
        "statistics": v2["statistics"],
        "query_budgets": v2["query_budgets"],
        "expected_revisions": {
            "embedding": depth["expected_revisions"]["embedding"],
            "rerank8b": depth["expected_revisions"]["rerank8b"],
        },
        "confirm": {
            "hypothesis": (
                "Building the candidate pool from longer per-signal lists (retrieval_k=200) but "
                "still reranking only the fused top 50 raises Hit@5 and MRR@10 for commit search, "
                "at no extra reranker cost. Evidence so far is post hoc on the v2 test split "
                "(Hit@5 +0.0152 [+0.0037, +0.0309], MRR +0.0113 [-0.0009, +0.0266])."
            ),
            "judge": "rerank8b",
            "rerank_depth": 50,
            "bm25_weight": 0.3,
            "arms": {"A": 50, "B": 200, "C": 100},
            "arm_roles": {
                "A": "control, current default",
                "B": "hypothesis",
                "C": "dose point, descriptive only",
            },
            "scoring": "all arms run the production search path with one shared score cache per "
            "question: a (query, document) pair is scored once, so arms differ only by which "
            "candidates enter the reranked top 50. Arm A is scored again from scratch (no cache) "
            "to measure GPU rerun noise.",
            "primary": "B minus A on all fresh questions: Hit@5 and MRR@10, paired evidence-group "
            "bootstrap (10,000 samples, seed 42)",
            "rule": {
                "min_hit5_gain": 0.01,
                "min_mrr_delta": 0.0,
                "lower_bound_floor": -0.01,
                "repo_guard_min_groups": 30,
                "max_repo_hit5_loss": 0.02,
                "max_ci_width_hit5": 0.06,
                "tentative_below_questions": 100,
                "noise": "|Hit@5(A rerun) - Hit@5(A)| must not exceed the observed B-A Hit@5 "
                "effect, otherwise inconclusive",
                "outcomes": "keep_50 when a check fails; inconclusive when the checks pass but the "
                "interval is wider than max_ci_width_hit5 or the noise check fails; "
                "eligible_pending_review otherwise. Nothing changes automatically.",
                "if_adopted": "add a per-kind commit setting (retrieval_k is shared by code, docs "
                "and commits, and only commits were measured); keep code and docs at 50",
            },
            "timing": {
                "questions": 40,
                "repetitions": 3,
                "arm_order": "rotate arm order by question index plus repetition",
                "warmup": "one unmeasured production call per arm and repo",
                "label": "warm A100 experiment timing, per-repo p50/p95 only; never averaged "
                "across repos; not laptop speed",
            },
            "dev_use": "none: every fresh question is evaluation data; split labels are ignored",
        },
        "supporting": {
            "dataset": "phase5_bugfix_v2",
            "split": "dev",
            "repos": v2["repos"],
            "freeze_sha256": {
                stem: depth["input_freeze_sha256"][stem]
                for stem in sorted(depth["input_freeze_sha256"])
            },
            "note": "66 questions; the nested-50 effect was never examined on dev, so this is a "
            "small unbiased check, reported separately",
        },
        "not_measured": [
            "generated answer quality",
            "similar past fix usefulness for a new bug",
            "independent label review",
            "laptop or deployment latency",
            "code and docs retrieval at retrieval_k=200",
            "other languages",
        ],
    }


def cmd_protocol(args: argparse.Namespace) -> None:
    pins = dict(item.split("=", 1) for item in args.pin)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT / "protocol.json"
    if (EXPERIMENT / "protocol.freeze.json").exists():
        raise FileExistsError("Protocol already frozen; preserve it")
    write(path, build_protocol(pins))
    write(
        EXPERIMENT / "protocol.freeze.json",
        {
            "protocol_sha256": digest(path),
            "stage": "frozen with repository pins before any sampling, retrieval or model loading",
        },
    )
    print(digest(path))


def compress_records(folder: Path) -> None:
    plain = folder / "records.jsonl"
    with (
        plain.open("rb") as source,
        (folder / "records.jsonl.gz").open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target,
    ):
        shutil.copyfileobj(source, target)
    plain.unlink()


DEVIATION = (
    "Direct closing references from commits that fail the message fix heuristic are ignored: "
    "those commits are not in the candidate pool, and prepare() otherwise aborts a repository "
    "(werkzeug tripped it). Decided after the abort, before any retrieval; click and httpx "
    "had no such links, so their output is unchanged."
)


def cmd_prepare(args: argparse.Namespace) -> None:
    import eval.prepare_bugfix_eval as sampling

    original = sampling.direct_references
    sampling.direct_references = lambda message, repo: (
        original(message, repo) if is_fix_message(message) else []
    )
    protocol = read(EXPERIMENT / "protocol.json")
    for repo in protocol["repos"]:
        output = EXPERIMENT / "repos" / stem_of(repo)
        if (output / "freeze.json").exists():
            print(repo, "already frozen")
            continue
        checkout = args.checkouts / stem_of(repo)
        try:
            prepare(
                repo,
                checkout,
                output,
                ref=protocol["pins"][repo],
                protocol_path=EXPERIMENT / "protocol.json",
            )
        except ValueError as error:
            if "No valid issue/fix pairs" not in str(error):
                raise
            print(repo, "yielded no valid questions:", error)
            continue
        shutil.rmtree(output / "cache", ignore_errors=True)
        compress_records(output)
        print(repo, "frozen")


def cmd_freeze(_: argparse.Namespace) -> None:
    protocol = read(EXPERIMENT / "protocol.json")
    if (EXPERIMENT / "freeze.json").exists():
        raise FileExistsError("Experiment already frozen; preserve it")
    reports, freezes, counts, pairs, excluded = {}, {}, {}, {}, {}
    missing = []
    for repo in protocol["repos"]:
        folder = EXPERIMENT / "repos" / stem_of(repo)
        if not (folder / "freeze.json").exists():
            missing.append(repo)
            continue
        report = leakage_report(folder)
        if report["violations"]["email_like_text_in_documents"]:
            raise ValueError(f"{repo}: email-like text in documents {report['examples']}")
        for qid in report["flagged_questions"]:
            excluded[qid] = "leakage check: " + ", ".join(
                k for k, v in report["examples"].items() if any(e.startswith(qid) for e in v)
            )
        reports[repo] = report
        freezes[stem_of(repo)] = digest(folder / "freeze.json")
        pairs[stem_of(repo)] = [
            p for p in read_lines(folder / "pairs.jsonl") if p["id"] not in excluded
        ]
        counts[stem_of(repo)] = len(pairs[stem_of(repo)])
    total = sum(counts.values())
    write(
        EXPERIMENT / "leakage_report.json",
        {
            "repos": reports,
            "missing_repos": missing,
            "questions": total,
            "groups": sum(r["groups"] for r in reports.values()),
            "tentative": total < protocol["confirm"]["rule"]["tentative_below_questions"],
            "excluded": excluded,
        },
    )
    write(
        EXPERIMENT / "freeze.json",
        {
            "stage": "frozen after sampling and leakage checks, before any retrieval or model use",
            "sha256": {
                "protocol.json": digest(EXPERIMENT / "protocol.json"),
                "protocol.freeze.json": digest(EXPERIMENT / "protocol.freeze.json"),
                "leakage_report.json": digest(EXPERIMENT / "leakage_report.json"),
            },
            "dataset_freeze_sha256": freezes,
            "excluded": excluded,
            "deviations": [DEVIATION],
            "counts": counts,
            "timing_ids": timing_ids(protocol, pairs),
        },
    )
    print(json.dumps({"questions": total, "per_repo": counts, "missing": missing}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("protocol")
    p.add_argument("--pin", action="append", required=True, help="owner/name=sha")
    p.set_defaults(func=cmd_protocol)
    p = sub.add_parser("prepare")
    p.add_argument("--checkouts", type=Path, required=True, help="folder of owner__name clones")
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("freeze")
    p.set_defaults(func=cmd_freeze)
    args = parser.parse_args()
    args.func(args)
