"""Re-score commit-question runs with a lenient rule next to the strict one.

Strict (the harness): the gold SHA appears in the answer. Lenient: a commit that names the same
issue number as the gold commit, or the reported issue itself, also counts. Several of the 113
questions have two commits that fix the same issue on the same day (a second contributor, a
follow-up); with one gold SHA each, the model that names the other one is marked wrong.

Read-only: it reads result files, the dataset, the pairs and metadata.db, and writes one JSON.
`uv run python eval/commit_lenient_score.py RESULT.json [RESULT.json ...]`
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "eval/datasets"
REPO = "psf/requests"
DEFAULT_DB = ROOT / "data" / "metadata.db"
SHA = re.compile(r"\b[0-9a-f]{7,40}\b")


def question_issues(repo: str = REPO) -> dict[str, int]:
    """commitfull-NNN -> reported issue number (the same pair order the harness uses)."""
    path = DATASETS / "phase5_bugfix_v2/repos" / repo.replace("/", "__") / "pairs.jsonl"
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    tests = sorted((p for p in pairs if p["split"] == "test"), key=lambda p: p["id"])
    return {f"commitfull-{n:03d}": int(p["issue_number"]) for n, p in enumerate(tests, start=1)}


def numbers_in(message: str) -> set[int]:
    return {int(n) for n in re.findall(r"(?:#|issues/|pull/)\s*(\d+)(?!\d)", message)}


def siblings(db: sqlite3.Connection, gold: list[str], issue: int, repo: str = REPO) -> set[str]:
    """12-character SHAs of commits that name the reported issue or any issue the gold names."""
    wanted = {issue}
    for sha in gold:
        row = db.execute(
            "select message from history_commits where repo=? and sha like ?", (repo, sha + "%")
        ).fetchone()
        if row:
            wanted |= numbers_in(row[0])
    found: set[str] = set()
    for sha, message in db.execute(
        "select sha, message from history_commits where repo=?", (repo,)
    ):
        if numbers_in(message) & wanted:
            found.add(sha[:12])
    return found


def cited(answer: str) -> set[str]:
    return {m[:12] for m in SHA.findall(answer or "")}


def score(result_path: Path, db: sqlite3.Connection) -> dict:
    info = json.loads(result_path.read_text(encoding="utf-8"))
    repo = info.get("repo", REPO)  # results written before --repo existed are psf/requests
    gold = {}
    for line in (DATASETS / info["dataset"]).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        gold[d["id"]] = [s[:12] for s in d["expect"]["shas"]]
    issues = question_issues(repo)
    rows = info["rows"]
    out = {"file": result_path.name, "n": len(rows), "strict": 0, "lenient": 0, "flipped": []}
    for r in rows:
        strict = bool(r["passed"])
        out["strict"] += strict
        ok = strict
        if not strict and not r["abstain"]:
            sib = siblings(db, gold[r["id"]], issues[r["id"]], repo)
            hit = cited(r["model_answer"]) & (sib | set(gold[r["id"]]))
            if hit:
                ok = True
                out["flipped"].append({"id": r["id"], "cited": sorted(hit)})
        out["lenient"] += ok
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.db.as_posix()}?mode=ro", uri=True)
    for path in args.results:
        res = score(path, db)
        print(
            f"{res['file']}: strict {res['strict']}/{res['n']}, lenient {res['lenient']}/{res['n']}"
        )
        for f in res["flipped"]:
            print("   counted by the lenient rule:", f["id"], f["cited"])


if __name__ == "__main__":
    main()
