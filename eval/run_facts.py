"""Answer the auto-generated fact set through the typed tools alone (no LLM) and score it.

    uv run python eval/run_facts.py eval/datasets/httpx_facts.jsonl \\
        --checkout data/repos/encode__httpx --data-dir <dir with metadata.db> [--out DIR]

Each fact names the tool and arguments an agent should call; this runner makes exactly that
call and compares the value with the expected one. Phase 7 reruns the same questions through
the agent, where the model has to choose the call itself.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from repo_rag.config import Settings
from repo_rag.history.git_tools import blame, releases_containing
from repo_rag.history.metadata_db import MetadataDB

TARGET = 0.95


def answer(fact: dict, db: MetadataDB, root: Path, settings: Settings) -> Any:
    repo, tool, args = fact["repo"], fact["tool"], fact["args"]
    if tool == "get_pr":
        pr = db.get_pr(repo, args["number"])
        return None if pr is None else str(getattr(pr, args["field"]))
    if tool == "get_issue_meta":
        issue = db.get_issue_meta(repo, args["number"])
        return None if issue is None else str(getattr(issue, args["field"]))
    if tool == "prs_closing_issue":
        return [str(pr.number) for pr in db.prs_closing_issue(repo, args["number"])]
    if tool == "release_first":
        pr = db.get_pr(repo, args["number"])
        if pr is None or not pr.merge_commit_sha:
            return None
        found = releases_containing(root, pr.merge_commit_sha, settings=settings)
        return found[0].tag if found else "none"
    if tool == "blame":
        groups = blame(root, args["path"], args["line"], args["line"])
        return groups[0].sha[:12] if groups else None
    if tool == "count_commits":
        return str(db.count_commits(repo, **args).count)
    if tool == "top_contributors":
        rows = db.top_contributors(repo, **args, limit=3).rows
        return rows[0].name if rows else None
    if tool == "change_frequency":
        file_rows = db.change_frequency(repo, **args, limit=3).rows
        return file_rows[0].path if file_rows else None
    if tool == "get_tag":
        tag = db.get_tag(repo, args["name"])
        return None if tag is None else tag.sha
    raise ValueError(f"unknown tool {tool!r}")


def matches(fact: dict, got: Any) -> bool:
    if fact["compare"] == "set":
        return isinstance(got, list) and sorted(got) == sorted(fact["expected"])
    return got == fact["expected"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("facts", type=Path)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    facts = [
        json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line
    ]
    settings = Settings(data_dir=args.data_dir)
    results: list[dict] = []
    with MetadataDB(settings.metadata_db_path) as db:
        for item in facts:
            try:
                got: Any = answer(item, db, args.checkout, settings)
                error = ""
            except Exception as exc:  # a tool error is a failed fact, not a crashed run
                got, error = None, f"{type(exc).__name__}: {exc}"
            results.append({**item, "got": got, "ok": matches(item, got), "error": error})
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["ok"])
    passed = sum(r["ok"] for r in results)
    rate = passed / len(results)
    lines = [f"Facts: {passed}/{len(results)} passed ({rate:.1%}); target {TARGET:.0%}.", ""]
    lines += [f"- {kind}: {sum(v)}/{len(v)}" for kind, v in sorted(by_type.items())]
    failures = [r for r in results if not r["ok"]]
    if failures:
        lines += ["", "Failures:"]
        for r in failures:
            lines.append(
                f"- {r['id']} — {r['question']} expected {r['expected']!r}, got {r['got']!r}"
                + (f" ({r['error']})" if r["error"] else "")
            )
    print("\n".join(lines))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
