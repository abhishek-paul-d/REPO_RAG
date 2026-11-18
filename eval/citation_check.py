"""Offline citation check on saved agent answers (no model): does each cited range show what the
sentence names, and what would `snap_citations` change?

    uv run python eval/citation_check.py [--answers FILE] [--checkout data/repos/psf__requests]

Reads the cited lines with `git show <sha>:<path>` at the pinned commit, so it needs no index of the
answers. Per answer it reports citation support (named identifiers missing from the cited lines)
and the snaps; with `--review CSV` it prints how the flags line up with a reviewer's faithfulness.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import judge_answers as ja  # noqa: E402

from repo_rag.agent.citation_snap import citation_support, snap_citations  # noqa: E402
from repo_rag.agent.citations import Chunk  # noqa: E402
from repo_rag.store.symbol_table import SymbolTable  # noqa: E402

SHA = "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--answers",
        default=str(ROOT / "eval/agent_eval_results/agent_eval_production_20261109T013636.json"),
    )
    parser.add_argument("--checkout", default=str(ROOT / "data/repos/psf__requests"))
    parser.add_argument("--db", default=str(ROOT / "data/metadata.db"))
    parser.add_argument("--repo", default="psf/requests")
    parser.add_argument("--review", help="review_sheet.csv with faithful verdicts")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--write-snapped", help="write the answers file with snapped citations (for the judge)"
    )
    args = parser.parse_args()

    checkout = Path(args.checkout)
    cache: dict[str, list[str] | None] = {}

    def read(path: str) -> list[str] | None:
        if path not in cache:
            cache[path] = ja.git_lines(checkout, SHA, path)
        return cache[path]

    verdict: dict[str, str] = {}
    if args.review:
        with open(args.review, encoding="utf-8", newline="") as fh:
            verdict = {r["id"]: r["faithful"] for r in csv.DictReader(fh)}
    rows = json.loads(Path(args.answers).read_text(encoding="utf-8"))["rows"]
    out = []
    snapped_answers: dict[str, str] = {}
    with SymbolTable(Path(args.db)) as table:

        def lookup(name: str, path: str) -> list[tuple[str, int, int]]:
            return [
                (r.qualname, r.start_line, r.end_line)
                for r in table.get_symbol(args.repo, name, path=path)
            ]

        for row in rows:
            if row["type"] not in ("code", "docs") or row.get("abstain"):
                continue
            answer = row["answer"]
            support = citation_support(answer, read, lookup)
            checked = [s for s in support if s.checked]
            # Retrieved chunks are not saved with the answers: the cited files stand in, so a snap
            # only needs the new range to sit in a file the answer already cites.
            chunks = [Chunk(s.ref.rpartition(":")[0], 1, 10**6) for s in support]
            snapped = snap_citations(answer, chunks, lookup)
            snapped_answers[row["id"]] = snapped.answer
            out.append(
                {
                    "id": row["id"],
                    "citations": len(support),
                    "checked": len(checked),
                    "unsupported": [
                        s.ref + " missing " + ",".join(s.missing) for s in checked if not s.ok
                    ],
                    "snaps": [(x.old, x.new, x.why) for x in snapped.snaps],
                    "review_faithful": verdict.get(row["id"], ""),
                }
            )
    if args.write_snapped:
        data = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        for row in data["rows"]:
            row["answer"] = snapped_answers.get(row["id"], row["answer"])
        data["citation_snap"] = (
            "citation_snap.snap_citations applied offline to code and docs answers"
        )
        Path(args.write_snapped).write_text(json.dumps(data, indent=1), encoding="utf-8")
    if args.json:
        print(json.dumps(out, indent=1))
        return
    flagged = [o for o in out if o["unsupported"]]
    print(
        f"{len(out)} answers, {sum(o['citations'] for o in out)} citations, "
        f"{len(flagged)} answers with a named identifier missing from its cited lines, "
        f"{sum(bool(o['snaps']) for o in out)} with a snap"
    )
    for o in out:
        if o["unsupported"] or o["snaps"]:
            print(f"{o['id']:9} review={o['review_faithful']:11} {o['unsupported']} {o['snaps']}")
    if verdict:
        for label, group in (
            ("flagged", flagged),
            ("not flagged", [o for o in out if not o["unsupported"]]),
        ):
            counts: dict[str, int] = {}
            for o in group:
                counts[o["review_faithful"]] = counts.get(o["review_faithful"], 0) + 1
            print(label, len(group), counts)


if __name__ == "__main__":
    main()
