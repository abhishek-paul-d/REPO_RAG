"""Hand-check `find_callers` against `git grep`: every textual call of a name must be found.

    uv run python eval/check_callers.py psf/requests data/repos/psf__requests \\
        merge_setting super_len HTTPAdapter.build_response Session.prepare_request

`git grep` lists lines containing `name(` that are not definitions; our rows are the call sites in
the symbol table (run `repo-rag index-symbols` first, same data dir). Lines that grep finds but
the table doesn't, and the reverse, are printed. A grep line in a comment or string, or a call
of a different function with the same name, shows up as a difference to review by hand.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from repo_rag.config import get_settings
from repo_rag.store.symbol_table import SymbolTable


def grep_calls(checkout: Path, name: str) -> set[tuple[str, int]]:
    out = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            rf"(^|[^A-Za-z0-9_]){re.escape(name)}\(",
            "--",
            "*.py",
            "*.js",
            "*.ts",
            "*.go",
            "*.java",
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    found = set()
    for line in out.splitlines():
        path, number, text = line.split(":", 2)
        if re.search(rf"\b(def|func|function)\s+(\([^)]*\)\s*)?{re.escape(name)}\b", text):
            continue
        found.add((path, int(number)))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("slug")
    parser.add_argument("checkout", type=Path)
    parser.add_argument("symbols", nargs="+")
    args = parser.parse_args()
    with SymbolTable(get_settings().metadata_db_path) as table:
        for symbol in args.symbols:
            name = symbol.rpartition(".")[2]
            ours_rows = table.find_callers(args.slug, symbol, limit=100_000)
            ours = {(r.path, r.line) for r in ours_rows}
            expected = grep_calls(args.checkout, name)
            strong = sum(r.how != "name only" for r in ours_rows)
            print(
                f"{symbol}: grep {len(expected)}, table {len(ours)} ({strong} better than name "
                f"only), grep-only {sorted(expected - ours)}, table-only {sorted(ours - expected)}"
            )


if __name__ == "__main__":
    main()
