"""Select the dev-only extra httpx facts for the Phase 7 bake-off v2 amendment.

The 60 frozen facts put only the easy fact types in dev. This adds a few dev-only facts of the
hard tool patterns (release chain, date-range count, blame, tag lookup, set answers, ranking)
about entities that are NOT in the 60, so the common prompt can be tuned on them without
touching test. Expected answers come from `eval/build_facts.py` (raw GitHub nodes and plain
git, never the typed tools). Rule, fixed before any answer model runs:

  1. run build_facts with seed 20261101 over the same snapshot and checkout;
  2. drop every fact whose (type, args) or id is in `eval/datasets/httpx_facts.jsonl`;
  3. for each type in EXTRA_TYPES keep the first EXTRA_PER_TYPE remaining facts in builder order.

    uv run python eval/build_dev_extra_facts.py --checkout data/repos/encode__httpx \\
        --data-dir <dir holding github_cache/sync/encode__httpx.*.json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261101
EXTRA_TYPES = (
    "release_first",
    "commits_in_month",
    "blame_line",
    "tag_commit",
    "issue_closed_by",
    "hot_file",
)
EXTRA_PER_TYPE = 2
ORIGINAL = ROOT / "eval/datasets/httpx_facts.jsonl"
OUT = ROOT / "eval/datasets/httpx_facts_dev_extra.jsonl"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def entity_key(fact: dict) -> tuple[str, str]:
    return fact["type"], json.dumps(fact["args"], sort_keys=True)


def select(generated: list[dict], original: list[dict]) -> list[dict]:
    """The registered rule: fresh facts only, first EXTRA_PER_TYPE of each type in builder order."""
    taken = {entity_key(f) for f in original}
    ids = {f["id"] for f in original}
    fresh = [f for f in generated if entity_key(f) not in taken and f["id"] not in ids]
    chosen: list[dict] = []
    for kind in EXTRA_TYPES:
        chosen += [f for f in fresh if f["type"] == kind][:EXTRA_PER_TYPE]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists; amendments need a new version, not an overwrite")
    original = rows(ORIGINAL)
    with tempfile.TemporaryDirectory() as tmp:
        generated_path = Path(tmp) / "generated.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "eval/build_facts.py"),
                "encode/httpx",
                "--checkout",
                str(args.checkout),
                "--data-dir",
                str(args.data_dir),
                "--seed",
                str(SEED),
                "--out",
                str(generated_path),
            ],
            check=True,
        )
        generated = rows(generated_path)
    chosen = select(generated, original)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for item in chosen:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} dev-only extra facts -> {OUT}")


if __name__ == "__main__":
    main()
