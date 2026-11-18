"""Select the 45 fresh httpx facts that become the Phase 7 bake-off v4 test facts.

The 71 facts of v3 (60 + 11 dev extras) have been seen, so v4 test needs facts about other
entities. Expected answers come from `eval/build_facts.py` (raw GitHub nodes and plain git, never
the typed tools). Rule, fixed before any answer model runs on v4:

  1. run build_facts with seed 20261102 over the same snapshot and checkout;
  2. drop every fact whose (type, args) or id is among the 71 existing facts;
  3. drop facts listed in EXCLUDED (the typed-tool audit on the capsule database disagreed with
     the gold answer; each entry says why);
  4. take facts round robin by type (types sorted by name, builder order within a type), one per
     type per round, until FRESH_COUNT facts are chosen.

    uv run python eval/build_v4_fresh_facts.py --checkout data/repos/encode__httpx \\
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
SEED = 20261102
FRESH_COUNT = 45
EXISTING = (
    ROOT / "eval/datasets/httpx_facts.jsonl",
    ROOT / "eval/datasets/httpx_facts_dev_extra.jsonl",
)
OUT = ROOT / "eval/datasets/httpx_facts_v4_fresh.jsonl"
EXCLUDED = {
    "issue_closed_by:3621": (
        "gold lists only PR 3761 (GitHub closedByPullRequestsReferences) while the typed tool, "
        "reading the PR side, returns 3759 and 3761; 3759 was closed without merging and 3761 "
        "is still open, so the question has no single defensible answer"
    ),
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def entity_key(fact: dict) -> tuple[str, str]:
    return fact["type"], json.dumps(fact["args"], sort_keys=True)


def select(generated: list[dict], existing: list[dict]) -> list[dict]:
    """The registered rule: fresh facts only, round robin by type until FRESH_COUNT."""
    taken = {entity_key(f) for f in existing}
    ids = {f["id"] for f in existing}
    fresh = [
        f
        for f in generated
        if entity_key(f) not in taken and f["id"] not in ids and f["id"] not in EXCLUDED
    ]
    by_type: dict[str, list[dict]] = {}
    for fact in fresh:
        by_type.setdefault(fact["type"], []).append(fact)
    chosen: list[dict] = []
    depth = 0
    while len(chosen) < FRESH_COUNT and any(len(v) > depth for v in by_type.values()):
        for kind in sorted(by_type):
            if len(by_type[kind]) > depth and len(chosen) < FRESH_COUNT:
                chosen.append(by_type[kind][depth])
        depth += 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists; changes need a new version, not an overwrite")
    existing = [fact for path in EXISTING for fact in rows(path)]
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
    chosen = select(generated, existing)
    if len(chosen) != FRESH_COUNT:
        raise SystemExit(f"Only {len(chosen)} fresh facts available, need {FRESH_COUNT}")
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for item in chosen:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} fresh facts -> {OUT}")


if __name__ == "__main__":
    main()
