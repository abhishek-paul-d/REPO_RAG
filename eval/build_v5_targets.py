"""Seeded target sampler for the Phase 7 bake-off v5 questions (docs/PHASE7_BAKEOFF_V5_DESIGN.md).

The v4 failures have been read, so question targets are not chosen by hand. This script draws
them from the capsule's indexed code with a fixed seed, excluding every gold file/symbol and
architecture file pair that v3/v4 already used. Question text and labels are then written from
source inspection without running retrieval or a model (`eval/build_v5_questions.py`). A target
that cannot be turned into a defensible question is dropped there and logged, never replaced by
hand, so the reserve below is consumed in order.

    uv run python eval/build_v5_targets.py [--show]
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAPSULE = ROOT / "notebooks/answer_bakeoff_inputs.zip"
V4_LABELS = ROOT / "eval/datasets/phase7_bakeoff_v4/labels.jsonl"
OUT = ROOT / "eval/datasets/phase7_bakeoff_v5_targets.json"
SEED = 20261103
REPOS = {"psf/requests": "requests", "pallets/flask": "flask"}
# Per repo and kind: (test + dev) wanted, and the number actually drawn (reserve for drops).
WANTED = {"code": 16, "architecture": 16, "probes": 11}
DRAWN = {"code": 24, "architecture": 40, "probes": 16}
MIN_LINES, MAX_LINES = 8, 90


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_chunks(archive: zipfile.ZipFile, short: str) -> list[dict]:
    data = json.loads(archive.read(f"bakeoff/code/{short}.json"))
    return [{**m, "text": t} for m, t in zip(data["metas"], data["texts"], strict=True)]


def used_before() -> dict:
    """What v3/v4 already asked about: code gold, architecture gold files, probe anchors."""
    code: set[tuple[str, str]] = set()
    files: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    anchors: set[str] = set()
    for row in read_rows(V4_LABELS):
        label = row["label"]
        if row["suite"] == "code":
            code.update((g["path"], g["symbol"]) for g in label.get("gold", []))
        elif row["suite"] == "architecture":
            gold = sorted(label["gold_files"])
            files.update(gold)
            pairs.update((a, b) for a in gold for b in gold if a < b)
        elif row["suite"] == "probes":
            anchors.add(label["boundary_anchor"]["path"])
    return {"code": code, "files": files, "pairs": pairs, "anchors": anchors}


def code_candidates(chunks: list[dict], used: dict) -> list[dict]:
    result = []
    for c in chunks:
        lines = c["end_line"] - c["start_line"] + 1
        name = c["symbol"].split(".")[-1]
        if (
            c["path"].startswith("src/")
            and c["symbol_kind"] in ("function", "method")
            and MIN_LINES <= lines <= MAX_LINES
            and not name.startswith("__")
            and (c["path"], c["symbol"]) not in used["code"]
        ):
            result.append(c)
    return sorted(result, key=lambda c: c["chunk_id"])


def architecture_candidates(repo: str, archive: zipfile.ZipFile, used: dict) -> list[dict]:
    """Cross-file call edges inside src/ whose callee name is defined in exactly one other file."""
    with tempfile.TemporaryDirectory() as tmp:
        database = Path(tmp) / "m.db"
        database.write_bytes(archive.read("bakeoff/metadata.db"))
        connection = sqlite3.connect(database)
        definitions: dict[str, set[str]] = {}
        for path, name in connection.execute(
            "select path, name from code_symbols where repo=? and path like 'src/%' "
            "and kind in ('function','method','class')",
            (repo,),
        ):
            definitions.setdefault(name, set()).add(path)
        edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for path, caller, callee in connection.execute(
            "select path, caller, callee from code_calls where repo=? and path like 'src/%'",
            (repo,),
        ):
            targets = definitions.get(callee.split(".")[-1], set())
            if len(targets) == 1 and (target := next(iter(targets))) != path:
                edges.setdefault((path, target), set()).add((caller, callee))
        connection.close()
    result = []
    for (source, target), calls in sorted(edges.items()):
        pair = tuple(sorted((source, target)))
        if pair in used["pairs"]:
            continue
        result.append({"caller_path": source, "callee_path": target, "calls": sorted(calls)[:6]})
    return result


def probe_candidates(chunks: list[dict], used: dict) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for c in sorted(chunks, key=lambda c: c["chunk_id"]):
        if c["path"].startswith("src/") and c["path"] not in used["anchors"] | seen:
            seen.add(c["path"])
            result.append(c)
    return result


def draw(items: list, count: int, tag: str) -> list:
    rng = random.Random(f"{SEED}:{tag}")
    pool = list(items)
    rng.shuffle(pool)
    return pool[:count]


def build() -> dict:
    used = used_before()
    targets: dict = {"seed": SEED, "wanted": WANTED, "drawn": DRAWN, "repos": {}}
    with zipfile.ZipFile(CAPSULE) as archive:
        for repo, short in REPOS.items():
            chunks = load_chunks(archive, short)
            pool = {
                "code": code_candidates(chunks, used),
                "architecture": architecture_candidates(repo, archive, used),
                "probes": probe_candidates(chunks, used),
            }
            targets["repos"][repo] = {
                "pool_sizes": {k: len(v) for k, v in pool.items()},
                "code": [
                    {k: c[k] for k in ("chunk_id", "path", "symbol", "start_line", "end_line")}
                    for c in draw(pool["code"], DRAWN["code"], f"{repo}:code")
                ],
                "architecture": draw(pool["architecture"], DRAWN["architecture"], f"{repo}:arch"),
                "probes": [
                    {k: c[k] for k in ("chunk_id", "path", "symbol", "start_line", "end_line")}
                    for c in draw(pool["probes"], DRAWN["probes"], f"{repo}:probe")
                ],
            }
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="print pool sizes and the draw")
    args = parser.parse_args()
    targets = build()
    OUT.write_text(json.dumps(targets, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for repo, body in targets["repos"].items():
        print(repo, body["pool_sizes"])
    if args.show:
        print(json.dumps(targets, indent=1)[:4000])
    print(f"written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
