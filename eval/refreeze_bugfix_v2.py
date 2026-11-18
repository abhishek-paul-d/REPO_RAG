"""Rebuild reviewed v2 from hash-verified v1 source data, without GitHub or retrieval.

V1 remains immutable. Queries, gold, groups and split bytes are reused exactly; this
corrects the formatter and protocol rather than creating an independent evaluation set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.prepare_bugfix_eval import (  # noqa: E402
    PROTOCOL,
    digest,
    read_lines,
    verify_corpus,
    verify_frozen,
    write,
    write_lines,
)
from repo_rag.config import Settings  # noqa: E402
from repo_rag.history.commit_indexer import commit_document, is_fix_message  # noqa: E402
from repo_rag.history.metadata_db import record_from_dict  # noqa: E402


def refreeze(source: Path, target: Path, protocol_path: Path = PROTOCOL) -> dict:
    old = verify_frozen(source)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(protocol_path.with_name("protocol.freeze.json").read_text(encoding="utf-8"))
    if protocol["version"] != 2 or lock["protocol_sha256"] != digest(protocol_path):
        raise ValueError("v2 protocol must be frozen first")
    if old["sha256"]["protocol.json"] != protocol["parent_protocol_sha256"]:
        raise ValueError("Wrong parent dataset protocol")
    if old["pin"] != protocol["pins"][old["repo"]]:
        raise ValueError("Parent pin mismatch")
    if target.exists():
        raise FileExistsError("Preserve existing datasets; use a new output")
    records = [record_from_dict(r) for r in read_lines(source / "records.jsonl")]
    if any(r.is_fix != is_fix_message(r.message) for r in records):
        raise ValueError("Stored candidate flags differ from message heuristic")
    eligible = {r.sha for r in records if r.is_fix}
    if any(not set(p["gold"]) <= eligible for p in read_lines(source / "pairs.jsonl")):
        raise ValueError("Gold candidate membership depends on labels")
    target.mkdir(parents=True)
    shutil.copy2(protocol_path, target / "protocol.json")
    for name in ("records.jsonl", "pairs.jsonl", "links.json", "issues.json", "selection.json"):
        path = source / name
        if not path.exists():
            path = Path(str(path) + ".gz")
        shutil.copy2(path, target / path.name)
    write(
        target / "preparation.json",
        {
            "repo": old["repo"],
            "pin": old["pin"],
            "parent_freeze_sha256": digest(source / "freeze.json"),
            "protocol_sha256": digest(protocol_path),
            "reuse": "v1 source records, queries, labels, groups and splits; no new API calls",
            "candidate_pool": "message heuristic only; every gold independently eligible",
        },
    )
    settings = Settings(commit_diff_chars=protocol["documents"]["commit_diff_chars"])
    docs = [commit_document(r, settings) for r in records if r.is_fix]
    write_lines(
        target / "corpus.jsonl", [{"text": d.page_content, "meta": d.metadata} for d in docs]
    )
    names = [
        "protocol.json",
        "records.jsonl",
        "pairs.jsonl",
        "links.json",
        "issues.json",
        "selection.json",
        "preparation.json",
        "corpus.jsonl",
    ]
    freeze = {
        "repo": old["repo"],
        "pin": old["pin"],
        "sha256": {name: digest(target / name) for name in names},
        "stage": "v2 frozen before retrieval; corrected after v1 review",
    }
    write(target / "freeze.json", freeze)
    return verify_corpus(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "eval/datasets/phase5_bugfix_v1/repos"
    )
    parser.add_argument("--output", type=Path, default=PROTOCOL.parent / "repos")
    args = parser.parse_args()
    for folder in sorted(args.source.iterdir()):
        print(refreeze(folder, args.output / folder.name), flush=True)
