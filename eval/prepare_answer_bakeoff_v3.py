"""Freeze/verify the Phase 7 bake-off v3 amendment (bounded PR/issue `body` field). Stdlib only.

v1 and v2 stay frozen and must keep verifying. v3 changes ONE thing: the typed PR/issue tools
return the record with its long free-text `body` cut to `tools.record_projection.max_chars`
characters, because whole records above the 12,000-character result budget were being dropped
entirely and their structured fields (author, merged_by, merge_commit_sha, state) became
unavailable to every model. Questions, labels, splits, gates, scoring, generation limits, timing
and the four approved models are byte-for-byte what v2 froze. Made before any answer model loaded.

    uv run python eval/prepare_answer_bakeoff_v3.py freeze|verify
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.prepare_answer_bakeoff_v2 import V1, V2
from eval.prepare_answer_bakeoff_v2 import verify as verify_v2

V3 = ROOT / "eval/datasets/phase7_bakeoff_v3"
COPIED = ["probes.jsonl", "labels.jsonl", "questions.jsonl", "splits.json"]
FILES = ["protocol.json", *COPIED]
UNCHANGED_PROTOCOL_KEYS = ("questions", "scope", "generation", "scoring", "selection", "timing")
PROJECTION = {
    "tools": ["get_pr", "get_issue_meta", "find_prs", "prs_closing_issue"],
    "field": "body",
    "max_chars": 2000,
    "marker": "...[body truncated: N more characters]",
    "rule": (
        "A `body` string longer than max_chars is replaced by its first max_chars characters "
        "followed by the marker with N set to the number of characters removed. Applied before "
        "redaction and before the result budget. Every other field, and the 12,000-character "
        "cumulative budget, is unchanged. No fact in the registered sets asks for a PR or issue "
        "body."
    ),
}
REASON = (
    "In the capsule database 35 of 500 httpx PR records and 10 of 500 issue records exceed the "
    "12,000-character per-question tool-result budget when returned whole, almost entirely "
    "because of `body`. The v2 context policy then omits the whole record, so author, "
    "merged_by, merge_commit_sha and state vanish for every model alike. The bound-tool audit "
    "on v2 dev passed 25/26 (issue_author:1428 omitted); the 41/45 test gate allows four "
    "misses. Prompt wording cannot fix this and the budget cannot be raised without hitting "
    "the 8,192-token input cap."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def freeze() -> None:
    if (V3 / "inputs.freeze.json").exists():
        raise SystemExit("v3 is frozen: use verify; a further change needs a v4 directory.")
    verify_v2()
    V3.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        (V3 / name).write_bytes((V2 / name).read_bytes())
    protocol = json.loads((V2 / "protocol.json").read_text(encoding="utf-8"))
    v2_protocol_sha = digest(V2 / "protocol.json")
    protocol["version"] = "phase7_bakeoff_v3"
    protocol["frozen_on"] = "2026-11-01"
    protocol["status"] = (
        "pre-model protocol v3 (bounded PR/issue body); candidate models approved, see approval.json"
    )
    protocol["tools"]["record_projection"] = PROJECTION
    protocol["amendment_v3"] = {
        "reason": REASON,
        "supersedes_protocol_sha256": v2_protocol_sha,
        "supersedes_inputs_freeze_sha256": digest(V2 / "inputs.freeze.json"),
        "made_before_any_answer_model_load": True,
        "changed": ["tools.record_projection (PR/issue body bounded)"],
        "unchanged": [
            "all 141 questions, labels and splits (byte-identical copies of v2)",
            "gates, scoring, generation limits and result budget, timing",
            "the four approved candidate revisions",
        ],
    }
    write_json(V3 / "protocol.json", protocol)
    write_json(
        V3 / "approval.json",
        {
            "approved_on": "2026-11-01",
            "models": "unchanged from phase7_bakeoff_v1/approval.json (same four revisions)",
            "approved_by": 'the user, in chat: "yea u do it" (build the v3 record-size fix)',
            "v1_approval_sha256": digest(V1 / "approval.json"),
            "v2_approval_sha256": digest(V2 / "approval.json"),
            "model_proposal_sha256": digest(V1 / "model_proposal.json"),
            "scope": (
                "The user asked for this amendment after the review found the budget problem. "
                "It changes no model, revision, gate, question or label, so the model approval "
                "carries over to protocol v3. No answer model has been loaded."
            ),
        },
    )
    v2_manifest = json.loads((V2 / "inputs.freeze.json").read_text(encoding="utf-8"))
    write_json(
        V3 / "inputs.freeze.json",
        {
            "version": 3,
            "frozen_on": "2026-11-01",
            "answer_model_loads": 0,
            "files": {name: digest(V3 / name) for name in FILES},
            "source_files": v2_manifest["source_files"],
            "preparer_sha256": digest(Path(__file__)),
            "supersedes": {
                "protocol_sha256": v2_protocol_sha,
                "inputs_freeze_sha256": digest(V2 / "inputs.freeze.json"),
            },
        },
    )
    verify()


def verify() -> None:
    verify_v2()
    manifest = json.loads((V3 / "inputs.freeze.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(V3 / name) != expected:
            raise SystemExit(f"Frozen v3 input changed: {name}")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise SystemExit(f"Source dataset changed: {name}")
    if digest(Path(__file__)) != manifest["preparer_sha256"]:
        raise SystemExit("v3 preparer changed; make a v4 directory instead")
    if digest(V2 / "protocol.json") != manifest["supersedes"]["protocol_sha256"]:
        raise SystemExit("v2 protocol changed")
    for name in COPIED:
        if (V3 / name).read_bytes() != (V2 / name).read_bytes():
            raise SystemExit(f"v3 {name} differs from v2")
    old = json.loads((V2 / "protocol.json").read_text(encoding="utf-8"))
    new = json.loads((V3 / "protocol.json").read_text(encoding="utf-8"))
    for key in UNCHANGED_PROTOCOL_KEYS:
        if old[key] != new[key]:
            raise SystemExit(f"v3 changed a section that must match v2: {key}")
    if {k: v for k, v in new["tools"].items() if k != "record_projection"} != old["tools"]:
        raise SystemExit("v3 changed the tool list")
    if new["tools"]["record_projection"] != PROJECTION:
        raise SystemExit("v3 record projection differs from the registered one")
    print("PASS v3: 141 questions, labels and splits identical to v2; gates and scoring unchanged.")
    print("PASS v3: only tools.record_projection (PR/issue body bounded) differs from v2.")
    print("No answer model loaded; labels remain hidden from model inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    args = parser.parse_args()
    freeze() if args.command == "freeze" else verify()
