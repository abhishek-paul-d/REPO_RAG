"""Freeze/verify the Phase 7 bake-off v4 protocol (fresh test facts, scoring corrections). Stdlib.

v1, v2 and v3 stay frozen and must keep verifying. v4 exists because the v3 run (A100, 2026-11-01)
showed that the v3 test facts, and part of the failure modes, have been seen. Changes:

  * dev = every v3 dev question + the 45 v3 test facts (all seen);
  * test facts = 45 fresh httpx facts (`eval/build_v4_fresh_facts.py`, seed 20261102);
  * test code, architecture and probe questions are the v3 ones, flagged as seen;
  * `scoring.v4_corrections`: three equivalence rules, reported next to the strict v3 score;
  * the common prompt gets worked examples (it lives in `agent/prompts.py`, hashed by the freeze
    of the run, not by this file).

Gates, tools, generation limits, timing, the four approved models and the question texts of reused
questions are unchanged. Made before any answer model ran on v4.

    uv run python eval/prepare_answer_bakeoff_v4.py freeze|verify
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_v4_fresh_facts import FRESH_COUNT, SEED, entity_key
from eval.prepare_answer_bakeoff_v2 import V1, V2
from eval.prepare_answer_bakeoff_v3 import V3
from eval.prepare_answer_bakeoff_v3 import verify as verify_v3

V4 = ROOT / "eval/datasets/phase7_bakeoff_v4"
FRESH = ROOT / "eval/datasets/httpx_facts_v4_fresh.jsonl"
EXPECTED_COUNTS = {
    "facts/dev": 26 + 45,
    "facts/test": FRESH_COUNT,
    "code/dev": 16,
    "code/test": 24,
    "architecture/test": 20,
    "probes/dev": 2,
    "probes/test": 8,
}
FILES = ["protocol.json", "probes.jsonl", "labels.jsonl", "questions.jsonl", "splits.json"]
UNCHANGED_PROTOCOL_KEYS = ("questions", "scope", "tools", "generation", "selection", "timing")
CORRECTIONS = {
    "path_prefix": (
        "Path-like tool arguments (`path`, `path_prefix`) compare equal after removing one "
        "trailing slash: `httpx` and `httpx/` are the same prefix."
    ),
    "blame_sha": (
        "A blame answer is correct if it equals the 12-character gold or is a full 40-hex SHA "
        "that starts with it. No shorter prefix is accepted."
    ),
    "single_member_set": (
        "For compare=set facts whose gold has exactly one member, a bare string equal to that "
        "member is accepted. Multi-member sets keep exact sorted-list equality."
    ),
    "reporting": (
        "Every report prints the strict v3 score (`fact_ok_strict`) and the corrected score "
        "(`fact_ok`) per model. Eligibility and dev selection use the corrected score. No case "
        "folding, whitespace normalisation or fuzzy matching is added."
    ),
}
REASON = (
    "The v3 run found no qualifying model. Format-driven failures dominated (missing `abstain`, "
    "tool name in `type`, abstaining without searching, no citations), two hot_file gold "
    "arguments disagree with the question wording, and the run skipped the registered dev prompt "
    "revisions. The v3 test failures were seen, so v3 test facts cannot be a clean test."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_labels() -> list[dict[str, Any]]:
    """v3 rows with the 45 v3 test facts moved to dev, then the fresh facts as test."""
    labels = []
    for row in read_rows(V3 / "labels.jsonl"):
        if row["suite"] == "facts" and row["split"] == "test":
            row = {**row, "split": "dev", "seen_in": "v3 test (failures seen)"}
        labels.append(row)
    for line, item in enumerate(read_rows(FRESH), 1):
        labels.append(
            {
                "suite": "facts",
                "split": "test",
                "source": "eval/datasets/httpx_facts_v4_fresh.jsonl",
                "source_line": line,
                "label": item,
                "id": item["id"],
                "question": item["question"],
                "repo": item["repo"],
            }
        )
    return labels


def freeze() -> None:
    if (V4 / "inputs.freeze.json").exists():
        raise SystemExit("v4 is frozen: use verify; a further change needs a v5 directory.")
    verify_v3()
    old_facts = [r["label"] for r in read_rows(V3 / "labels.jsonl") if r["suite"] == "facts"]
    fresh = read_rows(FRESH)
    taken = {entity_key(f) for f in old_facts}
    if len(fresh) != FRESH_COUNT or any(entity_key(f) in taken for f in fresh):
        raise SystemExit("Fresh facts are missing or reuse a v3 entity")
    if {f["id"] for f in fresh} & {f["id"] for f in old_facts}:
        raise SystemExit("A fresh fact reuses a v3 id")
    labels = build_labels()
    ids = [r["id"] for r in labels]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate question IDs")
    counts = Counter(f"{r['suite']}/{r['split']}" for r in labels)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected split counts: {counts}")

    V4.mkdir(parents=True, exist_ok=True)
    write_rows(V4 / "labels.jsonl", labels)
    write_rows(
        V4 / "questions.jsonl",
        [{k: r[k] for k in ("id", "suite", "split", "repo", "question")} for r in labels],
    )
    (V4 / "probes.jsonl").write_bytes((V3 / "probes.jsonl").read_bytes())
    splits = json.loads((V3 / "splits.json").read_text(encoding="utf-8"))
    splits["counts"] = EXPECTED_COUNTS
    splits["ids"] = {s: [r["id"] for r in labels if r["split"] == s] for s in ("dev", "test")}
    splits["fact_dev_types"] = sorted(
        {r["label"]["type"] for r in labels if r["suite"] == "facts" and r["split"] == "dev"}
    )
    splits["fact_test_types"] = sorted(
        {r["label"]["type"] for r in labels if r["suite"] == "facts" and r["split"] == "test"}
    )
    splits["fact_fresh_ids"] = [f["id"] for f in fresh]
    write_json(V4 / "splits.json", splits)

    protocol = json.loads((V3 / "protocol.json").read_text(encoding="utf-8"))
    v3_protocol_sha = digest(V3 / "protocol.json")
    protocol["version"] = "phase7_bakeoff_v4"
    protocol["frozen_on"] = "2026-11-01"
    protocol["status"] = (
        "pre-model protocol v4 (fresh test facts, scoring corrections); models approved"
    )
    protocol["inputs"]["facts"] = (
        "Dev: the 26 v3 dev facts plus the 45 v3 test facts (all seen, 71). Test: 45 fresh httpx "
        "facts about entities not among those 71 (eval/build_v4_fresh_facts.py, seed "
        f"{SEED}, round robin by type). Expected answers keep raw GitHub/git provenance."
    )
    protocol["inputs"]["facts_limit"] = (
        "Fresh facts are new entities on the same repository and the same fact types, not a new "
        "repository. Code, architecture and probe test questions are the v3 ones and were seen "
        "(aggregates and sampled transcripts); the report flags them. Architecture and probe "
        "labels remain unreviewed by a person."
    )
    protocol["inputs"]["total"] = len(labels)
    protocol["inputs"]["dev"] = sum(1 for r in labels if r["split"] == "dev")
    protocol["inputs"]["test"] = sum(1 for r in labels if r["split"] == "test")
    protocol["scoring"]["v4_corrections"] = CORRECTIONS
    protocol["amendment_v4"] = {
        "reason": REASON,
        "supersedes_protocol_sha256": v3_protocol_sha,
        "supersedes_inputs_freeze_sha256": digest(V3 / "inputs.freeze.json"),
        "made_before_any_answer_model_ran_on_v4": True,
        "made_after_seeing_v3_results": True,
        "v3_results": "eval/facts_results/answer_bakeoff_v3/",
        "changed": [
            "dev = v3 dev + v3 test facts",
            "test facts = 45 fresh httpx facts",
            "scoring.v4_corrections (strict and corrected reported together)",
            "common prompt gains worked examples; dev revisions are used, test is gated by "
            "CONFIRM_FREEZE in the notebook",
        ],
        "unchanged": [
            "gates, selection rule, tools and result bounds, generation limits, timing",
            "the four approved candidate revisions",
            "texts and labels of the reused code, architecture and probe questions",
        ],
    }
    write_json(V4 / "protocol.json", protocol)
    write_json(
        V4 / "approval.json",
        {
            "approved_on": "2026-11-01",
            "models": "unchanged from phase7_bakeoff_v1/approval.json (same four revisions)",
            "approved_by": 'the user, in chat: "Yes, design v4" then "Approve, build it"',
            "v1_approval_sha256": digest(V1 / "approval.json"),
            "v2_approval_sha256": digest(V2 / "approval.json"),
            "v3_approval_sha256": digest(V3 / "approval.json"),
            "model_proposal_sha256": digest(V1 / "model_proposal.json"),
            "design": "docs/PHASE7_BAKEOFF_V4_DESIGN.md",
            "scope": ("Changes no model, revision, gate or tool. No answer model has run on v4."),
        },
    )
    v3_manifest = json.loads((V3 / "inputs.freeze.json").read_text(encoding="utf-8"))
    write_json(
        V4 / "inputs.freeze.json",
        {
            "version": 4,
            "frozen_on": "2026-11-01",
            "answer_model_loads_on_v4": 0,
            "files": {name: digest(V4 / name) for name in FILES},
            "source_files": {
                **v3_manifest["source_files"],
                "eval/datasets/httpx_facts_v4_fresh.jsonl": digest(FRESH),
            },
            "preparer_sha256": digest(Path(__file__)),
            "supersedes": {
                "protocol_sha256": v3_protocol_sha,
                "inputs_freeze_sha256": digest(V3 / "inputs.freeze.json"),
            },
        },
    )
    verify()


def verify() -> None:
    verify_v3()
    manifest = json.loads((V4 / "inputs.freeze.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(V4 / name) != expected:
            raise SystemExit(f"Frozen v4 input changed: {name}")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise SystemExit(f"Source dataset changed: {name}")
    if digest(Path(__file__)) != manifest["preparer_sha256"]:
        raise SystemExit("v4 preparer changed; make a v5 directory instead")
    if digest(V3 / "protocol.json") != manifest["supersedes"]["protocol_sha256"]:
        raise SystemExit("v3 protocol changed")
    old = json.loads((V3 / "protocol.json").read_text(encoding="utf-8"))
    new = json.loads((V4 / "protocol.json").read_text(encoding="utf-8"))
    for key in UNCHANGED_PROTOCOL_KEYS:
        if old[key] != new[key]:
            raise SystemExit(f"v4 changed a section that must match v3: {key}")
    if {k: v for k, v in new["scoring"].items() if k != "v4_corrections"} != old["scoring"]:
        raise SystemExit("v4 changed strict scoring; it may only add v4_corrections")
    if new["scoring"]["v4_corrections"] != CORRECTIONS:
        raise SystemExit("v4 corrections differ from the registered ones")
    labels = read_rows(V4 / "labels.jsonl")
    if build_labels() != labels:
        raise SystemExit("v4 labels differ from the registered construction")
    counts = Counter(f"{r['suite']}/{r['split']}" for r in labels)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected v4 split counts: {counts}")
    for row in read_rows(V4 / "questions.jsonl"):
        if set(row) != {"id", "suite", "split", "repo", "question"}:
            raise SystemExit(f"Question file leaks fields: {sorted(row)}")
    dev = {entity_key(r["label"]) for r in labels if r["suite"] == "facts" and r["split"] == "dev"}
    test = {
        entity_key(r["label"]) for r in labels if r["suite"] == "facts" and r["split"] == "test"
    }
    if dev & test:
        raise SystemExit("A dev fact has the same type and arguments as a test fact")
    print(f"PASS v4: {len(labels)} questions; facts 71/45 (test facts fresh), 97 test questions.")
    print("PASS v4: frozen files, fresh-fact rule, v3 supersession, strict scoring unchanged.")
    print("No answer model ran on v4; labels remain hidden from model inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    args = parser.parse_args()
    freeze() if args.command == "freeze" else verify()
