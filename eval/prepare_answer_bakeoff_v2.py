"""Freeze/verify the Phase 7 bake-off v2 amendment (dev-only extra facts). Standard library only.

v1 stays frozen and must keep verifying. v2 changes ONLY the dev split: it adds the facts in
`eval/datasets/httpx_facts_dev_extra.jsonl` (hard tool patterns, entities not in the 60 frozen
facts) so the common prompt can be tuned on them. Test ids, test labels, gates, scoring,
generation, tools, timing and the approved models are byte-for-byte what v1 froze.
Made before any answer model was loaded.

    uv run python eval/prepare_answer_bakeoff_v2.py freeze|verify
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

from eval.build_dev_extra_facts import EXTRA_PER_TYPE, EXTRA_TYPES, SEED, entity_key
from eval.prepare_answer_bakeoff import SOURCES
from eval.prepare_answer_bakeoff import verify as verify_v1

V1 = ROOT / "eval/datasets/phase7_bakeoff_v1"
V2 = ROOT / "eval/datasets/phase7_bakeoff_v2"
EXTRA = ROOT / "eval/datasets/httpx_facts_dev_extra.jsonl"
EXPECTED_COUNTS = {
    "facts/dev": 15 + 11,
    "facts/test": 45,
    "code/dev": 16,
    "code/test": 24,
    "architecture/test": 20,
    "probes/dev": 2,
    "probes/test": 8,
}
# Everything below must be identical in v1 and v2.
UNCHANGED_PROTOCOL_KEYS = (
    "questions",
    "scope",
    "tools",
    "generation",
    "scoring",
    "selection",
    "timing",
)
FILES = ["protocol.json", "probes.jsonl", "labels.jsonl", "questions.jsonl", "splits.json"]
REASON = (
    "v1 put only pr_author, pr_state, issue_author and top_committer in dev, so the common prompt "
    "(at most two revisions) could not be tuned on the tool patterns the 41/45 gate depends on: "
    "the release chain, date-range counts, blame, tag lookup, set answers and rankings."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def entities(fact: dict) -> set[tuple[str, str]]:
    """What a fact is about: a PR/issue number, a file, a tag. Used to disclose overlap."""
    found = set()
    args = fact["args"]
    if args.get("number") is not None:
        found.add(("number", str(args["number"])))
    if args.get("path"):
        found.add(("path", str(args["path"])))
    if args.get("name"):
        found.add(("tag", str(args["name"])))
    return found


def freeze() -> None:
    if (V2 / "inputs.freeze.json").exists():
        raise SystemExit("v2 is frozen: use verify; a further change needs a v3 directory.")
    verify_v1()
    originals = read_rows(ROOT / SOURCES["facts"])
    extras = read_rows(EXTRA)
    if len(extras) != EXPECTED_COUNTS["facts/dev"] - 15:
        raise SystemExit(f"Expected 11 extra facts, found {len(extras)}")
    taken_keys = {entity_key(f) for f in originals}
    taken_ids = {f["id"] for f in originals}
    for item in extras:
        if item["type"] not in EXTRA_TYPES or entity_key(item) in taken_keys:
            raise SystemExit(f"Extra fact is not fresh or not an extra type: {item['id']}")
        if item["id"] in taken_ids:
            raise SystemExit(f"Extra fact id already used: {item['id']}")
    if len({f["id"] for f in extras}) != len(extras):
        raise SystemExit("Duplicate extra ids")
    per_type = Counter(f["type"] for f in extras)
    if any(count > EXTRA_PER_TYPE for count in per_type.values()):
        raise SystemExit("More than the registered number of extras for a type")

    v1_labels = read_rows(V1 / "labels.jsonl")
    new_labels = [
        {
            "suite": "facts",
            "split": "dev",
            "source": "eval/datasets/httpx_facts_dev_extra.jsonl",
            "source_line": line,
            "label": item,
            "id": item["id"],
            "question": item["question"],
            "repo": item["repo"],
        }
        for line, item in enumerate(extras, 1)
    ]
    test_entities = set()
    for row in v1_labels:
        if row["suite"] == "facts" and row["split"] == "test":
            test_entities |= entities(row["label"])
    overlap = sorted(
        f"{item['id']} shares {kind} {value} with a test fact"
        for item in extras
        for kind, value in sorted(entities(item) & test_entities)
    )

    V2.mkdir(parents=True, exist_ok=True)
    labels = v1_labels + new_labels
    ids = [r["id"] for r in labels]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate question IDs")
    counts = Counter(f"{r['suite']}/{r['split']}" for r in labels)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected split counts: {counts}")
    write_rows(V2 / "labels.jsonl", labels)
    write_rows(
        V2 / "questions.jsonl",
        [{k: r[k] for k in ("id", "suite", "split", "repo", "question")} for r in labels],
    )
    (V2 / "probes.jsonl").write_bytes((V1 / "probes.jsonl").read_bytes())

    splits = json.loads((V1 / "splits.json").read_text(encoding="utf-8"))
    splits["counts"] = EXPECTED_COUNTS
    splits["ids"] = {s: [r["id"] for r in labels if r["split"] == s] for s in ("dev", "test")}
    splits["fact_dev_types"] = sorted(
        {r["label"]["type"] for r in labels if r["suite"] == "facts" and r["split"] == "dev"}
    )
    splits["fact_dev_extra_ids"] = [f["id"] for f in extras]
    write_json(V2 / "splits.json", splits)

    protocol = json.loads((V1 / "protocol.json").read_text(encoding="utf-8"))
    v1_protocol_sha = digest(V1 / "protocol.json")
    protocol["version"] = "phase7_bakeoff_v2"
    protocol["frozen_on"] = "2026-11-01"
    protocol["status"] = (
        "pre-model protocol v2 (dev-split amendment); candidate models approved, see approval.json"
    )
    protocol["inputs"]["facts"] = (
        "All 60 original httpx_facts rows plus 11 dev-only extra facts. Whole types pr_author, "
        "pr_state, issue_author, top_committer are dev (15); the other original rows are test (45). "
        "The 11 extras (release_first, commits_in_month, blame_line, tag_commit, issue_closed_by, "
        "hot_file; entities not in the 60; httpx_facts_dev_extra.jsonl) are dev only, so those "
        "types are now in both splits with different entities. Expected answers retain raw "
        "GitHub/git provenance (build_facts.py); do not regenerate labels using tool results."
    )
    protocol["inputs"]["facts_limit"] = (
        "Test keeps held-out fact TYPES no longer: dev now has the hard types too, with different "
        "entities. This is not held-out entity or repository generalization. Entity overlap with "
        "test facts is listed in amendment_v2.entity_overlap_with_test. Reset model state for "
        "every question."
    )
    protocol["inputs"]["total"] = len(labels)
    protocol["inputs"]["dev"] = sum(1 for r in labels if r["split"] == "dev")
    protocol["inputs"]["test"] = sum(1 for r in labels if r["split"] == "test")
    protocol["amendment_v2"] = {
        "reason": REASON,
        "supersedes_protocol_sha256": v1_protocol_sha,
        "supersedes_inputs_freeze_sha256": digest(V1 / "inputs.freeze.json"),
        "made_before_any_answer_model_load": True,
        "selection_rule": (
            f"build_facts.py seed {SEED} on the frozen snapshot and checkout; drop facts whose "
            f"(type, args) or id is among the 60; keep the first {EXTRA_PER_TYPE} per type in "
            f"builder order for {', '.join(EXTRA_TYPES)}. hot_file had one fresh fact."
        ),
        "extra_ids": [f["id"] for f in extras],
        "entity_overlap_with_test": overlap,
        "changed": ["dev split (+11 facts)", "protocol inputs text and totals"],
        "unchanged": [
            "all 97 test questions and labels",
            "gates, scoring, generation, tools, timing",
            "the four approved candidate revisions",
        ],
    }
    write_json(V2 / "protocol.json", protocol)

    write_json(
        V2 / "approval.json",
        {
            "approved_on": "2026-11-01",
            "models": "unchanged from phase7_bakeoff_v1/approval.json (same four revisions)",
            "approved_by": 'the user, in chat: "Approve all four" and, later, "Amend dev split"',
            "v1_approval_sha256": digest(V1 / "approval.json"),
            "model_proposal_sha256": digest(V1 / "model_proposal.json"),
            "scope": (
                "The user asked for this dev-split amendment after approving the models. It "
                "changes no model, revision, gate or test question, so the model approval carries "
                "over to protocol v2. No answer model has been loaded."
            ),
        },
    )
    write_json(
        V2 / "inputs.freeze.json",
        {
            "version": 2,
            "frozen_on": "2026-11-01",
            "answer_model_loads": 0,
            "files": {name: digest(V2 / name) for name in FILES},
            "source_files": {
                **{name: digest(ROOT / name) for name in SOURCES.values()},
                "eval/datasets/httpx_facts_dev_extra.jsonl": digest(EXTRA),
            },
            "preparer_sha256": digest(Path(__file__)),
            "supersedes": {
                "protocol_sha256": v1_protocol_sha,
                "inputs_freeze_sha256": digest(V1 / "inputs.freeze.json"),
            },
            "approval": "models approved (approval.json); no model run, harness or notebook yet",
        },
    )
    verify()


def verify() -> None:
    verify_v1()
    manifest = json.loads((V2 / "inputs.freeze.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(V2 / name) != expected:
            raise SystemExit(f"Frozen v2 input changed: {name}")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise SystemExit(f"Source dataset changed: {name}")
    if digest(Path(__file__)) != manifest["preparer_sha256"]:
        raise SystemExit("v2 preparer changed; make a v3 directory instead")
    if digest(V1 / "protocol.json") != manifest["supersedes"]["protocol_sha256"]:
        raise SystemExit("v1 protocol changed")
    v1_protocol = json.loads((V1 / "protocol.json").read_text(encoding="utf-8"))
    v2_protocol = json.loads((V2 / "protocol.json").read_text(encoding="utf-8"))
    for key in UNCHANGED_PROTOCOL_KEYS:
        if v1_protocol[key] != v2_protocol[key]:
            raise SystemExit(f"v2 changed a section that must match v1: {key}")
    v1_labels = read_rows(V1 / "labels.jsonl")
    v2_labels = read_rows(V2 / "labels.jsonl")
    v1_test = [r for r in v1_labels if r["split"] == "test"]
    if v1_test != [r for r in v2_labels if r["split"] == "test"]:
        raise SystemExit("v2 test labels differ from v1")
    if [r for r in v1_labels if r["split"] == "dev"] != [
        r for r in v2_labels if r["split"] == "dev"
    ][: len([r for r in v1_labels if r["split"] == "dev"])]:
        raise SystemExit("v2 dev labels do not start with the v1 dev labels")
    counts = Counter(f"{r['suite']}/{r['split']}" for r in v2_labels)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected v2 split counts: {counts}")
    for row in read_rows(V2 / "questions.jsonl"):
        if set(row) != {"id", "suite", "split", "repo", "question"}:
            raise SystemExit(f"Question file leaks fields: {sorted(row)}")
    dev_facts = [r["label"] for r in v2_labels if r["suite"] == "facts" and r["split"] == "dev"]
    test_facts = [r["label"] for r in v2_labels if r["suite"] == "facts" and r["split"] == "test"]
    if {entity_key(f) for f in dev_facts} & {entity_key(f) for f in test_facts}:
        raise SystemExit("A dev fact has the same type and arguments as a test fact")
    print(f"PASS v2: {len(v2_labels)} questions; facts 26/45, test unchanged from v1 (97).")
    print(
        "PASS v2: frozen files, source datasets, v1 supersession and unchanged protocol sections."
    )
    print("No answer model loaded; labels remain hidden from model inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    args = parser.parse_args()
    freeze() if args.command == "freeze" else verify()
