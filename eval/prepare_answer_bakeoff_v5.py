"""Freeze/verify the Phase 7 bake-off v5 protocol (fresh code/architecture/probe questions). Stdlib.

v1-v4 stay frozen and must keep verifying. v5 exists because v4 chose no model: Qwen2.5-Coder-14B
failed one gate (answerable abstentions, 6 vs at most 4) and often omitted `path:start-end`
citations, and the v4 test questions were read to diagnose that. v5 asks whether a tuned prompt
(arms R2 and R3, `agent/prompts.py`) fixes it, on questions nobody has seen. Changes:

  * dev = 15 fresh answerable + 6 fresh probes; test = 45 fresh answerable + 16 fresh probes,
    all drawn with a fixed seed (`eval/build_v5_targets.py`) and written by
    `eval/build_v5_questions.py` from source inspection, no retrieval or model;
  * facts guard = the 45 v4 test facts as test facts (seen; regression check only), no dev facts;
  * two models (Coder subject, Qwen3-14B control), prompt arms by revision (0 = unchanged v4
    prompt, 1 = R2, 2 = R3), gates scaled to rates, one added gate (citation presence).

Tools, result bounds, generation limits, timing and the approved model revisions are unchanged.
Made before any answer model ran on v5.

    uv run python eval/prepare_answer_bakeoff_v5.py freeze|verify
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

from eval.prepare_answer_bakeoff_v2 import V1, V2
from eval.prepare_answer_bakeoff_v3 import V3
from eval.prepare_answer_bakeoff_v4 import V4
from eval.prepare_answer_bakeoff_v4 import verify as verify_v4

V5 = ROOT / "eval/datasets/phase7_bakeoff_v5"
AUTHORED = ROOT / "eval/datasets/phase7_bakeoff_v5_authored.jsonl"
TARGETS = ROOT / "eval/datasets/phase7_bakeoff_v5_targets.json"
SUBJECT = "Qwen/Qwen2.5-Coder-14B-Instruct"
CONTROL = "Qwen/Qwen3-14B"
EXPECTED_COUNTS = {
    "facts/test": 45,
    "code/dev": 8,
    "code/test": 24,
    "architecture/dev": 7,
    "architecture/test": 21,
    "probes/dev": 6,
    "probes/test": 16,
}
FILES = ["protocol.json", "labels.jsonl", "questions.jsonl", "splits.json"]
UNCHANGED_PROTOCOL_KEYS = ("questions", "scope", "tools", "generation", "timing")
GATES = {
    "grounded_fact_exactness_min": 0.95,
    "invalid_citation_rate_max": 0.05,
    "test_probe_abstention_min": 0.875,
    "answerable_Q2_abstention_max": 0.09,
    "citation_complete_min": 0.75,
}
DISCRETE = (
    "Subject model on the selected arm, test split: at least 43/45 facts (guard), at most 5% "
    "invalid citation references, at least 14/16 probes, at most 4/45 answerable abstentions, "
    "and at least 34/45 answerable questions with a valid citation in the raw output (the "
    "answer check is reported separately and is not a gate)."
)
ANSWER_CHECK = (
    "Reported beside the raw score, never instead of it: for a non-abstaining answer with no "
    "citation, `path:start-end` is attached for each retrieved search_code chunk whose file "
    "path occurs in the answer or whose symbol's last component (4+ characters) occurs as a "
    "whole word. The attached references are validated like model-written ones. The check cannot "
    "rescue an abstention, a failure, a probe, or a malformed citation."
)
REASON = (
    "v4 chose no model. On its 44 answerable test questions Qwen2.5-Coder-14B cited 14, named a "
    "file without a line range on 13, named no file on 11 and abstained on 6 (limit 4); "
    "Qwen3-14B cited 38 but answered 3 of 8 probes it should have refused. 21 of Coder's 24 "
    "answered-but-uncited answers name a retrieved symbol or file, and 5 of its 6 abstentions "
    "come after a single search on architecture questions. The v4 test questions were read to "
    "diagnose this, so claims about a tuned prompt need questions nobody has seen."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_labels() -> list[dict[str, Any]]:
    guard = []
    for row in read_rows(V4 / "labels.jsonl"):
        if row["suite"] == "facts" and row["split"] == "test":
            guard.append({**row, "seen_in": "v4 test (regression guard only)"})
    return [*read_rows(AUTHORED), *guard]


def prior_questions() -> dict[str, set]:
    """Everything v3/v4 asked: question strings, code gold, architecture file pairs, anchors."""
    strings: set[str] = set()
    code: set[tuple[str, str]] = set()
    pairs: set[tuple[str, str]] = set()
    anchors: set[str] = set()
    for row in read_rows(V4 / "labels.jsonl"):
        strings.add(row["question"])
        label = row["label"]
        if row["suite"] == "code":
            code.update((g["path"], g["symbol"]) for g in label["gold"])
        elif row["suite"] == "architecture":
            files = sorted(label["gold_files"])
            pairs.update((a, b) for a in files for b in files if a < b)
        elif row["suite"] == "probes":
            anchors.add(label["boundary_anchor"]["path"])
    return {"strings": strings, "code": code, "pairs": pairs, "anchors": anchors}


def check_disjoint(labels: list[dict[str, Any]]) -> None:
    prior = prior_questions()
    seen_ids, seen_text = set(), set()
    for row in labels:
        if row["id"] in seen_ids:
            raise SystemExit(f"Duplicate question id {row['id']}")
        seen_ids.add(row["id"])
        if row["suite"] == "facts":
            continue
        label = row["label"]
        if row["question"] in prior["strings"] or row["question"] in seen_text:
            raise SystemExit(f"Question repeats an earlier one: {row['question']}")
        seen_text.add(row["question"])
        if row["suite"] == "code" and any(
            (g["path"], g["symbol"]) in prior["code"] for g in label["gold"]
        ):
            raise SystemExit(f"Code gold reuses a v3/v4 target: {row['id']}")
        if row["suite"] == "architecture":
            files = sorted(label["gold_files"])
            if any((a, b) in prior["pairs"] for a in files for b in files if a < b):
                raise SystemExit(f"Architecture file pair reuses a v3/v4 pair: {row['id']}")
        if row["suite"] == "probes":
            if label["boundary_anchor"]["path"] in prior["anchors"]:
                raise SystemExit(f"Probe anchor reuses a v3/v4 anchor file: {row['id']}")
            if label["source_audit"]["hits"]:
                raise SystemExit(f"Probe term present in source: {row['id']}")


def freeze() -> None:
    if (V5 / "inputs.freeze.json").exists():
        raise SystemExit("v5 is frozen: use verify; a further change needs a v6 directory.")
    verify_v4()
    labels = build_labels()
    check_disjoint(labels)
    counts = Counter(f"{r['suite']}/{r['split']}" for r in labels)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected split counts: {counts}")

    V5.mkdir(parents=True, exist_ok=True)
    write_rows(V5 / "labels.jsonl", labels)
    write_rows(
        V5 / "questions.jsonl",
        [{k: r[k] for k in ("id", "suite", "split", "repo", "question")} for r in labels],
    )
    v4_splits = json.loads((V4 / "splits.json").read_text(encoding="utf-8"))
    write_json(
        V5 / "splits.json",
        {
            "counts": EXPECTED_COUNTS,
            "ids": {s: [r["id"] for r in labels if r["split"] == s] for s in ("dev", "test")},
            "repo_pins": v4_splits["repo_pins"],
            "seed": json.loads(TARGETS.read_text(encoding="utf-8"))["seed"],
        },
    )

    protocol = json.loads((V4 / "protocol.json").read_text(encoding="utf-8"))
    v4_protocol_sha = digest(V4 / "protocol.json")
    protocol["version"] = "phase7_bakeoff_v5"
    protocol["frozen_on"] = "2026-11-03"
    protocol["status"] = (
        "pre-model protocol v5 (fresh questions, tuned prompt arms); models approved"
    )
    protocol["subject"] = SUBJECT
    protocol["models"] = [SUBJECT, CONTROL]
    protocol["arms"] = {
        "0": "Unchanged v4 prompt, no reminder (paired baseline).",
        "1": "R2: rewritten code section (re-search before abstaining, citation per sentence) "
        "plus a fixed reminder user message after each search_code result.",
        "2": "R3: R2 plus a stricter citation sentence and a shorter reminder.",
        "where": "src/repo_rag/agent/prompts.py BAKEOFF_ARMS; revision number selects the arm.",
    }
    protocol["inputs"]["facts"] = (
        "Guard only: the 45 v4 test facts (seen). No dev facts; the facts rate is not part of the "
        "dev choice."
    )
    protocol["inputs"]["facts_limit"] = "Seen facts cannot support a claim; they detect regression."
    protocol["inputs"]["code"] = (
        "Fresh: per repo 16 methods/functions from src/ drawn by a fixed seed from the capsule "
        "chunks, excluding every v3/v4 gold file+symbol. Dev 4 + test 12 per repo."
    )
    protocol["inputs"]["architecture"] = (
        "Fresh: cross-file call pairs from the capsule call table, excluding every v3/v4 gold file "
        "pair. Requests 3 dev + 8 test, Flask 4 dev + 13 test (the unused pool is smaller than the "
        "design's 32; the split is skewed to Flask and the report says so)."
    )
    protocol["inputs"]["probes"] = (
        "Fresh: boundary-anchored unanswerable questions, 3 dev + 8 test per repo; every probe "
        "term is checked absent from the indexed src/ text."
    )
    protocol["inputs"]["total"] = len(labels)
    protocol["inputs"]["dev"] = sum(1 for r in labels if r["split"] == "dev")
    protocol["inputs"]["test"] = sum(1 for r in labels if r["split"] == "test")
    protocol["scoring"]["v5_answer_check"] = ANSWER_CHECK
    protocol["selection"]["gates"] = GATES
    protocol["selection"]["discrete_thresholds"] = DISCRETE
    protocol["selection"]["dev_rank"] = "v5"
    protocol["selection"]["ranking"] = (
        "Dev only, subject model: probe abstention rate, then citation-complete rate, then fewer "
        "answerable abstentions, then lower invalid citation rate, then the lower revision. Test "
        "runs the selected arm and the unchanged baseline (revision 0) once. The control model is "
        "reported, not gated. If the subject fails a gate nothing is adopted or relaxed."
    )
    protocol["amendment_v5"] = {
        "reason": REASON,
        "supersedes_protocol_sha256": v4_protocol_sha,
        "supersedes_inputs_freeze_sha256": digest(V4 / "inputs.freeze.json"),
        "made_before_any_answer_model_ran_on_v5": True,
        "made_after_seeing_v4_results": True,
        "v4_results": "eval/facts_results/answer_bakeoff_v4_test/",
        "design": "docs/PHASE7_BAKEOFF_V5_DESIGN.md",
        "changed": [
            "questions: fresh seeded code/architecture/probe sets; facts are a guard only",
            "models: Coder (subject) and Qwen3-14B (control)",
            "arms: prompt revisions 0/1/2 in agent/prompts.py; reminder after search_code",
            "gates scaled to rates for 45 answerable / 16 probes, plus citation presence",
            "answer check reported beside the raw score",
        ],
        "unchanged": [
            "tools and result bounds, generation limits, timing, the approved revisions",
            "strict single-JSON envelope; no repair and no retry",
            "mechanical scoring only",
        ],
    }
    write_json(V5 / "protocol.json", protocol)
    write_json(
        V5 / "approval.json",
        {
            "approved_on": "2026-11-03",
            "models": "two of the four revisions approved in phase7_bakeoff_v1/approval.json",
            "approved_by": 'the user, in chat: "Yes, design v5" then "Approve, build it"',
            "v1_approval_sha256": digest(V1 / "approval.json"),
            "v2_approval_sha256": digest(V2 / "approval.json"),
            "v3_approval_sha256": digest(V3 / "approval.json"),
            "v4_approval_sha256": digest(V4 / "approval.json"),
            "model_proposal_sha256": digest(V1 / "model_proposal.json"),
            "design": "docs/PHASE7_BAKEOFF_V5_DESIGN.md",
            "scope": "Changes no model, revision or tool. No answer model has run on v5.",
        },
    )
    v4_manifest = json.loads((V4 / "inputs.freeze.json").read_text(encoding="utf-8"))
    write_json(
        V5 / "inputs.freeze.json",
        {
            "version": 5,
            "frozen_on": "2026-11-03",
            "answer_model_loads_on_v5": 0,
            "files": {name: digest(V5 / name) for name in FILES},
            "source_files": {
                **v4_manifest["source_files"],
                "eval/datasets/phase7_bakeoff_v5_authored.jsonl": digest(AUTHORED),
                "eval/datasets/phase7_bakeoff_v5_targets.json": digest(TARGETS),
            },
            "preparer_sha256": digest(Path(__file__)),
            "supersedes": {
                "protocol_sha256": v4_protocol_sha,
                "inputs_freeze_sha256": digest(V4 / "inputs.freeze.json"),
            },
        },
    )
    verify()


def verify() -> None:
    verify_v4()
    manifest = json.loads((V5 / "inputs.freeze.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(V5 / name) != expected:
            raise SystemExit(f"Frozen v5 input changed: {name}")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise SystemExit(f"Source dataset changed: {name}")
    if digest(Path(__file__)) != manifest["preparer_sha256"]:
        raise SystemExit("v5 preparer changed; make a v6 directory instead")
    if digest(V4 / "protocol.json") != manifest["supersedes"]["protocol_sha256"]:
        raise SystemExit("v4 protocol changed")
    old = json.loads((V4 / "protocol.json").read_text(encoding="utf-8"))
    new = json.loads((V5 / "protocol.json").read_text(encoding="utf-8"))
    for key in UNCHANGED_PROTOCOL_KEYS:
        if old[key] != new[key]:
            raise SystemExit(f"v5 changed a section that must match v4: {key}")
    if {k: v for k, v in new["scoring"].items() if k != "v5_answer_check"} != old["scoring"]:
        raise SystemExit("v5 changed v4 scoring; it may only add v5_answer_check")
    if new["selection"]["gates"] != GATES or new["models"] != [SUBJECT, CONTROL]:
        raise SystemExit("v5 gates or models differ from the registered ones")
    labels = read_rows(V5 / "labels.jsonl")
    if build_labels() != labels:
        raise SystemExit("v5 labels differ from the registered construction")
    check_disjoint(labels)
    counts = Counter(f"{r['suite']}/{r['split']}" for r in labels)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected v5 split counts: {counts}")
    for row in read_rows(V5 / "questions.jsonl"):
        if set(row) != {"id", "suite", "split", "repo", "question"}:
            raise SystemExit(f"Question file leaks fields: {sorted(row)}")
    print(f"PASS v5: {len(labels)} questions; fresh answerable/probe sets disjoint from v3/v4.")
    print("PASS v5: frozen files, gates, models, unchanged v4 sections, strict scoring unchanged.")
    print("No answer model ran on v5; labels remain hidden from model inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    args = parser.parse_args()
    freeze() if args.command == "freeze" else verify()
