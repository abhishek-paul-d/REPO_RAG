"""Approval/input verification and separate question/label access for the v5 bakeoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from repo_rag.agent.bakeoff import canonical
from repo_rag.agent.bakeoff_tools import tool_schemas
from repo_rag.agent.prompts import ANSWER_BAKEOFF_PROMPT, BAKEOFF_ARMS

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/datasets/phase7_bakeoff_v5"
V4 = ROOT / "eval/datasets/phase7_bakeoff_v4"
V3 = ROOT / "eval/datasets/phase7_bakeoff_v3"
V2 = ROOT / "eval/datasets/phase7_bakeoff_v2"
PROPOSAL = ROOT / "eval/datasets/phase7_bakeoff_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify() -> tuple[dict, list[dict]]:
    """Hash labels without decoding them. No call to rows(labels) in execution code."""
    for folder in (PROPOSAL, V2, V3, V4, DATASET):
        frozen = load(folder / "inputs.freeze.json")
        for name, sha in frozen["files"].items():
            if digest(folder / name) != sha:
                raise ValueError(f"Frozen file changed: {folder.name}/{name}")
        for name, sha in frozen["source_files"].items():
            if digest(ROOT / name) != sha:
                raise ValueError("Original dataset hash changed")
        suffix = {PROPOSAL: "", V2: "_v2", V3: "_v3", V4: "_v4", DATASET: "_v5"}[folder]
        preparer = ROOT / "eval" / f"prepare_answer_bakeoff{suffix}.py"
        if digest(preparer) != frozen["preparer_sha256"]:
            raise ValueError("Frozen preparer changed")
    proposal = load(PROPOSAL / "model_proposal.json")
    pf = load(PROPOSAL / "model_proposal.freeze.json")
    for name, sha in pf["files"].items():
        if digest(PROPOSAL / name) != sha:
            raise ValueError("Frozen proposal changed")
    old, new = load(PROPOSAL / "approval.json"), load(DATASET / "approval.json")
    if digest(PROPOSAL / "approval.json") != new["v1_approval_sha256"]:
        raise ValueError("Carried approval changed")
    if digest(V2 / "approval.json") != new["v2_approval_sha256"]:
        raise ValueError("Carried v2 approval changed")
    if digest(V3 / "approval.json") != new["v3_approval_sha256"]:
        raise ValueError("Carried v3 approval changed")
    if digest(V4 / "approval.json") != new["v4_approval_sha256"]:
        raise ValueError("Carried v4 approval changed")
    if (
        digest(PROPOSAL / "model_proposal.json") != old["model_proposal_sha256"]
        or old["model_proposal_sha256"] != new["model_proposal_sha256"]
    ):
        raise ValueError("Approval does not bind proposal")
    candidates = proposal["candidates"]
    approved = {(r["model_id"], r["revision"]) for r in old["approved_models"]}
    if {(r["model_id"], r["revision"]) for r in candidates} != approved or len(candidates) != 4:
        raise ValueError("Candidates are not the approved four revisions")
    protocol = load(DATASET / "protocol.json")
    if set(tool_schemas()) != set(protocol["tools"]["allowlist"]):
        raise ValueError("Tool schema differs from protocol allowlist")
    if protocol.get("models"):  # v5 runs a registered subset of the four approved models
        candidates = [c for c in candidates if c["model_id"] in protocol["models"]]
        if {c["model_id"] for c in candidates} != set(protocol["models"]):
            raise ValueError("Protocol names a model that is not among the approved four")
    return protocol, candidates


def question_rows(split: str) -> list[dict]:
    if split not in ("dev", "test"):
        raise ValueError("Unknown split")
    return [r for r in rows(DATASET / "questions.jsonl") if r["split"] == split]


def identity() -> dict:
    """Hash effective LF source, as notebook source archives do, before any test run."""
    files = [
        *ROOT.joinpath("src/repo_rag").rglob("*.py"),
        *ROOT.joinpath("eval").glob("*answer_bakeoff*.py"),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "eval/run_facts.py",
    ]
    hashes = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(files)
    }
    return {
        "prompt_sha256": hashlib.sha256(ANSWER_BAKEOFF_PROMPT.encode()).hexdigest(),
        "arms_sha256": hashlib.sha256(
            canonical({str(k): list(v) for k, v in sorted(BAKEOFF_ARMS.items())}).encode()
        ).hexdigest(),
        "parser_sha256": hashes["src/repo_rag/agent/bakeoff.py"],
        "schemas_sha256": hashlib.sha256(canonical(tool_schemas()).encode()).hexdigest(),
        "protocol_sha256": digest(DATASET / "protocol.json"),
        "inputs_freeze_sha256": digest(DATASET / "inputs.freeze.json"),
        "v1_inputs_freeze_sha256": digest(PROPOSAL / "inputs.freeze.json"),
        "approval_sha256": digest(DATASET / "approval.json"),
        "source_hashes": hashes,
    }


def verify_test_freeze(freeze: Path, index_manifest: Path) -> dict:
    frozen = load(freeze)
    if frozen["identity"] != identity() or frozen["index_manifest_sha256"] != digest(
        index_manifest
    ):
        raise ValueError("Prompt/parser/source/index changed since dev freeze")
    if frozen["selection_split"] != "dev" or not 0 <= frozen["selected_revision"] <= 2:
        raise ValueError("Test requires dev-selected common prompt/parser freeze")
    for name, sha in frozen["dev_artifacts"].items():
        if digest(freeze.parent / name) != sha:
            raise ValueError("Dev transcript changed since freeze")
    return frozen
