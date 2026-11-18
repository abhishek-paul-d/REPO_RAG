"""Freeze/verify Phase 7 inputs using the standard library only; never load a model.

Run from the repository root: uv run python eval/prepare_answer_bakeoff.py freeze|verify
Existing freezes cannot be overwritten. Protocol amendments require a new version directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "eval/datasets/phase7_bakeoff_v1"
DEV_TYPES = {"pr_author", "pr_state", "issue_author", "top_committer"}
SOURCES = {
    "facts": "eval/datasets/httpx_facts.jsonl",
    "code": "eval/datasets/requests_qa.jsonl",
    "architecture_requests": "eval/datasets/architecture_v1/requests.jsonl",
    "architecture_flask": "eval/datasets/architecture_v1/flask.jsonl",
}
CHECKOUTS = {
    "psf/requests": "data/repos/psf__requests",
    "pallets/flask": "data/eval_repos/pallets__flask",
}
PINS = {
    "psf/requests": "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
    "pallets/flask": "d086db856be187255b8ec61ef409357393020f32",
    "encode/httpx": "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
}
# Authored by source inspection before any answer-model load or candidate search.
# Regex absence is a limited lexical audit, not a proof of semantic absence.
PROBES = [
    (
        "requests",
        "dev",
        "Where does this Requests implementation negotiate HTTP/3 over QUIC?",
        "http.?3|quic",
        "src/requests/adapters.py",
        158,
        190,
    ),
    (
        "requests",
        "test",
        "Where does this Requests implementation perform the built-in WebSocket upgrade and manage WebSocket frames?",
        "websocket",
        "src/requests/adapters.py",
        634,
        680,
    ),
    (
        "requests",
        "test",
        "Where does Requests implement its built-in persistent on-disk HTTP response cache and cache eviction?",
        "disk.{0,30}cache|cache.{0,30}evict",
        "src/requests/sessions.py",
        395,
        425,
    ),
    (
        "requests",
        "test",
        "Where does Requests implement built-in OAuth2 refresh-token rotation and token persistence?",
        "oauth2|refresh.token",
        "src/requests/models.py",
        670,
        698,
    ),
    (
        "requests",
        "test",
        "Where does Requests automatically open and commit SQL database transactions around HTTP requests?",
        "sqlalchemy|sqlite|sql.transaction",
        "src/requests/sessions.py",
        752,
        790,
    ),
    (
        "flask",
        "dev",
        "Where does Flask implement its built-in ORM schema migration planner and migration execution engine?",
        "alembic|migration",
        "src/flask/sansio/app.py",
        377,
        386,
    ),
    (
        "flask",
        "test",
        "Where does Flask implement built-in JWT issuance, signing-key rotation and refresh-token storage?",
        "jwt|jwk|refresh.token",
        "src/flask/app.py",
        992,
        1025,
    ),
    (
        "flask",
        "test",
        "Where does Flask implement its built-in Redis replication monitor and Sentinel failover algorithm?",
        r"\bredis\b|\bsentinel\s+failover\b",
        "src/flask/sansio/app.py",
        377,
        386,
    ),
    (
        "flask",
        "test",
        "Where does Flask compile a built-in GraphQL schema and execute GraphQL resolvers?",
        "graphql",
        "src/flask/app.py",
        992,
        1025,
    ),
    (
        "flask",
        "test",
        "Where does Flask implement its own HTTP/3 QUIC transport and packet-loss recovery?",
        "http.?3|quic",
        "src/flask/app.py",
        109,
        135,
    ),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def freeze() -> None:
    if (DEST / "inputs.freeze.json").exists():
        raise SystemExit("Already frozen: use verify; create a new version for amendments.")
    if not (DEST / "protocol.json").exists():
        raise SystemExit("Write the protocol before freezing.")
    labels: list[dict[str, Any]] = []
    for suite, name in SOURCES.items():
        for line, item in enumerate(read_rows(ROOT / name), 1):
            if suite.startswith("architecture") and item["split"] != "test":
                continue
            split = (
                ("dev" if item["type"] in DEV_TYPES else "test")
                if suite == "facts"
                else item["split"]
            )
            labels.append(
                {
                    "suite": "architecture" if suite.startswith("architecture") else suite,
                    "split": split,
                    "source": name,
                    "source_line": line,
                    "label": item,
                    "id": item["id"],
                    "question": item["question"],
                    "repo": item["repo"],
                }
            )
    probes: list[dict[str, Any]] = []
    for i, (short, split, question, pattern, path, start, end) in enumerate(PROBES, 1):
        repo = "psf/requests" if short == "requests" else "pallets/flask"
        checkout = ROOT / CHECKOUTS[repo]
        package = checkout / "src" / ("requests" if short == "requests" else "flask")
        inventory = {
            p.relative_to(checkout).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))
        }
        hits = [
            {"path": name, "line": number}
            for name in inventory
            for number, text in enumerate(
                (checkout / name).read_text(encoding="utf-8").splitlines(), 1
            )
            if re.search(pattern, text, re.IGNORECASE)
        ]
        if hits:
            raise SystemExit(f"Review lexical hits for probe {i}: {hits}")
        probe = {
            "id": f"unanswerable-{i:02d}",
            "repo": repo,
            "commit_sha": PINS[repo],
            "split": split,
            "question": question,
            "expected": "abstain",
            "author": "Codex: source inspection, no retrieval/model generation",
            "label_review": "unreviewed by a person",
            "scope": "Built-in implementation in pinned Python package only; excludes dependencies/extensions/user applications.",
            "source_audit": {"regex": pattern, "hits": hits, "package_sha256": inventory},
            "boundary_anchor": {
                "path": path,
                "start_line": start,
                "end_line": end,
                "sha256": digest(checkout / path),
            },
        }
        probes.append(probe)
        labels.append(
            {
                "suite": "probes",
                "split": split,
                "source": "probes.jsonl",
                "source_line": i,
                "id": probe["id"],
                "question": question,
                "repo": repo,
                "label": probe,
            }
        )
    ids = [r["id"] for r in labels]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate question IDs")
    counts = Counter(f"{r['suite']}/{r['split']}" for r in labels)
    expected = {
        "facts/dev": 15,
        "facts/test": 45,
        "code/dev": 16,
        "code/test": 24,
        "architecture/test": 20,
        "probes/dev": 2,
        "probes/test": 8,
    }
    if dict(counts) != expected:
        raise SystemExit(f"Unexpected split counts: {counts}")
    fact_types = {
        s: {r["label"]["type"] for r in labels if r["suite"] == "facts" and r["split"] == s}
        for s in ("dev", "test")
    }
    if fact_types["dev"] & fact_types["test"]:
        raise SystemExit("Fact type overlap")
    write_rows(DEST / "probes.jsonl", probes)
    write_rows(DEST / "labels.jsonl", labels)
    # Only `question` is model text; the other fields select/reset a scoped environment.
    write_rows(
        DEST / "questions.jsonl",
        [{k: r[k] for k in ("id", "suite", "split", "repo", "question")} for r in labels],
    )
    write_json(
        DEST / "splits.json",
        {
            "counts": expected,
            "fact_dev_types": sorted(DEV_TYPES),
            "fact_test_types": sorted(fact_types["test"]),
            "ids": {s: [r["id"] for r in labels if r["split"] == s] for s in ("dev", "test")},
            "repo_pins": PINS,
        },
    )
    files = ["protocol.json", "probes.jsonl", "labels.jsonl", "questions.jsonl", "splits.json"]
    write_json(
        DEST / "inputs.freeze.json",
        {
            "version": 1,
            "frozen_on": "2026-11-01",
            "answer_model_loads": 0,
            "files": {name: digest(DEST / name) for name in files},
            "source_files": {name: digest(ROOT / name) for name in SOURCES.values()},
            "preparer_sha256": digest(Path(__file__)),
            "approval": "pending; no downloads, model runs, harness or notebook until approval",
        },
    )
    verify()


def verify() -> None:
    manifest = json.loads((DEST / "inputs.freeze.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(DEST / name) != expected:
            raise SystemExit(f"Frozen input changed: {name}")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise SystemExit(f"Original dataset changed: {name}")
    if digest(Path(__file__)) != manifest["preparer_sha256"]:
        raise SystemExit("Preparer changed; preserve v1 and explicitly version amendments")
    for probe in read_rows(DEST / "probes.jsonl"):
        checkout = ROOT / CHECKOUTS[probe["repo"]]
        if checkout.exists():
            for name, expected in probe["source_audit"]["package_sha256"].items():
                if digest(checkout / name) != expected:
                    raise SystemExit(f"Probe source changed: {probe['repo']} {name}")
    print("PASS: 130 questions; facts 15/45, code 16/24, architecture 0/20, probes 2/8 dev/test.")
    print("PASS: 5 frozen files, 4 original datasets, preparer and available probe source hashes.")
    print("No answer model loaded; labels remain hidden from model inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    args = parser.parse_args()
    freeze() if args.command == "freeze" else verify()
