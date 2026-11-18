"""Freeze 15 cue-free, source-grounded stress questions before their new GPU comparison."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_followup_datasets import REPOS, gold_range  # noqa: E402
from eval.retrieval.ambiguous import AmbiguousCase, validate_cases  # noqa: E402
from eval.retrieval.dataset import Question, write_json  # noqa: E402

# Each seed: question, implementation proof, demonstration proof, related inspected v2 IDs.
# Explicit line anchors are reviewed against the pinned source, not retrieval output.
SEEDS = {
    "requests": [
        (
            "Why does putting a line break inside an HTTP header value cause a failure?",
            [
                ("src/requests/utils.py", 1093, 1095),
                ("src/requests/utils.py", 1103, 1119),
                ("src/requests/_internal_utils.py", 15, 19),
            ],
            [("tests/test_requests.py", 1817, 1840)],
            ["requests-036"],
        ),
        (
            "How can an already partially consumed upload be read again from its saved starting position?",
            [("src/requests/utils.py", 1139, 1155)],
            [("tests/test_requests.py", 2025, 2037)],
            ["requests-028"],
        ),
        (
            "Why can removing an HTTP header use different capitalization from when it was stored?",
            [("src/requests/structures.py", 59, 68)],
            [("tests/test_structures.py", 6, 10), ("tests/test_structures.py", 15, 27)],
            ["requests-020", "requests-fresh-004"],
        ),
        (
            "What happens when several response callbacks run in sequence and one returns no replacement value?",
            [("src/requests/hooks.py", 39, 48)],
            [("tests/test_hooks.py", 6, 18)],
            ["requests-fresh-002"],
        ),
        (
            "Can a copied HTTP header mapping contain the same values while being a separate object?",
            [("src/requests/structures.py", 49, 57), ("src/requests/structures.py", 89, 90)],
            [("tests/test_structures.py", 6, 10), ("tests/test_structures.py", 37, 40)],
            ["requests-fresh-005"],
        ),
    ],
    "express": [
        (
            "Why do spaces in a redirect destination become percent escapes?",
            [("lib/response.js", 797, 799)],
            [("test/res.location.js", 35, 47)],
            ["express-028"],
        ),
        (
            "Where can I follow how the client address is chosen behind a trusted proxy?",
            [("lib/request.js", 340, 343)],
            [("test/req.ip.js", 40, 53)],
            ["express-036"],
        ),
        (
            "Why does clearing a cookie expire it immediately even when a positive lifetime was supplied?",
            [("lib/response.js", 715, 722)],
            [("test/res.clearCookie.js", 36, 47)],
            ["express-fresh-004"],
        ),
        (
            "Why is a fractional HTTP response status rejected?",
            [("lib/response.js", 65, 69)],
            [("test/res.status.js", 144, 154)],
            ["express-fresh-006"],
        ),
        (
            "What happens if I try to read a request header without supplying its name?",
            [("lib/request.js", 63, 67)],
            [("test/req.get.js", 36, 46)],
            ["express-fresh-007"],
        ),
    ],
    "flask": [
        (
            "How can a prefix be removed from configuration keys while their letter case is normalized?",
            [("src/flask/config.py", 353, 364)],
            [("tests/test_config.py", 217, 226)],
            ["flask-fresh-007"],
        ),
        (
            "Why does supplying a list of HTTP methods to a method-specific route decorator fail?",
            [
                ("src/flask/sansio/scaffold.py", 284, 293),
                ("src/flask/sansio/scaffold.py", 295, 301),
            ],
            [("tests/test_basic.py", 70, 72)],
            ["flask-fresh-013"],
        ),
        (
            "Why is a dot forbidden in a blueprint name?",
            [("src/flask/sansio/blueprints.py", 198, 199)],
            [("tests/test_blueprints.py", 242, 244)],
            ["flask-fresh-016"],
        ),
        (
            "Why does a precise decimal number appear as a quoted string in JSON?",
            [("src/flask/json/provider.py", 112, 113)],
            [("tests/test_json.py", 191, 193)],
            ["flask-fresh-017"],
        ),
        (
            "How can an environment variable populate a deeply nested configuration dictionary?",
            [("src/flask/config.py", 152, 183)],
            [("tests/test_config.py", 79, 107)],
            ["flask-fresh-015"],
        ),
    ],
}


def build() -> None:
    output = ROOT / "eval/datasets/phase4_ambiguous"
    if (output / "manifest.json").exists():
        raise FileExistsError("stress labels are frozen; author a new version instead")
    output.mkdir(exist_ok=True)
    manifest: dict = {
        "status": "source-authored, same-assistant review; not independently reviewed",
        "selection": "post-hoc stress questions on previously inspected behaviours, not independent fresh evidence",
        "datasets": [],
    }
    for name, seeds in SEEDS.items():
        repo, sha, folder = REPOS[name]
        checkout = ROOT / folder
        cases = []
        for i, (text, source, demonstration, related) in enumerate(seeds, 1):
            alternatives = [
                [gold_range(checkout, path, f"lines:{a}:{b}") for path, a, b in ranges]
                for ranges in (source, demonstration)
            ]
            q = Question(
                id=f"{name}-ambiguous-{i:03}",
                repo=repo,
                commit_sha=sha,
                question=text,
                category="ambiguous_intent",
                split="test",
                group=f"{name}-ambiguous-{i:03}",
                provenance="New cue-free wording for inspected behaviours; stress set, no tuning",
                gold=alternatives[0],
                schema_version=2,
                cohort="legacy_regression",
            )
            if name == "requests" and i in (3, 5):
                q.group = "requests-header-mapping"  # shared fixture evidence
            cases.append(
                AmbiguousCase(
                    question=q,
                    accepted_evidence=alternatives,
                    rationale="Intent does not prescribe a file type. Either the behaviour's implementation or a complete demonstration is sufficient; ranges within one proof are jointly required.",
                    related_v2_ids=related,
                )
            )
        validate_cases(checkout, cases)
        path = output / f"{name}_qa.jsonl"
        path.write_text(
            "".join(json.dumps(c.model_dump(), ensure_ascii=False) + "\n" for c in cases),
            encoding="utf-8",
            newline="\n",
        )
        manifest["datasets"].append(
            {
                "repo": repo,
                "commit_sha": sha,
                "file": path.name,
                "questions": len(cases),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    build()
