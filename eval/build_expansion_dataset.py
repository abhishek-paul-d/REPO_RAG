"""Author the expansion questions from pinned source, then freeze before any retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from git import Repo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_comparison_dataset import assign_groups  # noqa: E402
from eval.retrieval.ambiguous import AmbiguousCase, read_cases, validate_cases  # noqa: E402
from eval.retrieval.dataset import GoldRange, Question, evidence_hash, overlaps, write_json  # noqa: E402

REPOS = {
    "requests": (
        "psf/requests",
        "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
        "data/repos/psf__requests",
    ),
    "express": (
        "expressjs/express",
        "0dee2c52c7c873498f8ed7d8ee4ddd0a04d3ada8",
        "data/eval_repos/expressjs__express",
    ),
    "flask": (
        "pallets/flask",
        "d086db856be187255b8ec61ef409357393020f32",
        "data/eval_repos/pallets__flask",
    ),
    "click": (
        "pallets/click",
        "2247b35ea1c47c727d7a06e51fa280e12a863ff6",
        "data/eval_repos/pallets__click",
    ),
}

# Each tuple is source-authored question + jointly required implementation ranges.
# No retrieval/expansion outputs are consulted here. Small explicit behavioural scopes avoid
# requiring whole classes or dependency internals as gold. All questions have one AND proof.
SEEDS: dict[str, list[tuple[str, list[tuple[str, int, int]]]]] = {
    "requests": [
        (
            "Trace how the top-level GET helper delegates a request and ensures its temporary session's adapters are closed afterwards.",
            [
                ("api.py", 74, 87),
                ("api.py", 67, 71),
                ("sessions.py", 505, 509),
                ("sessions.py", 883, 886),
            ],
        ),
        (
            "Trace how copying a prepared request copies its cookies, including the fallback for a generic cookie jar without its own copy method.",
            [("models.py", 456, 465), ("cookies.py", 479, 491)],
        ),
        (
            "Trace how environment proxy values fill missing request proxy entries and how the resulting map is combined with session settings and null entries removed.",
            [("sessions.py", 845, 851), ("sessions.py", 862, 868), ("sessions.py", 84, 105)],
        ),
        (
            "Trace how a digest-authentication challenge is saved and retried, and how the nonce counter becomes the eight-digit field in the new Authorization header.",
            [
                ("auth.py", 291, 312),
                ("auth.py", 162, 166),
                ("auth.py", 227, 237),
                ("auth.py", 250, 266),
            ],
        ),
        (
            "Trace how line-by-line response iteration obtains byte chunks and preserves a final unfinished line when a line crosses a chunk boundary; leave Unicode decoding disabled.",
            [("models.py", 1010, 1032), ("models.py", 935, 956), ("models.py", 967, 977)],
        ),
        (
            "Trace how an HTTP error status produces an exception that retains both the response and, when not supplied separately, the response's originating request.",
            [("models.py", 1160, 1171), ("exceptions.py", 21, 35), ("exceptions.py", 66, 67)],
        ),
    ],
    "express": [
        (
            "Trace how assigning the application's ETag setting selects and stores the weak or strong generator, up to its delegation to the external hashing library.",
            [
                ("application.js", 359, 366),
                ("utils.js", 130, 152),
                ("utils.js", 40, 40),
                ("utils.js", 51, 51),
                ("utils.js", 249, 257),
            ],
        ),
        (
            "Trace how setting a numeric proxy trust count stores a compiled trust function and how that function decides whether a hop is trusted.",
            [("application.js", 359, 360), ("application.js", 370, 379), ("utils.js", 194, 205)],
        ),
        (
            "Trace the extended-query-parser setting from application configuration through selection of its parser to the options passed to the external query-string parser.",
            [("application.js", 359, 369), ("utils.js", 162, 184), ("utils.js", 267, 271)],
        ),
        (
            "Trace how setting an attachment filename derives its content type and writes the attachment disposition, including how the derived type is assigned to the response.",
            [("response.js", 609, 618), ("response.js", 507, 514), ("response.js", 670, 692)],
        ),
        (
            "Trace how a redirect destination becomes the encoded Location header, is reused in the negotiated body, and yields no body for a HEAD request.",
            [("response.js", 797, 799), ("response.js", 838, 866)],
        ),
        (
            "Trace how application initialization installs a lazy router getter and how registering a route accesses that router rather than building a new one for each registration.",
            [("application.js", 59, 82), ("application.js", 256, 258)],
        ),
    ],
    "flask": [
        (
            "Trace a JSON response request with several positional values through argument normalization and the default provider's debug-mode formatting and response construction.",
            [
                ("json/__init__.py", 170, 170),
                ("json/provider.py", 75, 87),
                ("json/provider.py", 205, 215),
            ],
        ),
        (
            "Trace how loading configuration from a file hands the parsed mapping to the configuration object and accepts only uppercase keys.",
            [("config.py", 290, 302), ("config.py", 314, 321)],
        ),
        (
            "Trace how a categorized flash message is written to the session and later retrieved only once into request-local storage, filtered, and returned with or without categories.",
            [("helpers.py", 348, 350), ("helpers.py", 391, 399)],
        ),
        (
            "Trace how the file-response wrapper combines caller options with application and request defaults before handing them to Werkzeug; stop at that delegation boundary.",
            [("helpers.py", 528, 540), ("helpers.py", 402, 414)],
        ),
        (
            "Trace how a blueprint URL rule is deferred, invoked during registration, and given its URL prefix and namespaced endpoint before being added to the application.",
            [
                ("sansio/blueprints.py", 433, 441),
                ("sansio/blueprints.py", 334, 335),
                ("sansio/blueprints.py", 98, 116),
            ],
        ),
        (
            "Trace how a newly created Jinja environment obtains its default autoescape callback and how that callback handles an unnamed template or an HTML filename regardless of case.",
            [("app.py", 479, 494), ("sansio/app.py", 546, 548)],
        ),
    ],
    "click": [
        (
            "Trace how option decorators attach parameters to a callback and how command construction consumes that stored list in declaration order.",
            [
                ("decorators.py", 370, 377),
                ("decorators.py", 314, 321),
                ("decorators.py", 221, 230),
                ("decorators.py", 248, 250),
            ],
        ),
        (
            "Trace command forwarding: how the current context fills missing explicit arguments and how the invoked command fills any remaining defaults before calling its callback.",
            [("core.py", 952, 959), ("core.py", 907, 940)],
        ),
        (
            "Trace how a typed object-passing decorator finds the nearest matching parent context object and, with creation enabled, constructs and stores one when no match exists.",
            [("decorators.py", 76, 95), ("core.py", 770, 789)],
        ),
        (
            "Trace registration of a context cleanup callback through explicit context closure, including execution of the exit stack and replacement of that stack for reuse.",
            [("core.py", 707, 724), ("core.py", 738, 742)],
        ),
        (
            "Trace how creating a command context parses arguments using a parser populated with the command's parameters, up to parameter-result handling.",
            [("core.py", 1386, 1394), ("core.py", 1396, 1404), ("core.py", 1256, 1261)],
        ),
        (
            "Trace how a parameter falls back from command-line and environment input to a context default map, including invoking a callable stored default and recording the source of the value.",
            [("core.py", 2693, 2726), ("core.py", 825, 835), ("core.py", 798, 803)],
        ),
        (
            "Trace integer-range conversion when clamping is enabled and a value lies outside an open bound, showing how the numeric conversion and adjusted bound are computed.",
            [("types.py", 598, 610), ("types.py", 669, 686), ("types.py", 761, 765)],
        ),
        (
            "Trace case-insensitive choice conversion through normalization of the input and available choices, returning the original choice object rather than the normalized spelling.",
            [("types.py", 462, 470), ("types.py", 398, 404), ("types.py", 416, 424)],
        ),
        (
            "Trace how converting an eagerly opened file parameter schedules closing or flushing at context shutdown, and how the scheduled wrapper suppresses cleanup exceptions.",
            [
                ("types.py", 1006, 1021),
                ("core.py", 707, 717),
                ("utils.py", 36, 49),
                ("core.py", 719, 724),
                ("core.py", 738, 742),
            ],
        ),
        (
            "Trace how a group's command decorator constructs a command, registers its chosen name, and rejects registration without a command name.",
            [("core.py", 1881, 1892), ("decorators.py", 239, 250), ("core.py", 1835, 1843)],
        ),
        (
            "Trace how two-column help entries are measured and padded before their descriptions are wrapped to the available width.",
            [("formatting.py", 14, 28), ("formatting.py", 243, 269), ("formatting.py", 62, 107)],
        ),
        (
            "Trace how a missing filesystem path is accepted when existence is optional and then converted to the requested string, bytes, or path-object result type.",
            [("types.py", 1167, 1180), ("types.py", 1147, 1159)],
        ),
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build(output: Path) -> None:
    if (output / "manifest.json").exists():
        raise FileExistsError("labels frozen; author a new version instead of overwriting")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "version": 1,
        "authorship": "Codex source-authored and author-checked; independent review pending",
        "selection": "30 newly authored multi-block implementation questions before any new retrieval; 12 Click and 6 per earlier repo; not a random or independent holdout",
        "fresh_questions": 30,
        "regression_questions": 120,
        "datasets": [],
    }
    preview = ["# Frozen expansion questions", "", manifest["authorship"], ""]
    for short, (repo, sha, folder) in REPOS.items():
        checkout = ROOT / folder
        assert Repo(checkout).head.commit.hexsha == sha
        prefix = (
            f"src/{'requests' if short == 'requests' else 'flask' if short == 'flask' else 'click'}/"
            if short != "express"
            else "lib/"
        )
        cases = []
        for i, (text, specs) in enumerate(SEEDS[short], 1):
            gold = []
            for relative, start, end in specs:
                evidence_path = prefix + relative
                source = (checkout / evidence_path).read_text(encoding="utf-8")
                assert end <= len(source.splitlines()), (evidence_path, end)
                gold.append(
                    GoldRange(
                        path=evidence_path,
                        start_line=start,
                        end_line=end,
                        source_sha256=evidence_hash(source, start, end),
                        symbol=f"source lines {start}-{end}",
                    )
                )
            question = Question(
                id=f"{short}-expansion-{i:03}",
                repo=repo,
                commit_sha=sha,
                question=text,
                category="multi_block",
                split="test",
                group="pending",
                gold=gold,
                provenance="Codex source-authored before expansion eval; prior source exposure is disclosed, independent review pending",
                cohort="fresh",
                schema_version=2,
            )
            cases.append(
                AmbiguousCase(
                    question=question,
                    accepted_evidence=[gold],
                    rationale="All listed implementation ranges jointly establish the requested flow; external-library internals are outside scope.",
                    related_v2_ids=[],
                    stratum="multi_block",
                    topic=f"{short}-flow-{i}",
                )
            )
        assign_groups(cases)
        validate_cases(checkout, cases)
        previous = (
            read_cases(ROOT / f"eval/datasets/phase4_intent_v1/{short}_qa.jsonl")
            if short != "click"
            else []
        )
        overlap = {
            c.question.id: [
                p.question.id
                for p in previous
                if any(
                    overlaps(a, b)
                    for a in c.question.gold
                    for proof in p.accepted_evidence
                    for b in proof
                )
            ]
            for c in cases
        }
        path = output / f"{short}_qa.jsonl"
        path.write_text(
            "".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8", newline="\n"
        )
        read_cases(path)
        manifest["datasets"].append(
            {
                "repo": repo,
                "commit_sha": sha,
                "file": path.name,
                "questions": len(cases),
                "role": "fresh",
                "sha256": digest(path),
                "groups": len({c.question.group for c in cases}),
                "previous_regression_evidence_overlap": overlap,
            }
        )
        for case in cases:
            preview.extend(
                [
                    f"## {case.question.id}",
                    "",
                    case.question.question,
                    "",
                    "Required together: "
                    + ", ".join(
                        f"{g.path}:{g.start_line}-{g.end_line}" for g in case.question.gold
                    ),
                    "",
                ]
            )
        if previous:
            guard = output / f"{short}_regression.jsonl"
            guard.write_bytes(
                (ROOT / f"eval/datasets/phase4_intent_v1/{short}_qa.jsonl").read_bytes()
            )
            manifest["datasets"].append(
                {
                    "repo": repo,
                    "commit_sha": sha,
                    "file": guard.name,
                    "questions": len(previous),
                    "role": "regression",
                    "sha256": digest(guard),
                    "groups": len({c.question.group for c in previous}),
                }
            )
    write_json(output / "manifest.json", manifest)
    (output / "QUESTIONS.md").write_text("\n".join(preview), encoding="utf-8")
    print(
        json.dumps(
            {
                "fresh": sum(d["questions"] for d in manifest["datasets"] if d["role"] == "fresh"),
                "regression": sum(
                    d["questions"] for d in manifest["datasets"] if d["role"] == "regression"
                ),
                "fresh_groups": sum(
                    d["groups"] for d in manifest["datasets"] if d["role"] == "fresh"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "eval/datasets/phase4_expansion_v1")
    build(parser.parse_args().output)
