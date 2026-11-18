"""Rebuild every bakeoff table from raw turns; freeze common prompt/parser on dev only.

    uv run python eval/analyze_answer_bakeoff.py freeze --output DIR [--capsule DIR]
    uv run python eval/analyze_answer_bakeoff.py report --output DIR [--capsule DIR]

No model calls. Test labels are not decoded until the dev freeze has been verified.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.answer_bakeoff_io import (
    DATASET,
    digest,
    identity,
    load,
    question_rows,
    rows,
    save,
    verify,
    verify_test_freeze,
)
from eval.answer_bakeoff_scoring import score_trace, wilson
from repo_rag.agent.prompts import BAKEOFF_ARMS


def labels_for(split: str, unlocked: bool = False) -> dict[str, dict]:
    if split == "test" and not unlocked:
        raise ValueError("Test labels remain locked until dev freeze")
    ids = {r["id"] for r in question_rows(split)}
    result = {}
    # Frozen canonical JSONL puts the outer id first. Decode only that id, and then
    # only the selected split's rows: dev does not deserialize any test label object.
    for line in (DATASET / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        match = re.match(r'\{"id":\s*("(?:[^"\\]|\\.)*")', line)
        if not match:
            raise ValueError("Unexpected frozen JSONL layout")
        if json.loads(match[1]) in ids:
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError("Split mismatch")
            result[row["id"]] = row
    if set(result) != ids:
        raise ValueError("Incomplete label view")
    return result


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def proportion(items: list[dict], key: str) -> dict:
    num, den = sum(bool(r[key]) for r in items), len(items)
    return {"successes": num, "total": den, "rate": ratio(num, den), "wilson95": wilson(num, den)}


def aggregate(items: list[dict], protocol: dict) -> dict:
    facts = [r for r in items if r["suite"] == "facts"]
    answerable = [r for r in items if r["suite"] in ("code", "architecture")]
    probes = [r for r in items if r["suite"] == "probes"]
    citations = sum(r["citations"] for r in answerable)
    invalid = sum(r["invalid_citations"] for r in answerable)
    result = {
        "facts": proportion(facts, "fact_ok"),
        "facts_strict": proportion(facts, "fact_ok_strict"),
        "raw_exact": proportion(facts, "raw_exact"),
        "raw_exact_corrected": proportion(facts, "raw_exact_corrected"),
        "tool_names": proportion(facts, "tool_name_ok"),
        "arguments": proportion(facts, "arguments_ok"),
        "arguments_strict": proportion(facts, "arguments_ok_strict"),
        "arguments_given_names": proportion(
            [r for r in facts if r["tool_name_ok"]], "arguments_ok"
        ),
        "probe_abstention": proportion(probes, "probe_ok"),
        "answerable_abstention": proportion(answerable, "abstain"),
        "citation_complete": proportion(answerable, "citation_complete"),
        "citation_complete_checked": proportion(answerable, "citation_complete_checked"),
        "citations_attached": sum(r["citations_attached"] for r in answerable),
        "invalid_citations": invalid,
        "citations": citations,
        "invalid_citation_rate": invalid / citations if citations else 1.0,
        "no_citation_answers": sum(not r["citations"] for r in answerable),
        "no_citation_denominator": len(answerable),
        "lexical_violation_answers": sum(bool(r["lexical_violations"]) for r in answerable),
        "failure_counts": dict(Counter(r["failure"] or "none" for r in items)),
        "tool_call_count": sum(r["tool_calls"] for r in items),
        "extra_fact_calls_over_registered_minimum": sum(r["extra_fact_calls"] or 0 for r in facts),
        "facts_with_extra_calls": sum(bool(r["extra_fact_calls"]) for r in facts),
    }
    gates = protocol["selection"]["gates"]
    failed = []
    for name, rate, threshold, minimum in (
        ("fact_exactness", result["facts"]["rate"], gates["grounded_fact_exactness_min"], True),
        (
            "invalid_citations",
            result["invalid_citation_rate"],
            gates["invalid_citation_rate_max"],
            False,
        ),
        (
            "probe_abstention",
            result["probe_abstention"]["rate"],
            gates["test_probe_abstention_min"],
            True,
        ),
        (
            "answerable_abstention",
            result["answerable_abstention"]["rate"],
            gates["answerable_Q2_abstention_max"],
            False,
        ),
    ):
        if (rate < threshold) if minimum else (rate > threshold):
            failed.append(name)
    if "citation_complete_min" in gates and (
        result["citation_complete"]["rate"] < gates["citation_complete_min"]
    ):  # v5 gate, raw output without the answer check
        failed.append("citation_complete")
    result["failed_gates"] = failed
    result["eligible"] = not failed
    return result


def timing(items: list[dict]) -> dict:
    seconds = [r["seconds"] for r in items]
    generation = sum(r["generation_seconds"] for r in items)
    return {
        "questions": len(items),
        "p50_seconds": float(np.quantile(seconds, 0.5)) if seconds else None,
        "p95_seconds": float(np.quantile(seconds, 0.95)) if seconds else None,
        "generation_tokens_per_second": sum(r["output_tokens"] for r in items) / generation
        if generation
        else None,
        "peak_allocated_bytes": max(
            (r["memory"].get("peak_allocated_bytes", 0) for r in items), default=0
        ),
        "peak_reserved_bytes": max(
            (r["memory"].get("peak_reserved_bytes", 0) for r in items), default=0
        ),
        "includes_failures": True,
        "device_level_peak_sampled": False,
    }


def recompute_run(
    folder: Path, labels: dict, protocol: dict, candidates: list[dict]
) -> tuple[dict, list[dict], dict]:
    meta = load(folder / "run.json")
    if not meta["complete"] or meta["identity"]["protocol_sha256"] != digest(
        DATASET / "protocol.json"
    ):
        raise ValueError(
            "Incomplete or wrong-protocol matrix; retain fatal error, do not score a partial matrix"
        )
    expected = {c["model_id"]: c["revision"] for c in candidates}
    if meta["candidate_revisions"] != expected:
        raise ValueError("Not all four approved revisions are represented")
    arm_prompt, arm_reminder = BAKEOFF_ARMS[meta["revision"]]
    if meta["prompt"] != arm_prompt or meta.get("reminder") != arm_reminder:
        raise ValueError("Run prompt/reminder differs from the registered arm")
    if meta["identity"]["arms_sha256"] != identity()["arms_sha256"]:
        raise ValueError("Registered prompt arms changed since this run")
    summary, scored = {}, []
    order = None
    registered_order = [r["id"] for r in question_rows(meta["split"])]
    random.Random(42).shuffle(registered_order)
    for candidate in candidates:
        model_id = candidate["model_id"]
        path = folder / (model_id.replace("/", "__") + ".jsonl")
        if digest(path) != meta["raw_sha256"][path.name]:
            raise ValueError("Raw transcript hash changed")
        traces = rows(path)
        ids = [r["id"] for r in traces]
        if len(set(ids)) != len(ids) or set(ids) != set(labels):
            raise ValueError("Duplicate/missing/extra question in raw transcript")
        if ids != registered_order:
            raise ValueError("Raw question order differs from registered seed 42")
        if order is not None and order != ids:
            raise ValueError("Question order differs across models")
        order = ids
        result = []
        for trace in traces:
            row = labels[trace["id"]]
            if any(trace[k] != row[k] for k in ("repo", "split", "suite")):
                raise ValueError("Raw routing identity mismatch")
            score = score_trace(trace, row, protocol, meta["prompt"], meta.get("reminder"))
            score["model_id"] = model_id
            score["fact_type"] = row["label"].get("type", "") if row["suite"] == "facts" else ""
            result.append(score)
        scored += result
        report = aggregate(result, protocol)
        report["timing"] = {
            suite: timing([r for r in result if suite == "pooled" or r["suite"] == suite])
            for suite in ("pooled", "facts", "code", "architecture", "probes")
        }
        report["facts_by_type"] = {
            kind: aggregate([r for r in result if r["fact_type"] == kind], protocol)["facts"]
            for kind in sorted({r["fact_type"] for r in result if r["fact_type"]})
        }
        report["answerable_by_repo"] = {
            repo: aggregate(
                [r for r in result if r["repo"] == repo and r["suite"] in ("code", "architecture")],
                protocol,
            )
            for repo in sorted(
                {r["repo"] for r in result if r["suite"] in ("code", "architecture")}
            )
        }
        report["retrieval_by_suite"] = {
            suite: proportion([r for r in result if r["suite"] == suite], "retrieval_hit")
            for suite in ("code", "architecture")
        }
        report["load_seconds"] = meta["models"][model_id]["load_seconds"]
        report["parameters_total"] = candidate["parameters_total"]
        summary[model_id] = report
    return summary, scored, meta


def select_model(summary: dict) -> dict:
    eligible = [name for name, row in summary.items() if row["eligible"]]
    pool = eligible or list(summary)
    best = max(summary[name]["facts"]["successes"] for name in pool)
    tied = [name for name in pool if summary[name]["facts"]["successes"] >= best - 1]
    winner = sorted(
        tied,
        key=lambda name: (
            -summary[name]["citation_complete"]["rate"],
            summary[name]["invalid_citation_rate"],
            summary[name]["parameters_total"],
            name,
        ),
    )[0]
    return {
        "adopt": bool(eligible),
        "winner": winner if eligible else None,
        "best_descriptive": winner,
        "failed_gates": {name: row["failed_gates"] for name, row in summary.items()},
        "note": "Necessary mechanical gates only; semantic correctness/label review remain open.",
    }


def dev_models(protocol: dict, summary: dict) -> list[dict]:
    """Per-model dev rows with the model id attached, for the v5 subject-model rank."""
    return [{"model": name, **row} for name, row in summary.items()]


def rebuild_dev(
    output: Path, index: Path, protocol: dict, candidates: list[dict]
) -> tuple[list, dict, dict]:
    if any(
        p.is_dir() and p.name.startswith("dev_") and p.name not in {"dev_0", "dev_1", "dev_2"}
        for p in output.iterdir()
    ):
        raise ValueError("More than two shared dev revisions are not permitted")
    labels = labels_for("dev")
    options = []
    artifacts = {}
    summaries = {}
    for revision in range(3):
        folder = output / f"dev_{revision}"
        if not folder.exists():
            continue
        summary, scored, meta = recompute_run(folder, labels, protocol, candidates)
        if meta["split"] != "dev" or meta["revision"] != revision:
            raise ValueError("Wrong dev revision")
        if meta["index_manifest_sha256"] != digest(index):
            raise ValueError("Index changed across shared revisions")
        facts = [r for r in scored if r["suite"] == "facts"]
        answerable = [r for r in scored if r["suite"] == "code"]
        probes = [r for r in scored if r["suite"] == "probes"]
        refs = sum(r["citations"] for r in answerable)
        validity = 1 - sum(r["invalid_citations"] for r in answerable) / refs if refs else 0
        rank = (
            proportion(facts, "fact_ok")["rate"],
            proportion(answerable, "citation_complete")["rate"],
            validity,
            proportion(probes, "probe_ok")["rate"],
            -revision,
        )
        if protocol["selection"].get("dev_rank") == "v5":
            # Subject model only: probes first so a citation gain cannot buy broken refusals.
            subject = next(
                s for s in dev_models(protocol, summary) if s["model"] == protocol["subject"]
            )
            rank = (
                subject["probe_abstention"]["rate"],
                subject["citation_complete"]["rate"],
                -subject["answerable_abstention"]["rate"],
                -subject["invalid_citation_rate"],
                -revision,
            )
        options.append((rank, revision, meta))
        save(folder / "dev_summary.json", summary)
        save(folder / "dev_question_results.json", scored)
        summaries[str(revision)] = summary
        for path in [folder / "run.json", *folder.glob("*.jsonl")]:
            artifacts[path.relative_to(output).as_posix()] = digest(path)
    if not options or [o[1] for o in options] != list(range(len(options))):
        raise ValueError(
            "Need complete consecutive shared dev revisions starting at 0 (maximum 2 revisions)"
        )
    return options, artifacts, summaries


def freeze_dev(output: Path, capsule: Path) -> dict:
    protocol, candidates = verify()
    target = output / "prompt_parser.freeze.json"
    if target.exists() or any(output.glob("test_*/run.json")):
        raise ValueError("Dev freeze/test already exists; never overwrite or tune after test")
    options, artifacts, _ = rebuild_dev(
        output, capsule / "index_manifest.json", protocol, candidates
    )
    _, selected, meta = max(options, key=lambda x: x[0])
    if meta["identity"] != identity():
        raise ValueError(
            "Restore the dev-selected common prompt/parser/source before freezing; never test a different implementation"
        )
    frozen = {
        "selection_split": "dev",
        "selected_revision": selected,
        "identity": identity(),
        "frozen_utc": datetime.now(UTC).isoformat(),
        "dev_artifacts": artifacts,
        "index_manifest_sha256": digest(capsule / "index_manifest.json"),
        "dev_selection": [{"revision": r, "rank": list(rank)} for rank, r, _ in options],
        "test_labels_decoded": False,
        "candidate_revisions": meta["candidate_revisions"],
    }
    save(target, frozen)
    (output / "index_manifest.json").write_bytes((capsule / "index_manifest.json").read_bytes())
    print("Dev-selected common prompt/parser frozen before test label access:")
    print(
        json.dumps({k: frozen["identity"][k] for k in ("prompt_sha256", "parser_sha256")}, indent=2)
    )
    return frozen


def report(output: Path, capsule: Path) -> dict:
    protocol, candidates = verify()
    index = capsule / "index_manifest.json"
    if not index.exists():
        index = output / "index_manifest.json"
    frozen = verify_test_freeze(output / "prompt_parser.freeze.json", index)
    options, artifacts, dev_summaries = rebuild_dev(output, index, protocol, candidates)
    selected = max(options, key=lambda x: x[0])[1]
    if selected != frozen["selected_revision"] or artifacts != frozen["dev_artifacts"]:
        raise ValueError("Recomputed dev choice differs from the saved freeze")
    # This is the first operation in reporting that decodes test labels.
    labels = labels_for("test", unlocked=True)
    folder = output / f"test_{frozen['selected_revision']}"
    summary, scored, meta = recompute_run(folder, labels, protocol, candidates)
    if (
        meta["identity"] != frozen["identity"]
        or meta["index_manifest_sha256"] != frozen["index_manifest_sha256"]
    ):
        raise ValueError("Test implementation/index differs from frozen dev selection")
    if meta["split"] != "test" or meta["started_utc"] < frozen["frozen_utc"]:
        raise ValueError("Test was run before the common prompt/parser freeze")
    decision = select_model(summary)
    baseline = None
    baseline_folder = output / "test_0"
    if frozen["selected_revision"] != 0 and baseline_folder.exists():
        baseline, baseline_scored, base_meta = recompute_run(
            baseline_folder, labels, protocol, candidates
        )
        base_meta_ok = base_meta["identity"] == frozen["identity"]
        if not base_meta_ok or base_meta["started_utc"] < frozen["frozen_utc"]:
            raise ValueError("Baseline test run differs from the frozen implementation")
        save(output / "question_results_baseline.json", baseline_scored)
    result = {
        "baseline_revision_0": baseline,
        "selected_revision": frozen["selected_revision"],
        "models": summary,
        "dev_recomputed": dev_summaries,
        "selection": decision,
        "hardware": meta["hardware"],
        "cost_label": (
            "A100 experiment timing; not deployment"
            if "A100" in meta["hardware"].get("gpu", "")
            and 38 * 2**30 <= meta["hardware"].get("total_device_bytes", 0) <= 42 * 2**30
            else "Unregistered or simulated hardware; no A100 cost claim"
        ),
        "label_review": "architecture/probes unreviewed by a person",
        "interpretation": "Citation validity and lexical grounding do not prove correctness. Missing citations reported separately. Small strata and close comparisons descriptive.",
    }
    save(output / "summary.json", result)
    save(output / "question_results.json", scored)
    lines = [
        "# Phase 7 v5 — mechanical test results (tuned prompt vs the unchanged v4 prompt)",
        "",
        result["interpretation"],
        "",
        "Architecture/probe labels remain unreviewed by a person. Answerable and probe test "
        "questions are fresh (seeded draw); the facts guard reuses the 45 v4 test facts, which "
        "were seen. Prompt revision 0 is the unchanged v4 prompt; 1 is R2; 2 is R3.",
        "",
        f"Selected arm on dev: revision {frozen['selected_revision']}. Citation columns: raw is "
        "the model output; checked adds `path:start-end` for retrieved chunks the answer names.",
        "",
        "Exact facts = corrected score; strict = unchanged v3 rule.",
        "",
        "| Arm | Model | Facts (corr./strict) | Probes | Answerable abstentions | Citation complete raw | Citation complete checked | Invalid citations | Failed gates |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    arms = [(frozen["selected_revision"], summary)]
    if baseline is not None:
        arms.append((0, baseline))
    for revision, table in arms:
        for name, row in table.items():

            def count(key, row=row):
                item = row[key]
                return f"{item['successes']}/{item['total']}"

            lines.append(
                f"| R{revision} | {name} | {count('facts')} / {count('facts_strict')} | {count('probe_abstention')} | {count('answerable_abstention')} | {count('citation_complete')} | {count('citation_complete_checked')} | {row['invalid_citations']}/{row['citations']} ({row['invalid_citation_rate']:.3f}) | {', '.join(row['failed_gates']) or 'none'} |"
            )
    lines += [
        "",
        "95% Wilson intervals, type/repo/retrieval diagnostics and every scored question are in summary.json / question_results.json.",
        "",
        "## " + result["cost_label"] + " (includes failures, excludes cold model load)",
        "",
        "| Model | Suite | n | p50 s | p95 s | Generation tokens/s | Peak allocated GiB | Peak reserved GiB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in summary.items():
        for suite, stats in row["timing"].items():
            lines.append(
                f"| {name} | {suite} | {stats['questions']} | {stats['p50_seconds']} | {stats['p95_seconds']} | {stats['generation_tokens_per_second']} | {stats['peak_allocated_bytes'] / 2**30:.3f} | {stats['peak_reserved_bytes'] / 2**30:.3f} |"
            )
    subject = protocol.get("subject")
    if subject:
        failed = summary[subject]["failed_gates"]
        verdict = (
            f"{subject} on arm R{frozen['selected_revision']} passes every registered gate."
            if not failed
            else f"{subject} on arm R{frozen['selected_revision']} fails: {', '.join(failed)}."
        )
        verdict += " No default is changed; the control model is reported, not gated."
    else:
        verdict = (
            "Eligible model selected: " + str(decision["winner"])
            if decision["adopt"]
            else "No candidate qualifies. No default is changed; failed gates are retained above."
        )
    lines += ["", verdict, ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        "Rebuilt tables from raw generated turns/tool results; "
        + ("eligible candidate found." if decision["adopt"] else "no candidate qualifies.")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, default=ROOT / "bakeoff")
    args = parser.parse_args()
    (freeze_dev if args.action == "freeze" else report)(args.output, args.capsule)


if __name__ == "__main__":
    main()
