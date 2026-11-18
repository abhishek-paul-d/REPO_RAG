"""Recompute architecture tables and frozen gates from saved rankings, without a model.

uv run python eval/analyze_architecture.py eval/facts_results/architecture_v1_hybrid
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_architecture_dataset import OUT, REPOS, verify_frozen, write
from eval.retrieval.statistics import bootstrap_intervals, paired_intervals
from eval.run_architecture_eval import CODE_METRICS, decision, report, rows, score, select
from repo_rag.config import Settings


def rescore(saved: list[dict], questions: list[dict]) -> list[dict]:
    by_id = {q["id"]: q for q in questions}
    seen = set()
    results = []
    for row in saved:
        pair = row["id"], row["method"]
        if pair in seen:
            raise ValueError("Duplicate saved result")
        seen.add(pair)
        q = by_id[row["id"]]
        if any(row[k] != q[k] for k in ("repo", "group", "split")):
            raise ValueError("Saved labels/splits changed")
        gold = set(q.get("gold_files") or [g["path"] for g in q["gold"]])
        computed = score(row["ranking"], gold)
        if any(abs(row[key] - value) > 1e-12 for key, value in computed.items()):
            raise ValueError("Saved metrics do not match rankings")
        results.append({**row, **computed})
    for method in {r["method"] for r in saved}:
        if {r["id"] for r in saved if r["method"] == method} != by_id.keys():
            raise ValueError("Missing questions in saved result")
    return results


def analyze(directory: Path) -> dict:
    frozen = verify_frozen()
    run = json.loads((directory / "run.json").read_text())
    if run["freeze"] != frozen:
        raise ValueError("Run used other frozen inputs")
    protocol = json.loads((OUT / "protocol.json").read_text())
    questions = [q for name in REPOS for q in rows(OUT / f"{name}.jsonl")]
    dev_q = [q for q in questions if q["split"] == "dev"]
    test_q = [q for q in questions if q["split"] == "test"]
    code_q = rows(OUT / "requests_code_regression.jsonl")
    selection = json.loads((directory / "selection_E1.json").read_text())
    dev = rescore(json.loads((directory / "base/dev.json").read_text()), dev_q)
    router = select(dev, [a for a in protocol["arms"] if a.startswith("router")])
    if router != selection["router"]:
        raise ValueError("E1 router was not chosen on dev")
    baseline = rescore(json.loads((directory / "base/test.json").read_text()), test_q)
    code_base = [
        r
        for r in rescore(json.loads((directory / "base/code.json").read_text()), code_q)
        if r["method"] == "chunks"
    ]
    found_variants = [
        "base",
        *(
            v
            for v in protocol["fact_variants"]
            if v != "base" and (directory / v / "test.json").exists()
        ),
    ]
    results = {}
    dev_variants = []
    settings = Settings.model_construct(
        eval_bootstrap_samples=protocol["bootstrap"]["samples"],
        eval_bootstrap_seed=protocol["bootstrap"]["seed"],
        eval_bootstrap_confidence=protocol["bootstrap"]["confidence"],
    )
    for variant in found_variants:
        test = rescore(json.loads((directory / variant / "test.json").read_text()), test_q)
        code = rescore(json.loads((directory / variant / "code.json").read_text()), code_q)
        for row in rescore(json.loads((directory / variant / "dev.json").read_text()), dev_q):
            if row["method"] == router:
                dev_variants.append({**row, "method": variant})
        for method in dict.fromkeys(r["method"] for r in test):
            result = report(test, baseline, method, settings)
            measured_code = [r for r in code if r["method"] == method]
            result["code"] = bootstrap_intervals(measured_code, CODE_METRICS, settings)
            result["code_delta"] = paired_intervals(
                measured_code, code_base, CODE_METRICS, settings
            )
            result["decision"] = decision(
                result, result["code_delta"]["mrr@10"]["estimate"], run["mode"]
            )
            results[f"{variant}/{method}"] = result
    if len(found_variants) > 1:
        best = select(dev_variants, list(protocol["fact_variants"]))
        if best != json.loads((directory / "selection_E2.json").read_text())["variant"]:
            raise ValueError("E2 facts were not chosen on dev")
    else:
        best = "base"
    recomputed = {
        "label_review": "unreviewed by a person",
        "selected_variant": best,
        "selected_router": router,
        "results": results,
    }
    saved = json.loads((directory / "report.json").read_text())
    if any(recomputed[k] != saved[k] for k in recomputed):
        raise ValueError("Recomputed table differs from saved report")
    write(directory / "recomputed_report.json", recomputed)
    if len(found_variants) > 1:
        pairs = [
            ("F1", "base"),
            ("F2", "base"),
            ("F3", "base"),
            ("F4", "base"),
            ("F1F2", "F1"),
            ("F1F2F3", "F1F2"),
            ("F1F2F3F4", "F1F2F3"),
        ]
        changes = {}
        lines = [
            "Fact variants: labels **unreviewed by a person**. Fixed comparisons, no test selection.",
            "",
            f"Router fixed on dev: {router}.",
            "",
            "| Added / control | Before Recall@5 | After Recall@5 | Paired delta [95% CI] | Code MRR delta |",
            "|---|---|---|---|---|",
        ]
        for left, right in pairs:
            a = json.loads((directory / left / "test.json").read_text())
            b = json.loads((directory / right / "test.json").read_text())
            code_a = json.loads((directory / left / "code.json").read_text())
            code_b = json.loads((directory / right / "code.json").read_text())
            for arm in ("summaries", "fused", router):
                aa = [r for r in a if r["method"] == arm]
                bb = [r for r in b if r["method"] == arm]
                delta = paired_intervals(aa, bb, ["recall@5", "mrr@10"], settings)
                code_delta = paired_intervals(
                    [r for r in code_a if r["method"] == arm],
                    [r for r in code_b if r["method"] == arm],
                    CODE_METRICS,
                    settings,
                )
                changes[f"{left}-{right}/{arm}"] = {
                    "delta": delta,
                    "code_delta": code_delta,
                    "per_repo": {
                        slug: paired_intervals(
                            [r for r in aa if r["repo"] == slug],
                            [r for r in bb if r["repo"] == slug],
                            ["recall@5", "mrr@10"],
                            settings,
                        )
                        for slug, _, _ in REPOS.values()
                    },
                }
                if arm == router:
                    before = sum(r["recall@5"] for r in bb) / len(bb)
                    after = sum(r["recall@5"] for r in aa) / len(aa)
                    d = delta["recall@5"]
                    lines.append(
                        f"| {left} / {right} | {before:.4f} | {after:.4f} | {d['estimate']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] | {code_delta['mrr@10']['estimate']:+.4f} |"
                    )
        write(
            directory / "fact_deltas.json",
            {"label_review": "unreviewed by a person", "comparisons": changes},
        )
        (directory / "fact_deltas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Recomputed {len(results)} rows; all rankings, dev choices, intervals and gates match. Labels unreviewed by a person."
    )
    return recomputed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
