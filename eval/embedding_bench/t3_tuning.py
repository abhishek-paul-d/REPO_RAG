"""T3 tuning: query instructions x commit document formats for bug report -> fix commit.

Run 2 found Qwen3-Embedding-0.6B trailing nomic-embed-code on T3 (nDCG@10 0.488 vs 0.596).
This tries cheap fixes before reaching for a bigger model:

  * query side: the instruction Qwen3 is given (it is instruction-aware)
  * doc side:   what a commit document contains (message, files, touched symbols, diff)

To avoid fooling ourselves with 144 queries, issues are split 50/50 into dev and test
(fixed seed). The winner is picked on **dev** only and reported on **test**, with a paired
bootstrap confidence interval against the run-2 baseline.

Run: python t3_tuning.py --data-dir ./bench_data --out ./t3_results   (needs GITHUB_TOKEN)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import bench
import numpy as np

INSTRUCTIONS: dict[str, str] = {
    "baseline": bench.COMMIT_QUERY_INSTRUCTION,
    "issue_to_commit": (
        "Instruct: Given a GitHub issue describing a bug, retrieve the git commit "
        "(its message and code changes) that fixes it\nQuery:"
    ),
    "symptom_to_change": (
        "Instruct: Given a user's description of a problem, retrieve the code change "
        "that resolves it\nQuery:"
    ),
    "code_instruction": bench.CODE_QUERY_INSTRUCTION,  # one instruction for every route?
    "none": "",
}
BASELINE = ("baseline", "diff2000")

SELECT_DIMS = 512


def _split(n: int, seed: int = 0) -> tuple[list[int], list[int]]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    half = n // 2
    return sorted(order[:half]), sorted(order[half:])


def _subset_task(task: bench.Task, idx: list[int], name: str) -> bench.Task:
    return bench.Task(
        name=name,
        queries=[task.queries[i] for i in idx],
        gold=[task.gold[i] for i in idx],
        doc_ids=task.doc_ids,
        doc_texts=task.doc_texts,
        query_kind=task.query_kind,
    )


def _per_query_ndcg(task: bench.Task, rankings: list[list[int]]) -> list[float]:
    _, per_q = bench.score(task, rankings)
    return [p["ndcg@10"] for p in per_q]


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000, seed: int = 0) -> dict:
    """Mean(b - a) with a 95% percentile CI, resampling queries."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(b) - np.asarray(a)
    samples = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "mean_gain": round(float(diff.mean()), 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
        "p_gain_le_0": round(float((samples <= 0).mean()), 4),
    }


def _limit_corpus(
    ids: list[str],
    texts: dict[str, list[str]],
    gold: list[set[str]],
    max_commits: int,
    seed: int = 0,
) -> tuple[list[str], dict[str, list[str]]]:
    """Smoke-test helper: keep every gold commit plus a random sample of the rest."""
    needed = set().union(*gold) if gold else set()
    keep = [i for i, sha in enumerate(ids) if sha in needed]
    others = [i for i, sha in enumerate(ids) if sha not in needed]
    random.Random(seed).shuffle(others)
    keep = sorted(keep + others[: max(0, max_commits - len(keep))])
    return [ids[i] for i in keep], {f: [t[i] for i in keep] for f, t in texts.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("bench_data"))
    parser.add_argument("--out", type=Path, default=Path("t3_results"))
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--max-commits", type=int, default=0, help="smoke tests only")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required for T3 (issue text comes from the GitHub API).")
    args.out.mkdir(parents=True, exist_ok=True)

    repo_dir = bench.clone_repos(args.data_dir, {bench.BUG_REPO: bench.REPOS[bench.BUG_REPO]})[
        bench.BUG_REPO
    ]
    ids, texts, links = bench.commit_corpus_formats(repo_dir, bench.COMMIT_DOC_FORMATS)
    queries, gold, numbers = bench.bug_queries(
        links, token, args.data_dir / "issues_cache.json", args.max_pairs
    )
    if args.max_commits:
        ids, texts = _limit_corpus(ids, texts, gold, args.max_commits)
    dev_idx, test_idx = _split(len(queries))
    print(f"device={device} model={args.model}")
    print(
        f"T3: {len(queries)} queries (dev {len(dev_idx)}, test {len(test_idx)}) over {len(ids)} commits"
    )

    spec = bench.MODELS[args.model]
    enc = bench.Encoder(spec, device)

    doc_emb: dict[str, np.ndarray] = {}
    format_stats = []
    for fmt in bench.COMMIT_DOC_FORMATS:
        lengths = enc.token_lengths(texts[fmt])
        t0 = time.perf_counter()
        doc_emb[fmt] = enc.encode(texts[fmt])
        secs = time.perf_counter() - t0
        format_stats.append(
            {
                "doc_format": fmt,
                "median_tokens": int(statistics.median(lengths)),
                "encode_s": round(secs, 1),
                "docs_per_s": round(len(ids) / secs, 1),
            }
        )
        print(json.dumps(format_stats[-1]))

    q_emb = {
        name: enc.encode(queries, prompt, is_query=True) for name, prompt in INSTRUCTIONS.items()
    }

    full = bench.Task("T3", queries, gold, ids, texts["diff2000"], "commit")
    splits = {
        "dev": _subset_task(full, dev_idx, "dev"),
        "test": _subset_task(full, test_idx, "test"),
        "all": full,
    }
    split_idx = {"dev": dev_idx, "test": test_idx, "all": list(range(len(queries)))}

    # Shipped setting (512) and full size; a model smaller than 512 dims just uses its own.
    select_dims = min(SELECT_DIMS, enc.dim)
    dims_list = sorted({select_dims, enc.dim})

    rows, per_query = [], {}
    for inst in INSTRUCTIONS:
        for fmt in bench.COMMIT_DOC_FORMATS:
            for dims in dims_list:
                q, d = q_emb[inst], doc_emb[fmt]
                if dims < enc.dim:
                    q, d = bench.truncate_dims(q, dims), bench.truncate_dims(d, dims)
                rankings = bench.dense_rankings(q, d)
                per_query[(inst, fmt, dims)] = _per_query_ndcg(full, rankings)
                for split, task in splits.items():
                    agg, _ = bench.score(task, [rankings[i] for i in split_idx[split]])
                    rows.append(
                        {
                            "instruction": inst,
                            "doc_format": fmt,
                            "dims": dims,
                            "split": split,
                            "n": len(task.queries),
                            **{k: round(v, 4) for k, v in agg.items()},
                        }
                    )

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "t3_tuning.csv", index=False)
    pd.DataFrame(format_stats).to_csv(args.out / "t3_doc_formats.csv", index=False)

    sel = df[(df.split == "dev") & (df.dims == select_dims)]
    best = sel.sort_values("ndcg@10", ascending=False).iloc[0]
    best_key = (best.instruction, best.doc_format, select_dims)
    base_key = (*BASELINE, select_dims)

    def test_ndcg(key: tuple) -> list[float]:
        return [per_query[key][i] for i in test_idx]

    boot = paired_bootstrap(test_ndcg(base_key), test_ndcg(best_key))

    def pick(key: tuple, split: str) -> pd.Series:
        rows = df[
            (df.instruction == key[0])
            & (df.doc_format == key[1])
            & (df.dims == select_dims)
            & (df.split == split)
        ]
        return rows.iloc[0]

    verdict = {
        "model": args.model,
        "selected_on": f"dev nDCG@10 at {select_dims} dims",
        "best": {"instruction": best.instruction, "doc_format": best.doc_format},
        "baseline": {"instruction": BASELINE[0], "doc_format": BASELINE[1]},
        "test_ndcg@10": {
            "baseline": pick(base_key, "test")["ndcg@10"],
            "best": pick(best_key, "test")["ndcg@10"],
        },
        "test_hit@5": {
            "baseline": pick(base_key, "test")["hit@5"],
            "best": pick(best_key, "test")["hit@5"],
        },
        "all_ndcg@10": {
            "baseline": pick(base_key, "all")["ndcg@10"],
            "best": pick(best_key, "all")["ndcg@10"],
        },
        "paired_bootstrap_test": boot,
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "n_commits": len(ids),
    }
    (args.out / "t3_verdict.json").write_text(json.dumps(verdict, indent=2))
    split_issues = {"dev": [numbers[i] for i in dev_idx], "test": [numbers[i] for i in test_idx]}
    (args.out / "t3_split.json").write_text(json.dumps(split_issues))

    lines = [f"# T3 tuning ({args.model})", "", "## Dev nDCG@10 at 512 dims (selection grid)", ""]
    lines.append(
        sel.pivot(index="instruction", columns="doc_format", values="ndcg@10")
        .round(3)
        .to_markdown()
    )
    test = df[(df.split == "test") & (df.dims == select_dims)]
    lines += ["", "## Test nDCG@10 at 512 dims (reported, not used for selection)", ""]
    lines.append(
        test.pivot(index="instruction", columns="doc_format", values="ndcg@10")
        .round(3)
        .to_markdown()
    )
    lines += ["", "## Doc formats", "", pd.DataFrame(format_stats).to_markdown(index=False)]
    lines += ["", "## Verdict", "", "```json", json.dumps(verdict, indent=2), "```", ""]
    (args.out / "t3_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
