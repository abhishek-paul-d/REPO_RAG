"""T3 rerank experiment: can a reranker or a bigger embedder close the bug -> fix gap?

After T3 tuning, Qwen3-Embedding-0.6B still trails nomic-embed-code (7B) by ~0.1 nDCG@10 on
bug report -> fix commit. Two cheaper-to-deploy levers are tested here:

  * first stage: Qwen3-Embedding-0.6B @512d (production) vs Qwen3-Embedding-4B (@512d, full)
  * second stage: rerank each retriever's top-K with Qwen3-Reranker-0.6B / -4B or
    bge-reranker-v2-m3 (the reranker currently named in config.py)

Nothing is tuned on the data, so results are reported on all 144 issues as well as on the
same held-out test half used by t3_tuning.py, each with a paired bootstrap CI against the
production baseline. Rerank latency per query is measured too: a reranker too slow for chat
doesn't help.

Run: python t3_rerank.py --data-dir ./bench_data --out ./rerank_results   (needs GITHUB_TOKEN)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import bench
import numpy as np
import t3_tuning

DOC_FORMAT = "symbols_diff800"  # adopted after T3 tuning
QUERY_INSTRUCTION = bench.CODE_QUERY_INSTRUCTION  # adopted after T3 tuning
RERANK_INSTRUCTION = "Given a bug report, judge whether this commit's changes fix the reported bug"
TOP_K = 50
MAX_QUERY_CHARS = 2000


@dataclass(frozen=True)
class Retriever:
    key: str
    model: str  # bench.MODELS key
    dims: int | None  # None = full dimension


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    hf_id: str
    kind: str  # "qwen" (yes/no logits of a causal LM) or "cross" (sentence-transformers)
    batch_size: int
    max_length: int = 2048


RETRIEVERS = [
    Retriever("qwen3-0.6b@512", "qwen3-0.6b", 512),  # production baseline
    Retriever("qwen3-4b@512", "qwen3-4b", 512),
    Retriever("qwen3-4b@full", "qwen3-4b", None),
]
RETRIEVER_CHOICES = {
    r.key: r
    for r in [
        *RETRIEVERS,
        Retriever("nomic-embed-code", "nomic-embed-code", None),  # optional 7B reference
        Retriever("minilm-smoke", "minilm-smoke", None),  # CPU smoke tests only
    ]
}
BASELINE = ("qwen3-0.6b@512", "none")

RERANKERS = {
    s.key: s
    for s in [
        RerankerSpec("qwen3-reranker-0.6b", "Qwen/Qwen3-Reranker-0.6B", "qwen", batch_size=16),
        RerankerSpec("qwen3-reranker-4b", "Qwen/Qwen3-Reranker-4B", "qwen", batch_size=4),
        RerankerSpec("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3", "cross", 32, max_length=1024),
        # Small models for CPU smoke tests only.
        RerankerSpec("minilm-cross-smoke", "cross-encoder/ms-marco-MiniLM-L6-v2", "cross", 32, 512),
    ]
}
DEFAULT_RERANKERS = ["qwen3-reranker-0.6b", "qwen3-reranker-4b", "bge-reranker-v2-m3"]

# Verbatim from the Qwen3-Reranker model card.
_QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenReranker:
    """Qwen3-Reranker: P("yes") from the last-token logits, as in the model card."""

    def __init__(self, spec: RerankerSpec, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(spec.hf_id, padding_side="left")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=dtype).to(device)
        self.model.eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(_QWEN_PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(_QWEN_SUFFIX, add_special_tokens=False)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        out: list[float] = []
        budget = self.spec.max_length - len(self.prefix) - len(self.suffix)
        for i in range(0, len(pairs), self.spec.batch_size):
            texts = [
                f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"
                for q, d in pairs[i : i + self.spec.batch_size]
            ]
            enc = self.tok(
                texts,
                padding=False,
                truncation="longest_first",
                max_length=budget,
                return_attention_mask=False,
            )
            ids = [self.prefix + x + self.suffix for x in enc["input_ids"]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out += torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()
        return out


class CrossReranker:
    def __init__(self, spec: RerankerSpec, device: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, object] = {"max_length": spec.max_length, "device": device}
        if device == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        self.spec = spec
        self.model = CrossEncoder(spec.hf_id, **kwargs)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(pairs, batch_size=self.spec.batch_size, show_progress_bar=False)
        return [float(s) for s in np.asarray(scores, dtype=np.float32).ravel()]


def load_reranker(spec: RerankerSpec, device: str) -> QwenReranker | CrossReranker:
    return QwenReranker(spec, device) if spec.kind == "qwen" else CrossReranker(spec, device)


def _free(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def first_stage(
    retrievers: list[Retriever], queries: list[str], doc_texts: list[str], device: str, k: int
) -> tuple[dict[str, list[list[int]]], dict[str, dict]]:
    """Top-k doc indices per query for each retriever. Each model is loaded once."""
    rankings: dict[str, list[list[int]]] = {}
    stats: dict[str, dict] = {}
    for model_key in dict.fromkeys(r.model for r in retrievers):
        spec = bench.MODELS[model_key]
        enc = bench.Encoder(spec, device)
        prompt = (
            QUERY_INSTRUCTION
            if spec.code_query_prompt.startswith("Instruct:")
            else spec.code_query_prompt
        )
        t0 = time.perf_counter()
        d_full = enc.encode(doc_texts)
        index_s = time.perf_counter() - t0
        q_full = enc.encode(queries, prompt, is_query=True)
        for r in (r for r in retrievers if r.model == model_key):
            q, d = q_full, d_full
            if r.dims and r.dims < d.shape[1]:
                q, d = bench.truncate_dims(q, r.dims), bench.truncate_dims(d, r.dims)
            rankings[r.key] = bench.dense_rankings(q, d, k=k)
            stats[r.key] = {
                "dim": int(d.shape[1]),
                "index_docs_per_s": round(len(doc_texts) / index_s, 1),
            }
            print(f"first stage {r.key}: {json.dumps(stats[r.key])}")
        del enc, d_full, q_full
        _free(device)
    return rankings, stats


def rerank(
    spec: RerankerSpec,
    candidates: dict[str, list[list[int]]],
    queries: list[str],
    doc_texts: list[str],
    device: str,
) -> tuple[dict[str, list[list[int]]], dict]:
    """Rerank every retriever's candidates. Pairs shared between retrievers are scored once."""
    pairs_needed = sorted(
        {(qi, di) for cand in candidates.values() for qi, docs in enumerate(cand) for di in docs}
    )
    model = load_reranker(spec, device)
    model.score([(queries[0][:MAX_QUERY_CHARS], doc_texts[0])] * 2)  # warm-up
    t0 = time.perf_counter()
    scores = model.score(
        [(queries[qi][:MAX_QUERY_CHARS], doc_texts[di]) for qi, di in pairs_needed]
    )
    secs = time.perf_counter() - t0
    by_pair = dict(zip(pairs_needed, scores, strict=True))

    # Latency for one interactive query: score one query's top-K, as production would.
    q_docs = candidates.get(BASELINE[0], next(iter(candidates.values())))[0]
    t1 = time.perf_counter()
    model.score([(queries[0][:MAX_QUERY_CHARS], doc_texts[di]) for di in q_docs])
    one_query_ms = (time.perf_counter() - t1) * 1000

    del model
    _free(device)
    reranked = {
        r_key: [
            sorted(docs, key=lambda di, qi=qi: by_pair[(qi, di)], reverse=True)
            for qi, docs in enumerate(cand)
        ]
        for r_key, cand in candidates.items()
    }
    timing = {
        "reranker": spec.key,
        "pairs_scored": len(pairs_needed),
        "pairs_per_s": round(len(pairs_needed) / secs, 1),
        f"ms_per_query_top{len(q_docs)}": round(one_query_ms, 1),
    }
    print(json.dumps(timing))
    return reranked, timing


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("bench_data"))
    parser.add_argument("--out", type=Path, default=Path("rerank_results"))
    parser.add_argument("--rerankers", nargs="+", default=DEFAULT_RERANKERS)
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=[r.key for r in RETRIEVERS],
        choices=list(RETRIEVER_CHOICES),
    )
    parser.add_argument(
        "--with-7b", action="store_true", help="add nomic-embed-code as a first stage"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
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
    ids, texts, links = bench.commit_corpus_formats(repo_dir, (DOC_FORMAT,))
    queries, gold, _ = bench.bug_queries(
        links, token, args.data_dir / "issues_cache.json", args.max_pairs
    )
    if args.max_commits:
        ids, texts = t3_tuning._limit_corpus(ids, texts, gold, args.max_commits)
    doc_texts = texts[DOC_FORMAT]
    _, test_idx = t3_tuning._split(len(queries))  # same split as t3_tuning.py
    print(
        f"device={device} | T3: {len(queries)} queries (test {len(test_idx)}) over {len(ids)} commits"
    )

    all_retrievers = [RETRIEVER_CHOICES[key] for key in args.retrievers]
    if args.with_7b and "nomic-embed-code" not in args.retrievers:
        all_retrievers.append(RETRIEVER_CHOICES["nomic-embed-code"])
    candidates, first_stats = first_stage(all_retrievers, queries, doc_texts, device, args.top_k)

    systems: dict[tuple[str, str], list[list[int]]] = {
        (r, "none"): c for r, c in candidates.items()
    }
    timings = []
    for key in args.rerankers:
        reranked, timing = rerank(RERANKERS[key], candidates, queries, doc_texts, device)
        timings.append(timing)
        systems.update({(r, key): ranks for r, ranks in reranked.items()})

    task = bench.Task("T3", queries, gold, ids, doc_texts, "commit")
    per_query = {
        s: [p["ndcg@10"] for p in bench.score(task, ranks)[1]] for s, ranks in systems.items()
    }
    base = per_query.get(BASELINE)

    import pandas as pd

    rows = []
    for (r_key, rr_key), ranks in systems.items():
        recall = np.mean(
            [bool(set(task.doc_ids[i] for i in rk) & g) for rk, g in zip(ranks, gold, strict=True)]
        )
        row = {
            "retriever": r_key,
            "reranker": rr_key,
            f"recall@{args.top_k}": round(float(recall), 4),
        }
        for split, idx in (("all", list(range(len(queries)))), ("test", test_idx)):
            agg, _ = bench.score(
                bench.Task(
                    "s", [queries[i] for i in idx], [gold[i] for i in idx], ids, doc_texts, "commit"
                ),
                [ranks[i] for i in idx],
            )
            row[f"{split}_ndcg@10"] = round(agg["ndcg@10"], 4)
            row[f"{split}_hit@5"] = round(agg["hit@5"], 4)
            if base is not None and (r_key, rr_key) != BASELINE:
                b = t3_tuning.paired_bootstrap(
                    [base[i] for i in idx], [per_query[(r_key, rr_key)][i] for i in idx]
                )
                row[f"{split}_gain"] = b["mean_gain"]
                row[f"{split}_ci95"] = f"[{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}]"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("all_ndcg@10", ascending=False)
    df.to_csv(args.out / "t3_rerank.csv", index=False)
    pd.DataFrame(timings).to_csv(args.out / "rerank_timings.csv", index=False)
    meta = {
        "first_stage": first_stats,
        "top_k": args.top_k,
        "doc_format": DOC_FORMAT,
        "n_queries": len(queries),
        "n_test": len(test_idx),
        "n_commits": len(ids),
        "reference_nomic_7b_run2_all_ndcg@10": 0.596,
    }
    (args.out / "rerank_meta.json").write_text(json.dumps(meta, indent=2))

    lines = [
        "# T3 rerank experiment",
        "",
        f"Baseline = {BASELINE[0]} without reranking. Reference: nomic-embed-code (7B) scored "
        "0.596 nDCG@10 on all 144 in run 2.",
        "",
        "## Retrieval quality (sorted by all-144 nDCG@10)",
        "",
        df.to_markdown(index=False),
        "",
        "## Reranker speed",
        "",
        pd.DataFrame(timings).to_markdown(index=False),
        "",
        "## First stage",
        "",
        "```json",
        json.dumps(first_stats, indent=2),
        "```",
        "",
    ]
    (args.out / "t3_rerank_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
