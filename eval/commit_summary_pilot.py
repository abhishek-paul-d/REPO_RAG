"""Pilot: do plain-English commit summaries make the right commit easier to find?

Many psf/requests commits have a bare message ("Fixed #174", issue references are stripped from
the index), so a bug report written in English matches their diff poorly. This pilot adds a short
summary written by the answer model from the commit's own message and diff, and checks whether the
gold fix commit ranks higher. It is a small, cheap test run before summarising all 4,883 commits.

Steps (laptop unless noted):
    1. prepare    pick the 141-pair gold commits plus N random others from the local commit index
                  and write eval/commit_summary_pilot/{commits,queries}.jsonl
    2. summarize  ask the running 27B server for one summary per pilot commit (resumable; the server
                  handles one request at a time, so ~4 s each)
    3. package    zip commits + summaries + queries for the Colab notebook
    4. (Colab)    notebooks/commit_summary_pilot_on_colab.ipynb embeds both versions on a GPU and
                  returns pilot_vectors.npz
    5. score      rank the gold commit with and without the summary inside the pilot corpus

Leakage rule: the summariser sees only the commit's own indexed text (message and diff, issue
references already stripped). It never sees a bug report, an issue or a PR. Gold and random commits
get exactly the same treatment, so the summaries cannot favour the answers.

Ranks are within a ~1,100-commit corpus, not the full 4,883, so they show direction, not the final
score. Any gain must be re-measured through the agent before it counts.

    uv run python eval/commit_summary_pilot.py prepare --random 1000
    uv run python eval/commit_summary_pilot.py summarize --url U --token T
    uv run python eval/commit_summary_pilot.py package
    uv run python eval/commit_summary_pilot.py score --vectors path/to/pilot_vectors.npz
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPO = "psf/requests"
PAIRS = ROOT / "eval/datasets/phase5_bugfix_v2/repos/psf__requests/pairs.jsonl"
WORK = ROOT / "eval/commit_summary_pilot"
COMMITS = WORK / "commits.jsonl"
QUERIES = WORK / "queries.jsonl"
SUMMARIES = WORK / "summaries.jsonl"
PACKAGE = WORK / "pilot_input.zip"
RESULT = WORK / "result.json"

BUG_CHARS = 1200  # same cut as eval/agent_eval.py and eval/commit_retrieval_check.py
SUMMARY_INPUT_CHARS = 6000  # the commit text shown to the summariser (about 2,000 tokens)
SUMMARY_MAX_CHARS = 700  # a summary is cut to this before it is added to the indexed text
SEED = 20261118

SYSTEM = (
    "You write short, plain-English descriptions of git commits for a search index. "
    "Use only the commit shown. Never invent details that are not in it."
)
INSTRUCTION = (
    "Describe this commit in two or three plain sentences, at most 60 words. Say what behaviour "
    "or problem it changes (the symptom a user would notice, if it fixes a bug), and name the "
    "main function, class or module. Do not quote code or diff lines and do not mention issue "
    "numbers. Reply with the description only.\n\nCOMMIT\n"
)


# --- pure helpers (tested offline) -------------------------------------------------------


def gold_prefixes(pair: dict) -> list[str]:
    gold = pair["gold"]
    shas = ast.literal_eval(gold) if isinstance(gold, str) else gold
    return [s[:12] for s in shas]


def select_pilot(
    docs: Sequence[dict], golds: set[str], n_random: int, seed: int = SEED
) -> list[dict]:
    """Every gold commit present in `docs`, plus `n_random` others drawn with a fixed seed.

    Each doc is {"sha", "text"}. The output is sorted by sha so the order never depends on how the
    store happened to list them."""
    gold_docs = [d for d in docs if d["sha"][:12] in golds]
    others = sorted((d for d in docs if d["sha"][:12] not in golds), key=lambda d: d["sha"])
    picked = random.Random(seed).sample(others, min(n_random, len(others)))
    return sorted(gold_docs + picked, key=lambda d: d["sha"])


def summary_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTION + text[:SUMMARY_INPUT_CHARS]},
    ]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_summary(raw: str) -> str:
    """One line, no thinking block, no leading label, cut to SUMMARY_MAX_CHARS."""
    text = _THINK.sub("", raw).strip()
    text = re.sub(r"^(summary|description)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())[:SUMMARY_MAX_CHARS]


def with_summary(text: str, summary: str) -> str:
    """The indexed commit text with a `Summary:` line before `Files:` (after the message).

    It goes before the diff on purpose: the embedder truncates long inputs from the end, so a
    summary placed first is never cut off."""
    if not summary:
        return text
    line = f"Summary: {summary}"
    if "\nFiles: " in text:
        return text.replace("\nFiles: ", f"\n{line}\nFiles: ", 1)
    head, sep, rest = text.partition("\n")
    return f"{head}\n{line}{sep}{rest}" if sep else f"{head}\n{line}"


def ranks_for(order: Sequence[int], shas: Sequence[str], golds: Sequence[str]) -> int | None:
    """1-based rank of the first gold commit in a best-first list of document indexes."""
    wanted = set(golds)
    for position, index in enumerate(order, start=1):
        if shas[index][:12] in wanted:
            return position
    return None


def summarize_ranks(ranks: Sequence[int | None]) -> dict[str, Any]:
    n = len(ranks)

    def hit(k: int) -> int:
        return sum(1 for r in ranks if r is not None and r <= k)

    mrr = sum(1 / r for r in ranks if r is not None) / n if n else 0.0
    return {
        "n": n,
        "hit@1": hit(1),
        "hit@5": hit(5),
        "hit@10": hit(10),
        "hit@20": hit(20),
        "missing": sum(1 for r in ranks if r is None),
        "mrr": round(mrr, 3),
    }


def compare_ranks(before: Sequence[int | None], after: Sequence[int | None]) -> dict[str, int]:
    """Pairwise: how many gold commits moved up, down or stayed (None counts as worst)."""
    worst = 10**9
    up = down = same = 0
    for b, a in zip(before, after, strict=True):
        b, a = b or worst, a or worst
        if a < b:
            up += 1
        elif a > b:
            down += 1
        else:
            same += 1
    return {"better": up, "worse": down, "same": same}


def dense_order(doc_vectors: np.ndarray, query_vector: np.ndarray) -> list[int]:
    """Document indexes by cosine similarity (vectors are unit length), best first."""
    return list(np.argsort(-(doc_vectors @ query_vector), kind="stable"))


def hybrid_order(
    dense: Sequence[int], shas: Sequence[str], bm25_ids: Sequence[str], weights: dict, rrf_k: int
) -> list[int]:
    """Weighted RRF of the dense ranking and a BM25 ranking, the fusion the agent uses."""
    from repo_rag.retrieval.retrievers import weighted_rrf

    index_of = {sha: i for i, sha in enumerate(shas)}
    fused = weighted_rrf(
        {"dense": [shas[i] for i in dense], "bm25": list(bm25_ids)}, weights, rrf_k
    )
    ordered = [index_of[sha] for sha, _ in fused]
    seen = set(ordered)
    return ordered + [i for i in dense if i not in seen]


# --- commands -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def pairs() -> list[dict]:
    return [json.loads(x) for x in PAIRS.read_text(encoding="utf-8").splitlines()]


def prepare(n_random: int) -> None:
    from repo_rag.config import Settings
    from repo_rag.store.embeddings import get_embeddings
    from repo_rag.store.vectorstore import VectorStore

    settings = Settings()
    store = VectorStore(settings, get_embeddings(settings))
    docs = [
        {"sha": str(d.metadata["commit_sha"]), "text": d.page_content}
        for d in store.repo_documents("commits", REPO)
    ]
    all_pairs = pairs()
    golds = {g for p in all_pairs for g in gold_prefixes(p)}
    chosen = select_pilot(docs, golds, n_random)
    present = {d["sha"][:12] for d in chosen}
    usable = [p for p in all_pairs if any(g in present for g in gold_prefixes(p))]
    write_jsonl(COMMITS, chosen)
    write_jsonl(
        QUERIES,
        [
            {
                "id": p["id"],
                "split": p["split"],
                "query": p["query"][:BUG_CHARS],
                "gold": [g for g in gold_prefixes(p) if g in present],
            }
            for p in usable
        ],
    )
    print(
        f"commit index: {len(docs)} commits; gold shas wanted {len(golds)}, "
        f"found {len(golds & present)}\npilot corpus: {len(chosen)} commits; "
        f"pairs usable {len(usable)} of {len(all_pairs)}\nwrote {COMMITS} and {QUERIES}"
    )


def summarize(url: str, token: str, limit: int | None) -> None:
    from repo_rag.agent.remote_llm import RemoteChatModel

    model = RemoteChatModel(url, token, timeout=300.0)
    print("server:", model.health())
    commits = read_jsonl(COMMITS)
    if not commits:
        raise SystemExit("run `prepare` first")
    done = {r["sha"] for r in read_jsonl(SUMMARIES) if r.get("summary")}
    todo = [c for c in commits if c["sha"] not in done][:limit]
    print(f"{len(done)} done, {len(todo)} to do")
    started = time.perf_counter()
    for number, commit in enumerate(todo, start=1):
        row: dict[str, Any] = {"sha": commit["sha"]}
        for attempt in (1, 2):
            try:
                generation = model.generate(summary_messages(commit["text"]))
                row["summary"] = clean_summary(generation.text)
                row["seconds"] = round(generation.seconds, 2)
                break
            except Exception as exc:  # a budget or server error: retry once, then record it
                row["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                time.sleep(2 * attempt)
        with SUMMARIES.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if number % 25 == 0 or number == len(todo):
            rate = (time.perf_counter() - started) / number
            print(
                f"[{number}/{len(todo)}] {rate:.1f} s each, ~{rate * (len(todo) - number) / 60:.0f} min left"
            )


def package() -> None:
    summaries = {r["sha"]: r["summary"] for r in read_jsonl(SUMMARIES) if r.get("summary")}
    commits = read_jsonl(COMMITS)
    missing = [c["sha"][:12] for c in commits if c["sha"] not in summaries]
    if missing:
        raise SystemExit(f"{len(missing)} commits have no summary yet, e.g. {missing[:3]}")
    rows = [{**c, "summary": summaries[c["sha"]]} for c in commits]
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("commits.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n")
        archive.write(QUERIES, "queries.jsonl")
    print(f"wrote {PACKAGE} ({PACKAGE.stat().st_size / 1e6:.1f} MB, {len(rows)} commits)")


def score(vectors_path: Path) -> None:
    from langchain_core.documents import Document

    from repo_rag.config import Settings
    from repo_rag.store.bm25 import Bm25Index

    data = np.load(vectors_path, allow_pickle=False)
    shas = [str(s) for s in data["shas"]]
    queries = read_jsonl(QUERIES)
    commits = {c["sha"]: c["text"] for c in read_jsonl(COMMITS)}
    summaries = {r["sha"]: r["summary"] for r in read_jsonl(SUMMARIES) if r.get("summary")}
    settings = Settings()
    weights = {"dense": settings.dense_weight, "bm25": settings.bm25_weight}
    texts = {
        "baseline": [commits[s] for s in shas],
        "summary": [with_summary(commits[s], summaries.get(s, "")) for s in shas],
    }
    results: dict[str, Any] = {}
    per_pair: dict[str, dict[str, list[int | None]]] = {}
    for variant in ("baseline", "summary"):
        docs = [
            Document(page_content=t, metadata={"chunk_id": s})
            for t, s in zip(texts[variant], shas, strict=True)
        ]
        index = Bm25Index.build(docs)
        vectors = data[f"vectors_{variant}"]
        dense_ranks: list[int | None] = []
        hybrid_ranks: list[int | None] = []
        for q, qv in zip(queries, data["queries"], strict=True):
            order = dense_order(vectors, qv)
            dense_ranks.append(ranks_for(order, shas, q["gold"]))
            bm25 = [cid for cid, _ in index.search(q["query"], len(shas))]
            fused = hybrid_order(order, shas, bm25, weights, settings.rrf_k)
            hybrid_ranks.append(ranks_for(fused, shas, q["gold"]))
        per_pair[variant] = {"dense": dense_ranks, "hybrid": hybrid_ranks}
        results[variant] = {
            "dense": summarize_ranks(dense_ranks),
            "hybrid": summarize_ranks(hybrid_ranks),
        }
    for mode in ("dense", "hybrid"):
        results[f"change_{mode}"] = compare_ranks(
            per_pair["baseline"][mode], per_pair["summary"][mode]
        )
    results["corpus"] = len(shas)
    results["pairs"] = len(queries)
    results["summaries"] = len(summaries)
    RESULT.write_text(
        json.dumps(
            {"summary": results, "ranks": per_pair, "ids": [q["id"] for q in queries]}, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print(f"wrote {RESULT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--random", type=int, default=1000, help="random non-gold commits to add")
    s = sub.add_parser("summarize")
    s.add_argument("--url", required=True)
    s.add_argument("--token", required=True)
    s.add_argument("--limit", type=int, help="summarise at most this many more (to try it out)")
    sub.add_parser("package")
    c = sub.add_parser("score")
    c.add_argument("--vectors", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.random)
    elif args.command == "summarize":
        summarize(args.url, args.token, args.limit)
    elif args.command == "package":
        package()
    else:
        score(args.vectors)


if __name__ == "__main__":
    main()
