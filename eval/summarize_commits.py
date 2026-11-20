"""Write the 27B's plain-English summary for EVERY commit in the index (resumable, parallel).

The pilot (eval/commit_summary_pilot.py) summarised 1,143 commits one at a time. This does all of
them and sends several requests at once: vLLM batches them on the GPU, so start the server with
CONCURRENCY = 8 (notebooks/serve_model_27b_on_colab.ipynb) and use --workers 8 here.

Summaries come only from the 27B. The 67 the 27B already wrote for the pilot are reused; the
1,076 Codex wrote are NOT (they are a different writer). Output: eval/commit_summary_pilot/
summaries_27b.jsonl, rows {"sha", "summary", "seconds"}. Same prompt and cleaner as the pilot.

    uv run python eval/summarize_commits.py prepare
    uv run python eval/summarize_commits.py run --url U --token T --workers 8
    uv run python eval/summarize_commits.py status
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import threading
import time
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import commit_summary_pilot as pilot  # noqa: E402
from commit_summary_pilot import (  # noqa: E402
    BUG_CHARS,
    REPO,
    SUMMARIES,
    WORK,
    clean_summary,
    read_jsonl,
    summary_messages,
    with_summary,
)

DEFAULT_WORK = pilot.WORK
PILOT_SUMMARIES = (
    SUMMARIES  # the 67 the 27B wrote in the psf/requests pilot (none for another repo)
)
QUERIES_FILE = pilot.QUERIES
ALL_COMMITS = WORK / "all_commits.jsonl"
SUMMARIES_27B = WORK / "summaries_27b.jsonl"
MERGED_SUMMARIES = WORK / "summaries_27b_all.jsonl"
ALL_PACKAGE = WORK / "all_input.zip"
RESULT_ALL = WORK / "result_all.json"
EXPERIMENT_DIR = ROOT / "data" / "exp_summaries"
DATA_DIR: Path | None = None


def configure(repo: str, data_dir: Path | None = None) -> None:
    """Work on another repository: its own folder under eval/commit_summary_pilot/, its own
    bug-report queries for the Colab step, and its own summary index folder. psf/requests keeps
    every original path."""
    global REPO, WORK, SUMMARIES, QUERIES_FILE, ALL_COMMITS, SUMMARIES_27B
    global MERGED_SUMMARIES, ALL_PACKAGE, RESULT_ALL, EXPERIMENT_DIR, DATA_DIR
    DATA_DIR = data_dir
    if repo == "psf/requests":
        REPO = "psf/requests"
        WORK = DEFAULT_WORK
        SUMMARIES = PILOT_SUMMARIES
        QUERIES_FILE = pilot.QUERIES
        ALL_COMMITS = WORK / "all_commits.jsonl"
        SUMMARIES_27B = WORK / "summaries_27b.jsonl"
        MERGED_SUMMARIES = WORK / "summaries_27b_all.jsonl"
        ALL_PACKAGE = WORK / "all_input.zip"
        RESULT_ALL = WORK / "result_all.json"
        EXPERIMENT_DIR = ROOT / "data" / "exp_summaries"
        return
    slug = repo.replace("/", "__")
    REPO = repo
    WORK = DEFAULT_WORK / slug
    SUMMARIES = WORK / "pilot_summaries_none.jsonl"  # there was no pilot for this repository
    QUERIES_FILE = WORK / "queries.jsonl"
    ALL_COMMITS = WORK / "all_commits.jsonl"
    SUMMARIES_27B = WORK / "summaries_27b.jsonl"
    MERGED_SUMMARIES = WORK / "summaries_27b_all.jsonl"
    ALL_PACKAGE = WORK / "all_input.zip"
    RESULT_ALL = WORK / "result_all.json"
    EXPERIMENT_DIR = ROOT / "data" / f"exp_summaries_{slug}"


def write_queries(repo: str) -> int:
    """The repository's bug reports (test and dev pairs) in the shape the Colab step reads, so
    the notebook can print a first dense-search answer. Gold SHAs absent from the index are cut."""
    path = ROOT / "eval/datasets/phase5_bugfix_v2/repos" / repo.replace("/", "__") / "pairs.jsonl"
    indexed = {c["sha"][:12] for c in read_jsonl(ALL_COMMITS)}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        gold = p["gold"]
        shas = ast.literal_eval(gold) if isinstance(gold, str) else gold
        present = [g[:12] for g in shas if g[:12] in indexed]
        if present:
            rows.append({"id": p["id"], "split": p["split"], "query": p["query"][:BUG_CHARS],
                         "gold": present})  # fmt: skip
    pilot.write_jsonl(QUERIES_FILE, rows)
    return len(rows)


def finished_27b(pilot_rows: Sequence[dict], own_rows: Sequence[dict]) -> dict[str, dict]:
    """sha -> row for every commit the 27B has summarised: its own file plus the pilot's 67.

    Codex rows (source == "codex") and rows without a summary are never counted."""
    done: dict[str, dict] = {}
    for row in [*pilot_rows, *own_rows]:
        if row.get("summary") and row.get("source") != "codex":
            done[row["sha"]] = row
    return done


def pending(commits: Sequence[dict], done: dict[str, dict]) -> list[dict]:
    return [c for c in commits if c["sha"] not in done]


def prepare() -> None:
    from repo_rag.config import Settings
    from repo_rag.store.vectorstore import VectorStore

    settings = Settings(data_dir=DATA_DIR) if DATA_DIR is not None else Settings()
    store = VectorStore(settings, _NoEmbedding())  # type: ignore[arg-type]
    docs = sorted(
        (
            {"sha": str(d.metadata["commit_sha"]), "text": d.page_content}
            for d in store.repo_documents("commits", REPO)
        ),
        key=lambda d: d["sha"],
    )
    if not docs:
        raise SystemExit(
            f"no indexed commits for {REPO} in {settings.data_dir}; "
            "pass --data-dir or set REPO_RAG_DATA_DIR"
        )
    WORK.mkdir(parents=True, exist_ok=True)
    ALL_COMMITS.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    print(f"wrote {ALL_COMMITS} ({len(docs)} commits)")
    if REPO != "psf/requests":
        print(f"wrote {QUERIES_FILE} ({write_queries(REPO)} bug reports with an indexed gold)")


def status() -> None:
    commits = read_jsonl(ALL_COMMITS)
    done = finished_27b(read_jsonl(SUMMARIES), read_jsonl(SUMMARIES_27B))
    left = pending(commits, done)
    print(f"{len(commits)} commits, {len(done)} done by the 27B, {len(left)} to do")


def run(url: str, token: str, workers: int, limit: int | None) -> None:
    from repo_rag.agent.remote_llm import RemoteChatModel

    commits = read_jsonl(ALL_COMMITS)
    if not commits:
        raise SystemExit("run `prepare` first")
    print("server:", RemoteChatModel(url, token, timeout=300.0).health())
    done = finished_27b(read_jsonl(SUMMARIES), read_jsonl(SUMMARIES_27B))
    todo = pending(commits, done)[:limit]
    print(f"{len(done)} done, {len(todo)} to do, {workers} at once")
    local = threading.local()
    write_lock = threading.Lock()
    counter = {"n": 0, "failed": 0}
    started = time.perf_counter()

    def one(commit: dict) -> None:
        if not hasattr(local, "model"):
            local.model = RemoteChatModel(url, token, timeout=300.0)
        row: dict[str, Any] = {"sha": commit["sha"]}
        for attempt in (1, 2):
            try:
                generation = local.model.generate(summary_messages(commit["text"]))
                row = {
                    "sha": commit["sha"],
                    "summary": clean_summary(generation.text),
                    "seconds": round(generation.seconds, 2),
                }
                break
            except Exception as exc:  # retry once, then record the failure so a rerun retries it
                row["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                time.sleep(2 * attempt)
        with write_lock:
            with SUMMARIES_27B.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            counter["n"] += 1
            counter["failed"] += 0 if row.get("summary") else 1
            n = counter["n"]
            if n % 50 == 0 or n == len(todo):
                each = (time.perf_counter() - started) / n
                left = each * (len(todo) - n) / 60
                print(f"[{n}/{len(todo)}] {each:.1f} s each overall, ~{left:.0f} min left, "
                      f"{counter['failed']} failed")  # fmt: skip

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    print(f"finished; {counter['failed']} failed (rerun to retry them)")


def merged_summaries() -> dict[str, str]:
    done = finished_27b(read_jsonl(SUMMARIES), read_jsonl(SUMMARIES_27B))
    return {sha: row["summary"] for sha, row in done.items()}


def package_all() -> None:
    """Zip all commits with their 27B summaries, in the shape the pilot's Colab notebook reads."""
    commits = read_jsonl(ALL_COMMITS)
    summaries = merged_summaries()
    missing = [c["sha"][:12] for c in commits if c["sha"] not in summaries]
    if missing:
        raise SystemExit(f"{len(missing)} commits have no 27B summary yet, e.g. {missing[:3]}")
    rows = [{**c, "summary": summaries[c["sha"]]} for c in commits]
    with zipfile.ZipFile(ALL_PACKAGE, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("commits.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n")
        archive.write(QUERIES_FILE, "queries.jsonl")
    print(f"wrote {ALL_PACKAGE} ({ALL_PACKAGE.stat().st_size / 1e6:.1f} MB, {len(rows)} commits)")


def score_all(vectors: Path) -> None:
    """The pilot's scoring, on the full commit corpus and the 27B's summaries only."""
    merged = [{"sha": sha, "summary": s} for sha, s in merged_summaries().items()]
    pilot.write_jsonl(MERGED_SUMMARIES, merged)
    saved = (pilot.COMMITS, pilot.QUERIES, pilot.SUMMARIES, pilot.RESULT)
    pilot.COMMITS, pilot.QUERIES, pilot.SUMMARIES, pilot.RESULT = (
        ALL_COMMITS,
        QUERIES_FILE,
        MERGED_SUMMARIES,
        RESULT_ALL,
    )
    try:
        pilot.score(vectors)
    finally:
        pilot.COMMITS, pilot.QUERIES, pilot.SUMMARIES, pilot.RESULT = saved


class _NoEmbedding:
    """Stands in for the embedder: building the experiment index re-embeds nothing."""

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("the experiment index is built from stored vectors")

    embed_documents = embed_query


def build_index(vectors_path: Path, out: Path, force: bool) -> None:
    """Copy the data folder to `out`, then swap in summary-bearing commit text and vectors there.

    The real data folder is only read. Run the agent against `out` with
    REPO_RAG_DATA_DIR=<out> to test the summaries without changing the shipped index."""
    import shutil

    import numpy as np
    from langchain_core.documents import Document

    from repo_rag.config import Settings
    from repo_rag.store.bm25 import Bm25Index, bm25_path
    from repo_rag.store.vectorstore import VectorStore

    source = (DATA_DIR or Settings().data_dir).resolve()
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force to rebuild it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("chroma", "bm25", "repos", "parser_cache", "github_cache"):
        if (source / name).exists():
            shutil.copytree(source / name, out / name)
    for name in ("manifest.json", "metadata.db"):
        shutil.copy2(source / name, out / name)

    data = np.load(vectors_path, allow_pickle=False)
    new_vectors = dict(zip((str(s) for s in data["shas"]), data["vectors_summary"], strict=True))
    summaries = merged_summaries()
    settings = Settings(data_dir=out)
    store = VectorStore(settings, _NoEmbedding())  # type: ignore[arg-type]
    ids, texts, metas, vectors = store.export_records("commits", REPO)
    shas = [str(m["commit_sha"]) for m in metas]
    missing = [s[:12] for s in shas if s not in new_vectors or s not in summaries]
    if missing:
        raise SystemExit(f"{len(missing)} commits lack a vector or summary, e.g. {missing[:3]}")
    new_texts = [with_summary(t, summaries[s]) for t, s in zip(texts, shas, strict=True)]
    stacked = np.asarray([new_vectors[s] for s in shas], dtype=np.float32)
    store.replace_records("commits", REPO, ids, new_texts, metas, stacked)
    docs = [Document(page_content=t, metadata=m) for t, m in zip(new_texts, metas, strict=True)]
    Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, REPO.replace("/", "__"), "commits"))
    changed = sum(1 for a, b in zip(texts, new_texts, strict=True) if a != b)
    print(f"{out}: {len(ids)} commits, {changed} now carry a Summary line "
          f"(mean vector change {float(np.mean(np.linalg.norm(stacked - vectors, axis=1))):.3f})")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="psf/requests", help="owner/name (default psf/requests)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="folder with this repository's index (default: REPO_RAG_DATA_DIR / .env)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    sub.add_parser("package")
    sc = sub.add_parser("score")
    sc.add_argument("--vectors", type=Path, required=True)
    bi = sub.add_parser("build-index")
    bi.add_argument("--vectors", type=Path, required=True)
    bi.add_argument("--out", type=Path, default=None, help="default data/exp_summaries[_repo]")
    bi.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--token", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, help="summarise at most this many (to try it out)")
    args = parser.parse_args()
    configure(args.repo, args.data_dir)
    if args.command == "prepare":
        prepare()
    elif args.command == "status":
        status()
    elif args.command == "package":
        package_all()
    elif args.command == "score":
        score_all(args.vectors)
    elif args.command == "build-index":
        build_index(args.vectors, args.out or EXPERIMENT_DIR, args.force)
    else:
        run(args.url, args.token, args.workers, args.limit)


if __name__ == "__main__":
    main()
