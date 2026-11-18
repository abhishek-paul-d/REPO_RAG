"""Frozen architecture E1/E2 evaluation through the production HybridRetriever.

uv run python eval/run_architecture_eval.py --mode hybrid --stage E1
uv run python eval/run_architecture_eval.py --mode hybrid --stage E2
BM25 is a labelled preflight only. No neural claims come from fake embeddings.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_architecture_dataset import OUT, REPOS, digest, verify_frozen, write
from eval.retrieval.statistics import bootstrap_intervals, paired_intervals
from repo_rag.config import Settings
from repo_rag.history.github_sync import GitHubSync
from repo_rag.history.metadata_db import MetadataDB, record_from_dict
from repo_rag.ingest.indexer import chunk_repo
from repo_rag.retrieval.retrievers import HybridRetriever, weighted_rrf
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.embeddings import get_embeddings
from repo_rag.store.vectorstore import VectorStore
from repo_rag.summaries.facts import history_ancestors
from repo_rag.summaries.repo_map import build_summaries

METRICS = ["recall@5", "recall@10", "hit@3", "mrr@10", "directory_hit@3"]
CODE_METRICS = ["hit@1", "hit@5", "mrr@10"]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score(ranked: list[str], gold: set[str]) -> dict:
    ranked = list(dict.fromkeys(ranked))
    first = next((i for i, path in enumerate(ranked, 1) if path in gold), None)
    dirs = list(dict.fromkeys(str(PurePosixPath(p).parent) for p in ranked))
    gold_dirs = {str(PurePosixPath(p).parent) for p in gold}
    return {
        **{f"recall@{k}": len(set(ranked[:k]) & gold) / len(gold) for k in (5, 10)},
        **{f"hit@{k}": float(first is not None and first <= k) for k in (1, 3, 5)},
        "mrr@10": 1 / first if first and first <= 10 else 0.0,
        "directory_hit@3": float(bool(set(dirs[:3]) & gold_dirs)),
    }


def collapse(hits) -> list[str]:
    return list(dict.fromkeys(str(h.document.metadata["path"]) for h in hits))


def rank_arms(retriever, slug: str, question: str, protocol: dict) -> dict[str, list[str]]:
    depth = protocol["depth"]
    code = collapse(retriever.search(slug, question, kind="code", k=depth))
    files = collapse(
        retriever.search(
            slug, question, kind="summaries", k=depth, filters={"symbol_kind": "file_summary"}
        )
    )
    directories = collapse(
        retriever.search(
            slug, question, kind="summaries", k=depth, filters={"symbol_kind": "dir_summary"}
        )
    )
    result = {
        "chunks": code,
        "summaries": files,
        "fused": [
            p for p, _ in weighted_rrf({"c": code, "s": files}, {}, retriever.settings.rrf_k)
        ],
    }
    for kind, ranking in (("file", files), ("dir", directories)):
        for n in (1, 2, 3):
            result[f"router_{kind}_{n}"] = collapse(
                retriever.search(slug, question, kind="code", k=depth, paths=ranking[:n])
            )
    return result


def select(dev: list[dict], arms: list[str]) -> str:
    """Only dev rows are accepted, making accidental test-driven selection fail closed."""
    if not dev or any(row["split"] != "dev" for row in dev):
        raise ValueError("selection requires dev rows only")

    def key(arm):
        sample = [r for r in dev if r["method"] == arm]
        if not sample:
            raise ValueError(f"Missing dev arm {arm}")
        return (
            np.mean([r["recall@5"] for r in sample]),
            np.mean([r["mrr@10"] for r in sample]),
            -arms.index(arm),
        )

    return max(arms, key=key)


class DiskEmbeddings(Embeddings):
    """Cache by text, query/document mode and full embedding settings; no duplicate work."""

    def __init__(self, directory: Path, settings: Settings, inner: Embeddings):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "embeddings.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY, value BLOB)")
        self.inner = inner
        self.batch_size = settings.eval_summary_embedding_batch_size
        self.identity = json.dumps(
            {
                k: v
                for k, v in settings.model_dump(mode="json").items()
                if k.startswith(("embedding_", "local_embedding_"))
            },
            sort_keys=True,
        )

    def key(self, text: str, query: bool) -> str:
        return hashlib.sha256(f"{self.identity}\0{query}\0{text}".encode()).hexdigest()

    def seed(self, texts: list[str], vectors: np.ndarray) -> None:
        self.db.executemany(
            "INSERT OR IGNORE INTO vectors VALUES(?,?)",
            [
                (self.key(t, False), np.asarray(v, dtype=np.float32).tobytes())
                for t, v in zip(texts, vectors, strict=True)
            ],
        )
        self.db.commit()

    def _encode(self, texts: list[str], query: bool) -> list[list[float]]:
        keys = [self.key(text, query) for text in texts]
        found: dict[str, list[float]] = {}
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            row = self.db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
            if row:
                found[key] = np.frombuffer(row[0], dtype=np.float32).tolist()
            else:
                missing[key] = text
        pending = list(missing)
        batch_size = self.batch_size if not query else 1
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            print(
                f"Embedding {'queries' if query else 'documents'} {start + 1}-{start + len(batch)}/{len(pending)}",
                flush=True,
            )
            vectors = (
                [self.inner.embed_query(missing[key]) for key in batch]
                if query
                else self.inner.embed_documents([missing[key] for key in batch])
            )
            for key, vector in zip(batch, vectors, strict=True):
                self.db.execute(
                    "INSERT INTO vectors VALUES(?,?)",
                    (key, np.asarray(vector, dtype=np.float32).tobytes()),
                )
                found[key] = vector
            self.db.commit()
        return [found[key] for key in keys]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, False)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], True)[0]


class NoEmbeddings(Embeddings):
    """BM25 preflight uses constant vectors with dense weight zero; no model download."""

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class Bm25Store(VectorStore):
    def search(self, kind, repo, query, n, filters=None):
        return []


def prepare_history(settings: Settings) -> dict:
    """Load existing frozen history and PR snapshots into a scratch metadata.db only."""
    coverage = {}
    with MetadataDB(settings.metadata_db_path, readonly=False) as db:
        for name, (slug, location, _) in REPOS.items():
            source = (
                ROOT
                / "eval/datasets/phase5_bugfix_v2/repos"
                / slug.replace("/", "__")
                / "records.jsonl.gz"
            )
            records = [
                record_from_dict(json.loads(line))
                for line in gzip.decompress(source.read_bytes()).decode().split("\n")
                if line
            ]
            db.sync_history(slug, records, [], [])
            ancestors, ancestry_source = history_ancestors(slug, ROOT / location, settings)
            snapshot = ROOT / "eval/datasets" / f"{name}_pr_snapshot"
            prs = []
            if snapshot.exists():
                target = settings.github_cache_dir / "sync"
                target.mkdir(parents=True, exist_ok=True)
                for path in snapshot.glob("*.json"):
                    shutil.copyfile(path, target / path.name)
                sync = GitHubSync(settings, None)  # type: ignore[arg-type]  # cache-only reads
                prs = sync.pull_requests(slug)
                db.sync_github(
                    slug, prs, sync.issues(slug), prs_complete=False, issues_complete=False
                )
            coverage[slug] = {
                "indexed_commits": len(records),
                "synced_prs": len(prs),
                "ancestry_checkout": ancestry_source.relative_to(ROOT).as_posix(),
                "checkout_ancestors": len(ancestors),
                "indexed_ancestor_commits": sum(r.sha in ancestors for r in records),
                "history_source": source.relative_to(ROOT).as_posix(),
                "history_sha256": digest(source),
                "pr_source": snapshot.relative_to(ROOT).as_posix() if snapshot.exists() else None,
                "limitation": "bounded frozen history; PR file links require exact merge SHA in history; ancestors of checkout only",
            }
    return coverage


def prepare_code(
    slug: str,
    name: str,
    checkout: Path,
    pin: str,
    settings: Settings,
    store: VectorStore,
    embeddings: Embeddings,
    mode: str,
) -> dict:
    code, _ = chunk_repo(slug, checkout, pin, settings)
    docs = code["code"]
    reused = 0
    if mode == "hybrid" and name == "requests":
        original = VectorStore(
            Settings(data_dir=ROOT / "data", embedding_dtype="float32"), embeddings
        )
        ids, texts, metas, vectors = original.export_records("code", slug)
        if {d.metadata["chunk_id"]: d.page_content for d in docs} != dict(
            zip(ids, texts, strict=True)
        ):
            raise ValueError("Requests code index differs from pinned production chunks")
        store.replace_records("code", slug, ids, texts, metas, vectors)
        reused = len(ids)
    elif mode == "hybrid" and isinstance(embeddings, DiskEmbeddings):
        old = ROOT / "data/phase4_followup_colab/pallets__flask"
        old_docs = (
            json.loads((old / "chunks.json").read_text(encoding="utf-8")) if old.exists() else []
        )
        if old_docs:
            run = json.loads((old / "run.json").read_text(encoding="utf-8"))
            assert run["commit_sha"] == pin and run["matrix"]["embedding_dtype"] == "float32"
            legacy = run["settings"]
            identity = {
                "texts": [d["text"] for d in old_docs],
                "model": legacy["local_embedding_model"],
                "provider": legacy["embedding_provider"],
                "dims": legacy["embedding_dims"],
                "instruction": None,
                "tokens": legacy["embedding_max_tokens"],
                "dtype": legacy["embedding_dtype"],
                "device": legacy["compute_device"],
                "implementation": "LocalEmbeddings",
                "embedding_source": run["source_sha256"]["src/repo_rag/store/embeddings.py"],
            }
            cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            vectors = np.load(old / "vector_cache" / f"{cache_key}.npy", allow_pickle=False)
            if (
                vectors.shape != (len(old_docs), settings.embedding_dims)
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("Invalid Flask vector cache")
            expected = {d.metadata["chunk_id"]: d.page_content for d in docs}
            if expected != {d["metadata"]["chunk_id"]: d["text"] for d in old_docs}:
                raise ValueError("Flask cached chunks differ from pinned production chunks")
            embeddings.seed([d["text"] for d in old_docs], vectors)
            reused = len(old_docs)
        store.sync("code", slug, docs)
    else:
        store.sync("code", slug, docs)
    Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, "code"))
    return {"chunks": len(docs), "reused_vectors": reused}


def evaluate(
    retriever, questions: list[dict], protocol: dict, variant: str, arms: list[str]
) -> list[dict]:
    results = []
    for q in questions:
        ranking = rank_arms(retriever, q["repo"], q["question"], protocol)
        gold = set(q.get("gold_files") or [g["path"] for g in q["gold"]])
        for arm in arms:
            results.append(
                {
                    "id": q["id"],
                    "repo": q["repo"],
                    "group": q["group"],
                    "split": q["split"],
                    "method": arm,
                    "variant": variant,
                    "ranking": ranking[arm],
                    **score(ranking[arm], gold),
                }
            )
    return results


def report(all_rows: list[dict], baseline: list[dict], method: str, settings: Settings) -> dict:
    test = [r for r in all_rows if r["split"] == "test" and r["method"] == method]
    control = [r for r in baseline if r["split"] == "test" and r["method"] == "chunks"]
    repos = sorted({r["repo"] for r in test})
    return {
        "method": method,
        "label_review": "unreviewed by a person",
        "pooled_test": bootstrap_intervals(test, METRICS, settings),
        "delta": paired_intervals(test, control, METRICS, settings),
        "per_repo": {
            slug: {
                "metrics": bootstrap_intervals(
                    [r for r in test if r["repo"] == slug], METRICS, settings
                ),
                "delta": paired_intervals(
                    [r for r in test if r["repo"] == slug],
                    [r for r in control if r["repo"] == slug],
                    METRICS,
                    settings,
                ),
            }
            for slug in repos
        },
    }


def decision(result: dict, code_delta: float, mode: str) -> dict:
    delta = result["delta"]["recall@5"]
    checks = {
        "gain_at_least_005": delta["estimate"] >= 0.05 - 1e-12,
        "lower_bound_positive": delta["low"] > 0,
        "neither_repo_worse": all(
            r["delta"]["recall@5"]["estimate"] >= -1e-12 for r in result["per_repo"].values()
        ),
        "code_mrr_drop_at_most_001": code_delta >= -0.01 - 1e-12,
    }
    return {
        "checks": checks,
        "eligible": mode == "hybrid" and all(checks.values()),
        "code_mrr10_delta": code_delta,
        "defaults_changed": False,
        "reason": "BM25 preflight is descriptive only"
        if mode != "hybrid"
        else "apply frozen rule; human label review pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--stage", choices=["E1", "E2"], default="E1")
    parser.add_argument("--work", type=Path, default=ROOT / "data/architecture_v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    frozen = verify_frozen()
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    output = args.out or ROOT / f"eval/facts_results/architecture_v1_{args.mode}"
    work = args.work / args.mode
    work.mkdir(parents=True, exist_ok=True)
    defaults = Settings.model_construct()
    settings = Settings(
        data_dir=work,
        embedding_provider="huggingface",
        local_embedding_model=protocol["embedding"]["model"],
        embedding_dims=protocol["embedding"]["dims"],
        embedding_dtype=protocol["embedding"]["dtype"],
        embedding_max_tokens=defaults.embedding_max_tokens,
        embedding_query_instruction=defaults.embedding_query_instruction,
        embedding_batch_size=defaults.embedding_batch_size,
        eval_summary_embedding_batch_size=defaults.eval_summary_embedding_batch_size,
        retrieval_k=protocol["depth"],
        rrf_k=defaults.rrf_k,
        eval_bootstrap_samples=protocol["bootstrap"]["samples"],
        eval_bootstrap_seed=protocol["bootstrap"]["seed"],
        eval_bootstrap_confidence=protocol["bootstrap"]["confidence"],
        summary_max_chars=protocol["summary_caps"]["base"],
        summary_fact_max_chars=protocol["summary_caps"]["facts"],
        summary_history_repos_dir=ROOT / "data/phase5_repos",
        compute_device="auto",
        dense_weight=0.0 if args.mode == "bm25" else defaults.dense_weight,
        bm25_weight=1.0 if args.mode == "bm25" else defaults.bm25_weight,
    )
    if args.mode == "hybrid":
        import torch

        torch.set_num_threads(4)
        embeddings: Embeddings = DiskEmbeddings(work / "cache", settings, get_embeddings(settings))
    else:
        embeddings = NoEmbeddings()
    store = (Bm25Store if args.mode == "bm25" else VectorStore)(settings, embeddings)
    history = prepare_history(settings)
    questions = [q for name in REPOS for q in rows(OUT / f"{name}.jsonl")]
    for name, (slug, location, pin) in REPOS.items():
        checkout = ROOT / location
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != pin:
            raise ValueError(f"Wrong checkout pin {slug}")
        for q in (q for q in questions if q["repo"] == slug):
            for label in q["label_sources"]:
                if digest(checkout / label["path"]) != label["sha256"]:
                    raise ValueError("Label source changed")
        history[slug]["code"] = prepare_code(
            slug, name, checkout, pin, settings, store, embeddings, args.mode
        )
    write(
        output / "run.json",
        {
            "freeze": frozen,
            "started_at": datetime.now(UTC).isoformat(),
            "mode": args.mode,
            "history": history,
            "label_review": "unreviewed by a person",
            "settings": {
                k: v
                for k, v in settings.model_dump(mode="json").items()
                if not k.endswith(("key", "token"))
            },
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): digest(p)
                for p in [*sorted((ROOT / "src/repo_rag").rglob("*.py")), Path(__file__).resolve()]
            },
        },
    )
    variants = {"base": []} if args.stage == "E1" else protocol["fact_variants"]
    if args.stage == "E2" and not (output / "selection_E1.json").exists():
        raise ValueError("Complete E1 before E2")
    retriever = HybridRetriever(store, settings)
    code_questions = rows(OUT / "requests_code_regression.jsonl")
    baseline: list[dict] = []
    code_base: list[dict] = []
    dev_by_variant: list[dict] = []
    for variant, features in variants.items():
        print(f"Preparing {variant}", flush=True)
        cap = protocol["summary_caps"]["facts" if features else "base"]
        summary_settings = settings.model_copy(
            update={"summary_fact_features": tuple(features), "summary_max_chars": cap}
        )
        for name, (slug, location, _) in REPOS.items():
            docs, _ = build_summaries(slug, ROOT / location, summary_settings)
            if any(
                q["question"].casefold() in d.page_content.casefold()
                for q in questions
                for d in docs
            ):
                raise ValueError("Question text leaked into summary facts")
            store.sync("summaries", slug, docs)
            Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, "summaries"))
            write(
                output / variant / f"{name}_summaries.json",
                [{"text": d.page_content, "metadata": d.metadata} for d in docs],
            )
            audit = {
                "documents": len(docs),
                "max_chars": cap,
                "char_truncated": sum(bool(d.metadata.get("truncated")) for d in docs),
                "files_with_commit_subjects": sum(
                    "Commit subject:" in d.page_content for d in docs
                ),
                "files_with_pr_titles": sum("Merged PR title:" in d.page_content for d in docs),
                "files_with_doc_sections": sum("Documentation [" in d.page_content for d in docs),
                "exact_question_leaks": 0,
                "features": features,
            }
            if isinstance(embeddings, DiskEmbeddings):
                tokenizer = embeddings.inner.model.tokenizer  # type: ignore[attr-defined]
                lengths = [len(tokenizer.encode(d.page_content)) for d in docs]
                audit["embedding_token_cap"] = settings.embedding_max_tokens
                audit["embedding_token_truncated"] = sum(
                    n > settings.embedding_max_tokens for n in lengths
                )
            write(output / variant / f"{name}_facts_audit.json", audit)
        retriever.forget_cached_indexes()
        dev = evaluate(
            retriever,
            [q for q in questions if q["split"] == "dev"],
            protocol,
            variant,
            protocol["arms"],
        )
        write(output / variant / "dev.json", dev)
        if variant == "base":
            router = select(dev, [a for a in protocol["arms"] if a.startswith("router")])
            if args.stage == "E1":
                write(
                    output / "selection_E1.json",
                    {
                        "router": router,
                        "only_split": "dev",
                        "dev_sha256": digest(output / variant / "dev.json"),
                    },
                )
            else:
                previous = json.loads((output / "selection_E1.json").read_text())
                if previous["router"] != router:
                    raise ValueError("E1 dev selection changed")
        else:
            router = json.loads((output / "selection_E1.json").read_text())["router"]
        # All E2 settings are selected on dev BEFORE any E2 test scoring (second pass below).
        for r in dev:
            if r["method"] == router:
                dev_by_variant.append({**r, "method": variant})
    if args.stage == "E2":
        best = select(dev_by_variant, list(variants))
        write(
            output / "selection_E2.json", {"variant": best, "router": router, "only_split": "dev"}
        )
    else:
        best = "base"
    reports = {}
    for variant in ["base", *(v for v in variants if v != "base")]:
        for name, (slug, _, _) in REPOS.items():
            from langchain_core.documents import Document

            docs = [
                Document(page_content=d["text"], metadata=d["metadata"])
                for d in json.loads((output / variant / f"{name}_summaries.json").read_text())
            ]
            store.sync("summaries", slug, docs)
            Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, "summaries"))
        retriever.forget_cached_indexes()
        measured_arms = ["chunks", "summaries", "fused", router]
        test = evaluate(
            retriever,
            [q for q in questions if q["split"] == "test"],
            protocol,
            variant,
            measured_arms,
        )
        code = evaluate(retriever, code_questions, protocol, variant, measured_arms)
        write(output / variant / "test.json", test)
        write(output / variant / "code.json", code)
        if variant == "base":
            baseline, code_base = test, [r for r in code if r["method"] == "chunks"]
        for arm in measured_arms:
            result = report(test, baseline, arm, settings)
            left = [r for r in code if r["method"] == arm]
            result["code"] = bootstrap_intervals(left, CODE_METRICS, settings)
            result["code_delta"] = paired_intervals(left, code_base, CODE_METRICS, settings)
            result["decision"] = decision(
                result, result["code_delta"]["mrr@10"]["estimate"], args.mode
            )
            reports[f"{variant}/{arm}"] = result
    final_report = {
        "label_review": "unreviewed by a person",
        "stage": args.stage,
        "selected_variant": best,
        "selected_router": router,
        "results": reports,
    }
    write(output / "report.json", final_report)
    write(output / f"report_{args.stage}.json", final_report)
    lines = [
        "Architecture labels: **unreviewed by a person**. Pointwise group-bootstrap intervals; small sample, exploratory.",
        f"Mode: {args.mode}; dev-selected router: {router}; dev-selected facts: {best}.",
        "",
        "| Variant / arm | Test Recall@5 | Delta [95% CI] | Code MRR@10 | Eligible |",
        "|---|---|---|---|---|",
    ]
    for name, result in reports.items():
        d = result["delta"]["recall@5"]
        lines.append(
            f"| {name} | {result['pooled_test']['recall@5']['estimate']:.4f} | {d['estimate']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] | {result['code']['mrr@10']['estimate']:.4f} | {result['decision']['eligible']} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
