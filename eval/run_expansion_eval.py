"""Frozen expansion matrix through production search. Quality only; no GPU timing claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.ambiguous import AmbiguousCase, read_cases, score_case, validate_cases  # noqa: E402
from eval.retrieval.dataset import write_json  # noqa: E402
from eval.retrieval.expansion_eval import METHODS, fixed_settings, validate_multiblock, volume  # noqa: E402
from eval.retrieval.followup import production_corpus  # noqa: E402
from eval.retrieval.policies import shared_factory  # noqa: E402
from eval.retrieval.runner import cached_vectors, environment  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.rerank import QwenReranker  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever, SearchHit  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.embeddings import LocalEmbeddings  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402


def digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_labels(labels: Path, repo: str):
    protocol = json.loads((labels / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((labels / "manifest.json").read_text(encoding="utf-8"))
    freeze = json.loads((labels / "freeze.json").read_text(encoding="utf-8"))
    if freeze["protocol_sha256"] != digest(labels / "protocol.json") or freeze[
        "manifest_sha256"
    ] != digest(labels / "manifest.json"):
        raise ValueError("frozen labels/protocol were changed")
    entries = [e for e in manifest["datasets"] if e["repo"] == repo]
    if not entries:
        raise ValueError("repository absent from frozen manifest")
    cases: list[tuple[AmbiguousCase, str]] = []
    for entry in entries:
        path = labels / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("dataset checksum mismatch")
        current = read_cases(path)
        if len(current) != entry["questions"]:
            raise ValueError("question count mismatch")
        cases.extend((c, entry["role"]) for c in current)
    if len({c.question.id for c, _ in cases}) != len(cases):
        raise ValueError("question ids overlap between new and regression sets")
    return protocol, entries, cases


class TracedStore(VectorStore):
    def search(self, *args, **kwargs):
        hits = super().search(*args, **kwargs)
        self.last_dense = [
            {"chunk_id": str(d.metadata["chunk_id"]), "score": float(s)} for d, s in hits
        ]
        return hits


class TracedRetriever(HybridRetriever):
    ranked: list[SearchHit]
    pairs: list[dict]

    def _rerank(self, query, hits, kind="code"):
        self.pairs = [{"chunk_id": h.chunk_id, "fused_score": h.score} for h in hits]
        self.ranked = super()._rerank(query, hits, kind)
        scores = {h.chunk_id: h.rerank_score for h in self.ranked}
        for pair in self.pairs:
            pair["score"] = scores[pair["chunk_id"]]
        return self.ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "bm25", "full"), default="full")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh output; preserve any partial measurements")
    protocol, entries, cases = load_labels(args.labels, args.repo)
    validate_cases(args.checkout, [case for case, _ in cases])
    settings = Settings(
        data_dir=args.output / "isolated_index",
        compute_device=args.device,
        **fixed_settings(protocol),
    )
    q0 = cases[0][0].question
    docs = production_corpus(args.checkout, q0.repo, q0.commit_sha, settings)
    validation = validate_multiblock([c for c, role in cases if role == "fresh"], docs)
    if args.mode == "validate":
        print({"repo": args.repo, "chunks": len(docs), "fresh_validation": validation})
        return
    args.output.mkdir(parents=True)
    chunks = args.output / "chunks.jsonl"
    chunks.write_text(
        "".join(
            json.dumps({"page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False)
            + "\n"
            for d in docs
        ),
        encoding="utf-8",
        newline="\n",
    )
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    index = Bm25Index.build(docs)
    embedding = scorer = retriever = store = None
    gpu = None
    if args.mode == "full":
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use the A100 notebook")
        gpu = torch.cuda.get_device_name() if args.device == "cuda" else None
        embedding = LocalEmbeddings(settings)
        vectors = cached_vectors(
            [d.page_content for d in docs],
            settings,
            args.output / "vector_cache",
            embedding,
            query=False,
        )
        store = TracedStore(settings, embedding)
        store.replace_records(
            "code",
            q0.repo,
            list(by_id),
            [d.page_content for d in docs],
            [d.metadata for d in docs],
            vectors,
        )
        index.save(bm25_path(settings.bm25_dir, q0.repo, "code"))
        scorer = QwenReranker(settings, instruction=settings.code_rerank_instruction)
        retriever = TracedRetriever(store, settings, reranker_factory=shared_factory(scorer))
    rows, traces = [], []
    for case, role in cases:
        query = case.question.question
        lexical = index.search(query, settings.retrieval_k)
        for method in METHODS if retriever else ("bm25",):
            if retriever is not None:
                retriever.settings = settings.model_copy(
                    update={"evidence_expansion": "none" if method == "plain" else method}
                )
                hits = retriever.search(q0.repo, query, k=10, rerank=True)
                ranked = retriever.ranked[:10]
                measured = volume(hits, ranked)
                final_docs = [h.document for h in hits]
                pair_trace = retriever.pairs
                dense = store.last_dense if store else []
                ranked_ids = [h.chunk_id for h in ranked]
            else:
                final_docs = [by_id[cid] for cid, _ in lexical[:10]]
                ranked_ids = [str(d.metadata["chunk_id"]) for d in final_docs]
                measured = {
                    "returned_chunks": len(final_docs),
                    "returned_chars": sum(len(d.page_content) for d in final_docs),
                    "expanded_ids": [],
                    "displaced_ids": [],
                }
                pair_trace, dense = [], []
            ranking = [str(d.metadata["chunk_id"]) for d in final_docs]
            rows.append(
                {
                    "id": case.question.id,
                    "group": case.question.group,
                    "split": "test",
                    "cohort": role,
                    "stratum": case.stratum,
                    "method": method,
                    **score_case(case, final_docs, settings),
                    **measured,
                    "results": [{"chunk_id": cid, **by_id[cid].metadata} for cid in ranking],
                }
            )
            traces.append(
                {
                    "id": case.question.id,
                    "method": method,
                    "ranked_top10": ranked_ids,
                    "final_order": ranking,
                    "pairs": pair_trace,
                    "dense": dense,
                    "bm25": [{"chunk_id": cid, "score": float(s)} for cid, s in lexical],
                }
            )
        write_json(args.output / "per_question.json", rows)
        print(f"Scored {role} {case.question.id}", flush=True)
    write_json(args.output / "traces.json", traces)
    write_json(
        args.output / "run.json",
        {
            "mode": args.mode,
            "repo": q0.repo,
            "commit_sha": q0.commit_sha,
            "methods": list(METHODS) if retriever else ["bm25"],
            "protocol_sha256": digest(args.labels / "protocol.json"),
            "manifest_sha256": digest(args.labels / "manifest.json"),
            "dataset_sha256": {e["file"]: e["sha256"] for e in entries},
            "chunks_sha256": digest(chunks),
            "source_only_validation": validation,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): digest(p)
                for p in [
                    *ROOT.joinpath("src/repo_rag").rglob("*.py"),
                    *ROOT.joinpath("eval/retrieval").rglob("*.py"),
                    Path(__file__),
                    ROOT / "eval/analyze_expansion_run.py",
                    ROOT / "eval/build_expansion_dataset.py",
                ]
            },
            "settings": settings.model_dump(
                mode="json", exclude={"anthropic_api_key", "github_token", "voyage_api_key"}
            ),
            "gpu": gpu,
            "environment": environment(),
            "quality_only": True,
            "score_sharing": protocol["score_sharing"],
            "embedding_model_revision": getattr(
                getattr(embedding.model._first_module().auto_model, "config", None),
                "_commit_hash",
                None,
            )
            if embedding
            else None,
            "reranker_model_revision": getattr(
                getattr(scorer.model, "config", None), "_commit_hash", None
            )
            if scorer
            else None,
            "independent_review_complete": False,
            "production_default_changed": False,
        },
    )


if __name__ == "__main__":
    main()
