"""Judge comparison: Qwen3-Reranker 4B vs 8B through production search. Quality (+ rough timing).

Same first stage and candidates for every judge; only the cross-encoder differs. Protocol:
eval/datasets/phase4_judge_v1/protocol.json (frozen before the run). Labels are the frozen
expansion set (30 fresh multi-block + 120 regression questions).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.ambiguous import score_case, validate_cases  # noqa: E402
from eval.retrieval.dataset import write_json  # noqa: E402
from eval.retrieval.expansion_eval import fixed_settings, validate_multiblock  # noqa: E402
from eval.retrieval.followup import production_corpus  # noqa: E402
from eval.retrieval.policies import shared_factory  # noqa: E402
from eval.retrieval.runner import cached_vectors, environment  # noqa: E402
from eval.run_expansion_eval import TracedRetriever, TracedStore, digest, load_labels  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.rerank import QwenReranker  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.embeddings import LocalEmbeddings  # noqa: E402

DEPTH = 20


def tag(model: str) -> str:
    return model.rsplit("-", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "full"), default="full")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh output; preserve any partial measurements")
    judge_protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    judges: list[str] = judge_protocol["judges"]
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
        print({"repo": args.repo, "chunks": len(docs), "judges": judges, "fresh": validation})
        return
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the A100 notebook")
    args.output.mkdir(parents=True)
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    index = Bm25Index.build(docs)
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

    scorers: dict[str, QwenReranker] = {}
    retrievers: dict[str, TracedRetriever] = {}
    for model in judges:
        print(f"Loading judge {model} (from the Hugging Face cache)", flush=True)
        judge_settings = settings.model_copy(update={"reranker_model": model})
        scorer = QwenReranker(judge_settings, instruction=settings.code_rerank_instruction)
        scorers[tag(model)] = scorer
        retrievers[tag(model)] = TracedRetriever(
            store, judge_settings, reranker_factory=shared_factory(scorer)
        )
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    warm_query = cases[0][0].question.question
    for name, retriever in retrievers.items():  # load and warm every judge before timing
        retriever.search(q0.repo, warm_query, k=10, rerank=True)
        print(f"Judge {name} ready", flush=True)

    rows, traces = [], []
    for case, role in cases:
        query = case.question.question
        for method in ["hybrid", *[f"rerank@{t}" for t in retrievers]]:
            rerank = method != "hybrid"
            retriever = (
                retrievers[method.split("@")[1]] if rerank else next(iter(retrievers.values()))
            )
            retriever.settings = retriever.settings.model_copy(
                update={"rerank_candidate_depth": DEPTH, "evidence_expansion": "none"}
            )
            sync()
            started = time.perf_counter()
            hits = retriever.search(q0.repo, query, k=10, rerank=rerank)
            sync()
            elapsed = (time.perf_counter() - started) * 1000
            final_docs = [h.document for h in hits]
            ranking = [str(d.metadata["chunk_id"]) for d in final_docs]
            rows.append(
                {
                    "id": case.question.id,
                    "group": case.question.group,
                    "split": "test",
                    "cohort": role,
                    "stratum": case.stratum,
                    "method": method,
                    "elapsed_ms": elapsed,
                    **score_case(case, final_docs, settings),
                    "results": [{"chunk_id": cid, **by_id[cid].metadata} for cid in ranking],
                }
            )
            traces.append(
                {
                    "id": case.question.id,
                    "method": method,
                    "final_order": ranking,
                    "pairs": retriever.pairs if rerank else [],
                }
            )
        write_json(args.output / "per_question.json", rows)
        print(f"Scored {role} {case.question.id}", flush=True)
    write_json(args.output / "traces.json", traces)
    write_json(
        args.output / "run.json",
        {
            "repo": q0.repo,
            "commit_sha": q0.commit_sha,
            "judges": judges,
            "judge_protocol_sha256": digest(args.protocol),
            "labels_protocol_sha256": digest(args.labels / "protocol.json"),
            "manifest_sha256": digest(args.labels / "manifest.json"),
            "dataset_sha256": {e["file"]: e["sha256"] for e in entries},
            "source_only_validation": validation,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): digest(p)
                for p in [
                    *ROOT.joinpath("src/repo_rag").rglob("*.py"),
                    *ROOT.joinpath("eval/retrieval").rglob("*.py"),
                    Path(__file__),
                ]
            },
            "settings": settings.model_dump(
                mode="json", exclude={"anthropic_api_key", "github_token", "voyage_api_key"}
            ),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "environment": environment(),
            "reranker_model_revisions": {
                t: getattr(getattr(s.model, "config", None), "_commit_hash", None)
                for t, s in scorers.items()
            },
            "embedding_model_revision": getattr(
                getattr(embedding.model._first_module().auto_model, "config", None),
                "_commit_hash",
                None,
            ),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()
            if args.device == "cuda"
            else None,
            "independent_review_complete": False,
            "production_default_changed": False,
        },
    )


if __name__ == "__main__":
    main()
