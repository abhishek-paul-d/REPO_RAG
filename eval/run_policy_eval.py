"""Phase 4 policy run: rerank ordering and task-instruction policies on a GPU; no tuning.

One resident reranker serves five pre-registered policies through the production retriever
(`eval/retrieval/policies.py`). Quality only: no timing is taken here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.ambiguous import read_cases, score_case, validate_cases  # noqa: E402
from eval.retrieval.comparison_protocol import protocol_settings  # noqa: E402
from eval.retrieval.dataset import write_json  # noqa: E402
from eval.retrieval.followup import production_corpus  # noqa: E402
from eval.retrieval.policies import (  # noqa: E402
    BASELINE,
    EXTRA_POLICIES,
    POLICIES,
    policy_settings,
    shared_factory,
)
from eval.retrieval.runner import cached_vectors, environment  # noqa: E402
from repo_rag.config import Settings  # noqa: E402
from repo_rag.retrieval.intent import detect_intent  # noqa: E402
from repo_rag.retrieval.rerank import QwenReranker  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.embeddings import LocalEmbeddings  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402


class TracedRetriever(HybridRetriever):
    """Production algorithm; also records the candidate pool and the probabilities it got."""

    pair_trace: list[dict[str, Any]]

    def _rerank(self, query, hits, kind="code"):  # type: ignore[no-untyped-def]
        self.pair_trace = [{"chunk_id": h.chunk_id} for h in hits]
        result = super()._rerank(query, hits, kind)
        probabilities = {h.chunk_id: h.rerank_score for h in result}
        for pair in self.pair_trace:
            pair["score"] = probabilities[pair["chunk_id"]]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "full"), default="full")
    parser.add_argument(
        "--policies", default=",".join(POLICIES), help="comma list; 'hybrid' = no reranking"
    )
    parser.add_argument(
        "--header-defines",
        action="store_true",
        help="chunk with the `# defines:` header line (Settings.header_defines)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    cases = read_cases(args.dataset)
    validate_cases(args.checkout, cases)
    if args.mode == "validate":
        print(f"Validated {len(cases)} cases")
        return
    if args.output.exists():
        raise FileExistsError("choose a new output to preserve completed or partial measurements")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    output = args.output
    names = [n for n in args.policies.split(",") if n]
    if not names or any(n not in {**POLICIES, **EXTRA_POLICIES} for n in names):
        raise ValueError(f"unknown policy in {args.policies!r}")
    settings = Settings(
        data_dir=output / "isolated_index",
        compute_device=args.device,
        header_defines=args.header_defines,
        **protocol_settings(protocol),
    )
    q0 = cases[0].question
    docs = production_corpus(args.checkout, q0.repo, q0.commit_sha, settings)
    by_id = {str(d.metadata["chunk_id"]): d for d in docs}
    output.mkdir(parents=True)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the A100 notebook or explicitly choose CPU")
    embedding = LocalEmbeddings(settings)
    vectors = cached_vectors(
        [d.page_content for d in docs], settings, output / "vector_cache", embedding, query=False
    )
    store = VectorStore(settings, embedding)
    store.replace_records(
        "code",
        q0.repo,
        list(by_id),
        [d.page_content for d in docs],
        [d.metadata for d in docs],
        vectors,
    )
    Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, q0.repo, "code"))
    scorer = QwenReranker(settings, instruction=settings.code_rerank_instruction)
    factory = shared_factory(scorer)
    retriever = TracedRetriever(store, settings, reranker_factory=factory)

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for case in cases:
        query = case.question.question
        for name in names:
            retriever.settings = policy_settings(settings, name)
            factory.log.clear()  # type: ignore[attr-defined]
            retriever.pair_trace = []
            hits = retriever.search(q0.repo, query, k=10, rerank=name != "hybrid")
            ranking = [h.chunk_id for h in hits]
            scores = score_case(case, [by_id[c] for c in ranking], settings)
            instruction = factory.log[-1] if factory.log else None  # type: ignore[attr-defined]
            rows.append(
                {
                    "id": case.question.id,
                    "group": case.question.group,
                    "split": "test",
                    "stratum": case.stratum,
                    "paraphrase": case.question.id.endswith("-para"),
                    "detected_intent": detect_intent(query),
                    "method": name,
                    **scores,
                    "results": [{"chunk_id": c, **by_id[c].metadata} for c in ranking],
                }
            )
            traces.append(
                {
                    "id": case.question.id,
                    "method": name,
                    "instruction": instruction,
                    "final_order": ranking,
                    "pairs": retriever.pair_trace,
                }
            )
        write_json(output / "per_question.json", rows)
        print(f"Scored {case.question.id}", flush=True)
    write_json(output / "traces.json", traces)
    write_json(
        output / "run.json",
        {
            "repo": q0.repo,
            "commit_sha": q0.commit_sha,
            "environment": environment(),
            "dataset_sha256": hashlib.sha256(
                args.dataset.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "protocol_sha256": hashlib.sha256(
                args.protocol.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for p in [
                    *ROOT.joinpath("src/repo_rag").rglob("*.py"),
                    *ROOT.joinpath("eval/retrieval").rglob("*.py"),
                    Path(__file__),
                ]
            },
            "settings": settings.model_dump(
                mode="json", exclude={"anthropic_api_key", "github_token", "voyage_api_key"}
            ),
            "policies": {n: {**POLICIES, **EXTRA_POLICIES}[n] for n in names},
            "header_defines": args.header_defines,
            "baseline": BASELINE,
            "rerank_depth": policy_settings(settings, BASELINE).rerank_candidate_depth,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "platform": platform.platform(),
            "embedding_model_revision": getattr(
                getattr(embedding.model._first_module().auto_model, "config", None),
                "_commit_hash",
                None,
            ),
            "reranker_model_revision": getattr(
                getattr(scorer.model, "config", None), "_commit_hash", None
            ),
            "independent_review_complete": False,
        },
    )
    lines = [
        "# Phase 4 rerank policy run",
        "",
        "| Policy | Questions | MRR@10 | Complete proof@10 |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        sel = [r for r in rows if r["method"] == name]
        mrr = sum(r["mrr@10"] for r in sel) / len(sel)
        proof = sum(r["any_complete_proof@10"] for r in sel) / len(sel)
        lines.append(f"| {name} | {len(sel)} | {mrr:.3f} | {proof:.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
