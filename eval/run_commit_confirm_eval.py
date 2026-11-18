"""Run the pre-registered retrieval_k confirmation for one repository, one stage per process.

`confirm` runs arms A/B/C through the production search path with a shared per-question score
cache, then scores arm A again from scratch. `timing` makes real uncached production calls.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.confirm_lib import (  # noqa: E402
    arm_settings,
    hashes,
    load_pairs,
    question_rows,
    read,
    verify_experiment,
)
from eval.prepare_bugfix_eval import digest, write  # noqa: E402
from eval.run_bugfix_eval import TracedRetriever, TracedStore, synchronize  # noqa: E402
from repo_rag.retrieval.rerank import QwenReranker, Reranker  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402


class SharedScores:
    """Score each (query, document) once per question; `fresh` bypasses the cache entirely."""

    def __init__(self, inner: Reranker) -> None:
        self.inner = inner
        self.cache: dict[tuple[str, str], float] = {}
        self.fresh = False

    def reset(self) -> None:
        self.cache.clear()

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if self.fresh:
            return self.inner.score(query, texts)
        missing = list(dict.fromkeys(t for t in texts if (query, t) not in self.cache))
        if missing:
            for text, value in zip(missing, self.inner.score(query, missing), strict=True):
                self.cache[(query, text)] = value
        return [self.cache[(query, t)] for t in texts]


def dataset_folder(root: Path, group: str, stem: str) -> Path:
    return root / "datasets" / ("v2" if group == "v2dev" else group) / stem


def revisions(retriever: Any, inner: Any) -> dict:
    model = getattr(
        getattr(retriever.store.embeddings, "_model", None), "_first_module", lambda: None
    )()
    embedding = getattr(
        getattr(getattr(model, "auto_model", None), "config", None), "_commit_hash", None
    )
    judge = getattr(getattr(getattr(inner, "model", None), "config", None), "_commit_hash", None)
    return {"embedding": embedding, "reranker": judge}


def run(
    root: Path,
    group: str,
    stem: str,
    stage: str,
    *,
    device: str = "cuda",
    experiment: Path | None = None,
) -> dict:
    import torch

    from repo_rag.store.embeddings import get_embeddings

    experiment = experiment or root / "experiment"
    protocol = verify_experiment(experiment, root / "datasets")  # BEFORE any model loads
    freeze = read(experiment / "freeze.json")
    if group not in {"fresh", "v2dev"} or stage not in {"confirm", "timing"}:
        raise ValueError("Unknown group or stage")
    if stage == "timing" and group != "fresh":
        raise ValueError("Timing uses the fresh holdout only")
    folder = dataset_folder(root, group, stem)
    split = "dev" if group == "v2dev" else None
    exclude = set(freeze["excluded"]) if group == "fresh" else set()
    docs, pairs = load_pairs(folder, protocol, split, exclude)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    output = root / "runs" / group / stem / stage
    if output.exists():
        raise FileExistsError("Preserve completed or incomplete runs; use a new output")
    output.mkdir(parents=True)
    confirm = protocol["confirm"]
    arms: dict[str, int] = confirm["arms"]
    config = arm_settings(protocol, root / "runs" / group / stem / "index", device, arms["A"])
    repo = str(docs[0].metadata["repo"])
    store_class = VectorStore if stage == "timing" else TracedStore
    store = store_class(config, get_embeddings(config))
    store.sync("commits", repo, docs)
    lexical = Bm25Index.build(docs)
    lexical.save(bm25_path(config.bm25_dir, repo, "commits"))
    ids = {str(d.metadata["chunk_id"]): str(d.metadata["commit_sha"]) for d in docs}
    filters = {"is_fix": "true"}
    traces: list[dict] = []
    rows: list[dict] = []
    if stage == "confirm":
        inner = QwenReranker(config, config.rerank_instruction)
        scorer = SharedScores(inner)
        retriever = TracedRetriever(store, config, reranker=scorer)

        def call(pair: dict, k: int) -> dict:
            retriever.settings = config.model_copy(update={"retrieval_k": k})
            retriever.pair_scores = []
            retriever.search(
                repo, pair["query"], kind="commits", k=10, filters=filters, rerank=True
            )
            return {
                "retrieval_k": k,
                "dense": store.dense_scores,  # type: ignore[attr-defined]
                "bm25": [
                    {"chunk_id": cid, "sha": ids[cid], "score": value}
                    for cid, value in lexical.search(pair["query"], k)
                ],
                "pairs": retriever.pair_scores,
            }

        for i, pair in enumerate(pairs):
            scorer.reset()
            trace: dict[str, Any] = {"id": pair["id"], "arms": {}}
            for arm, k in arms.items():
                trace["arms"][arm] = call(pair, k)
            scorer.fresh = True  # independent second scoring of arm A: GPU rerun noise
            trace["rerun"] = {"arm": "A", "pairs": call(pair, arms["A"])["pairs"]}
            scorer.fresh = False
            traces.append(trace)
            if i % 20 == 0:
                print(group, stem, "confirm", i + 1, "/", len(pairs), flush=True)
        for pair, trace in zip(pairs, traces, strict=True):
            rows.extend(question_rows(pair, trace, protocol))
        model_revisions = revisions(retriever, inner)
    else:
        retriever_plain = HybridRetriever(store, config)
        by_id = {p["id"]: p for p in pairs}
        sample = [by_id[qid] for qid in freeze["timing_ids"] if qid in by_id]
        if not sample:
            raise ValueError("No frozen timing questions for this repository")
        names = list(arms)

        def timed(pair: dict, arm: str) -> tuple[float, list[str]]:
            retriever_plain.settings = config.model_copy(update={"retrieval_k": arms[arm]})
            synchronize(device)
            start = time.perf_counter()
            hits = retriever_plain.search(
                repo, pair["query"], kind="commits", k=10, filters=filters, rerank=True
            )
            synchronize(device)
            return time.perf_counter() - start, [
                str(h.document.metadata["commit_sha"]) for h in hits
            ]

        for arm in names:
            timed(sample[0], arm)  # unmeasured warmup per arm
        for repetition in range(confirm["timing"]["repetitions"]):
            print(stem, "timing repetition", repetition + 1, flush=True)
            for pair in sample:
                offset = (freeze["timing_ids"].index(pair["id"]) + repetition) % len(names)
                for position, arm in enumerate(names[offset:] + names[:offset]):
                    seconds, ranking = timed(pair, arm)
                    rows.append(
                        {
                            "id": pair["id"],
                            "repo": repo,
                            "arm": arm,
                            "repetition": repetition,
                            "position": position,
                            "elapsed_seconds": seconds,
                            "ranking": ranking,
                        }
                    )
        scorer_model = next(iter(retriever_plain._task_rerankers.values()), None)
        model_revisions = revisions(retriever_plain, scorer_model)
    info = {
        "repo": repo,
        "group": group,
        "stage": stage,
        "dataset_freeze_sha256": digest(folder / "freeze.json"),
        "experiment_freeze_sha256": digest(experiment / "freeze.json"),
        "protocol_sha256": digest(experiment / "protocol.json"),
        "settings": config.model_dump(
            mode="json", exclude={"github_token", "anthropic_api_key", "voyage_api_key"}
        ),
        "source_sha256": hashes(),
        "model_revisions": model_revisions,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "offline CPU fixture",
        "rows": len(rows),
        "timing": confirm["timing"]["label"] if stage == "timing" else "quality only",
    }
    write(output / "per_question.json", rows)
    write(output / "traces.json", traces)
    write(output / "run.json", info)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--group", choices=["fresh", "v2dev"], required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--stage", choices=["confirm", "timing"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    run(
        args.root,
        args.group,
        args.stem,
        args.stage,
        device=args.device,
        experiment=args.experiment,
    )
