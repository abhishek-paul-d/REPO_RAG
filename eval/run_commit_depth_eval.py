"""Run the pre-registered commit-depth/weight experiment, one GPU model per process."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.prepare_bugfix_eval import digest, write  # noqa: E402
from eval.retrieval.commit_depth import (  # noqa: E402
    PROTOCOL,
    depth_rows,
    hashes,
    labels,
    prefix_ranking,
    read,
    score,
    select_weight,
    settings,
    verify_protocol,
    weight_rows,
)
from eval.run_bugfix_eval import TracedRetriever, TracedStore, synchronize  # noqa: E402
from repo_rag.retrieval.retrievers import HybridRetriever  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402


def revisions(retriever, config) -> dict:
    model = getattr(
        getattr(retriever.store.embeddings, "_model", None), "_first_module", lambda: None
    )()
    embedding = getattr(
        getattr(getattr(model, "auto_model", None), "config", None), "_commit_hash", None
    )
    scorer = retriever._task_rerankers.get(config.rerank_instruction)
    judge = getattr(getattr(getattr(scorer, "model", None), "config", None), "_commit_hash", None)
    return {"embedding": embedding, "reranker": judge}


def selected(root: Path, protocol: dict) -> dict:
    rows = [
        r
        for stem in protocol["input_freeze_sha256"]
        for r in read(root / "runs" / stem / "weights/per_question.json")
    ]
    return select_weight(rows, protocol)


def run(
    root: Path,
    stem: str,
    stage: str,
    *,
    judge: str = "rerank8b",
    device: str = "cuda",
    experiment: Path = PROTOCOL,
    weight: float | None = None,
) -> dict:
    import torch

    from repo_rag.store.embeddings import get_embeddings

    protocol = verify_protocol(experiment, root / "datasets")
    cohorts = read(experiment / "cohorts.json")
    dataset = root / "datasets" / stem
    docs, pairs = labels(dataset, protocol, cohorts)  # BEFORE model initialization
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if stage == "selected":
        choice = selected(root, protocol)
        if read(root / "selection.json") != choice:
            raise ValueError("Saved selection does not rebuild from dev")
        if choice["selected"] == protocol["weight"]["baseline"]:
            return {"skipped": "dev retained 0.3"}
        if weight not in (choice["selected"], protocol["weight"]["baseline"]):
            raise ValueError("Conditional weight outside dev-selected comparison")
    elif weight is not None:
        raise ValueError("Depth/weight experiment settings are frozen")
    if stage not in {"depth", "weights", "selected", "timing"} or judge not in {
        "rerank8b",
        "rerank4b",
    }:
        raise ValueError("Unknown stage/judge")
    depth = (
        max(protocol["depth"]["depths"]) if stage == "depth" else min(protocol["depth"]["depths"])
    )
    retrieval_k = (
        protocol["depth"]["retrieval_k"]
        if stage in {"depth", "timing"}
        else protocol["weight"]["retrieval_k"]
    )
    folder_name = (
        "weights"
        if stage == "weights"
        else "timing"
        if stage == "timing"
        else judge
        if stage == "depth"
        else f"selected_{weight:g}"
    )
    output = root / "runs" / stem / folder_name
    if output.exists():
        raise FileExistsError("Preserve completed/incomplete runs; use a new output")
    output.mkdir(parents=True)
    config = settings(
        protocol,
        root / "runs" / stem / "index",
        "dense" if stage == "weights" else judge,
        device,
        depth=depth,
        weight=protocol["depth"]["bm25_weight"] if weight is None else weight,
        retrieval_k=retrieval_k,
    )
    store = (VectorStore if stage == "timing" else TracedStore)(config, get_embeddings(config))
    repo = str(docs[0].metadata["repo"])
    if repo not in protocol["repos"]:
        raise ValueError("Repository outside frozen experiment")
    store.sync("commits", repo, docs)
    lexical = Bm25Index.build(docs)
    lexical.save(bm25_path(config.bm25_dir, repo, "commits"))
    retriever = (HybridRetriever if stage == "timing" else TracedRetriever)(store, config)
    ids = {str(d.metadata["chunk_id"]): str(d.metadata["commit_sha"]) for d in docs}
    traces, rows = [], []
    if stage == "timing":
        if judge != protocol["timing"]["judge"]:
            raise ValueError("Timing is prechosen for the primary 8B only")
        by_id = {p["id"]: p for p in pairs}
        sample = [by_id[qid] for qid in cohorts["timing_ids"] if qid in by_id]
        if not sample:
            raise ValueError("No frozen timing questions for this repo")
        for d in protocol["depth"]["depths"]:
            retriever.settings = config.model_copy(update={"rerank_candidate_depth": d})
            retriever.search(
                repo,
                sample[0]["query"],
                kind="commits",
                k=10,
                filters={"is_fix": "true"},
                rerank=True,
            )
        for repetition in range(protocol["timing"]["repetitions"]):
            print(repo, "timing repetition", repetition + 1, flush=True)
            for pair in sample:
                depths = protocol["depth"]["depths"]
                offset = (cohorts["timing_ids"].index(pair["id"]) + repetition) % len(depths)
                for position, d in enumerate(depths[offset:] + depths[:offset]):
                    retriever.settings = config.model_copy(update={"rerank_candidate_depth": d})
                    synchronize(device)
                    start = time.perf_counter()
                    hits = retriever.search(
                        repo,
                        pair["query"],
                        kind="commits",
                        k=10,
                        filters={"is_fix": "true"},
                        rerank=True,
                    )
                    synchronize(device)
                    rows.append(
                        {
                            "id": pair["id"],
                            "repo": repo,
                            "depth": d,
                            "repetition": repetition,
                            "position": position,
                            "elapsed_seconds": time.perf_counter() - start,
                            "ranking": [h.document.metadata["commit_sha"] for h in hits],
                        }
                    )
    else:
        for i, pair in enumerate(pairs):
            retriever.search(
                repo,
                pair["query"],
                kind="commits",
                k=depth,
                filters={"is_fix": "true"},
                rerank=stage != "weights",
            )
            keyword = [
                {"chunk_id": cid, "sha": ids[cid], "score": value}
                for cid, value in lexical.search(pair["query"], retrieval_k)
            ]
            trace = {
                "id": pair["id"],
                "dense": store.dense_scores,  # type: ignore[attr-defined]
                "bm25": keyword,
                "pairs": getattr(retriever, "pair_scores", []),
            }
            traces.append(trace)
            if stage == "depth":
                rows.extend(depth_rows(pair, trace["pairs"], judge, protocol))
            elif stage == "weights":
                rows.extend(weight_rows(pair, trace["dense"], keyword, ids, protocol))
            else:
                ranking = prefix_ranking(trace["pairs"], depth)
                rows.append(
                    {
                        **pair,
                        "method": f"selected_{weight:g}",
                        "weight": weight,
                        "ranking": ranking,
                        **score(ranking, pair["gold"]),
                        "pool_recall": float(any(p["sha"] in pair["gold"] for p in trace["pairs"])),
                    }
                )
            if i % 20 == 0:
                print(repo, folder_name, i + 1, "/", len(pairs), flush=True)
    info = {
        "repo": repo,
        "stage": stage,
        "judge": judge,
        "dataset_freeze_sha256": digest(dataset / "freeze.json"),
        "experiment_freeze_sha256": digest(experiment / "freeze.json"),
        "protocol_sha256": digest(experiment / "protocol.json"),
        "settings": config.model_dump(
            mode="json", exclude={"github_token", "anthropic_api_key", "voyage_api_key"}
        ),
        "source_sha256": hashes(),
        "model_revisions": revisions(retriever, config),
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "offline CPU fixture",
        "rows": len(rows),
        "timing": protocol["timing"]
        if stage == "timing"
        else "quality only; shared prefix scores are not latency measurements",
    }
    write(output / "per_question.json", rows)
    write(output / "traces.json", traces)
    write(output / "run.json", info)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--experiment", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--stage", choices=["weights", "depth", "selected", "timing", "select"], required=True
    )
    parser.add_argument("--repo", default="")
    parser.add_argument("--judge", choices=["rerank8b", "rerank4b"], default="rerank8b")
    parser.add_argument("--weight", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.stage == "select":
        p = verify_protocol(args.experiment, args.root / "datasets")
        choice = selected(args.root, p)
        if (args.root / "selection.json").exists():
            if read(args.root / "selection.json") != choice:
                raise ValueError("Existing selection changed")
        else:
            write(args.root / "selection.json", choice)
        print(choice)
    else:
        run(
            args.root,
            args.repo,
            args.stage,
            judge=args.judge,
            weight=args.weight,
            device=args.device,
            experiment=args.experiment,
        )
