"""E3: approved model sees only frozen facts; audit every paragraph; test fixed router.

Run from an extracted summary_prose_inputs.zip capsule on Colab. No model judge.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_architecture_dataset import OUT, REPOS, verify_frozen, write
from eval.retrieval.statistics import bootstrap_intervals, paired_intervals
from eval.run_architecture_eval import CODE_METRICS, METRICS, decision, evaluate, report, rows
from repo_rag.agent.prompts import SUMMARY_PROMPT
from repo_rag.config import Settings
from repo_rag.retrieval.retrievers import HybridRetriever
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.vectorstore import VectorStore
from repo_rag.summaries.prose import CHECK_VERSION, TransformersSummarizer
from repo_rag.summaries.repo_map import _Cache


class QueryEmbeddings(Embeddings):
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_documents(self, texts):
        raise AssertionError("code vectors are frozen; summary vectors use the real embedder")

    def embed_query(self, text):
        return self.vectors[text]


def generate(facts: list[dict], summarizer, cache: _Cache, cap: int) -> list[Document]:
    documents = []
    for row in facts:
        text, meta = row["text"], row["metadata"]
        kind, path = meta["symbol_kind"], meta["path"]
        key = cache.key(summarizer.name, kind, path, text)
        cached = key in cache.data
        if key not in cache.data:
            cache.data[key] = summarizer.summarize(kind, path, text)
            cache.save()
        paragraph = summarizer.validate(kind, path, text, cache.data[key])
        summarizer.records[-1]["cache_hit"] = cached
        content = f"Summary: {paragraph}\n{text}" if paragraph else text
        metadata = {**meta, "prose_chars": len(paragraph), "truncated": len(content) > cap}
        documents.append(Document(page_content=content[:cap], metadata=metadata))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-model", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "summary_prose_results")
    args = parser.parse_args()
    verify_frozen()
    manifest = json.loads((ROOT / "capsule_manifest.json").read_text())
    for name, sha in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Changed capsule input: {name}")
    proposal = json.loads((ROOT / "model_proposal.json").read_text())
    if (
        proposal["check_version"] != CHECK_VERSION
        or proposal["prompt_sha256"] != hashlib.sha256(SUMMARY_PROMPT.encode()).hexdigest()
    ):
        raise ValueError("Prose prompt or grounding check differs from the frozen proposal")
    if args.approved_model != proposal["model"]:
        raise PermissionError("Approval must name the proposed model exactly")
    selection = json.loads((ROOT / "selection.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    defaults = Settings.model_construct()
    settings = Settings(
        data_dir=args.output / "index",
        embedding_provider="huggingface",
        compute_device="cuda",
        embedding_dtype="float32",
        embedding_max_tokens=defaults.embedding_max_tokens,
        embedding_query_instruction=defaults.embedding_query_instruction,
        local_embedding_model=protocol["embedding"]["model"],
        embedding_dims=protocol["embedding"]["dims"],
        retrieval_k=protocol["depth"],
        dense_weight=defaults.dense_weight,
        bm25_weight=defaults.bm25_weight,
        rrf_k=defaults.rrf_k,
        eval_bootstrap_samples=protocol["bootstrap"]["samples"],
        eval_bootstrap_seed=protocol["bootstrap"]["seed"],
        eval_bootstrap_confidence=protocol["bootstrap"]["confidence"],
        summary_model=proposal["model"],
        summary_prose_max_tokens=proposal["decoding"]["max_new_tokens"],
        summary_max_chars=protocol["summary_caps"]["facts"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("E3 requires a Colab CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    runtime = {
        "label_review": "unreviewed by a person",
        "approved_model": args.approved_model,
        "revision": proposal["revision"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "capsule_manifest_sha256": hashlib.sha256(
            (ROOT / "capsule_manifest.json").read_bytes()
        ).hexdigest(),
        "model_judge": False,
    }
    model = TransformersSummarizer(settings, approved=True, revision=proposal["revision"])
    cache = _Cache(args.output / "raw_prose_cache.json")
    generated = {}
    for name in REPOS:
        facts = json.loads((ROOT / f"facts/{name}.json").read_text())
        generated[name] = generate(facts, model, cache, settings.summary_max_chars)
        write(
            args.output / f"{name}_prose_summaries.json",
            [{"text": d.page_content, "metadata": d.metadata} for d in generated[name]],
        )
    write(
        args.output / "prose_audit.json",
        {
            "paragraphs": model.audit.paragraphs,
            "rejected": model.audit.rejected,
            "violation_rate": model.audit.violation_rate,
            "cache_hits": sum(r["cache_hit"] for r in model.records),
            "records": model.records,
            "check_version": CHECK_VERSION,
            "limitation": "lexical grounding only, not semantic entailment",
        },
    )
    runtime["prose_model_loaded"] = model.model is not None
    runtime["prose_peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write(args.output / "runtime.json", runtime)
    # Release the prose model before loading the embedder; these are separate models.
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    from repo_rag.store.embeddings import get_embeddings

    encoder = get_embeddings(settings)
    query_embeddings = QueryEmbeddings(json.loads((ROOT / "queries.json").read_text()))
    store = VectorStore(settings, query_embeddings)
    for name, (slug, _, _) in REPOS.items():
        code = json.loads((ROOT / f"code/{name}.json").read_text())
        store.replace_records(
            "code",
            slug,
            code["ids"],
            code["texts"],
            code["metas"],
            np.load(ROOT / f"code/{name}.npy", allow_pickle=False),
        )
        Bm25Index.build(
            [
                Document(page_content=t, metadata=m)
                for t, m in zip(code["texts"], code["metas"], strict=True)
            ]
        ).save(bm25_path(settings.bm25_dir, slug, "code"))
    questions = [q for name in REPOS for q in rows(OUT / f"{name}.jsonl") if q["split"] == "test"]
    code_q = rows(OUT / "requests_code_regression.jsonl")
    ret = HybridRetriever(store, settings)
    results = {}
    for variant in ("facts", "prose"):
        for name, (slug, _, _) in REPOS.items():
            docs = (
                generated[name]
                if variant == "prose"
                else [
                    Document(page_content=r["text"], metadata=r["metadata"])
                    for r in json.loads((ROOT / f"facts/{name}.json").read_text())
                ]
            )
            vectors = np.asarray(
                encoder.embed_documents([d.page_content for d in docs]), dtype=np.float32
            )
            store.replace_records(
                "summaries",
                slug,
                [d.metadata["chunk_id"] for d in docs],
                [d.page_content for d in docs],
                [d.metadata for d in docs],
                vectors,
            )
            Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, "summaries"))
        ret.forget_cached_indexes()
        arms = ["chunks", selection["router"]]
        arch = evaluate(ret, questions, protocol, variant, arms)
        code = evaluate(ret, code_q, protocol, variant, arms)
        write(args.output / f"{variant}_test.json", arch)
        write(args.output / f"{variant}_code.json", code)
        results[variant] = {"arch": arch, "code": code}
    arm = selection["router"]
    result = report(results["prose"]["arch"], results["facts"]["arch"], arm, settings)
    left = [r for r in results["prose"]["code"] if r["method"] == arm]
    control = [r for r in results["facts"]["code"] if r["method"] == "chunks"]
    result["code"] = bootstrap_intervals(left, CODE_METRICS, settings)
    result["code_delta"] = paired_intervals(left, control, CODE_METRICS, settings)
    result["delta_over_facts"] = paired_intervals(
        [r for r in results["prose"]["arch"] if r["method"] == arm],
        [r for r in results["facts"]["arch"] if r["method"] == arm],
        METRICS,
        settings,
    )
    result["decision"] = decision(result, result["code_delta"]["mrr@10"]["estimate"], "hybrid")
    write(
        args.output / "report.json",
        {
            "label_review": "unreviewed by a person",
            "proposal": proposal,
            "selection": selection,
            "approval": {"approved_model": args.approved_model, "mechanism": "explicit CLI flag"},
            "result": result,
        },
    )
    print(
        json.dumps(
            {
                "delta_over_facts": result["delta_over_facts"]["recall@5"],
                "decision": result["decision"],
                "label_review": "unreviewed by a person",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
