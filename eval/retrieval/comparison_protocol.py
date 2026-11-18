"""Lock the expanded experiment's settings before any scoring; production is unchanged."""

from __future__ import annotations

from typing import Any


def protocol_settings(protocol: dict) -> dict[str, Any]:
    if protocol["methods"] != ["hybrid", "rerank20", "rerank50"] or protocol[
        "rerank_candidate_depths"
    ] != [20, 50]:
        raise ValueError("unsupported comparison matrix")
    if protocol["quality_repetition"] != 0 or protocol["timing_repetitions"] != 3:
        raise ValueError("quality must use repetition zero of three timed calls")
    values = {
        key: protocol[key]
        for key in (
            "embedding_dims",
            "embedding_dtype",
            "embedding_max_tokens",
            "reranker_model",
            "rerank_max_tokens",
            "retrieval_k",
            "dense_weight",
            "bm25_weight",
            "rrf_k",
            "code_rerank_instruction",
            "embedding_query_instruction",
            "max_chunk_chars",
        )
    }
    return {
        **values,
        "embedding_provider": "huggingface",
        "local_embedding_model": protocol["embedding_model"],
        "reranker_mode": "on",
        # The frozen comparison scores plain P(yes) with the one code instruction.
        "rerank_first_stage_weight": 0.0,
        "rerank_instruction_mode": "default",
        "rerank_candidate_depth": 50,
        "eval_cutoffs": tuple(protocol["cutoffs"]),
        "eval_coverage_threshold": 0.9,
        "eval_bootstrap_samples": protocol["bootstrap"]["samples"],
        "eval_bootstrap_seed": protocol["bootstrap"]["seed"],
        "eval_bootstrap_confidence": protocol["bootstrap"]["confidence"],
    }
