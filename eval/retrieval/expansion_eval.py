"""Frozen evidence-expansion protocol and measurements; no automatic production changes."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from eval.retrieval.ambiguous import AmbiguousCase
from eval.retrieval.comparison_protocol import protocol_settings
from eval.retrieval.metrics import coverage_scores

METHODS = ("plain", "callee", "both")
METRICS = ("mrr@10", "any_complete_proof@10")
GUARD_STRATA = ("ambiguous", "multi_block", "implementation", "test", "example")


def fixed_settings(protocol: dict) -> dict[str, Any]:
    if protocol["methods"] != list(METHODS):
        raise ValueError("unsupported expansion matrix")
    return {
        **protocol_settings(protocol["retrieval"]),
        "header_defines": False,
        "rerank_candidate_depth": 20,
        "evidence_expansion": "none",
        **{
            f"expansion_{key}": protocol["expansion"][key]
            for key in ("keep", "seeds", "per_seed", "max_definers")
        },
    }


def validate_multiblock(cases: list[AmbiguousCase], docs: list[Document]) -> dict:
    """Source/corpus answerability only; no query search or expansion output is inspected."""
    for case in cases:
        if case.stratum != "multi_block" or len(case.accepted_evidence) != 1:
            raise ValueError(
                "fresh questions must each define one jointly required multi-block proof"
            )
        q = case.question
        if coverage_scores(q, docs, 0.9)[1] != 1:
            raise ValueError(f"required proof missing from full corpus: {q.id}")
        if any(coverage_scores(q, [d], 0.9)[1] == 1 for d in docs):
            raise ValueError(f"proof fits in one indexed chunk: {q.id}")
    return {"questions": len(cases), "all_corpus_answerable": True, "single_chunk_sufficient": 0}


def volume(hits, ranked) -> dict:
    """Count actual returned text including contextual headers; also expose displacement."""
    original = {h.chunk_id for h in ranked}
    final = {h.chunk_id for h in hits}
    return {
        "returned_chunks": len(hits),
        "returned_chars": sum(len(h.document.page_content) for h in hits),
        "expanded_ids": [h.chunk_id for h in hits if h.chunk_id not in original],
        "displaced_ids": [h.chunk_id for h in ranked if h.chunk_id not in final],
    }


def success_rule(fresh_delta: dict, guards: dict, growth: dict, protocol: dict) -> dict:
    """Apply the pre-chosen pointwise rule. This is eligibility, never a default mutation."""
    rule = protocol["success_rule"]
    proof = fresh_delta["any_complete_proof@10"]
    checks = {
        "fresh_proof_gain": proof["estimate"] >= rule["min_fresh_proof_gain"] - 1e-12,
        "fresh_proof_lower_positive": proof["low"] > 0,
        "regression_strata": all(
            guards[stratum][m]["estimate"] >= -rule["max_regression_loss"] - 1e-12
            for stratum in GUARD_STRATA
            for m in METRICS
        ),
        "returned_text_growth": all(
            growth[cohort] <= rule["max_text_growth"] + 1e-12 for cohort in ("fresh", "regression")
        ),
    }
    return {
        "checks": checks,
        "eligible_callee": all(checks.values()),
        "production_default_changed": False,
        "next_step": "independent label review and eventual deployment assessment"
        if all(checks.values())
        else "reject expansion; proceed to Phase 5 without further tuning",
        "scope": "retrieval completeness, not generated-answer quality; both is secondary only",
    }
