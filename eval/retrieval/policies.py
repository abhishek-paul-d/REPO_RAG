"""Pre-registered rerank policies for the Phase 4 policy run, and a shared-model adapter.

Every policy reranks the same fused top 20 through the production `HybridRetriever._rerank`.
The adapter lets one resident Qwen3-Reranker model serve all task instructions and caches scores
per (instruction, query, candidates), so policies that share an instruction share exact scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from repo_rag.config import Settings

POLICIES: dict[str, dict[str, Any]] = {
    "plain": {"rerank_first_stage_weight": 0.0, "rerank_instruction_mode": "default"},
    "tiebreak": {"rerank_first_stage_weight": 0.25, "rerank_instruction_mode": "default"},
    "intent": {"rerank_first_stage_weight": 0.0, "rerank_instruction_mode": "intent"},
    "source_aware": {"rerank_first_stage_weight": 0.0, "rerank_instruction_mode": "source_aware"},
    "source_aware_tiebreak": {
        "rerank_first_stage_weight": 0.25,
        "rerank_instruction_mode": "source_aware",
    },
}
# Not part of the pre-registered five: first stage only, for the header comparison.
EXTRA_POLICIES: dict[str, dict[str, Any]] = {
    "hybrid": {"rerank_first_stage_weight": 0.0, "rerank_instruction_mode": "default"},
}
BASELINE = "plain"
DEPTH = 20


def policy_settings(settings: Settings, name: str) -> Settings:
    return settings.model_copy(
        update={**{**POLICIES, **EXTRA_POLICIES}[name], "rerank_candidate_depth": DEPTH}
    )


class SharedScorer:
    """Adapts one scorer object to the retriever's per-instruction factory."""

    def __init__(
        self, scorer: Any, instruction: str, cache: dict[tuple, list[float]], log: list[str]
    ) -> None:
        self.scorer = scorer
        self.instruction = instruction
        self.cache = cache
        self.log = log

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.log.append(self.instruction)
        key = (self.instruction, query, tuple(texts))
        if key not in self.cache:
            self.scorer.instruction = self.instruction
            self.cache[key] = list(self.scorer.score(query, texts))
        return list(self.cache[key])


def shared_factory(scorer: Any) -> Any:
    cache: dict[tuple, list[float]] = {}
    log: list[str] = []

    def factory(settings: Settings, instruction: str) -> SharedScorer:
        return SharedScorer(scorer, instruction, cache, log)

    factory.cache = cache  # type: ignore[attr-defined]
    factory.log = log  # type: ignore[attr-defined]
    return factory
