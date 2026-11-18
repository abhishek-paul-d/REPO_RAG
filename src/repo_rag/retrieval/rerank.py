"""Qwen3-Reranker: P("yes") for each (query, document) pair, as on the model card.

Measured on bug report -> fix commit and in Phase 4 code experiments. Code reranking remains
opt-in. HybridRetriever selects the task instruction; the constructor default is for commits.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Protocol

from repo_rag.config import Settings, get_settings
from repo_rag.store.embeddings import resolve_device

log = logging.getLogger(__name__)

# Verbatim from the Qwen3-Reranker model card.
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_MAX_QUERY_CHARS = 2000


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


def format_pair(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query[:_MAX_QUERY_CHARS]}\n<Document>: {document}"


def combine_with_first_stage(probabilities: Sequence[float], weight: float) -> list[float]:
    """Log-odds of P(yes) minus weight * ln(first-stage rank). Rank is the 1-based list position.

    Log-odds alone orders exactly like the probability, but it keeps differences visible where
    the probability saturates near 1 (bfloat16 scores of 0.9991 and 0.9990 are a coin flip).
    """
    eps = 1e-6
    out = []
    for rank, p in enumerate(probabilities, start=1):
        p = min(max(p, eps), 1 - eps)
        out.append(math.log(p / (1 - p)) - weight * math.log(rank))
    return out


def reranking_enabled(settings: Settings) -> bool:
    """`on`/`off` are absolute; `auto` means only when a CUDA GPU is present."""
    if settings.reranker_mode == "auto":
        return resolve_device(settings.compute_device) == "cuda"
    return settings.reranker_mode == "on"


class QwenReranker:
    def __init__(
        self,
        settings: Settings | None = None,
        instruction: str | None = None,
        batch_size: int = 4,
        *,
        record_diagnostics: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.instruction = instruction or self.settings.rerank_instruction
        self.batch_size = batch_size
        self._loaded = False
        self.record_diagnostics = record_diagnostics
        self.last_pair_diagnostics: list[dict[str, int]] = []

    def pair_token_counts(self, query: str, texts: Sequence[str]) -> list[dict[str, int]]:
        """Audit token budgets outside timed model calls, after the scorer is loaded."""
        if not self._loaded:
            raise RuntimeError("load/score the ranker before auditing token budgets")
        wrappers = len(self.prefix) + len(self.suffix)
        budget = self.settings.rerank_max_tokens - wrappers
        raw = self.tokenizer(
            [format_pair(self.instruction, query, text) for text in texts],
            padding=False,
            truncation=False,
        )["input_ids"]
        return [
            {
                "raw_tokens": len(ids) + wrappers,
                "used_tokens": min(len(ids), budget) + wrappers,
                "truncated_tokens": max(0, len(ids) - budget),
            }
            for ids in raw
        ]

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = self.settings.reranker_model
        self.device = resolve_device(self.settings.compute_device)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        # transformers wraps `.to` with a signature that mypy reads as expecting a model.
        self.model = model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)
        self._loaded = True

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        import torch

        if not self._loaded:
            self._load()
        budget = self.settings.rerank_max_tokens - len(self.prefix) - len(self.suffix)
        if budget < 1:
            raise ValueError("rerank_max_tokens must fit the prompt prefix and suffix")
        self.last_pair_diagnostics = []
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            pairs = [
                format_pair(self.instruction, query, text)
                for text in texts[start : start + self.batch_size]
            ]
            encoded = self.tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                max_length=budget,
                return_attention_mask=False,
            )
            if self.record_diagnostics:
                raw = self.tokenizer(pairs, padding=False, truncation=False)["input_ids"]
                self.last_pair_diagnostics.extend(
                    {
                        "raw_tokens": len(full) + len(self.prefix) + len(self.suffix),
                        "used_tokens": len(short) + len(self.prefix) + len(self.suffix),
                        "truncated_tokens": len(full) - len(short),
                    }
                    for full, short in zip(raw, encoded["input_ids"], strict=True)
                )
            ids = [self.prefix + x + self.suffix for x in encoded["input_ids"]]
            batch = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in batch.items()}
            with torch.no_grad():
                # Only the last position is read. Without `logits_to_keep` the model builds
                # logits for every position (batch x 2,048 x 152k), a ~2 GB spike that ran a
                # 40 GB GPU out of memory next to the 14B answer model.
                logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            scores += torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()
        return scores
