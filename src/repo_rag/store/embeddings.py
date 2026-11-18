"""Local embeddings: Qwen3-Embedding via sentence-transformers, as a LangChain `Embeddings`.

This reproduces the setup `eval/embedding_bench` measured (see eval/RESULTS.md):
documents are encoded without a prompt, queries with the task instruction, left padding, a
2,048-token cap, then Matryoshka truncation to `embedding_dims` and re-normalisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from langchain_core.embeddings import Embeddings

from repo_rag.config import Settings, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(setting: str, device: str) -> str:
    """The torch dtype name: `auto` is bfloat16 on a GPU and float32 on a CPU."""
    if setting != "auto":
        return setting
    return "bfloat16" if device == "cuda" else "float32"


def truncate_and_normalise(vectors: np.ndarray, dims: int | None) -> np.ndarray:
    """Keep the first `dims` components (Matryoshka) and re-normalise to unit length."""
    if dims and dims < vectors.shape[1]:
        vectors = vectors[:, :dims]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    return (vectors / norms).astype(np.float32)


class LocalEmbeddings(Embeddings):
    """Loads the model on first use, so importing or constructing this is free."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = resolve_device(self.settings.compute_device)
            dtype = resolve_dtype(self.settings.embedding_dtype, device)
            # Qwen3 pools the last token, so padding must be on the left. (sentence-transformers
            # 5.7 renamed `tokenizer_kwargs` to `processor_kwargs`; pyproject pins >=5.7.)
            self._model = SentenceTransformer(
                self.settings.local_embedding_model,
                device=device,
                model_kwargs={"dtype": getattr(torch, dtype)},
                processor_kwargs={"padding_side": "left"},
            )
            self._model.max_seq_length = self.settings.embedding_max_tokens
        return self._model

    def _encode(self, texts: list[str], prompt: str | None) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return truncate_and_normalise(vectors, self.settings.embedding_dims).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, prompt=None)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], prompt=self.settings.embedding_query_instruction)[0]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    if settings.embedding_provider != "huggingface":
        raise NotImplementedError(
            f"embedding_provider={settings.embedding_provider!r} is not wired up yet. "
            "Only the local Qwen3 model has been benchmarked; use 'huggingface'."
        )
    return LocalEmbeddings(settings)
