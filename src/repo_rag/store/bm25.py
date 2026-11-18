"""BM25 keyword index over chunks, persisted per (repo, kind).

Complements dense search for exact names (`MAX_POOL_SIZE`, `rebuild_auth`) that embeddings
blur. The index holds only chunk ids and a few filterable fields; texts live in Chroma.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_FORMAT_VERSION = 1
_WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PARTS = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
_FILTER_FIELDS = ("path", "language", "symbol_kind", "doc_kind", "is_fix", "state")

MetaFilter = Callable[[dict[str, str]], bool]


def code_tokens(text: str) -> list[str]:
    """Whole identifiers plus their camelCase / snake_case parts, lowercased.

    `HTTPAdapter.send_request` -> httpadapter, http, adapter, send_request, send, request
    """
    tokens: list[str] = []
    for word in _WORDS.findall(text):
        tokens.append(word.lower())
        parts = [p.lower() for piece in word.split("_") for p in _PARTS.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


@dataclass
class Bm25Index:
    ids: list[str]
    metas: list[dict[str, str]]
    bm25: BM25Okapi

    @classmethod
    def build(cls, docs: Sequence[Document]) -> Bm25Index:
        ids = [str(d.metadata["chunk_id"]) for d in docs]
        metas = [{f: str(d.metadata.get(f, "")) for f in _FILTER_FIELDS} for d in docs]
        # BM25Okapi can't be built from an empty corpus.
        corpus = [code_tokens(d.page_content) or ["<empty>"] for d in docs] or [["<empty>"]]
        return cls(ids=ids, metas=metas, bm25=BM25Okapi(corpus))

    def __len__(self) -> int:
        return len(self.ids)

    def search(
        self, query: str, n: int, accept: MetaFilter | None = None
    ) -> list[tuple[str, float]]:
        """Top `n` (chunk_id, score) with a positive score, best first."""
        if not self.ids:
            return []
        scores = np.asarray(self.bm25.get_scores(code_tokens(query)), dtype=np.float64)
        if accept is not None:
            mask = np.fromiter((accept(m) for m in self.metas), dtype=bool, count=len(self.ids))
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:n]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "ids": self.ids,
            "metas": self.metas,
            "bm25": self.bm25,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)  # never leave a half-written index behind

    @classmethod
    def load(cls, path: Path) -> Bm25Index | None:
        """The saved index, or None if it is missing or from another format version."""
        if not path.exists():
            return None
        payload = pickle.loads(path.read_bytes())  # our own file, written by save()
        if payload.get("version") != _FORMAT_VERSION:
            return None
        return cls(ids=payload["ids"], metas=payload["metas"], bm25=payload["bm25"])


def bm25_path(directory: Path, repo: str, kind: str) -> Path:
    return directory / f"{repo.replace('/', '__')}__{kind}.pkl"
