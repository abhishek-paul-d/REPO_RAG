"""Chroma-backed chunk store (via langchain-chroma), one collection per kind.

Collections are named after the embedding space (`code_chunks__Qwen3-Embedding-0.6B_512`), so
vectors from different models or dimensions can never be mixed. Every collection holds many
repos, told apart by the `repo` metadata field that all queries filter on.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from repo_rag.config import Settings, get_settings

COLLECTIONS = {"code": "code_chunks", "docs": "docs"}
# File indexing/legacy transfers still iterate COLLECTIONS. History has its own snapshot.
SEARCH_COLLECTIONS = {
    **COLLECTIONS,
    "commits": "commits",
    "pull_requests": "pull_requests",
    "issues": "issues",
    "summaries": "summaries",
}
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class SyncStats:
    added: int = 0
    updated: int = 0  # text changed, so re-embedded
    metadata_only: int = 0  # same text, new metadata (e.g. a new commit sha); not re-embedded
    unchanged: int = 0
    removed: int = 0
    reused: int = 0  # of added + updated: the vector was copied from a chunk with the same text

    @property
    def embedded(self) -> int:
        """Chunks that actually went through the embedding model."""
        return self.added + self.updated - self.reused


def build_where(repo: str, filters: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A Chroma `where` clause: always the repo, plus exact-match metadata filters."""
    clauses: list[dict[str, Any]] = [{"repo": repo}]
    clauses += [{key: value} for key, value in (filters or {}).items() if value]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _same_text_donors(
    to_embed: Sequence[str], by_id: Mapping[str, Document], old_text: Mapping[str, str]
) -> dict[str, str]:
    """{chunk id: id of a stored chunk with identical text} for chunks that need a vector."""
    wanted = {by_id[i].page_content for i in to_embed}
    donor_of_text: dict[str, str] = {}
    for chunk_id, text in old_text.items():
        if text in wanted:
            donor_of_text.setdefault(text, chunk_id)
    return {
        chunk_id: donor_of_text[by_id[chunk_id].page_content]
        for chunk_id in to_embed
        if by_id[chunk_id].page_content in donor_of_text
    }


class VectorStore:
    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if embeddings is None:
            from repo_rag.store.embeddings import get_embeddings

            embeddings = get_embeddings(self.settings)
        self.embeddings = embeddings
        self._client = client
        self._stores: dict[str, Chroma] = {}

    # ---------------------------------------------------------------- collections

    def collection_name(self, kind: str) -> str:
        suffix = _UNSAFE_NAME_CHARS.sub("-", self.settings.embedding_id)
        return f"{SEARCH_COLLECTIONS[kind]}__{suffix}"

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        return self._client

    def _read(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        return self._store(kind).get(**kwargs)

    def _store(self, kind: str) -> Chroma:
        if kind not in self._stores:
            threshold = self.settings.chroma_sync_threshold
            store = Chroma(
                collection_name=self.collection_name(kind),
                embedding_function=self.embeddings,
                client=self.client,
                collection_configuration={"hnsw": {"space": "cosine", "sync_threshold": threshold}},
            )
            # Collections made before this setting existed keep Chroma's lazy default (1000):
            # switch them too, so every read can find the vector index on disk.
            store._collection.modify(configuration={"hnsw": {"sync_threshold": threshold}})
            self._stores[kind] = store
        return self._stores[kind]

    # ---------------------------------------------------------------- writing

    def sync(
        self,
        kind: str,
        repo: str,
        docs: Sequence[Document],
        *,
        force: bool = False,
        dry_run: bool = False,
        batch_size: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SyncStats:
        """Make the repo's chunks in this collection equal `docs`, embedding only what changed.

        `dry_run` writes nothing and loads no model: it returns the counts the real call would
        produce, so a caller can see how many chunks need embedding first.

        Unchanged chunks are neither re-embedded nor rewritten. Chunks no longer produced are
        deleted. Safe to run repeatedly.

        A chunk id includes its start line, so code that only moved (a line added above it)
        arrives under a new id with identical text. Its vector is copied from the stored chunk
        with that text instead of being computed again (`SyncStats.reused`); `force` disables this.
        """
        store = self._store(kind)
        by_id: dict[str, Document] = {}
        for doc in docs:
            by_id[str(doc.metadata["chunk_id"])] = doc  # ids are unique by construction

        existing = self._read(kind, where=build_where(repo), include=["documents", "metadatas"])
        old_text = dict(zip(existing["ids"], existing["documents"], strict=True))
        old_meta = dict(zip(existing["ids"], existing["metadatas"], strict=True))

        stats = SyncStats()
        to_embed: list[str] = []
        meta_only: list[str] = []
        for chunk_id, doc in by_id.items():
            if chunk_id not in old_text:
                to_embed.append(chunk_id)
                stats.added += 1
            elif force or old_text[chunk_id] != doc.page_content:
                to_embed.append(chunk_id)
                stats.updated += 1
            elif old_meta[chunk_id] != doc.metadata:
                meta_only.append(chunk_id)
                stats.metadata_only += 1
            else:
                stats.unchanged += 1

        size = batch_size or self.settings.index_write_batch
        donor_of = {} if force else _same_text_donors(to_embed, by_id, old_text)
        stats.reused = len(donor_of)
        stale = [chunk_id for chunk_id in old_text if chunk_id not in by_id]
        if dry_run:
            stats.removed = len(stale)
            return stats

        # Read the donors' vectors before the stale chunks are deleted: a moved chunk's old
        # copy is exactly what gets removed.
        reuse = self._read_vectors(kind, donor_of, size)
        fresh = [chunk_id for chunk_id in to_embed if chunk_id not in reuse]
        stats.reused = len(reuse)

        if stale:
            store.delete(ids=stale)
            stats.removed = len(stale)
        if meta_only:
            store._collection.update(
                ids=meta_only, metadatas=[by_id[i].metadata for i in meta_only]
            )

        done = 0
        reused_ids = list(reuse)
        for start in range(0, len(reused_ids), size):
            batch = reused_ids[start : start + size]
            store._collection.upsert(
                ids=batch,
                embeddings=[reuse[i] for i in batch],
                documents=[by_id[i].page_content for i in batch],
                metadatas=[by_id[i].metadata for i in batch],
            )
            done += len(batch)
            if on_progress:
                on_progress(done, len(to_embed))
        for start in range(0, len(fresh), size):
            batch = fresh[start : start + size]
            store.add_texts(
                [by_id[i].page_content for i in batch],
                metadatas=[by_id[i].metadata for i in batch],
                ids=batch,
            )
            done += len(batch)
            if on_progress:
                on_progress(done, len(to_embed))
        return stats

    def _read_vectors(
        self, kind: str, donor_of: Mapping[str, str], size: int
    ) -> dict[str, np.ndarray]:
        """{chunk id: vector} copied from each chunk's donor (a stored chunk with the same text)."""
        donors = sorted(set(donor_of.values()))
        vectors: dict[str, np.ndarray] = {}
        for start in range(0, len(donors), size):
            got = self._read(kind, ids=donors[start : start + size], include=["embeddings"])
            for donor_id, vector in zip(got["ids"], got["embeddings"], strict=True):
                vectors[donor_id] = np.asarray(vector, dtype=np.float32)
        return {
            chunk_id: vectors[donor_id]
            for chunk_id, donor_id in donor_of.items()
            if donor_id in vectors
        }

    def export_records(
        self, kind: str, repo: str
    ) -> tuple[list[str], list[str], list[dict[str, Any]], np.ndarray]:
        """Everything stored for a repo, sorted by id: (ids, texts, metadatas, vectors)."""
        got = self._read(
            kind, where=build_where(repo), include=["documents", "metadatas", "embeddings"]
        )
        order = sorted(range(len(got["ids"])), key=lambda i: got["ids"][i])
        ids = [got["ids"][i] for i in order]
        texts = [got["documents"][i] for i in order]
        metas = [dict(got["metadatas"][i]) for i in order]
        vectors = np.asarray([got["embeddings"][i] for i in order], dtype=np.float32)
        return ids, texts, metas, vectors

    def replace_records(
        self,
        kind: str,
        repo: str,
        ids: Sequence[str],
        texts: Sequence[str],
        metas: Sequence[Mapping[str, Any]],
        vectors: np.ndarray,
        *,
        batch_size: int = 512,
    ) -> int:
        """Make the repo's chunks exactly these records, using the given vectors as they are
        (nothing is re-embedded). Returns how many stale chunks were deleted."""
        store = self._store(kind)
        keep = set(ids)
        current = self._read(kind, where=build_where(repo), include=[])["ids"]
        stale = [i for i in current if i not in keep]
        if stale:
            store.delete(ids=stale)
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            store._collection.upsert(
                ids=list(ids[start:end]),
                documents=list(texts[start:end]),
                metadatas=[dict(m) for m in metas[start:end]],
                embeddings=vectors[start:end].tolist(),
            )
        return len(stale)

    def delete_repo(self, repo: str) -> None:
        for kind in COLLECTIONS:
            ids = self._read(kind, where=build_where(repo), include=[])["ids"]
            if ids:
                self._store(kind).delete(ids=ids)

    # ---------------------------------------------------------------- reading

    def count(self, kind: str, repo: str) -> int:
        return len(self._read(kind, where=build_where(repo), include=[])["ids"])

    def search(
        self,
        kind: str,
        repo: str,
        query: str,
        n: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[Document, float]]:
        """Nearest chunks as (document, cosine similarity), best first."""
        if n <= 0:
            return []
        vector = self.embeddings.embed_query(query)
        results = self._store(kind).similarity_search_by_vector_with_relevance_scores(
            vector, k=n, filter=build_where(repo, filters)
        )
        return [(doc, 1.0 - float(distance)) for doc, distance in results]

    def get_documents(self, kind: str, repo: str, ids: Sequence[str]) -> dict[str, Document]:
        """Chunks by id (ids missing from the collection are simply absent)."""
        if not ids:
            return {}
        found = self._read(
            kind, ids=list(ids), where=build_where(repo), include=["documents", "metadatas"]
        )
        return {
            chunk_id: Document(page_content=text, metadata=meta, id=chunk_id)
            for chunk_id, text, meta in zip(
                found["ids"], found["documents"], found["metadatas"], strict=True
            )
        }

    def get_records(
        self, kind: str, repo: str, ids: Sequence[str], *, batch_size: int = 512
    ) -> tuple[list[str], list[Document], np.ndarray]:
        """Fetch chunks and their stored float32 vectors for a subset of ids (no re-embedding)."""
        unique = list(dict.fromkeys(ids))
        if not unique:
            return [], [], np.empty((0, 0), dtype=np.float32)
        found_ids: list[str] = []
        found_docs: list[Document] = []
        found_vecs: list[np.ndarray] = []
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            got = self._read(
                kind,
                ids=batch,
                where=build_where(repo),
                include=["documents", "metadatas", "embeddings"],
            )
            if not got["ids"]:
                continue
            for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=True):
                found_ids.append(str(cid))
                found_docs.append(Document(page_content=text, metadata=meta, id=str(cid)))
            found_vecs.append(np.asarray(got["embeddings"], dtype=np.float32))
        if not found_vecs:
            return [], [], np.empty((0, 0), dtype=np.float32)
        return found_ids, found_docs, np.vstack(found_vecs)

    def repo_documents(
        self, kind: str, repo: str, filters: Mapping[str, str] | None = None
    ) -> list[Document]:
        """Text and metadata only, scoped to a repo; no embedding or model calls."""
        found = self._read(
            kind, where=build_where(repo, filters), include=["documents", "metadatas"]
        )
        docs = [
            Document(page_content=text, metadata=meta, id=cid)
            for cid, text, meta in zip(
                found["ids"], found["documents"], found["metadatas"], strict=True
            )
        ]
        return sorted(
            docs,
            key=lambda d: (
                str(d.metadata["path"]),
                int(d.metadata["start_line"]),
                str(d.metadata["chunk_id"]),
            ),
        )
