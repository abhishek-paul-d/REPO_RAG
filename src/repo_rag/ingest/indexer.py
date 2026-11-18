"""Index a cloned repo: walk files, chunk them, embed new or changed chunks, build BM25.

Code (any language) goes to the `code` collection through the AST chunker, which falls back to
the text splitter by itself. Docs and config files go to the `docs` collection. Re-running is
cheap: unchanged chunks are not re-embedded and vanished ones are deleted.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from repo_rag.chunking.ast_chunker import AstChunker
from repo_rag.chunking.docs_chunker import chunk_doc_file
from repo_rag.config import Settings, get_settings
from repo_rag.ingest.file_walker import walk_repo
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.manifest import Manifest, RepoEntry
from repo_rag.store.vectorstore import SyncStats, VectorStore

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]  # (stage, done, total)


@dataclass
class IndexResult:
    slug: str
    commit_sha: str
    files: Counter[str] = field(default_factory=Counter)
    chunks: dict[str, int] = field(default_factory=dict)
    sync: dict[str, SyncStats] = field(default_factory=dict)
    chunkers: Counter[str] = field(default_factory=Counter)
    seconds: float = 0.0


def _dedupe(docs: list[Document]) -> list[Document]:
    unique: dict[str, Document] = {}
    for doc in docs:
        unique[str(doc.metadata["chunk_id"])] = doc
    if len(unique) != len(docs):
        log.warning("dropped %d chunks with duplicate ids", len(docs) - len(unique))
    return list(unique.values())


def chunk_repo(
    slug: str,
    root: Path,
    commit_sha: str,
    settings: Settings,
    on_progress: ProgressFn | None = None,
) -> tuple[dict[str, list[Document]], Counter[str]]:
    """Chunk every indexable file: {"code": [...], "docs": [...]} plus file counts by kind."""
    chunker = AstChunker(settings)
    files = list(walk_repo(root, max_file_bytes=settings.max_file_bytes))
    chunks: dict[str, list[Document]] = {"code": [], "docs": []}
    counts: Counter[str] = Counter()
    for i, f in enumerate(files, start=1):
        counts[f.kind] += 1
        source = f.abs_path.read_text(encoding="utf-8", errors="replace")
        if f.kind == "code":
            chunks["code"] += chunker.chunk(
                source, repo=slug, commit_sha=commit_sha, path=f.path, language=f.language
            )
        else:
            chunks["docs"] += chunk_doc_file(
                source, repo=slug, commit_sha=commit_sha, path=f.path, settings=settings
            )
        if on_progress:
            on_progress("chunking", i, len(files))
    return {kind: _dedupe(docs) for kind, docs in chunks.items()}, counts


def index_repo(
    slug: str,
    root: Path,
    commit_sha: str,
    *,
    store: VectorStore,
    settings: Settings | None = None,
    force: bool = False,
    on_progress: ProgressFn | None = None,
) -> IndexResult:
    settings = settings or get_settings()
    start = time.perf_counter()
    chunks, counts = chunk_repo(slug, root, commit_sha, settings, on_progress)

    result = IndexResult(slug=slug, commit_sha=commit_sha, files=counts)
    for kind, docs in chunks.items():
        result.chunks[kind] = len(docs)
        result.chunkers.update(d.metadata["chunker"] for d in docs)

        def progress(done: int, total: int, kind: str = kind) -> None:
            if on_progress:
                on_progress(f"embedding {kind}", done, total)

        result.sync[kind] = store.sync(kind, slug, docs, force=force, on_progress=progress)
        Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, kind))

    manifest = Manifest(settings.manifest_path)
    previous = manifest.get(slug)
    if previous and previous.embedding_id == settings.embedding_id and "commits" in previous.chunks:
        result.chunks["commits"] = previous.chunks["commits"]
    manifest.put(
        RepoEntry(
            slug=slug,
            commit_sha=commit_sha,
            embedding_id=settings.embedding_id,
            indexed_at=datetime.now(UTC).isoformat(timespec="seconds"),
            clone_path=str(root),
            history_sha=(
                previous.history_sha
                if previous and previous.embedding_id == settings.embedding_id
                else ""
            ),
            files=dict(counts),
            chunks=result.chunks,
        )
    )
    result.seconds = time.perf_counter() - start
    return result
