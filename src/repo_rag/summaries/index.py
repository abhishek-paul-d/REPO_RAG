"""Embed the repo map into the `summaries` collection (like `index_pr_issues`)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from repo_rag.config import Settings
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.manifest import Manifest, RepoEntry
from repo_rag.store.vectorstore import SyncStats, VectorStore
from repo_rag.summaries.repo_map import Summarizer, SummaryResult, build_summaries


@dataclass(frozen=True)
class SummaryIndexResult:
    summary: SummaryResult
    sync: SyncStats


def index_summaries(
    slug: str,
    checkout: Path,
    settings: Settings,
    store: VectorStore,
    *,
    summarizer: Summarizer | None = None,
) -> SummaryIndexResult:
    """Make the `summaries` collection equal the checkout's repo map, embedding only changes."""
    manifest = Manifest(settings.manifest_path)
    previous = manifest.get(slug)
    if previous and previous.embedding_id != settings.embedding_id:
        raise ValueError("Existing index uses another embedding space; use a separate data_dir")
    docs, result = build_summaries(slug, checkout, settings, summarizer=summarizer)
    if not result.files:
        raise ValueError(f"no code files to summarise in {checkout}")
    stats = store.sync("summaries", slug, docs)
    Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, "summaries"))
    entry = previous or RepoEntry(slug, "", settings.embedding_id, "", "")
    entry.chunks["summaries"] = len(docs)
    entry.indexed_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.put(entry)
    return SummaryIndexResult(result, stats)
