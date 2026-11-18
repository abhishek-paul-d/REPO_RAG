"""Searchable `pull_requests` and `issues` collections, built from the synced GitHub rows.

The text we embed is the title, labels and body (capped). Exact fields (state, author, merged
by, dates) stay in metadata.db and in the chunk metadata for filters; questions about them go
to the typed tools, not to search. Emails are redacted before anything is embedded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from langchain_core.documents import Document

from repo_rag.config import Settings
from repo_rag.history.commit_indexer import _EMAIL
from repo_rag.history.github_sync import IssueMeta, PullRequest
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.manifest import Manifest, RepoEntry
from repo_rag.store.vectorstore import SyncStats, VectorStore

KINDS = ("pull_requests", "issues")


@dataclass(frozen=True)
class PrIndexResult:
    pull_requests: int
    issues: int
    sync: dict[str, SyncStats]


def _clean(text: str, limit: int) -> str:
    return _EMAIL.sub("<email omitted>", text.replace("\r\n", "\n"))[:limit].strip()


def _text(label: str, number: int, title: str, labels: tuple[str, ...], body: str, cap: int) -> str:
    head = f"{label} #{number}: {_clean(title, 300)}"
    if labels:
        head += f"\nLabels: {', '.join(labels)}"
    body = _clean(body, cap)
    return f"{head}\n\n{body}" if body else head


def _base(repo: str, kind: str, number: int, text: str) -> dict[str, str | int]:
    chunk_id = hashlib.sha1(f"{repo}:{kind}:{number}".encode()).hexdigest()
    return {
        "repo": repo,
        "chunk_id": chunk_id,
        "number": number,
        "path": "",
        "language": "text",
        "start_line": 1,
        "end_line": max(1, len(text.split("\n"))),
        "symbol": "",
        "symbol_kind": kind,
        "parent_symbol": "",
        "signature": "",
        "symbols": "",
        "chunker": kind,
    }


def pr_document(pr: PullRequest, settings: Settings) -> Document:
    text = _text("PR", pr.number, pr.title, pr.labels, pr.body, settings.pr_body_chars)
    meta = _base(pr.repo, "pull_request", pr.number, text)
    meta.update(
        state=pr.state,
        author=pr.author,
        merged_by=pr.merged_by,
        labels=",".join(pr.labels),
        created_at=pr.created_at,
        merged_at=pr.merged_at,
    )
    return Document(page_content=text, metadata=meta)


def issue_document(issue: IssueMeta, settings: Settings) -> Document:
    text = _text(
        "Issue", issue.number, issue.title, issue.labels, issue.body, settings.pr_body_chars
    )
    meta = _base(issue.repo, "issue", issue.number, text)
    meta.update(
        state=issue.state,
        author=issue.author,
        labels=",".join(issue.labels),
        created_at=issue.created_at,
        closed_at=issue.closed_at,
    )
    return Document(page_content=text, metadata=meta)


def index_pr_issues(slug: str, settings: Settings, store: VectorStore) -> PrIndexResult:
    """Make both collections equal the synced rows, embedding only changed text."""
    with MetadataDB(settings.metadata_db_path) as db:
        prs = db.find_prs(slug, limit=1_000_000)
        issues = db.find_issues(slug, limit=1_000_000)
    if not prs and not issues:
        raise ValueError(f"no synced PRs or issues for {slug}; run repo-rag sync-github first")
    docs = {
        "pull_requests": [pr_document(p, settings) for p in prs],
        "issues": [issue_document(i, settings) for i in issues],
    }
    manifest = Manifest(settings.manifest_path)
    previous = manifest.get(slug)
    if previous and previous.embedding_id != settings.embedding_id:
        raise ValueError("Existing index uses another embedding space; use a separate data_dir")
    stats: dict[str, SyncStats] = {}
    for kind, items in docs.items():
        stats[kind] = store.sync(kind, slug, items)
        Bm25Index.build(items).save(bm25_path(settings.bm25_dir, slug, kind))
    entry = previous or RepoEntry(slug, "", settings.embedding_id, "", "")
    for kind, items in docs.items():
        entry.chunks[kind] = len(items)
    entry.indexed_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.put(entry)
    return PrIndexResult(len(prs), len(issues), stats)
