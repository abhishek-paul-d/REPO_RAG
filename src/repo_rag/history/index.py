"""Commit collection indexing, separate from the code/docs indexing command."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from repo_rag.config import Settings
from repo_rag.history.commit_indexer import commit_document, iter_commits, prefetch_blobs
from repo_rag.history.issue_linker import GitHubClient, Issue, IssueLink, link_commit
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.manifest import Manifest, RepoEntry
from repo_rag.store.vectorstore import SyncStats, VectorStore


@dataclass(frozen=True)
class HistoryIndexResult:
    commits: int
    fix_candidates: int
    direct_links: int
    via_pr_links: int
    sync: SyncStats
    prefetched_blobs: int = 0


def index_history(
    slug: str,
    root: Path,
    settings: Settings,
    store: VectorStore,
    *,
    ref: str = "HEAD",
    client: GitHubClient | None = None,
) -> HistoryIndexResult:
    from git import Repo

    with Repo(root) as repo:
        pinned = repo.commit(ref).hexsha
    manifest = Manifest(settings.manifest_path)
    previous = manifest.get(slug)
    if previous and previous.embedding_id != settings.embedding_id:
        raise ValueError("Existing code/docs use another embedding space; use a separate data_dir")
    prefetched = prefetch_blobs(root, settings, ref=pinned)
    records = list(iter_commits(root, slug, settings, ref=pinned))
    links: list[IssueLink] = []
    issues: dict[tuple[str, int], Issue] = {}
    if client:
        # An apparently ordinary message can still close an issue through its PR.
        for record in records:
            found, fetched = link_commit(slug, record.sha, record.message, client)
            links.extend(found)
            issues.update(((issue.repo, issue.number), issue) for issue in fetched)
    elif settings.metadata_db_path.exists():
        # An offline refresh preserves already verified links for retained SHAs.
        with MetadataDB(settings.metadata_db_path) as db:
            retained = {r.sha for r in records}
            links = [link for link in db.links(slug) if link.commit_sha in retained]
            for link in links:
                issue = db.get_issue(link.issue_repo, link.issue_number)
                if issue:
                    issues[(issue.repo, issue.number)] = issue
    linked = {link.commit_sha for link in links}
    docs = [commit_document(r, settings, linked_fix=r.sha in linked) for r in records]
    sync = store.sync("commits", slug, docs)
    Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, "commits"))
    with MetadataDB(settings.metadata_db_path, readonly=False) as db:
        db.sync_history(slug, records, links, list(issues.values()))
    entry = previous or RepoEntry(slug, pinned, settings.embedding_id, "", str(root))
    entry.chunks["commits"] = len(docs)
    entry.history_sha = pinned
    entry.indexed_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.put(entry)
    return HistoryIndexResult(
        len(records),
        sum(d.metadata["is_fix"] == "true" for d in docs),
        sum(link.link_type == "direct" for link in links),
        sum(link.link_type == "via_pr" for link in links),
        sync,
        prefetched,
    )
