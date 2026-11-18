"""Sync a repository's PRs and issues from GitHub and load them into metadata.db."""

from __future__ import annotations

from dataclasses import dataclass

from repo_rag.config import Settings
from repo_rag.history.github_sync import GitHubSync, SyncResult
from repo_rag.history.issue_linker import GitHubClient
from repo_rag.history.metadata_db import MetadataDB


@dataclass(frozen=True)
class GitHubIndexResult:
    prs: SyncResult
    issues: SyncResult


def sync_github_metadata(
    slug: str,
    settings: Settings,
    client: GitHubClient,
    *,
    max_prs: int | None = None,
    max_issues: int | None = None,
) -> GitHubIndexResult:
    """Fetch (incrementally), then replace this repo's PR/issue rows in one transaction.

    The raw-node store is only updated by a pass that finished, so a failed sync leaves both
    it and metadata.db as they were.
    """
    sync = GitHubSync(settings, client)
    prs = sync.sync_prs(slug, cap=max_prs)
    issues = sync.sync_issues(slug, cap=max_issues)
    with MetadataDB(settings.metadata_db_path, readonly=False) as db:
        db.sync_github(
            slug,
            sync.pull_requests(slug),
            sync.issues(slug),
            prs_complete=prs.complete,
            issues_complete=issues.complete,
        )
    return GitHubIndexResult(prs, issues)
