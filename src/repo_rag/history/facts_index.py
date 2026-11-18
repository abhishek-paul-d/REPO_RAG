"""Load tags, name merges and login links from a local checkout into metadata.db."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repo_rag.config import Settings
from repo_rag.history.git_facts import (
    read_identity_map,
    read_login_links,
    read_tags,
)
from repo_rag.history.metadata_db import MetadataDB


@dataclass(frozen=True)
class FactsResult:
    tags: int
    merged_names: int
    ambiguous_names: int
    login_links: int


def index_git_facts(
    slug: str, checkout: Path, settings: Settings, *, ref: str = "HEAD"
) -> FactsResult:
    """Replace this repo's tags and identity rows. Login links use the PRs already synced."""
    tags = read_tags(checkout, slug)
    identities = read_identity_map(checkout, ref=ref, max_commits=settings.max_commits)
    with MetadataDB(settings.metadata_db_path, readonly=False) as db:
        prs = db.find_prs(slug, state="MERGED", limit=1_000_000)
        links = read_login_links(checkout, slug, prs, identities)
        db.sync_git_facts(slug, tags, identities, links)
    return FactsResult(len(tags), len(identities.canonical), len(identities.ambiguous), len(links))
