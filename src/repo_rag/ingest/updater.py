"""Bring an already indexed repository up to its latest commit, embedding only what changed.

`index_repo` is already incremental where it matters: `VectorStore.sync` embeds only chunks whose
text is new, copies the vector of a chunk that merely moved, and deletes vanished ones. What this
module adds is the plan before the work (how many chunks would be embedded, so the caller can stop
before a slow CPU run) and the refresh of everything derived from the same checkout (symbols,
summaries, commit history) that was built for this repo before.

Chunking the whole tree is cheap (about a second for psf/requests), so unchanged files are not
skipped: that would save nothing measurable and would make the index depend on a diff being right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from git import GitCommandError, Repo

from repo_rag.config import Settings
from repo_rag.ingest.github_client import clone_repo, head_commit, parse_repo_url
from repo_rag.ingest.indexer import ProgressFn, chunk_repo, index_repo
from repo_rag.store.manifest import Manifest, RepoEntry
from repo_rag.store.vectorstore import SyncStats, VectorStore


class UpdateError(Exception):
    """The repository cannot be updated (not indexed, clone missing, pull failed)."""


@dataclass
class UpdatePlan:
    slug: str
    root: Path
    old_sha: str
    new_sha: str
    changed_files: int | None  # None when the old commit is not in the clone
    sync: dict[str, SyncStats] = field(default_factory=dict)

    @property
    def up_to_date(self) -> bool:
        return self.old_sha == self.new_sha

    @property
    def to_embed(self) -> int:
        return sum(stats.embedded for stats in self.sync.values())


def indexed_entry(slug: str, settings: Settings, checkout: Path | None = None) -> RepoEntry:
    """The manifest entry to update. `checkout` names the clone to use when none is recorded
    (an index imported from Colab has no clone) and is recorded by the update that follows."""
    entry = Manifest(settings.manifest_path).get(slug)
    if entry is None:
        raise UpdateError(f"{slug} is not indexed; run `repo-rag index` or import an index first")
    if entry.embedding_id != settings.embedding_id:
        raise UpdateError(f"{slug} was indexed under another embedding ({entry.embedding_id})")
    if checkout is not None:
        if not (checkout / ".git").exists():
            raise UpdateError(f"{checkout} is not a git clone")
        entry.clone_path = str(checkout.resolve())
    if not entry.clone_path or not Path(entry.clone_path).is_dir():
        raise UpdateError(
            f"No local clone for {slug} (recorded: {entry.clone_path or 'none'}). "
            "An imported index has none: pass --checkout <path to a clone of the repository>."
        )
    return entry


def fast_forward(slug: str, root: Path) -> None:
    """Fetch the default branch into the existing clone (fast-forward only)."""
    try:
        clone_repo(parse_repo_url(slug).clone_url, root)
    except (GitCommandError, FileExistsError) as exc:
        raise UpdateError(f"Could not update the clone of {slug}: {exc}") from exc


def count_changed_files(root: Path, old_sha: str, new_sha: str) -> int | None:
    try:
        with Repo(root) as repo:
            return len(repo.git.diff("--name-only", old_sha, new_sha).splitlines())
    except GitCommandError:
        return None  # e.g. the recorded commit is not in this clone


def plan_update(
    entry: RepoEntry,
    settings: Settings,
    store: VectorStore,
    on_progress: ProgressFn | None = None,
) -> UpdatePlan:
    """What an update would do, without writing anything or loading the embedding model."""
    root = Path(entry.clone_path)
    new_sha = head_commit(root)
    plan = UpdatePlan(
        slug=entry.slug,
        root=root,
        old_sha=entry.commit_sha,
        new_sha=new_sha,
        changed_files=count_changed_files(root, entry.commit_sha, new_sha),
    )
    chunks, _ = chunk_repo(entry.slug, root, new_sha, settings, on_progress)
    for kind, docs in chunks.items():
        plan.sync[kind] = store.sync(kind, entry.slug, docs, dry_run=True)
    return plan


@dataclass
class UpdateResult:
    plan: UpdatePlan
    embedded: dict[str, int]
    refreshed: list[str]  # derived indexes rebuilt from the new checkout


def apply_update(
    plan: UpdatePlan,
    entry: RepoEntry,
    settings: Settings,
    store: VectorStore,
    on_progress: ProgressFn | None = None,
) -> UpdateResult:
    """Index the new commit, then refresh the symbol table, summaries and history if they exist."""
    result = index_repo(
        plan.slug, plan.root, plan.new_sha, store=store, settings=settings, on_progress=on_progress
    )
    refreshed: list[str] = []

    from repo_rag.ingest.symbol_indexer import index_symbols
    from repo_rag.store.symbol_table import SymbolTable

    had_symbols = False
    if settings.metadata_db_path.exists():
        with SymbolTable(settings.metadata_db_path) as table:
            had_symbols = table.counts(plan.slug)[0] > 0
    if had_symbols:
        index_symbols(plan.slug, plan.root, settings)
        refreshed.append("symbols")

    if store.count("summaries", plan.slug) > 0:
        from repo_rag.summaries.index import index_summaries

        index_summaries(plan.slug, plan.root, settings, store)
        refreshed.append("summaries")

    if entry.history_sha:
        from repo_rag.history.index import index_history

        # Offline: links already verified are kept; new commits are not linked to issues here.
        index_history(plan.slug, plan.root, settings, store)
        refreshed.append("history")

    return UpdateResult(
        plan, {kind: stats.embedded for kind, stats in result.sync.items()}, refreshed
    )
