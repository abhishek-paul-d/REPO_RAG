"""`repo-rag` command-line interface."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from git.exc import GitCommandError, GitError
from pydantic import SecretStr
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from repo_rag import __version__
from repo_rag.chunking.common import split_header
from repo_rag.config import Settings, get_settings
from repo_rag.history.facts_index import index_git_facts
from repo_rag.history.github_index import sync_github_metadata
from repo_rag.history.index import index_history
from repo_rag.history.issue_linker import GitHubClient, GitHubLookupError
from repo_rag.history.metadata_db import HistoryWindow, MetadataDB
from repo_rag.history.pr_index import index_pr_issues
from repo_rag.ingest.github_client import clone, head_commit, parse_repo_url
from repo_rag.ingest.indexer import index_repo
from repo_rag.ingest.symbol_indexer import index_symbols
from repo_rag.retrieval.bug_match import search_fixes
from repo_rag.retrieval.rerank import reranking_enabled
from repo_rag.retrieval.retrievers import HybridRetriever, SearchHit
from repo_rag.store.embeddings import get_embeddings
from repo_rag.store.manifest import Manifest
from repo_rag.store.symbol_table import SymbolTable
from repo_rag.store.transfer import TransferError, TransferResult, export_index, import_index
from repo_rag.store.vectorstore import COLLECTIONS, SEARCH_COLLECTIONS, VectorStore
from repo_rag.summaries.index import index_summaries

app = typer.Typer(
    name="repo-rag",
    help="Code-aware RAG assistant for GitHub repositories.",
    no_args_is_help=True,
)
console = Console()

_PREVIEW_LINES = 12


def _not_implemented(feature: str, phase: int) -> None:
    console.print(f"[yellow]{feature} is not implemented yet (plan.md Phase {phase}).[/yellow]")
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"repo-rag {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Code-aware RAG assistant for GitHub repositories."""


@app.command()
def index(
    url: Annotated[
        str, typer.Argument(help="GitHub repository URL, e.g. https://github.com/psf/requests")
    ],
    history: Annotated[
        bool,
        typer.Option(
            "--history/--no-history", help="Fetch full commit history (blobless) or depth 1."
        ),
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", help="Re-embed every chunk, even unchanged ones.")
    ] = False,
    commit: Annotated[
        str | None,
        typer.Option("--commit", help="Index this revision (needs history) instead of the tip."),
    ] = None,
) -> None:
    """Clone a repository, chunk it, and build its search indexes."""
    settings = get_settings()
    try:
        with console.status(f"Cloning {url} ..."):
            ref, repo_dir = clone(url, settings, with_history=history, commit=commit)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except (GitCommandError, FileExistsError) as exc:
        console.print(f"[red]Clone failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    commit_sha = head_commit(repo_dir)
    console.print(f"[green]{ref.slug}[/green] at {commit_sha[:12]} -> {repo_dir}")
    store = VectorStore(settings, get_embeddings(settings))

    with Progress(
        TextColumn("[progress.description]{task.description:<22}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        tasks: dict[str, TaskID] = {}

        def on_progress(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = progress.add_task(stage, total=total)
            progress.update(tasks[stage], completed=done, total=total)

        result = index_repo(
            ref.slug,
            repo_dir,
            commit_sha,
            store=store,
            settings=settings,
            force=force,
            on_progress=on_progress,
        )

    table = Table("Collection", "Files", "Chunks", "New", "Re-embedded", "Unchanged", "Removed")
    files_for = {"code": result.files["code"], "docs": result.files["doc"] + result.files["config"]}
    for kind in COLLECTIONS:
        s = result.sync[kind]
        table.add_row(
            kind,
            str(files_for[kind]),
            str(result.chunks[kind]),
            str(s.added),
            str(s.updated),
            str(s.unchanged + s.metadata_only),
            str(s.removed),
        )
    console.print(table)
    chunkers = ", ".join(f"{n} {name}" for name, n in sorted(result.chunkers.items()))
    console.print(f"Chunkers: {chunkers}. Done in {result.seconds:.1f}s.")


@app.command()
def update(
    repo: Annotated[str, typer.Argument(help="GitHub owner/name that is already indexed.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be embedded; change nothing.")
    ] = False,
    pull: Annotated[
        bool, typer.Option("--pull/--no-pull", help="Fast-forward the local clone first.")
    ] = True,
    yes: Annotated[
        bool, typer.Option("--yes", help="Embed even above the CPU guard (update_embed_limit).")
    ] = False,
    checkout: Annotated[
        Path | None,
        typer.Option(
            "--checkout",
            exists=True,
            file_okay=False,
            help="Clone to use when none is recorded (an index imported from Colab has none).",
        ),
    ] = None,
) -> None:
    """Bring an indexed repo up to its latest commit, embedding only changed chunks.

    Chunks whose text is unchanged keep their vectors, even if they moved to other lines.
    Symbols, summaries and commit history are refreshed too when they were built before.
    """
    from repo_rag.ingest.updater import (
        UpdateError,
        apply_update,
        fast_forward,
        indexed_entry,
        plan_update,
    )

    settings = get_settings()
    try:
        slug = parse_repo_url(repo).slug
        entry = indexed_entry(slug, settings, checkout)
        if pull:
            with console.status(f"Updating the clone of {slug} ..."):
                fast_forward(slug, Path(entry.clone_path))
        store = VectorStore(settings, get_embeddings(settings))
        plan = plan_update(entry, settings, store)
    except (ValueError, UpdateError) as exc:
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(code=1) from exc

    changed = "unknown (the old commit is not in the clone)"
    if plan.changed_files is not None:
        changed = str(plan.changed_files)
    console.print(
        f"{slug}: indexed {plan.old_sha[:12]}, clone at {plan.new_sha[:12]}; "
        f"files changed in git: {changed}."
    )
    table = Table("Collection", "Chunks to embed", "Vector reused", "Removed", "Unchanged")
    for kind, s in plan.sync.items():
        table.add_row(
            kind, str(s.embedded), str(s.reused), str(s.removed), str(s.unchanged + s.metadata_only)
        )
    console.print(table)
    if plan.up_to_date and plan.to_embed == 0 and not any(s.removed for s in plan.sync.values()):
        console.print("Already up to date.")
        return
    if dry_run:
        console.print("Dry run: nothing was changed.")
        return
    if plan.to_embed > settings.update_embed_limit and not yes:
        console.print(
            f"[yellow]{plan.to_embed} chunks need embedding, above the limit of "
            f"{settings.update_embed_limit} for this machine.[/yellow] Embedding on a CPU is slow: "
            "build the index on a GPU with notebooks/index_on_colab.ipynb and run `import-index`, "
            "or pass --yes to embed here anyway."
        )
        raise typer.Exit(code=1)

    with Progress(
        TextColumn("[progress.description]{task.description:<22}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        tasks: dict[str, TaskID] = {}

        def on_progress(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = progress.add_task(stage, total=total)
            progress.update(tasks[stage], completed=done, total=total)

        try:
            result = apply_update(plan, entry, settings, store, on_progress)
        except (ValueError, GitError, UpdateError) as exc:
            console.print(str(exc), markup=False, style="red")
            raise typer.Exit(code=1) from exc
    embedded = ", ".join(f"{kind} {n}" for kind, n in result.embedded.items())
    refreshed = ", ".join(result.refreshed) or "none"
    console.print(f"Updated to {plan.new_sha[:12]}. Embedded: {embedded}. Refreshed: {refreshed}.")
    if "history" in result.refreshed:
        console.print(
            "Commit history was refreshed offline: new commits are not linked to issues yet. "
            "Run `repo-rag index-history --github` for that."
        )


def _resolve_repo(settings: Settings, repo: str | None) -> str:
    entries = Manifest(settings.manifest_path).entries()
    if not entries:
        console.print("[red]Nothing is indexed yet.[/red] Run `repo-rag index <url>` first.")
        raise typer.Exit(code=1)
    if repo is None:
        if len(entries) > 1:
            names = ", ".join(sorted(entries))
            console.print(f"[red]Several repos are indexed; pick one with --repo:[/red] {names}")
            raise typer.Exit(code=2)
        repo = next(iter(entries))
    else:
        try:
            repo = parse_repo_url(repo).slug
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
    entry = entries.get(repo)
    if entry is None:
        console.print(f"[red]{repo} is not indexed.[/red] Indexed: {', '.join(sorted(entries))}")
        raise typer.Exit(code=1)
    if entry.embedding_id != settings.embedding_id:
        console.print(
            f"[red]{repo} was indexed with {entry.embedding_id}, but the current setting is "
            f"{settings.embedding_id}.[/red] Re-run `repo-rag index` to rebuild it."
        )
        raise typer.Exit(code=1)
    return repo


def _print_hit(rank: int, hit: SearchHit) -> None:
    meta = hit.document.metadata
    if meta.get("symbol_kind") == "commit":
        sha = str(meta["commit_sha"])
        console.print(
            f"{rank}. {sha[:12]}  https://github.com/{meta['repo']}/commit/{sha}", markup=False
        )
        console.print(hit.document.page_content, markup=False)
        return
    if str(meta.get("symbol_kind", "")).endswith("_summary"):
        console.print(
            f"{rank}. {meta['symbol_kind']}  {meta['path'] or '(repository)'}", markup=False
        )
        console.print(hit.document.page_content[:900], markup=False)
        return
    if meta.get("symbol_kind") in ("pull_request", "issue"):
        noun = "pull" if meta["symbol_kind"] == "pull_request" else "issues"
        url = f"https://github.com/{meta['repo']}/{noun}/{meta['number']}"
        console.print(f"{rank}. {url}  [{meta['state']}]", markup=False)
        console.print(hit.document.page_content[:600], markup=False)
        return
    _, body = split_header(hit.document.page_content)
    lines = body.split("\n")
    shown = "\n".join(lines[:_PREVIEW_LINES])
    if len(lines) > _PREVIEW_LINES:
        shown += f"\n... ({len(lines) - _PREVIEW_LINES} more lines)"
    where = f"{meta['path']}:{meta['start_line']}-{meta['end_line']}"
    title = f"{rank}. {where}"
    if meta["symbol"]:
        title += f"  {meta['symbol']} ({meta['symbol_kind']})"
    if hit.expansion_mode is not None:
        detail = "related evidence (unscored)"
    elif hit.rerank_score is not None:
        detail = f"rerank {hit.rerank_score:.3f}"
    else:
        dense = f"#{hit.dense_rank}" if hit.dense_rank else "-"
        keyword = f"#{hit.bm25_rank}" if hit.bm25_rank else "-"
        detail = f"fused {hit.score:.4f}  dense {dense}  keyword {keyword}"
    lexer = str(meta["language"]) or "text"
    console.print(
        Panel(
            Syntax(shown, lexer, line_numbers=True, start_line=int(meta["start_line"])),
            title=title,
            subtitle=detail,
            title_align="left",
            subtitle_align="right",
        )
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    repo: Annotated[
        str | None,
        typer.Option("--repo", "-r", help="owner/name; optional if only one repo is indexed."),
    ] = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Number of results.")] = 8,
    kind: Annotated[
        str, typer.Option("--kind", help="code, docs, commits, pull_requests, issues or summaries.")
    ] = "code",
    state: Annotated[
        str, typer.Option("--state", help="PRs/issues only: OPEN, CLOSED or MERGED.")
    ] = "",
    language: Annotated[str, typer.Option("--language", help="Only this language.")] = "",
    symbol_kind: Annotated[
        str, typer.Option("--symbol-kind", help="function, method, class, module or block.")
    ] = "",
    path: Annotated[str, typer.Option("--path", help="Only paths starting with this.")] = "",
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Cross-encoder rerank (needs a GPU)."),
    ] = False,
    rerank_depth: Annotated[
        int | None,
        typer.Option("--rerank-depth", min=1, help="Candidates to rerank; e.g. 20 or 50."),
    ] = None,
) -> None:
    """Search indexed chunks (no LLM): dense + keyword, fused."""
    if kind not in SEARCH_COLLECTIONS:
        console.print(f"[red]--kind must be one of: {', '.join(SEARCH_COLLECTIONS)}[/red]")
        raise typer.Exit(code=2)
    settings = get_settings()
    if rerank_depth is not None:
        settings = settings.model_copy(update={"rerank_candidate_depth": rerank_depth})
    if rerank and reranking_enabled(settings) and k > settings.rerank_candidate_depth:
        console.print("[red]--k cannot exceed the rerank candidate depth.[/red]")
        raise typer.Exit(code=2)
    slug = _resolve_repo(settings, repo)
    if kind == "commits":
        _require_history(settings, slug)
    if rerank and not reranking_enabled(settings):
        console.print("[yellow]Reranking skipped: reranker_mode=auto needs a CUDA GPU.[/yellow]")
    retriever = HybridRetriever(VectorStore(settings, get_embeddings(settings)), settings)
    with console.status("Searching ..."):
        hits = retriever.search(
            slug,
            query,
            kind=kind,
            k=k,
            filters={
                "language": language,
                "symbol_kind": symbol_kind,
                "path_prefix": path,
                "state": state.upper(),
            },
            rerank=rerank,
        )
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        raise typer.Exit(code=1)
    for rank, hit in enumerate(hits, start=1):
        _print_hit(rank, hit)


def _slugs(repos: list[str] | None) -> list[str] | None:
    try:
        return [parse_repo_url(r).slug for r in repos] if repos else None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _print_transfer(result: TransferResult, verb: str) -> None:
    table = Table("Repo", "Code chunks", "Docs chunks", "Commits")
    for slug, counts in result.chunks.items():
        table.add_row(
            slug,
            str(counts.get("code", 0)),
            str(counts.get("docs", 0)),
            str(counts.get("commits", 0)),
        )
    console.print(table)
    console.print(f"{verb} {len(result.repos)} repo(s).")


@app.command("export-index")
def export_index_command(
    out: Annotated[Path, typer.Argument(help="Zip file to write.")],
    repo: Annotated[
        list[str] | None,
        typer.Option("--repo", "-r", help="owner/name; repeat for several. Default: all."),
    ] = None,
) -> None:
    """Pack indexed repos (text, metadata, vectors) into a zip to move to another machine."""
    settings = get_settings()
    store = VectorStore(settings, get_embeddings(settings))
    try:
        result = export_index(out, store=store, settings=settings, repos=_slugs(repo))
    except TransferError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    _print_transfer(result, f"Wrote {out} with")


@app.command("import-index")
def import_index_command(
    archive: Annotated[Path, typer.Argument(help="Zip made by `repo-rag export-index`.")],
    repo: Annotated[
        list[str] | None,
        typer.Option("--repo", "-r", help="owner/name; repeat for several. Default: all."),
    ] = None,
) -> None:
    """Merge an exported index into this machine's data. Nothing is re-embedded."""
    settings = get_settings()
    store = VectorStore(settings, get_embeddings(settings))
    try:
        with console.status("Importing ..."):
            result = import_index(archive, store=store, settings=settings, repos=_slugs(repo))
    except TransferError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    _print_transfer(result, "Imported")
    if result.removed:
        console.print(f"Replaced the old copies ({result.removed} stale chunks removed).")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about the indexed repository.")],
    repo: Annotated[str | None, typer.Option("--repo", "-r")] = None,
    bug: Annotated[
        bool, typer.Option("--bug", help="Find past fix commits for a bug report.")
    ] = False,
    rerank: Annotated[bool, typer.Option("--rerank", help="Opt-in GPU reranking.")] = False,
    k: Annotated[int, typer.Option("--k", "-k", min=1)] = 5,
) -> None:
    """Find cited past fixes with --bug. General question answering arrives in Phase 7."""
    if not bug:
        _not_implemented("General question answering (use ask --bug for commit evidence)", 7)
    if not question.strip():
        console.print("Bug report must not be blank.")
        raise typer.Exit(code=2)
    settings = get_settings()
    slug = _resolve_repo(settings, repo)
    _require_history(settings, slug)
    if rerank and not reranking_enabled(settings):
        console.print("[yellow]Reranking skipped: reranker_mode=auto needs a CUDA GPU.[/yellow]")
    if rerank and reranking_enabled(settings) and k > settings.rerank_candidate_depth:
        console.print("[red]--k cannot exceed the rerank candidate depth.[/red]")
        raise typer.Exit(code=2)
    retriever = HybridRetriever(VectorStore(settings, get_embeddings(settings)), settings)
    hits = search_fixes(retriever, slug, question, k=k, rerank=rerank)
    if not hits:
        console.print("No matching past fixes found.")
        raise typer.Exit(code=1)
    console.print("Retrieved past-fix candidates (relevance is not confirmation of the same bug):")
    for rank, hit in enumerate(hits, 1):
        _print_hit(rank, hit)


def _require_history(settings: Settings, slug: str) -> None:
    entry = Manifest(settings.manifest_path).get(slug)
    if entry is None or "commits" not in entry.chunks:
        console.print("Commit history is not indexed; run repo-rag index-history first.")
        raise typer.Exit(code=1)


@app.command("index-history")
def index_history_command(
    repo: Annotated[str, typer.Argument(help="GitHub owner/name.")],
    checkout: Annotated[Path, typer.Option("--checkout", exists=True, file_okay=False)],
    ref: Annotated[str, typer.Option("--ref", help="Pin a local commit or tag.")] = "HEAD",
    github: Annotated[
        bool, typer.Option("--github/--offline", help="Resolve cached/live issue links.")
    ] = False,
) -> None:
    """Index a local clone's bounded history; --github enables read-only GitHub lookups."""
    settings = get_settings()
    try:
        slug = parse_repo_url(repo).slug
        if github and not settings.github_token:
            raise GitHubLookupError("Set GITHUB_TOKEN in .env before live GitHub lookups")
        result = index_history(
            slug,
            checkout,
            settings,
            VectorStore(settings, get_embeddings(settings)),
            ref=ref,
            client=GitHubClient(settings) if github else None,
        )
    except (ValueError, GitHubLookupError, GitError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    console.print(
        f"Indexed {result.commits} commits; {result.fix_candidates} fix candidates. "
        f"Links: {result.direct_links} direct, {result.via_pr_links} via PR. "
        f"Embedded {result.sync.embedded}; unchanged {result.sync.unchanged}. "
        f"Prefetched {result.prefetched_blobs} file versions."
    )


@app.command("sync-github")
def sync_github_command(
    repo: Annotated[str, typer.Argument(help="GitHub owner/name.")],
    max_prs: Annotated[int | None, typer.Option(help="First-sync cap (newest PRs).")] = None,
    max_issues: Annotated[int | None, typer.Option(help="First-sync cap (newest issues).")] = None,
) -> None:
    """Fetch PRs (with reviews) and issues, read-only; later runs only fetch what changed."""
    settings = get_settings()
    try:
        slug = parse_repo_url(repo).slug
        result = sync_github_metadata(
            slug, settings, GitHubClient(settings), max_prs=max_prs, max_issues=max_issues
        )
    except (ValueError, GitHubLookupError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    for part in (result.prs, result.issues):
        scope = "all" if part.complete else "the newest only (cap reached)"
        console.print(
            f"{part.kind}: fetched {part.fetched}, stored {part.total}, covering {scope}."
        )


@app.command("index-prs")
def index_prs_command(
    repo: Annotated[str, typer.Argument(help="GitHub owner/name (already synced).")],
) -> None:
    """Embed the synced PRs and issues so `search --kind pull_requests|issues` works."""
    settings = get_settings()
    try:
        slug = parse_repo_url(repo).slug
        result = index_pr_issues(slug, settings, VectorStore(settings, get_embeddings(settings)))
    except ValueError as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    for kind, count in (("pull_requests", result.pull_requests), ("issues", result.issues)):
        stats = result.sync[kind]
        console.print(f"{kind}: {count} (embedded {stats.embedded}, unchanged {stats.unchanged}).")


def _open_metadata(settings: Settings) -> MetadataDB:
    if not settings.metadata_db_path.exists():
        console.print("No metadata.db yet; run repo-rag sync-github first.")
        raise typer.Exit(code=1)
    return MetadataDB(settings.metadata_db_path)


@app.command("show-pr")
def show_pr_command(
    number: Annotated[int, typer.Argument(help="Pull request number.")],
    repo: Annotated[str, typer.Option("--repo", help="GitHub owner/name.")],
) -> None:
    """Print one synced pull request: who authored and merged it, reviews and closed issues."""
    settings = get_settings()
    slug = parse_repo_url(repo).slug
    with _open_metadata(settings) as db:
        pr = db.get_pr(slug, number)
        coverage = db.github_coverage(slug)
    if pr is None:
        note = (
            "" if coverage.get("prs", {}).get("complete") else " (index holds only the newest PRs)"
        )
        console.print(f"PR #{number} not found in {slug}{note}.", markup=False)
        raise typer.Exit(code=1)
    lines = [
        f"#{pr.number} {pr.title}",
        f"state: {pr.state}   base: {pr.base}   head: {pr.head}",
        f"author: {pr.author or 'unknown'}   merged by: {pr.merged_by or 'not merged'}"
        "   (git does not record who pushed)",
        f"created: {pr.created_at}   merged: {pr.merged_at or '-'}   merge commit: "
        f"{pr.merge_commit_sha[:12] or '-'}",
        f"files changed: {pr.changed_files}   +{pr.additions} -{pr.deletions}",
        f"labels: {', '.join(pr.labels) or '-'}",
        "closes issues: " + (", ".join(f"#{n}" for n in pr.closes_issues) or "-"),
    ]
    if pr.reviews:
        cut = (
            f" (showing {len(pr.reviews)} of {pr.reviews_total})"
            if pr.reviews_total > len(pr.reviews)
            else ""
        )
        lines.append(f"reviews{cut}:")
        lines += [f"  {r.reviewer or 'unknown'}: {r.state} at {r.submitted_at}" for r in pr.reviews]
    console.print("\n".join(lines), markup=False)


@app.command("index-facts")
def index_facts_command(
    repo: Annotated[str, typer.Argument(help="GitHub owner/name.")],
    checkout: Annotated[Path, typer.Option("--checkout", exists=True, file_okay=False)],
    ref: Annotated[
        str, typer.Option("--ref", help="Local commit or tag to read names from.")
    ] = "HEAD",
) -> None:
    """Load tags, .mailmap name merges and PR-author login links into metadata.db."""
    settings = get_settings()
    try:
        result = index_git_facts(parse_repo_url(repo).slug, checkout, settings, ref=ref)
    except (ValueError, GitError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    console.print(
        f"Tags: {result.tags}. Names merged by .mailmap: {result.merged_names}"
        f" ({result.ambiguous_names} ambiguous, left unmerged). "
        f"Author-login links: {result.login_links}."
    )


@app.command("index-summaries")
def index_summaries_command(
    repo: Annotated[str, typer.Argument(help="GitHub owner/name.")],
    checkout: Annotated[Path, typer.Option("--checkout", exists=True, file_okay=False)],
) -> None:
    """Build and embed the repo map: file, directory and repository summaries (no LLM)."""
    settings = get_settings()
    try:
        result = index_summaries(
            parse_repo_url(repo).slug,
            checkout,
            settings,
            VectorStore(settings, get_embeddings(settings)),
        )
    except ValueError as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    s = result.summary
    console.print(
        f"{s.documents} summaries ({s.files} files, {s.directories} directories, 1 repository); "
        f"embedded {result.sync.added + result.sync.updated}, unchanged {result.sync.unchanged}."
    )


@app.command("index-symbols")
def index_symbols_command(
    repo: Annotated[str, typer.Argument(help="GitHub owner/name.")],
    checkout: Annotated[Path, typer.Option("--checkout", exists=True, file_okay=False)],
) -> None:
    """Extract definitions and call sites from a checkout into metadata.db (no model, no GPU)."""
    settings = get_settings()
    try:
        result = index_symbols(parse_repo_url(repo).slug, checkout, settings)
    except ValueError as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    console.print(
        f"{result.definitions} definitions and {result.calls} call sites from {result.files} files."
    )


def _symbol_table(settings: Settings) -> SymbolTable:
    try:
        return SymbolTable(settings.metadata_db_path)
    except sqlite3.OperationalError as exc:
        console.print("No metadata.db yet; run repo-rag index-symbols first.")
        raise typer.Exit(code=1) from exc


@app.command("symbol")
def symbol_command(
    name: Annotated[str, typer.Argument(help="Qualified (Session.send) or bare (send) name.")],
    repo: Annotated[str, typer.Option("--repo", help="GitHub owner/name.")],
    path: Annotated[str, typer.Option("--path", help="Only this file.")] = "",
) -> None:
    """Where a function, method or class is defined."""
    settings = get_settings()
    slug = parse_repo_url(repo).slug
    try:
        with _symbol_table(settings) as table:
            rows = table.get_symbol(slug, name, path=path or None)
    except (ValueError, sqlite3.OperationalError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    if not rows:
        console.print("[yellow]No definition found.[/yellow]")
        raise typer.Exit(code=1)
    for row in rows:
        console.print(
            f"{row.qualname} ({row.kind}) {row.path}:{row.start_line}-{row.end_line}", markup=False
        )
        console.print(f"    {row.signature}", markup=False)


@app.command("callers")
def callers_command(
    name: Annotated[str, typer.Argument(help="Qualified (Session.send) or bare (send) name.")],
    repo: Annotated[str, typer.Option("--repo", help="GitHub owner/name.")],
    path: Annotated[str, typer.Option("--path", help="Definition is in this file.")] = "",
    limit: Annotated[int, typer.Option("--limit", min=1)] = 30,
) -> None:
    """Call sites that plausibly call a symbol. Matching is by name: see the `how` column."""
    settings = get_settings()
    slug = parse_repo_url(repo).slug
    try:
        with _symbol_table(settings) as table:
            rows = table.find_callers(slug, name, path=path or None, limit=limit)
    except (ValueError, sqlite3.OperationalError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    if not rows:
        console.print("[yellow]No callers found.[/yellow]")
        raise typer.Exit(code=1)
    for row in rows:
        console.print(
            f"{row.path}:{row.line}  in {row.caller}  calls {row.expression}  [{row.how}]",
            markup=False,
        )
    if any(r.same_name_definitions > 1 for r in rows):
        console.print(
            "Several definitions share this name, so 'name only' rows may call a different "
            "function. Matching is by name, not full type resolution.",
            markup=False,
        )


@app.command("callees")
def callees_command(
    name: Annotated[str, typer.Argument(help="Qualified (Session.send) or bare (send) name.")],
    repo: Annotated[str, typer.Option("--repo", help="GitHub owner/name.")],
    path: Annotated[str, typer.Option("--path", help="Definition is in this file.")] = "",
    limit: Annotated[int, typer.Option("--limit", min=1)] = 60,
    external: Annotated[bool, typer.Option(help="Also list calls to libraries/builtins.")] = False,
) -> None:
    """What a symbol calls. Matching is by name: see the `how` column."""
    settings = get_settings()
    slug = parse_repo_url(repo).slug
    try:
        with _symbol_table(settings) as table:
            rows = table.find_callees(slug, name, path=path or None, limit=limit * 20)
    except (ValueError, sqlite3.OperationalError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc
    if not rows:
        console.print("[yellow]No definition or calls found.[/yellow]")
        raise typer.Exit(code=1)
    shown = [r for r in rows if external or r.how != "external"][:limit]
    for row in shown:
        target = f"{row.resolves_to} ({row.resolves_path})" if row.resolves_to else "external"
        console.print(
            f"line {row.line}  in {row.inside}  calls {row.expression} -> {target}  [{row.how}]",
            markup=False,
        )
    hidden = sum(r.how == "external" for r in rows) if not external else 0
    if hidden:
        console.print(
            f"{hidden} calls to libraries/builtins hidden (use --external).", markup=False
        )
    if any(r.same_name_definitions > 1 for r in shown):
        console.print(
            "Several definitions share some names, so 'name only' rows may call a different "
            "function. Matching is by name, not full type resolution.",
            markup=False,
        )


def _window_line(window: HistoryWindow) -> str:
    return (
        f"Based on {window.commits} indexed non-merge commits, {window.oldest[:10] or '-'} to "
        f"{window.newest[:10] or '-'}; older history is not counted."
    )


@app.command("contributors")
def contributors_command(
    repo: Annotated[str, typer.Option("--repo", help="GitHub owner/name.")],
    path: Annotated[str | None, typer.Option(help="File or directory.")] = None,
    since: Annotated[str | None, typer.Option(help="YYYY-MM-DD, inclusive.")] = None,
    until: Annotated[str | None, typer.Option(help="YYYY-MM-DD, exclusive.")] = None,
    limit: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    """Top authors by commit count in the indexed history (names and handles, never emails)."""
    settings = get_settings()
    with _open_metadata(settings) as db:
        try:
            answer = db.top_contributors(
                parse_repo_url(repo).slug, path=path, since=since, until=until, limit=limit
            )
        except ValueError as exc:
            console.print(str(exc), markup=False)
            raise typer.Exit(code=1) from exc
    lines = [
        f"{i}. {c.name}{f' (@{c.login})' if c.login else ''}: {c.commits} commits, "
        f"+{c.additions} -{c.deletions}, last {c.last_commit[:10]}"
        for i, c in enumerate(answer.rows, 1)
    ]
    console.print("\n".join([*lines, _window_line(answer.window)]), markup=False)


@app.command("hot-files")
def hot_files_command(
    repo: Annotated[str, typer.Option("--repo", help="GitHub owner/name.")],
    path: Annotated[str | None, typer.Option(help="Directory prefix.")] = None,
    since: Annotated[str | None, typer.Option(help="YYYY-MM-DD, inclusive.")] = None,
    until: Annotated[str | None, typer.Option(help="YYYY-MM-DD, exclusive.")] = None,
    limit: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Most frequently changed files in the indexed history."""
    settings = get_settings()
    with _open_metadata(settings) as db:
        try:
            answer = db.change_frequency(
                parse_repo_url(repo).slug, path_prefix=path, since=since, until=until, limit=limit
            )
        except ValueError as exc:
            console.print(str(exc), markup=False)
            raise typer.Exit(code=1) from exc
    lines = [
        f"{f.commits:4d} commits  {f.authors} authors  last {f.last_changed[:10]}  {f.path}"
        for f in answer.rows
    ]
    console.print("\n".join([*lines, _window_line(answer.window)]), markup=False)


@app.command("config")
def show_config() -> None:
    """Show the effective configuration (secrets masked)."""
    settings = get_settings()
    table = Table("Setting", "Value")
    for name, value in settings.model_dump().items():
        if isinstance(value, SecretStr):
            shown = "set" if value.get_secret_value() else "empty"
        elif value is None:
            shown = "-"
        else:
            shown = str(value)
        table.add_row(name, shown)
    console.print(table)


if __name__ == "__main__":
    app()
