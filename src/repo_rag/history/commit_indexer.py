"""Local, bounded git history records and leakage-safe symbols_diff800 documents.

No checkout mutation. A blobless clone may hydrate historical blobs from its remote.
Hunks use exact old/new AST spans; deleted lines are mapped on the parent version.
Merge commits are excluded, as specified by the Phase 5 protocol.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from git import NULL_TREE, Repo
from git.objects import Blob, Commit
from langchain_core.documents import Document

from repo_rag.chunking.ast_chunker import AstChunker
from repo_rag.chunking.languages import language_for_path
from repo_rag.config import Settings
from repo_rag.history.issue_linker import strip_issue_references

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_FIX = re.compile(r"\b(?:fix(?:e[sd])?|bug|issue|regression|crash|error)\b|#\d+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int


@dataclass(frozen=True)
class FileChange:
    path: str
    old_path: str
    change_type: str
    additions: int
    deletions: int
    patch: str
    patch_truncated: bool
    binary: bool
    hunks: tuple[Hunk, ...]
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class CommitRecord:
    repo: str
    sha: str
    message: str
    author_name: str
    committer_name: str
    author_date: str
    commit_date: str
    is_fix: bool
    files: tuple[FileChange, ...]


def is_fix_message(message: str) -> bool:
    """Broad candidate heuristic, not a ground-truth bug-fix label."""
    return bool(_FIX.search(message))


def _source(blob: Blob | None, settings: Settings) -> str | None:
    if blob is None or blob.size > settings.max_file_bytes:
        return None
    data = blob.data_stream.read()
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace")


def _touched(
    chunker: AstChunker, source: str | None, path: str, ranges: list[tuple[int, int]]
) -> set[str]:
    if source is None:
        return set()
    definitions = chunker.definitions(source, language_for_path(path))
    return {
        d.symbol
        for d in definitions
        for start, count in ranges
        if count > 0 and d.start_line <= start + count - 1 and d.end_line >= start
    }


def read_commit(repo: Repo, slug: str, commit: Commit, settings: Settings) -> CommitRecord:
    if len(commit.parents) > 1:
        raise ValueError("Merge commits are excluded from the Phase 5 corpus")
    parent = commit.parents[0] if commit.parents else None
    diffs = (
        parent.diff(commit, create_patch=True, unified=0)
        if parent
        else commit.diff(NULL_TREE, create_patch=True, unified=0)
    )
    chunker = AstChunker(settings)
    changes: list[FileChange] = []
    for diff in diffs:
        path = diff.b_path or diff.a_path or ""
        old_path = diff.a_path or path
        raw = diff.diff
        patch = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        hunks = tuple(
            Hunk(
                int(m[1]),
                int(m[2]) if m[2] is not None else 1,
                int(m[3]),
                int(m[4]) if m[4] is not None else 1,
            )
            for m in _HUNK.finditer(patch)
        )
        old_ranges = [(h.old_start, h.old_count) for h in hunks]
        new_ranges = [(h.new_start, h.new_count) for h in hunks]
        # Read source only for supported AST languages. Unknown language changes still carry
        # a diff and paths, and are not dropped from the bounded corpus.
        symbols: set[str] = set()
        if language_for_path(path) or language_for_path(old_path):
            symbols |= _touched(chunker, _source(diff.a_blob, settings), old_path, old_ranges)
            symbols |= _touched(chunker, _source(diff.b_blob, settings), path, new_ranges)
        change_type = (
            "A"
            if diff.new_file
            else "D"
            if diff.deleted_file
            else "R"
            if diff.renamed_file
            else "M"
        )
        changes.append(
            FileChange(
                path,
                old_path,
                change_type,
                sum(h.new_count for h in hunks),
                sum(h.old_count for h in hunks),
                patch[: settings.commit_patch_chars],
                len(patch) > settings.commit_patch_chars,
                "Binary files " in patch or "GIT binary patch" in patch,
                hunks,
                tuple(sorted(symbols)),
            )
        )
    message = str(commit.message)
    return CommitRecord(
        slug,
        commit.hexsha,
        message,
        commit.author.name or "",
        commit.committer.name or "",
        commit.authored_datetime.isoformat(),
        commit.committed_datetime.isoformat(),
        is_fix_message(message),
        tuple(sorted(changes, key=lambda f: f.path)),
    )


def _git(root: Path, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True)


def _history_blobs(root: Path, settings: Settings, ref: str) -> set[str]:
    """Every file version the bounded history diffs (both sides of each change), by blob id.

    `log --raw` needs only trees, so it works in a blobless clone. Renames are not detected here:
    both sides of a rename appear as a delete and an add, which covers what rename detection reads.
    """
    log = _git(
        root, "log", "--no-merges", "--root", "--no-renames", "--raw", "--no-abbrev",
        "--format=", f"--max-count={settings.max_commits}", ref,
    )  # fmt: skip
    if log.returncode:
        return set()
    blobs: set[str] = set()
    for line in log.stdout.decode(errors="replace").splitlines():
        if not line.startswith(":"):
            continue
        old_mode, new_mode, old_id, new_id = line[1:].split("\t")[0].split()[:4]
        for mode, blob in ((old_mode, old_id), (new_mode, new_id)):
            if mode != "160000" and set(blob) != {"0"}:  # not a submodule, not "no file"
                blobs.add(blob)
    return blobs


def prefetch_blobs(root: Path, settings: Settings, *, ref: str = "HEAD") -> int:
    """Fetch the file versions the bounded history will read, in a few requests.

    In a blobless clone each missing blob otherwise costs its own round trip to the remote while
    commits are read one by one (psf/requests, 60 commits: 60.6 s lazily against about 6 s with
    one batch fetch first). Best effort: any failure leaves reading to fetch on demand.
    Returns how many blobs were requested.
    """
    wanted = _history_blobs(root, settings, ref)
    local = _git(root, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
    if not wanted or local.returncode:
        return 0
    missing = sorted(wanted - set(local.stdout.decode().split()))
    fetch = [
        "-c", "fetch.negotiationAlgorithm=noop", "fetch", "origin", "--no-tags",
        "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none",
    ]  # fmt: skip
    size = settings.history_prefetch_chunk
    for start in range(0, len(missing), size):
        batch = missing[start : start + size]
        done = _git(root, *fetch, "--stdin", stdin=("\n".join(batch) + "\n").encode())
        if done.returncode:  # git without `fetch --stdin`: pass the ids as arguments
            done = _git(root, *fetch, *batch)
        if done.returncode:
            return start
    return len(missing)


def iter_commits(
    root: Path, slug: str, settings: Settings, *, ref: str = "HEAD"
) -> Iterator[CommitRecord]:
    """Most recent capped non-merge history. The working tree is never consulted."""
    with Repo(root) as repo:
        for commit in repo.iter_commits(ref, no_merges=True, max_count=settings.max_commits):
            yield read_commit(repo, slug, commit, settings)


def commit_document(
    record: CommitRecord, settings: Settings, *, linked_fix: bool = False
) -> Document:
    """One stable repo+SHA document; issue text and reference IDs never enter retrieval.

    Reference-bearing messages and links remain in metadata.db with email redaction. The searchable
    metadata omits issue numbers, avoiding leakage through future contextual headers.
    """
    paths = ", ".join(f.path for f in record.files)
    symbols = ", ".join(f"{f.path}::{s}" for f in record.files for s in f.symbols)
    diff = "\n".join(f"--- {f.old_path}\n+++ {f.path}\n{f.patch}" for f in record.files)
    message = _EMAIL.sub("<email omitted>", strip_issue_references(record.message))
    text = _EMAIL.sub(
        "<email omitted>",
        (
            f"{message}\nFiles: {strip_issue_references(paths)}\n"
            f"Symbols: {strip_issue_references(symbols)}\n"
            f"{strip_issue_references(diff)[: settings.commit_diff_chars]}"
        ).strip(),
    )
    chunk_id = hashlib.sha1(f"{record.repo}:commit:{record.sha}".encode()).hexdigest()
    return Document(
        page_content=text,
        metadata={
            "repo": record.repo,
            "commit_sha": record.sha,
            "chunk_id": chunk_id,
            "path": record.files[0].path if record.files else "",
            "language": "diff",
            "start_line": 1,
            "end_line": max(1, len(text.splitlines())),
            "symbol": "",
            "symbol_kind": "commit",
            "parent_symbol": "",
            "signature": "",
            "symbols": symbols,
            "chunker": "commit",
            "files_changed": paths,
            "is_fix": "true" if record.is_fix or linked_fix else "false",
            "author_name": _EMAIL.sub("<email omitted>", record.author_name),
            "committer_name": _EMAIL.sub("<email omitted>", record.committer_name),
            "author_date": record.author_date,
            "commit_date": record.commit_date,
        },
    )
