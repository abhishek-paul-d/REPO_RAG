"""Typed git tools for exact questions (Phase 5b): blame, history, diffs and releases.

Every function runs fixed git commands on a local clone: no shell, arguments validated, and
the LLM never composes a command. Results carry the commit SHA so answers can cite them.
Only author names are returned, never emails. Git records an author and a committer, not
who pushed; callers must say so (CLAUDE.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from git import GitCommandError, Repo

from repo_rag.config import Settings, get_settings

_SEP = "\x1f"
_MARK = "\x1e"
_PR_NUMBER = re.compile(r"\(#(\d+)\)\s*$")
_REF = re.compile(r"^[\w][\w./+@{}^~-]*$")
_FUNCTION = re.compile(r"^[\w.$]+$")


class GitToolError(ValueError):
    """A bad argument or a ref/path git cannot resolve; the message is safe to show."""


@dataclass(frozen=True)
class BlameGroup:
    sha: str
    author_name: str
    author_date: str
    summary: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class HistoryEntry:
    sha: str
    author_name: str
    author_date: str
    subject: str
    pr_number: int | None  # parsed from a trailing "(#N)" in the subject, else None


@dataclass(frozen=True)
class FileDelta:
    path: str
    additions: int | None  # None for binary files
    deletions: int | None


@dataclass(frozen=True)
class DiffSummary:
    ref_a: str
    ref_b: str
    commits: tuple[HistoryEntry, ...]
    files: tuple[FileDelta, ...]
    commits_truncated: bool
    files_truncated: bool


@dataclass(frozen=True)
class Release:
    tag: str
    date: str


def _open(root: Path) -> Repo:
    try:
        return Repo(root)
    except Exception as error:
        raise GitToolError(f"not a git repository: {root.name}") from error


def _check_ref(value: str, name: str) -> str:
    if not _REF.match(value) or ".." in value:
        raise GitToolError(f"invalid {name}: {value!r}")
    return value


def _check_path(value: str) -> str:
    if not value or value.startswith(("-", "/")) or ".." in Path(value).parts or "\x00" in value:
        raise GitToolError(f"invalid path: {value!r}")
    return value


def _resolve(repo: Repo, ref: str, name: str) -> str:
    try:
        return repo.git.rev_parse(
            "--verify", "--end-of-options", f"{_check_ref(ref, name)}^{{commit}}"
        )
    except GitCommandError as error:
        raise GitToolError(f"unknown {name}: {ref}") from error


def _iso(timestamp: str | int) -> str:
    return datetime.fromtimestamp(int(timestamp), UTC).isoformat()


def _entries(text: str) -> list[HistoryEntry]:
    entries = []
    for line in text.split("\n"):  # not splitlines(): it also breaks on the record marker
        if not line.startswith(_MARK):
            continue  # patch text from `git log -L`
        sha, name, date, subject = line[len(_MARK) :].split(_SEP, 3)
        match = _PR_NUMBER.search(subject)
        entries.append(
            HistoryEntry(sha, name, date, subject, int(match.group(1)) if match else None)
        )
    return entries


_LOG_FORMAT = "--format=%x1e%H%x1f%an%x1f%aI%x1f%s"


def blame(root: Path, path: str, start: int, end: int, *, ref: str = "HEAD") -> list[BlameGroup]:
    """Who last changed each run of lines start..end (1-based, inclusive) of `path` at `ref`."""
    if start < 1 or end < start:
        raise GitToolError("line range must satisfy 1 <= start <= end")
    repo = _open(root)
    sha = _resolve(repo, ref, "ref")
    try:
        text = repo.git.blame("--porcelain", "-L", f"{start},{end}", sha, "--", _check_path(path))
    except GitCommandError as error:
        raise GitToolError(f"cannot blame {path} lines {start}-{end} at {ref}") from error
    meta: dict[str, dict[str, str]] = {}
    lines: list[tuple[int, str]] = []
    current = ""
    for line in text.splitlines():
        if line.startswith("\t"):
            continue
        head = line.split(" ")
        if len(head[0]) == 40 and len(head) >= 3 and head[2].isdigit():
            current = head[0]
            meta.setdefault(current, {})
            lines.append((int(head[2]), current))
        elif current and " " in line:
            key, value = line.split(" ", 1)
            meta[current].setdefault(key, value)
    groups: list[BlameGroup] = []
    for number, commit in lines:
        if groups and groups[-1].sha == commit and groups[-1].end_line == number - 1:
            last = groups.pop()
            groups.append(BlameGroup(**{**last.__dict__, "end_line": number}))
            continue
        info = meta[commit]
        groups.append(
            BlameGroup(
                commit,
                info.get("author", ""),
                _iso(info["author-time"]) if "author-time" in info else "",
                info.get("summary", ""),
                number,
                number,
            )
        )
    return groups


def file_history(
    root: Path,
    path: str,
    *,
    function: str | None = None,
    lines: tuple[int, int] | None = None,
    limit: int | None = None,
    ref: str = "HEAD",
    settings: Settings | None = None,
) -> list[HistoryEntry]:
    """Newest-first commits touching a file (following renames), a function in it, or a line range.

    `function` goes to `git log -L :name:path`, which only finds a definition that git's default
    function-name rule recognises: a top-level one. An indented method gives "no match" (seen on
    `prepare_url` in psf/requests), so a caller that knows the lines passes `lines` instead."""
    settings = settings or get_settings()
    cap = min(limit or settings.git_tool_max_results, settings.git_tool_max_results)
    repo = _open(root)
    sha = _resolve(repo, ref, "ref")
    path = _check_path(path)
    if lines is not None:
        first, last = lines
        if not (1 <= first <= last):
            raise GitToolError(f"invalid line range: {first}-{last}")
        args = ["-L", f"{first},{last}:{path}", f"--max-count={cap}", _LOG_FORMAT, sha]
    elif function is not None:
        if not _FUNCTION.match(function):
            raise GitToolError(f"invalid function name: {function!r}")
        args = ["-L", f":{function}:{path}", f"--max-count={cap}", _LOG_FORMAT, sha]
    else:
        args = ["--follow", f"--max-count={cap}", _LOG_FORMAT, "--no-patch", sha, "--", path]
    try:
        text = repo.git.log(*args)
    except GitCommandError as error:
        target = f"{function} in {path}" if function else path
        if lines is not None:
            target = f"lines {lines[0]}-{lines[1]} of {path}"
        raise GitToolError(f"no history for {target} at {ref}") from error
    return _entries(text)


def pr_commits(root: Path, merge_sha: str, *, limit: int = 10) -> list[HistoryEntry]:
    """The commits a merge brought in: `first_parent..second_parent`, newest first.

    A merge commit holds no change of its own, so a "which commit fixed this" answer should name
    one of these. A single-parent commit (squash or rebase merge) has none: it is the change.
    """
    repo = _open(root)
    sha = _resolve(repo, merge_sha, "merge commit")
    parents = repo.git.rev_list("--parents", "-n1", sha).split()[1:]
    if len(parents) != 2:
        return []
    try:
        text = repo.git.log(
            f"--max-count={max(1, limit)}", _LOG_FORMAT, "--no-patch", f"{parents[0]}..{parents[1]}"
        )
    except GitCommandError as error:
        raise GitToolError(f"no commits for {merge_sha}") from error
    return _entries(text)


def exact_renames(root: Path, ref: str = "HEAD") -> dict[str, str]:
    """old path -> new path for every rename git sees as 100% identical, over the history of `ref`.

    Only exact renames: finding a changed-and-moved file needs blob contents, and on a blobless
    clone that fetches every blob. Nothing is fetched here (`GIT_NO_LAZY_FETCH`), so a rename with
    edits is not followed. When a path was renamed twice the chain is old -> middle -> new.
    """
    repo = _open(root)
    sha = _resolve(repo, ref, "ref")
    with repo.git.custom_environment(GIT_NO_LAZY_FETCH="1"):
        try:
            text = repo.git.log("--diff-filter=R", "-M100%", "--name-status", "--format=", sha)
        except GitCommandError as error:
            raise GitToolError(f"cannot read renames at {ref}") from error
    renames: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("	")
        if len(parts) == 3 and parts[0] == "R100":
            renames.setdefault(parts[1], parts[2])  # newest rename of an old name wins
    return renames


def commits_between(
    root: Path,
    ref_a: str,
    ref_b: str,
    *,
    limit: int | None = None,
    settings: Settings | None = None,
) -> tuple[int, list[HistoryEntry]]:
    """(how many commits `ref_b` has that `ref_a` lacks, the newest `limit` of them).

    Unlike `diff_between` it reads no file contents, so on a blobless clone it never fetches blobs.
    """
    settings = settings or get_settings()
    cap = min(limit or settings.git_tool_max_results, settings.git_tool_max_results)
    repo = _open(root)
    a, b = _resolve(repo, ref_a, "ref_a"), _resolve(repo, ref_b, "ref_b")
    total = int(repo.git.rev_list("--count", f"{a}..{b}"))
    text = repo.git.log(f"--max-count={cap}", _LOG_FORMAT, "--no-patch", f"{a}..{b}")
    return total, _entries(text)


def diff_between(
    root: Path,
    ref_a: str,
    ref_b: str,
    *,
    path: str | None = None,
    settings: Settings | None = None,
) -> DiffSummary:
    """Commits reachable from ref_b but not ref_a, plus per-file line counts a to b."""
    settings = settings or get_settings()
    cap = settings.git_tool_max_results
    repo = _open(root)
    a, b = _resolve(repo, ref_a, "ref_a"), _resolve(repo, ref_b, "ref_b")
    scope = ["--", _check_path(path)] if path else []
    log = repo.git.log(f"--max-count={cap + 1}", _LOG_FORMAT, "--no-patch", f"{a}..{b}", *scope)
    stat = repo.git.diff("--numstat", "-z", a, b, *scope)
    files: list[FileDelta] = []
    for item in filter(None, stat.split("\x00")):
        added, deleted, name = item.split("\t", 2)
        binary = added == "-"
        files.append(
            FileDelta(name, None if binary else int(added), None if binary else int(deleted))
        )
    commits = _entries(log)
    return DiffSummary(
        ref_a,
        ref_b,
        tuple(commits[:cap]),
        tuple(files[:cap]),
        len(commits) > cap,
        len(files) > cap,
    )


def releases_containing(root: Path, sha: str, *, settings: Settings | None = None) -> list[Release]:
    """Tags whose history contains the commit, earliest first. Empty means unreleased."""
    settings = settings or get_settings()
    repo = _open(root)
    commit = _resolve(repo, sha, "sha")
    text = repo.git.tag(
        "--contains",
        commit,
        "--sort=creatordate",
        "--format=%(refname:short)" + _SEP + "%(creatordate:iso-strict)",
    )
    releases = [Release(*line.split(_SEP, 1)) for line in text.split("\n") if _SEP in line]
    return releases[: settings.git_tool_max_results]
