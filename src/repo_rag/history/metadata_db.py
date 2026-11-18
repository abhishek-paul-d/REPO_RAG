"""Phase 5 history/link tables plus the Phase 5b PR/review/issue tables, by repository.

Query handles use SQLite mode=ro; only typed, parameterized methods are exposed. Identity
merging and tags remain Phase 5b work. New snapshots redact email-like strings
throughout commit/issue payloads. Reads also redact legacy payloads; reindex to replace old
database contents. Frozen evaluation source records deliberately retain historical bytes.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Generic, TypeVar

from repo_rag.history.commit_indexer import CommitRecord, FileChange, Hunk
from repo_rag.history.git_facts import IdentityMap, LoginLink, TagRecord
from repo_rag.history.github_sync import IssueMeta, PullRequest, Review
from repo_rag.history.issue_linker import Issue, IssueLink

_SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS history_commits (
 repo TEXT NOT NULL, sha TEXT NOT NULL, author_name TEXT NOT NULL,
 committer_name TEXT NOT NULL, author_date TEXT NOT NULL, commit_date TEXT NOT NULL,
 message TEXT NOT NULL, is_fix INTEGER NOT NULL, record_json TEXT NOT NULL,
 PRIMARY KEY(repo, sha));
CREATE TABLE IF NOT EXISTS history_files (
 repo TEXT NOT NULL, sha TEXT NOT NULL, path TEXT NOT NULL, old_path TEXT NOT NULL,
 additions INTEGER NOT NULL, deletions INTEGER NOT NULL,
 PRIMARY KEY(repo,sha,path),
 FOREIGN KEY(repo,sha) REFERENCES history_commits(repo,sha) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS history_symbols (
 repo TEXT NOT NULL, sha TEXT NOT NULL, path TEXT NOT NULL, symbol TEXT NOT NULL,
 PRIMARY KEY(repo,sha,path,symbol),
 FOREIGN KEY(repo,sha) REFERENCES history_commits(repo,sha) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS history_issues (
 repo TEXT NOT NULL, number INTEGER NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
 state TEXT NOT NULL, url TEXT NOT NULL, PRIMARY KEY(repo,number));
CREATE TABLE IF NOT EXISTS issue_links (
 repo TEXT NOT NULL, commit_sha TEXT NOT NULL, issue_repo TEXT NOT NULL,
 issue_number INTEGER NOT NULL, pr_number INTEGER NOT NULL DEFAULT 0,
 link_type TEXT NOT NULL CHECK(link_type IN ('direct','via_pr')),
 PRIMARY KEY(repo,commit_sha,issue_repo,issue_number,pr_number,link_type),
 FOREIGN KEY(repo,commit_sha) REFERENCES history_commits(repo,sha) ON DELETE CASCADE,
 FOREIGN KEY(issue_repo,issue_number) REFERENCES history_issues(repo,number));
CREATE TABLE IF NOT EXISTS pull_requests (
 repo TEXT NOT NULL, number INTEGER NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
 state TEXT NOT NULL, is_draft INTEGER NOT NULL, author TEXT NOT NULL, merged_by TEXT NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, merged_at TEXT NOT NULL,
 closed_at TEXT NOT NULL, base TEXT NOT NULL, head TEXT NOT NULL,
 merge_commit_sha TEXT NOT NULL, labels TEXT NOT NULL, changed_files INTEGER NOT NULL,
 additions INTEGER NOT NULL, deletions INTEGER NOT NULL, reviews_total INTEGER NOT NULL,
 PRIMARY KEY(repo, number));
CREATE TABLE IF NOT EXISTS pr_reviews (
 repo TEXT NOT NULL, pr_number INTEGER NOT NULL, reviewer TEXT NOT NULL, state TEXT NOT NULL,
 submitted_at TEXT NOT NULL,
 FOREIGN KEY(repo, pr_number) REFERENCES pull_requests(repo, number) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS pr_closes (
 repo TEXT NOT NULL, pr_number INTEGER NOT NULL, issue_number INTEGER NOT NULL,
 PRIMARY KEY(repo, pr_number, issue_number),
 FOREIGN KEY(repo, pr_number) REFERENCES pull_requests(repo, number) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS issues (
 repo TEXT NOT NULL, number INTEGER NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
 state TEXT NOT NULL, author TEXT NOT NULL, labels TEXT NOT NULL, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL, closed_at TEXT NOT NULL, closed_by_prs TEXT NOT NULL,
 PRIMARY KEY(repo, number));
CREATE TABLE IF NOT EXISTS github_sync (
 repo TEXT NOT NULL, kind TEXT NOT NULL, stored INTEGER NOT NULL, complete INTEGER NOT NULL,
 PRIMARY KEY(repo, kind));
CREATE TABLE IF NOT EXISTS tags (
 repo TEXT NOT NULL, name TEXT NOT NULL, sha TEXT NOT NULL, date TEXT NOT NULL,
 annotated INTEGER NOT NULL, PRIMARY KEY(repo, name));
CREATE TABLE IF NOT EXISTS identities (
 repo TEXT NOT NULL, raw_name TEXT NOT NULL, canonical_name TEXT NOT NULL,
 PRIMARY KEY(repo, raw_name));
CREATE TABLE IF NOT EXISTS author_logins (
 repo TEXT NOT NULL, canonical_name TEXT NOT NULL, login TEXT NOT NULL,
 evidence INTEGER NOT NULL, conflicts INTEGER NOT NULL, PRIMARY KEY(repo, canonical_name));
"""

_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$")
_T = TypeVar("_T")


@dataclass(frozen=True)
class HistoryWindow:
    """The part of history metadata.db holds. Counts and rankings only cover this window."""

    commits: int
    oldest: str
    newest: str


@dataclass(frozen=True)
class Answer(Generic[_T]):
    rows: tuple[_T, ...]
    window: HistoryWindow


@dataclass(frozen=True)
class CommitSummary:
    sha: str
    author: str  # canonical name
    author_date: str
    subject: str
    is_fix: bool


@dataclass(frozen=True)
class Contributor:
    name: str
    login: str  # "" when no PR evidence links this name to a GitHub login
    commits: int
    additions: int
    deletions: int
    first_commit: str
    last_commit: str


@dataclass(frozen=True)
class FileFrequency:
    path: str
    commits: int
    authors: int
    additions: int
    deletions: int
    last_changed: str


@dataclass(frozen=True)
class CommitCount:
    count: int
    window: HistoryWindow


def _check_date(value: str | None, name: str) -> str | None:
    if value is not None and not _DATE.match(value):
        raise ValueError(f"{name} must look like YYYY-MM-DD or YYYY-MM-DDTHH:MM[:SS]")
    return value


def _clean_path(path: str | None) -> str | None:
    if path is None:
        return None
    path = path.replace("\\", "/").strip().removeprefix("./").strip("/")
    return path or None


def _path_match(paths: Sequence[str], column: str = "f.path") -> tuple[str, list[Any]]:
    """SQL for "the file is one of `paths`, or lies under one of them" and its bound values."""
    parts: list[str] = []
    params: list[Any] = []
    for path in paths:
        parts.append(f"({column}=? OR substr({column},1,?)=?)")
        params += [path, len(path) + 1, path + "/"]
    return "(" + " OR ".join(parts) + ")", params


def follow_renames(path: str, renames: Mapping[str, str]) -> str:
    """The latest name of `path`: follow old -> new renames to the end (bounded, no cycles)."""
    for _ in range(50):
        newer = renames.get(path)
        if newer is None or newer == path:
            break
        path = newer
    return path


def names_of(path: str, renames: Mapping[str, str]) -> list[str]:
    """Every name the file `path` had, itself first: all old names whose latest name is its."""
    latest = follow_renames(path, renames)
    older = sorted(old for old in renames if follow_renames(old, renames) == latest)
    return [path, *[name for name in older if name != path]]


def _safe_payload(value: dict) -> dict:
    """Redact nested textual fields, including trailers and patches, before persistence/output."""

    def redact(item: Any) -> Any:
        if isinstance(item, str):
            return _EMAIL.sub("<email omitted>", item)
        if isinstance(item, dict):
            return {key: redact(text) for key, text in item.items()}
        if isinstance(item, (tuple, list)):
            return [redact(text) for text in item]
        return item

    return {key: redact(item) for key, item in value.items()}


def record_from_dict(data: dict) -> CommitRecord:
    data = dict(data)
    files = []
    for item in data.pop("files"):
        item = dict(item)
        item["hunks"] = tuple(Hunk(**h) for h in item["hunks"])
        item["symbols"] = tuple(item["symbols"])
        files.append(FileChange(**item))
    return CommitRecord(**data, files=tuple(files))


_MENTION = re.compile(
    r"(?<![\w/#])#(\d{1,7})\b|github\.com/([\w.-]+/[\w.-]+)/(?:pull|issues)/(\d{1,7})"
)


def referenced_numbers(text: str, repo: str) -> set[int]:
    """Same-repository PR/issue numbers written as `#N` or as a github.com/<repo>/pull/N URL."""
    found: set[int] = set()
    for short, slug, linked in _MENTION.findall(text):
        if short:
            found.add(int(short))
        elif slug.lower() == repo.lower():
            found.add(int(linked))
    return found


class MetadataDB:
    def __init__(self, path: Path, *, readonly: bool = True) -> None:
        self.path = path
        self.readonly = readonly
        if readonly:
            self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
            self.connection.executescript(_SCHEMA)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        tables = {r[0] for r in self.connection.execute("SELECT name FROM sqlite_master")}
        if "identities" not in tables:  # database written before name merging existed
            self._CANON = "c.author_name"
        self._has_logins = "author_logins" in tables

    def __enter__(self) -> MetadataDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def sync_history(
        self, repo: str, records: list[CommitRecord], links: list[IssueLink], issues: list[Issue]
    ) -> None:
        """Atomically replace one bounded repo snapshot; preserve every other repository."""
        if self.readonly:
            raise PermissionError("MetadataDB is open read-only")
        if any(r.repo != repo for r in records) or any(link.repo != repo for link in links):
            raise ValueError("cross-repository history snapshot")
        linked = {link.commit_sha for link in links}
        records = [record_from_dict(_safe_payload(asdict(r))) for r in records]
        issues = [Issue(**_safe_payload(asdict(i))) for i in issues]
        with self.connection:
            self.connection.execute("DELETE FROM history_commits WHERE repo=?", (repo,))
            self.connection.executemany(
                "INSERT INTO history_commits VALUES(?,?,?,?,?,?,?,?,?)",
                [
                    (
                        repo,
                        r.sha,
                        r.author_name,
                        r.committer_name,
                        r.author_date,
                        r.commit_date,
                        r.message,
                        int(r.is_fix or r.sha in linked),
                        json.dumps(asdict(replace(r, is_fix=r.is_fix or r.sha in linked))),
                    )
                    for r in records
                ],
            )
            self.connection.executemany(
                "INSERT INTO history_files VALUES(?,?,?,?,?,?)",
                [
                    (repo, r.sha, f.path, f.old_path, f.additions, f.deletions)
                    for r in records
                    for f in r.files
                ],
            )
            self.connection.executemany(
                "INSERT INTO history_symbols VALUES(?,?,?,?)",
                [
                    (repo, r.sha, f.path, symbol)
                    for r in records
                    for f in r.files
                    for symbol in f.symbols
                ],
            )
            self.connection.executemany(
                "INSERT INTO history_issues VALUES(?,?,?,?,?,?) ON CONFLICT(repo,number) "
                "DO UPDATE SET title=excluded.title,body=excluded.body,"
                "state=excluded.state,url=excluded.url",
                [(i.repo, i.number, i.title, i.body, i.state, i.url) for i in issues],
            )
            self.connection.executemany(
                "INSERT INTO issue_links VALUES(?,?,?,?,?,?)",
                [
                    (
                        link.repo,
                        link.commit_sha,
                        link.issue_repo,
                        link.issue_number,
                        link.pr_number or 0,
                        link.link_type,
                    )
                    for link in links
                ],
            )

    def get_commit(self, repo: str, sha: str) -> CommitRecord | None:
        row = self.connection.execute(
            "SELECT record_json FROM history_commits WHERE repo=? AND sha=?", (repo, sha)
        ).fetchone()
        return record_from_dict(_safe_payload(json.loads(row[0]))) if row else None

    def query_commits(
        self, repo: str, *, is_fix: bool | None = None, limit: int = 50
    ) -> list[CommitRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            "SELECT record_json FROM history_commits WHERE repo=? AND (? IS NULL OR is_fix=?) "
            "ORDER BY commit_date DESC,sha LIMIT ?",
            (repo, is_fix, is_fix, limit),
        )
        return [record_from_dict(_safe_payload(json.loads(row[0]))) for row in rows]

    def links(self, repo: str, *, sha: str | None = None) -> list[IssueLink]:
        rows = self.connection.execute(
            "SELECT * FROM issue_links WHERE repo=? AND (? IS NULL OR commit_sha=?) "
            "ORDER BY commit_sha,issue_repo,issue_number,link_type,pr_number",
            (repo, sha, sha),
        )
        return [
            IssueLink(
                row["repo"],
                row["commit_sha"],
                row["issue_repo"],
                row["issue_number"],
                row["link_type"],
                row["pr_number"] or None,
            )
            for row in rows
        ]

    def get_issue(self, repo: str, number: int) -> Issue | None:
        row = self.connection.execute(
            "SELECT * FROM history_issues WHERE repo=? AND number=?", (repo, number)
        ).fetchone()
        return Issue(**_safe_payload(dict(row))) if row else None

    # --- Phase 5b: pull requests, reviews and issues from the GitHub sync ---

    def sync_github(
        self,
        repo: str,
        prs: list[PullRequest],
        issues: list[IssueMeta],
        *,
        prs_complete: bool,
        issues_complete: bool,
    ) -> None:
        """Atomically replace one repo's PR/issue rows; other repositories are untouched."""
        if self.readonly:
            raise PermissionError("MetadataDB is open read-only")
        if any(p.repo != repo for p in prs) or any(i.repo != repo for i in issues):
            raise ValueError("cross-repository GitHub snapshot")
        prs = [replace(p, **_safe_payload({"title": p.title, "body": p.body})) for p in prs]
        issues = [replace(i, **_safe_payload({"title": i.title, "body": i.body})) for i in issues]
        with self.connection:
            for table in ("pull_requests", "issues", "github_sync"):
                self.connection.execute(f"DELETE FROM {table} WHERE repo=?", (repo,))
            self.connection.executemany(
                "INSERT INTO pull_requests VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        repo,
                        p.number,
                        p.title,
                        p.body,
                        p.state,
                        int(p.is_draft),
                        p.author,
                        p.merged_by,
                        p.created_at,
                        p.updated_at,
                        p.merged_at,
                        p.closed_at,
                        p.base,
                        p.head,
                        p.merge_commit_sha,
                        ",".join(p.labels),
                        p.changed_files,
                        p.additions,
                        p.deletions,
                        p.reviews_total,
                    )
                    for p in prs
                ],
            )
            self.connection.executemany(
                "INSERT INTO pr_reviews VALUES(?,?,?,?,?)",
                [
                    (repo, p.number, r.reviewer, r.state, r.submitted_at)
                    for p in prs
                    for r in p.reviews
                ],
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO pr_closes VALUES(?,?,?)",
                [(repo, p.number, n) for p in prs for n in p.closes_issues],
            )
            self.connection.executemany(
                "INSERT INTO issues VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        repo,
                        i.number,
                        i.title,
                        i.body,
                        i.state,
                        i.author,
                        ",".join(i.labels),
                        i.created_at,
                        i.updated_at,
                        i.closed_at,
                        ",".join(str(n) for n in i.closed_by_prs),
                    )
                    for i in issues
                ],
            )
            self.connection.executemany(
                "INSERT INTO github_sync VALUES(?,?,?,?)",
                [
                    (repo, "prs", len(prs), int(prs_complete)),
                    (repo, "issues", len(issues), int(issues_complete)),
                ],
            )

    def github_coverage(self, repo: str) -> dict[str, dict[str, int | bool]]:
        """How many PRs/issues are stored and whether that reaches the oldest one."""
        try:
            rows = self.connection.execute(
                "SELECT kind, stored, complete FROM github_sync WHERE repo=?", (repo,)
            ).fetchall()
        except sqlite3.OperationalError:
            return {}  # database written before the GitHub tables existed
        return {r["kind"]: {"stored": r["stored"], "complete": bool(r["complete"])} for r in rows}

    def _pr_from_row(self, row: sqlite3.Row) -> PullRequest:
        repo, number = row["repo"], row["number"]
        reviews = self.connection.execute(
            "SELECT reviewer, state, submitted_at FROM pr_reviews WHERE repo=? AND pr_number=? "
            "ORDER BY submitted_at, reviewer",
            (repo, number),
        )
        closes = self.connection.execute(
            "SELECT issue_number FROM pr_closes WHERE repo=? AND pr_number=? ORDER BY 1",
            (repo, number),
        )
        data = dict(row)
        data["labels"] = tuple(filter(None, data["labels"].split(",")))
        data["is_draft"] = bool(data["is_draft"])
        return PullRequest(
            **data,
            closes_issues=tuple(r[0] for r in closes),
            reviews=tuple(Review(*r) for r in reviews),
        )

    def get_pr(self, repo: str, number: int) -> PullRequest | None:
        row = self.connection.execute(
            "SELECT * FROM pull_requests WHERE repo=? AND number=?", (repo, number)
        ).fetchone()
        return self._pr_from_row(row) if row else None

    def find_prs(
        self,
        repo: str,
        *,
        state: str | None = None,
        author: str | None = None,
        merged_by: str | None = None,
        label: str | None = None,
        title_contains: str | None = None,
        limit: int = 50,
    ) -> list[PullRequest]:
        """Filtered PR list, newest number first. Filters are exact except the title substring."""
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            "SELECT * FROM pull_requests WHERE repo=? AND (? IS NULL OR state=?) "
            "AND (? IS NULL OR author=?) AND (? IS NULL OR merged_by=?) "
            "AND (? IS NULL OR instr(','||labels||',', ','||?||',')>0) "
            "AND (? IS NULL OR instr(lower(title), lower(?))>0) ORDER BY number DESC LIMIT ?",
            (
                repo,
                state,
                state,
                author,
                author,
                merged_by,
                merged_by,
                label,
                label,
                title_contains,
                title_contains,
                limit,
            ),
        ).fetchall()
        return [self._pr_from_row(r) for r in rows]

    def pr_for_commit(self, repo: str, sha: str) -> PullRequest | None:
        """The PR whose merge commit is `sha` (full SHA or a prefix of 7+ characters)."""
        if len(sha) < 7:
            raise ValueError("use at least 7 characters of the SHA")
        rows = self.connection.execute(
            "SELECT * FROM pull_requests WHERE repo=? AND merge_commit_sha!='' "
            "AND substr(merge_commit_sha,1,?)=? ORDER BY number LIMIT 2",
            (repo, len(sha), sha.lower()),
        ).fetchall()
        if len(rows) > 1:
            raise ValueError("ambiguous SHA prefix: matches more than one PR; use more characters")
        return self._pr_from_row(rows[0]) if rows else None

    def references(self, repo: str, number: int, *, limit: int = 25) -> dict[str, object]:
        """Textual `#N` links of a PR or issue, in both directions, from synced titles and bodies.

        Outgoing: numbers this item's title or body mention. Incoming: synced PRs and issues whose
        title or body mention it. These are written mentions only, not GitHub's timeline
        cross-reference events, and only synced items can be found. `closes_issues` is the
        structured closing link GitHub reports.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        kinds = {"pr": "pull_requests", "issue": "issues"}

        def fetch(kind: str, where: str, args: tuple) -> list[sqlite3.Row]:
            return self.connection.execute(
                f"SELECT number, title, body, state FROM {kinds[kind]} WHERE repo=? AND {where}",
                (repo, *args),
            ).fetchall()

        def describe(item: int) -> dict[str, object]:
            for kind in ("pr", "issue"):
                row = fetch(kind, "number=?", (item,))
                if row:
                    return {
                        "number": item,
                        "kind": kind,
                        "title": row[0]["title"],
                        "state": row[0]["state"],
                    }
            return {"number": item, "kind": "unknown", "title": "", "state": ""}

        own = next(
            (r for kind in ("pr", "issue") for r in fetch(kind, "number=?", (number,))), None
        )
        mentions: dict[int, set[str]] = {}
        if own is not None:
            for place, text in (("title", own["title"]), ("body", own["body"])):
                for found in referenced_numbers(text, repo) - {number}:
                    mentions.setdefault(found, set()).add(place)
        incoming: dict[tuple[str, int], dict[str, object]] = {}
        needles = (f"#{number}", f"/pull/{number}", f"/issues/{number}")
        condition = " OR ".join("instr(title,?)>0 OR instr(body,?)>0" for _ in needles)
        for kind in kinds:
            rows = fetch(
                kind, f"({condition}) AND number!=?", (*(n for n in needles for _ in "ab"), number)
            )
            for row in rows:
                places = [
                    place
                    for place, text in (("title", row["title"]), ("body", row["body"]))
                    if number in referenced_numbers(text, repo)
                ]
                if places:
                    incoming[(kind, row["number"])] = {
                        "number": row["number"],
                        "kind": kind,
                        "title": row["title"],
                        "state": row["state"],
                        "found_in": places,
                    }
        outgoing = [
            {**describe(n), "found_in": sorted(places)} for n, places in sorted(mentions.items())
        ]
        mentioned_by = sorted(incoming.values(), key=lambda item: int(item["number"]))  # type: ignore[call-overload]
        closes = [
            r[0]
            for r in self.connection.execute(
                "SELECT issue_number FROM pr_closes WHERE repo=? AND pr_number=? ORDER BY 1",
                (repo, number),
            )
        ]
        return {
            "number": number,
            "kind": "pr" if fetch("pr", "number=?", (number,)) else ("issue" if own else "unknown"),
            "mentions": outgoing[:limit],
            "mentioned_by": mentioned_by[:limit],
            "closes_issues": closes,
            "truncated": len(outgoing) > limit or len(mentioned_by) > limit,
            "note": "Written #N mentions in synced titles and bodies; not GitHub timeline events.",
        }

    def file_summary_titles(
        self, repo: str, path: str, allowed_shas: set[str], *, limit: int = 12
    ) -> dict[str, list[str]]:
        """Titles for exact touched paths, restricted to ancestors of the summary checkout.

        PR association requires a synced merge SHA touching the file in indexed history;
        a title mentioning a filename alone is not evidence. No author fields are returned.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        tables = {r[0] for r in self.connection.execute("SELECT name FROM sqlite_master")}
        if not {"history_files", "history_commits"} <= tables:
            return {"commits": [], "prs": []}
        rows = self.connection.execute(
            "SELECT c.sha, c.message FROM history_commits c JOIN history_files f "
            "ON c.repo=f.repo AND c.sha=f.sha WHERE c.repo=? AND (f.path=? OR f.old_path=?) "
            "ORDER BY datetime(c.commit_date) DESC, c.sha",
            (repo, path, path),
        ).fetchall()
        commits: list[str] = []
        prs: list[str] = []
        for row in rows:
            if row["sha"] not in allowed_shas:
                continue
            subject = _EMAIL.sub("<email omitted>", row["message"].strip().split("\n", 1)[0])
            if subject and subject not in commits and len(commits) < limit:
                commits.append(subject)
            if "pull_requests" in tables and len(prs) < limit:
                pr = self.pr_for_commit(repo, row["sha"])
                if pr and pr.state == "MERGED" and pr.title not in prs:
                    prs.append(_EMAIL.sub("<email omitted>", pr.title))
        return {"commits": commits, "prs": prs}

    def prs_closing_issue(self, repo: str, number: int) -> list[PullRequest]:
        rows = self.connection.execute(
            "SELECT p.* FROM pull_requests p JOIN pr_closes c "
            "ON c.repo=p.repo AND c.pr_number=p.number "
            "WHERE c.repo=? AND c.issue_number=? ORDER BY p.number",
            (repo, number),
        ).fetchall()
        return [self._pr_from_row(r) for r in rows]

    def _issue_from_row(self, row: sqlite3.Row) -> IssueMeta:
        data = dict(row)
        data["labels"] = tuple(filter(None, data["labels"].split(",")))
        data["closed_by_prs"] = tuple(
            int(n) for n in filter(None, data["closed_by_prs"].split(","))
        )
        return IssueMeta(**data)

    def get_issue_meta(self, repo: str, number: int) -> IssueMeta | None:
        row = self.connection.execute(
            "SELECT * FROM issues WHERE repo=? AND number=?", (repo, number)
        ).fetchone()
        return self._issue_from_row(row) if row else None

    def find_issues(
        self,
        repo: str,
        *,
        state: str | None = None,
        author: str | None = None,
        label: str | None = None,
        title_contains: str | None = None,
        limit: int = 50,
    ) -> list[IssueMeta]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            "SELECT * FROM issues WHERE repo=? AND (? IS NULL OR state=?) "
            "AND (? IS NULL OR author=?) "
            "AND (? IS NULL OR instr(','||labels||',', ','||?||',')>0) "
            "AND (? IS NULL OR instr(lower(title), lower(?))>0) ORDER BY number DESC LIMIT ?",
            (
                repo,
                state,
                state,
                author,
                author,
                label,
                label,
                title_contains,
                title_contains,
                limit,
            ),
        ).fetchall()
        return [self._issue_from_row(r) for r in rows]

    # --- Phase 5b: tags, identities and commit-level exact tools ---

    def sync_git_facts(
        self,
        repo: str,
        tags: list[TagRecord],
        identities: IdentityMap,
        links: list[LoginLink],
    ) -> None:
        """Atomically replace one repo's tags, name merges and login links."""
        if self.readonly:
            raise PermissionError("MetadataDB is open read-only")
        if any(t.repo != repo for t in tags) or any(link.repo != repo for link in links):
            raise ValueError("cross-repository git facts")
        with self.connection:
            for table in ("tags", "identities", "author_logins"):
                self.connection.execute(f"DELETE FROM {table} WHERE repo=?", (repo,))
            self.connection.executemany(
                "INSERT OR REPLACE INTO tags VALUES(?,?,?,?,?)",
                [(repo, t.name, t.sha, t.date, int(t.annotated)) for t in tags],
            )
            self.connection.executemany(
                "INSERT INTO identities VALUES(?,?,?)",
                [(repo, raw, canonical) for raw, canonical in identities.canonical.items()],
            )
            self.connection.executemany(
                "INSERT INTO author_logins VALUES(?,?,?,?,?)",
                [(repo, k.canonical_name, k.login, k.evidence, k.conflicts) for k in links],
            )

    def list_tags(self, repo: str, *, limit: int = 100) -> list[TagRecord]:
        """Tags oldest first (by tag date)."""
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            "SELECT * FROM tags WHERE repo=? ORDER BY datetime(date), name LIMIT ?", (repo, limit)
        )
        return [
            TagRecord(r["repo"], r["name"], r["sha"], r["date"], bool(r["annotated"])) for r in rows
        ]

    def get_tag(self, repo: str, name: str) -> TagRecord | None:
        r = self.connection.execute(
            "SELECT * FROM tags WHERE repo=? AND name=?", (repo, name)
        ).fetchone()
        return (
            TagRecord(r["repo"], r["name"], r["sha"], r["date"], bool(r["annotated"]))
            if r
            else None
        )

    def tags_for_commit(self, repo: str, sha: str) -> list[TagRecord]:
        """Tags that point exactly at this commit (use git_tools.releases_containing for 'first
        released in')."""
        if len(sha) < 7:
            raise ValueError("use at least 7 characters of the SHA")
        rows = self.connection.execute(
            "SELECT * FROM tags WHERE repo=? AND substr(sha,1,?)=? ORDER BY datetime(date), name",
            (repo, len(sha), sha.lower()),
        )
        return [
            TagRecord(r["repo"], r["name"], r["sha"], r["date"], bool(r["annotated"])) for r in rows
        ]

    def history_window(self, repo: str) -> HistoryWindow:
        row = self.connection.execute(
            "SELECT count(*), min(datetime(author_date)), max(datetime(author_date)) "
            "FROM history_commits WHERE repo=?",
            (repo,),
        ).fetchone()
        return HistoryWindow(row[0], row[1] or "", row[2] or "")

    _CANON = (
        "COALESCE((SELECT canonical_name FROM identities i "
        "WHERE i.repo=c.repo AND i.raw_name=c.author_name), c.author_name)"
    )

    def _login_sql(self) -> str:
        if not self._has_logins:
            return "NULL"
        return (
            "(SELECT login FROM author_logins l WHERE l.repo=c.repo "
            f"AND l.canonical_name={self._CANON})"
        )

    def _author_names(self, repo: str, who: str) -> list[str]:
        """Canonical names that `who` denotes: a canonical name or a linked GitHub login."""
        who = who.strip().casefold()
        if not who:
            raise ValueError("author must not be empty")
        names = {
            r[0]
            for r in self.connection.execute(
                f"SELECT DISTINCT {self._CANON} FROM history_commits c WHERE c.repo=?", (repo,)
            )
            if r[0].casefold() == who
        }
        with suppress(sqlite3.OperationalError):  # database written before the identity tables
            names |= {
                r[0]
                for r in self.connection.execute(
                    "SELECT canonical_name FROM author_logins WHERE repo=? AND lower(login)=?",
                    (repo, who),
                )
            }
        return sorted(names)

    def _commit_filter(
        self,
        repo: str,
        *,
        author: str | None,
        path: str | None,
        symbol: str | None,
        since: str | None,
        until: str | None,
        is_fix: bool | None,
        alt_paths: Sequence[str] = (),
    ) -> tuple[str, list[Any]]:
        """WHERE clause over history_commits aliased `c`; every value is a bound parameter.

        `alt_paths` are other names of `path` (it was renamed): a commit matches on any of them.
        """
        clauses: list[str] = ["c.repo=?"]
        params: list[Any] = [repo]
        if author is not None:
            names = self._author_names(repo, author)
            if not names:
                return "0", []
            clauses.append(f"{self._CANON} IN ({','.join('?' * len(names))})")
            params += names
        path = _clean_path(path)
        if path is not None:
            match, match_params = _path_match([path, *alt_paths])
            clauses.append(
                "EXISTS (SELECT 1 FROM history_files f WHERE f.repo=c.repo AND f.sha=c.sha "
                f"AND {match})"
            )
            params += match_params
        if symbol is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM history_symbols s WHERE s.repo=c.repo AND s.sha=c.sha "
                "AND (s.symbol=? OR substr(s.symbol,-?)=?))"
            )
            params += [symbol, len(symbol) + 1, "." + symbol]
        if _check_date(since, "since"):
            clauses.append("datetime(c.author_date) >= datetime(?)")
            params.append(since)
        if _check_date(until, "until"):
            clauses.append("datetime(c.author_date) < datetime(?)")
            params.append(until)
        if is_fix is not None:
            clauses.append("c.is_fix=?")
            params.append(int(is_fix))
        return " AND ".join(clauses), params

    def find_commits(
        self,
        repo: str,
        *,
        author: str | None = None,
        path: str | None = None,
        symbol: str | None = None,
        since: str | None = None,
        until: str | None = None,
        is_fix: bool | None = None,
        limit: int = 50,
    ) -> Answer[CommitSummary]:
        """Filtered log, newest first. `since` is inclusive and `until` exclusive (UTC).

        `author` is a canonical name or a linked GitHub login; `path` is a file or a directory;
        `symbol` is a qualified name or its last component. Merge commits are not in the index.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        where, params = self._commit_filter(
            repo, author=author, path=path, symbol=symbol, since=since, until=until, is_fix=is_fix
        )
        rows = self.connection.execute(
            f"SELECT c.sha, {self._CANON} AS author, c.author_date, c.message, c.is_fix "
            f"FROM history_commits c WHERE {where} "
            "ORDER BY datetime(c.author_date) DESC, c.sha LIMIT ?",
            [*params, limit],
        ).fetchall()
        summaries = tuple(
            CommitSummary(
                r["sha"],
                r["author"],
                r["author_date"],
                _EMAIL.sub("<email omitted>", r["message"].strip().split("\n", 1)[0]),
                bool(r["is_fix"]),
            )
            for r in rows
        )
        return Answer(summaries, self.history_window(repo))

    def count_commits(
        self,
        repo: str,
        *,
        author: str | None = None,
        path: str | None = None,
        symbol: str | None = None,
        since: str | None = None,
        until: str | None = None,
        is_fix: bool | None = None,
        alt_paths: Sequence[str] = (),
    ) -> CommitCount:
        where, params = self._commit_filter(
            repo,
            author=author,
            path=path,
            symbol=symbol,
            since=since,
            until=until,
            is_fix=is_fix,
            alt_paths=alt_paths,
        )
        (count,) = self.connection.execute(
            f"SELECT count(*) FROM history_commits c WHERE {where}", params
        ).fetchone()
        return CommitCount(count, self.history_window(repo))

    def top_contributors(
        self,
        repo: str,
        *,
        path: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 10,
        alt_paths: Sequence[str] = (),
    ) -> Answer[Contributor]:
        """Authors ranked by commits (ties: more added lines, then name), identities merged.

        With `path`, additions/deletions count only files under that path. `alt_paths` are older
        names of `path`, counted with it (a rename does not reset the history).
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        where, params = self._commit_filter(
            repo, author=None, path=None, symbol=None, since=since, until=until, is_fix=None
        )
        path = _clean_path(path)
        file_match: str = "1"
        file_params: list[Any] = []
        if path is not None:
            file_match, file_params = _path_match([path, *alt_paths])
        rows = self.connection.execute(
            f"SELECT {self._CANON} AS name, count(DISTINCT c.sha) AS commits, "
            "COALESCE(sum(f.additions),0) AS additions, COALESCE(sum(f.deletions),0) AS deletions, "
            "min(datetime(c.author_date)) AS first_commit, "
            "max(datetime(c.author_date)) AS last_commit, "
            f"{self._login_sql()} AS login "
            "FROM history_commits c JOIN history_files f ON f.repo=c.repo AND f.sha=c.sha "
            f"AND {file_match} WHERE {where} GROUP BY name "
            "ORDER BY commits DESC, additions DESC, name LIMIT ?",
            [*file_params, *params, limit],
        ).fetchall()
        return Answer(
            tuple(
                Contributor(
                    r["name"],
                    r["login"] or "",
                    r["commits"],
                    r["additions"],
                    r["deletions"],
                    r["first_commit"],
                    r["last_commit"],
                )
                for r in rows
            ),
            self.history_window(repo),
        )

    def _change_frequency_following(
        self,
        repo: str,
        renames: Mapping[str, str],
        *,
        path_prefix: str | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> Answer[FileFrequency]:
        where, params = self._commit_filter(
            repo, author=None, path=None, symbol=None, since=since, until=until, is_fix=None
        )
        rows = self.connection.execute(
            f"SELECT f.path AS path, c.sha AS sha, {self._CANON} AS author, "
            "f.additions AS additions, f.deletions AS deletions, "
            "datetime(c.author_date) AS changed "
            "FROM history_commits c JOIN history_files f ON f.repo=c.repo AND f.sha=c.sha "
            f"WHERE {where}",
            params,
        ).fetchall()
        prefix = _clean_path(path_prefix)
        grouped: dict[str, dict[str, Any]] = {}
        for r in rows:
            latest = follow_renames(r["path"], renames)
            if prefix is not None and not (latest == prefix or latest.startswith(prefix + "/")):
                continue
            g = grouped.setdefault(
                latest, {"shas": set(), "authors": set(), "add": 0, "del": 0, "last": ""}
            )
            g["shas"].add(r["sha"])
            g["authors"].add(r["author"])
            g["add"] += r["additions"] or 0
            g["del"] += r["deletions"] or 0
            g["last"] = max(g["last"], r["changed"] or "")
        ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]["shas"]), item[0]))[:limit]
        return Answer(
            tuple(
                FileFrequency(
                    path, len(g["shas"]), len(g["authors"]), g["add"], g["del"], g["last"]
                )
                for path, g in ranked
            ),
            self.history_window(repo),
        )

    def change_frequency(
        self,
        repo: str,
        *,
        path_prefix: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
        renames: Mapping[str, str] | None = None,
    ) -> Answer[FileFrequency]:
        """Most-changed files (distinct commits touching the path as named at that commit).

        A file renamed during the window is counted under each name it had, unless `renames`
        (old path -> new path, from `git_tools.exact_renames`) is given: then every name is
        counted under the file's latest name, and a commit touching two of its names counts once.
        `path_prefix` then applies to the latest name.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        if renames:
            return self._change_frequency_following(
                repo, renames, path_prefix=path_prefix, since=since, until=until, limit=limit
            )
        where, params = self._commit_filter(
            repo, author=None, path=None, symbol=None, since=since, until=until, is_fix=None
        )
        prefix = _clean_path(path_prefix)
        file_match, file_params = ("1", [])
        if prefix is not None:
            file_match = "(f.path=? OR substr(f.path,1,?)=?)"
            file_params = [prefix, len(prefix) + 1, prefix + "/"]
        rows = self.connection.execute(
            "SELECT f.path AS path, count(DISTINCT c.sha) AS commits, "
            f"count(DISTINCT {self._CANON}) AS authors, sum(f.additions) AS additions, "
            "sum(f.deletions) AS deletions, max(datetime(c.author_date)) AS last_changed "
            "FROM history_commits c JOIN history_files f ON f.repo=c.repo AND f.sha=c.sha "
            f"WHERE {where} AND {file_match} GROUP BY f.path "
            "ORDER BY commits DESC, f.path LIMIT ?",
            [*params, *file_params, limit],
        ).fetchall()
        return Answer(
            tuple(
                FileFrequency(
                    r["path"],
                    r["commits"],
                    r["authors"],
                    r["additions"],
                    r["deletions"],
                    r["last_changed"],
                )
                for r in rows
            ),
            self.history_window(repo),
        )
