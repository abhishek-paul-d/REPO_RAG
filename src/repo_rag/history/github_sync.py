"""Read-only GraphQL sync of pull requests (with reviews) and issues (Phase 5b).

Newest-first by `updatedAt`. The first sync takes the newest `max_prs` / `max_issues`; later
syncs fetch only what changed since the saved watermark and merge it by number. Raw nodes are
kept in `data/github_cache/sync/` and a pass is saved only if it finished, so a failure never
leaves a half-written or silently shortened index. `complete` records whether the sync reached
the oldest item, so tools can say the index covers only the newest N.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repo_rag.config import Settings
from repo_rag.history.issue_linker import GitHubClient, GitHubLookupError

_PR_FIELDS = """number title body state isDraft createdAt updatedAt mergedAt closedAt
 baseRefName headRefName changedFiles additions deletions
 author{login} mergedBy{login} mergeCommit{oid}
 labels(first:30){nodes{name}}
 closingIssuesReferences(first:20){nodes{number repository{nameWithOwner}}}
 reviews(first:$reviews){totalCount nodes{author{login} state submittedAt}}"""

_ISSUE_FIELDS = """number title body state createdAt updatedAt closedAt
 author{login}
 labels(first:30){nodes{name}}
 closedByPullRequestsReferences(first:5,includeClosedPrs:false){nodes{number}}"""

PR_QUERY = (
    "query($owner:String!,$name:String!,$size:Int!,$reviews:Int!,$cursor:String){"
    "repository(owner:$owner,name:$name){pullRequests(first:$size,after:$cursor,"
    "orderBy:{field:UPDATED_AT,direction:DESC}){nodes{" + _PR_FIELDS + "}"
    "pageInfo{hasNextPage endCursor}}}}"
)
ISSUE_QUERY = (
    "query($owner:String!,$name:String!,$size:Int!,$cursor:String){"
    "repository(owner:$owner,name:$name){issues(first:$size,after:$cursor,"
    "orderBy:{field:UPDATED_AT,direction:DESC}){nodes{" + _ISSUE_FIELDS + "}"
    "pageInfo{hasNextPage endCursor}}}}"
)


@dataclass(frozen=True)
class Review:
    reviewer: str
    state: str
    submitted_at: str


@dataclass(frozen=True)
class PullRequest:
    repo: str
    number: int
    title: str
    body: str
    state: str  # OPEN | CLOSED | MERGED
    is_draft: bool
    author: str
    merged_by: str
    created_at: str
    updated_at: str
    merged_at: str
    closed_at: str
    base: str
    head: str
    merge_commit_sha: str
    labels: tuple[str, ...]
    changed_files: int
    additions: int
    deletions: int
    closes_issues: tuple[int, ...]
    reviews: tuple[Review, ...]
    reviews_total: int  # more than len(reviews) means the review list was cut


@dataclass(frozen=True)
class IssueMeta:
    repo: str
    number: int
    title: str
    body: str
    state: str  # OPEN | CLOSED
    author: str
    labels: tuple[str, ...]
    created_at: str
    updated_at: str
    closed_at: str
    closed_by_prs: tuple[int, ...]


@dataclass(frozen=True)
class SyncResult:
    kind: str
    fetched: int  # items downloaded in this pass
    total: int  # items now stored
    complete: bool  # reached the oldest item (not cut by the cap)


def _login(node: dict[str, Any] | None) -> str:
    return str((node or {}).get("login") or "")  # deleted users come back as null


def _names(connection: dict[str, Any] | None) -> tuple[str, ...]:
    return tuple(n["name"] for n in (connection or {}).get("nodes") or [])


def parse_pr(repo: str, node: dict[str, Any]) -> PullRequest:
    reviews = node.get("reviews") or {}
    return PullRequest(
        repo=repo,
        number=int(node["number"]),
        title=node["title"],
        body=node.get("body") or "",
        state=node["state"],
        is_draft=bool(node.get("isDraft")),
        author=_login(node.get("author")),
        merged_by=_login(node.get("mergedBy")),
        created_at=node["createdAt"],
        updated_at=node["updatedAt"],
        merged_at=node.get("mergedAt") or "",
        closed_at=node.get("closedAt") or "",
        base=node.get("baseRefName") or "",
        head=node.get("headRefName") or "",
        merge_commit_sha=(node.get("mergeCommit") or {}).get("oid") or "",
        labels=_names(node.get("labels")),
        changed_files=int(node.get("changedFiles") or 0),
        additions=int(node.get("additions") or 0),
        deletions=int(node.get("deletions") or 0),
        closes_issues=tuple(
            int(n["number"])
            for n in (node.get("closingIssuesReferences") or {}).get("nodes") or []
            if n["repository"]["nameWithOwner"] == repo  # only same-repo issues are keyed here
        ),
        reviews=tuple(
            Review(_login(r.get("author")), r["state"], r.get("submittedAt") or "")
            for r in reviews.get("nodes") or []
        ),
        reviews_total=int(reviews.get("totalCount") or 0),
    )


def parse_issue(repo: str, node: dict[str, Any]) -> IssueMeta:
    return IssueMeta(
        repo=repo,
        number=int(node["number"]),
        title=node["title"],
        body=node.get("body") or "",
        state=node["state"],
        author=_login(node.get("author")),
        labels=_names(node.get("labels")),
        created_at=node["createdAt"],
        updated_at=node["updatedAt"],
        closed_at=node.get("closedAt") or "",
        closed_by_prs=tuple(
            int(n["number"])
            for n in (node.get("closedByPullRequestsReferences") or {}).get("nodes") or []
        ),
    )


class GitHubSync:
    """Sync one repository's PRs and issues into a local raw-node store."""

    def __init__(self, settings: Settings, client: GitHubClient) -> None:
        self.settings = settings
        self.client = client

    def _state_path(self, repo: str, kind: str) -> Path:
        return self.settings.github_cache_dir / "sync" / f"{repo.replace('/', '__')}.{kind}.json"

    def load(self, repo: str, kind: str) -> dict[str, Any]:
        path = self._state_path(repo, kind)
        if not path.exists():
            return {"watermark": "", "complete": False, "items": {}}
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def _sync(
        self,
        repo: str,
        kind: str,
        query: str,
        field: str,
        extra: dict[str, Any],
        cap: int,
    ) -> SyncResult:
        owner, name = repo.split("/")
        state = self.load(repo, kind)
        watermark = state["watermark"]
        items: dict[str, Any] = dict(state["items"])
        fetched, cursor = 0, None
        complete = bool(state["complete"])  # stays true once the oldest item has been seen
        while True:
            data = self.client.fetch(
                query,
                {
                    "owner": owner,
                    "name": name,
                    "size": self.settings.github_sync_page_size,
                    "cursor": cursor,
                    **extra,
                },
            )
            repository = data.get("repository")
            if repository is None:
                raise GitHubLookupError(f"Repository unavailable: {repo}")
            connection = repository[field]
            nodes, page = connection["nodes"], connection["pageInfo"]
            stop = False
            for position, node in enumerate(nodes):
                if watermark and node["updatedAt"] < watermark:
                    stop = True  # older than the last sync: already stored
                    break
                items[str(node["number"])] = node
                fetched += 1
                if not watermark and fetched >= cap:
                    more = position + 1 < len(nodes) or page["hasNextPage"]
                    complete = not more  # cut by the cap unless that was the very last item
                    stop = True
                    break
            if stop:
                break
            if not page["hasNextPage"]:
                complete = True
                break
            cursor = page["endCursor"]
        newest = max((n["updatedAt"] for n in items.values()), default="")
        path = self._state_path(repo, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"watermark": newest, "complete": complete, "items": items}),
            encoding="utf-8",
        )
        tmp.replace(path)
        return SyncResult(kind, fetched, len(items), complete)

    def sync_prs(self, repo: str, *, cap: int | None = None) -> SyncResult:
        return self._sync(
            repo,
            "prs",
            PR_QUERY,
            "pullRequests",
            {"reviews": self.settings.github_review_page},
            cap or self.settings.max_prs,
        )

    def sync_issues(self, repo: str, *, cap: int | None = None) -> SyncResult:
        return self._sync(
            repo, "issues", ISSUE_QUERY, "issues", {}, cap or self.settings.max_issues
        )

    def pull_requests(self, repo: str) -> list[PullRequest]:
        items = self.load(repo, "prs")["items"].values()
        return sorted((parse_pr(repo, n) for n in items), key=lambda p: -p.number)

    def issues(self, repo: str) -> list[IssueMeta]:
        items = self.load(repo, "issues")["items"].values()
        return sorted((parse_issue(repo, n) for n in items), key=lambda i: -i.number)
