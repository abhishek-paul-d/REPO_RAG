"""Typed issue links and a cached, read-only GitHub GraphQL client.

A bare #123 or squash PR suffix is never treated as an issue-closing link. Direct
references are verified with GitHub's Issue type; PR links use closingIssuesReferences.
No issue text is passed to commit document construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from repo_rag.config import Settings

_CLOSING = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b\s*:?\s*"
    r"(?P<refs>(?:(?:[\w.-]+/[\w.-]+)?#\d+|https://github\.com/[\w.-]+/[\w.-]+/issues/\d+)"
    r"(?:\s*(?:,|and)\s*(?:(?:[\w.-]+/[\w.-]+)?#\d+|https://github\.com/[\w.-]+/[\w.-]+/issues/\d+))*)",
    re.IGNORECASE,
)
_REFERENCE = re.compile(
    r"https://github\.com/(?P<urlrepo>[\w.-]+/[\w.-]+)/issues/(?P<urlnum>\d+)"
    r"|(?:(?P<repo>[\w.-]+/[\w.-]+))?#(?P<num>\d+)"
)
_ISSUE_NUMBER = re.compile(r"(?<![\w])(?:[\w.-]+/[\w.-]+)?#\d+")
_ISSUE_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+\S*")
_ISSUE_ROLE = re.compile(r":(?:issue|pr):`\d+`", re.IGNORECASE)
_ISSUE_WORD = re.compile(r"\b(?:issue|pull request|pr)\s+#?\d+\b", re.IGNORECASE)
_ISSUE_ANCHOR = re.compile(r"(?m)^\s*\.\.\s+_#\d+:\s*$")


@dataclass(frozen=True)
class IssueRef:
    repo: str
    number: int


@dataclass(frozen=True)
class Issue:
    repo: str
    number: int
    title: str
    body: str
    state: str
    url: str


@dataclass(frozen=True)
class IssueLink:
    repo: str
    commit_sha: str
    issue_repo: str
    issue_number: int
    link_type: Literal["direct", "via_pr"]
    pr_number: int | None = None


def direct_references(message: str, repo: str) -> list[IssueRef]:
    refs: dict[tuple[str, int], IssueRef] = {}
    for closing in _CLOSING.finditer(message):
        for match in _REFERENCE.finditer(closing["refs"]):
            owner = match["urlrepo"] or match["repo"] or repo
            number = int(match["urlnum"] or match["num"])
            if number > 0:
                refs[(owner, number)] = IssueRef(owner, number)
    return list(refs.values())


def strip_issue_references(text: str) -> str:
    """Remove reference spans while retaining adjacent message/diff descriptions.

    Apply to the entire document input (including diff lines), not just the subject.
    Linked issue title/body are never appended, in production or evaluation.
    """
    text = _CLOSING.sub("", text)
    for pattern in (_ISSUE_ANCHOR, _ISSUE_ROLE, _ISSUE_WORD, _ISSUE_URL, _ISSUE_NUMBER):
        text = pattern.sub("", text)
    return text.strip()


class GitHubLookupError(RuntimeError):
    """A lookup is incomplete or failed; it must not silently become an empty link set."""


Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class GitHubClient:
    """Cache successful GraphQL pages. Inject transport for fully offline fixture tests."""

    def __init__(self, settings: Settings, *, transport: Transport | None = None) -> None:
        self.settings = settings
        self._transport = transport or self._request

    def _request(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        secret = self.settings.github_token
        if not secret or not secret.get_secret_value():
            raise GitHubLookupError("Set GITHUB_TOKEN in .env before live GitHub lookups")
        request = Request(
            "https://api.github.com/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Authorization": f"Bearer {secret.get_secret_value()}",
                "Content-Type": "application/json",
                "User-Agent": "repo-rag",
            },
        )
        try:
            with urlopen(request, timeout=self.settings.github_timeout_seconds) as response:
                return dict(json.load(response))
        except HTTPError as exc:
            # Do not include request headers, API response bodies or tokens in exceptions.
            raise GitHubLookupError(f"GitHub HTTP {exc.code}; lookup not cached") from None
        except (URLError, TimeoutError) as exc:
            raise GitHubLookupError(f"GitHub connection failed ({type(exc).__name__})") from None

    def fetch(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """One uncached, error-checked call, for syncs that must see current data."""
        payload = self._transport(query, variables)
        if payload.get("errors") or not isinstance(payload.get("data"), dict):
            raise GitHubLookupError("GitHub returned GraphQL errors; sync not saved")
        return dict(payload["data"])

    def query(
        self, query: str, variables: dict[str, Any], *, allow_missing_issue: bool = False
    ) -> dict[str, Any]:
        key = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
        target = self.settings.github_cache_dir / f"{key}.json"
        if target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
        else:
            payload = self._transport(query, variables)
            errors = payload.get("errors") or []
            repository = (payload.get("data") or {}).get("repository")
            if (
                allow_missing_issue
                and errors
                and repository is not None
                and "issueOrPullRequest" in repository
                and repository["issueOrPullRequest"] is None
                and all(error.get("type") == "NOT_FOUND" for error in errors)
            ):
                # A deleted/absent issue in a visible repository is an explicit missing
                # record, not a rate-limit/auth/network failure. Cache the tombstone.
                payload = {"data": payload["data"], "missing_issue": True}
            if payload.get("errors") or not isinstance(payload.get("data"), dict):
                raise GitHubLookupError("GitHub returned GraphQL errors; lookup not cached")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        return dict(payload["data"])

    def issue(self, ref: IssueRef) -> Issue | None:
        owner, name = ref.repo.split("/")
        data = self.query(
            """query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){
            issueOrPullRequest(number:$number){__typename ... on Issue{
            number title body state url}}}}""",
            {"owner": owner, "name": name, "number": ref.number},
            allow_missing_issue=True,
        )
        repository = data.get("repository")
        if repository is None:
            raise GitHubLookupError(f"Repository unavailable: {ref.repo}")
        node = repository.get("issueOrPullRequest")
        if not node or node.get("__typename") != "Issue":
            return None
        return Issue(
            ref.repo, int(node["number"]), node["title"], node["body"], node["state"], node["url"]
        )

    def closing_issues(self, repo: str, sha: str) -> list[tuple[int, IssueRef]]:
        """Page both associated PRs and each PR's closing issues. Never infer from #suffix."""
        owner, name = repo.split("/")
        cursor = None
        prs: list[int] = []
        while True:
            data = self.query(
                """query($owner:String!,$name:String!,$sha:String!,$cursor:String){
                repository(owner:$owner,name:$name){object(expression:$sha){... on Commit{
                associatedPullRequests(first:100,after:$cursor){nodes{number mergedAt
                baseRepository{nameWithOwner}} pageInfo{hasNextPage endCursor}}}}}}""",
                {"owner": owner, "name": name, "sha": sha, "cursor": cursor},
            )
            repository = data.get("repository")
            obj = repository.get("object") if repository else None
            if not obj or "associatedPullRequests" not in obj:
                raise GitHubLookupError(f"Commit unavailable on GitHub: {repo}@{sha}")
            connection = obj["associatedPullRequests"]
            prs.extend(
                int(n["number"])
                for n in connection["nodes"]
                if n.get("mergedAt") and n.get("baseRepository", {}).get("nameWithOwner") == repo
            )
            cursor = _next_cursor(connection)
            if cursor is None:
                break
        links: list[tuple[int, IssueRef]] = []
        for number in sorted(set(prs)):
            cursor = None
            while True:
                data = self.query(
                    """query($owner:String!,$name:String!,$number:Int!,$cursor:String){
                    repository(owner:$owner,name:$name){pullRequest(number:$number){
                    closingIssuesReferences(first:100,after:$cursor){nodes{number
                    repository{nameWithOwner}} pageInfo{hasNextPage endCursor}}}}}""",
                    {"owner": owner, "name": name, "number": number, "cursor": cursor},
                )
                repository = data.get("repository")
                pr = repository.get("pullRequest") if repository else None
                if not pr:
                    raise GitHubLookupError(f"PR unavailable: {repo}#{number}")
                connection = pr["closingIssuesReferences"]
                links.extend(
                    (number, IssueRef(n["repository"]["nameWithOwner"], int(n["number"])))
                    for n in connection["nodes"]
                )
                cursor = _next_cursor(connection)
                if cursor is None:
                    break
        return links

    def linked_pull_requests(self, repo: str, *, limit: int) -> list[tuple[int, str]]:
        """Discover merged PRs with issue links, then verify them through closing_issues.

        Search is label discovery only; never use PR bodies as retrieval documents. GitHub
        search has a 1,000-result ceiling, so this bounded helper refuses larger requests.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("linked PR discovery limit must be between 1 and 1000")
        candidates: list[tuple[int, str]] = []
        cursor = None
        while len(candidates) < limit:
            data = self.query(
                """query($query:String!,$cursor:String){search(query:$query,type:ISSUE,
                first:100,after:$cursor){nodes{... on PullRequest{number mergedAt mergeCommit{oid}}}
                pageInfo{hasNextPage endCursor}}}""",
                {
                    "query": f"repo:{repo} is:pr is:merged linked:issue sort:updated-desc",
                    "cursor": cursor,
                },
            )
            connection = data["search"]
            candidates.extend(
                (int(node["number"]), str(node["mergeCommit"]["oid"]))
                for node in connection["nodes"]
                if node.get("mergedAt") and node.get("mergeCommit")
            )
            cursor = _next_cursor(connection)
            if cursor is None:
                break
        return candidates[:limit]


def _next_cursor(connection: dict[str, Any]) -> str | None:
    page = connection["pageInfo"]
    if not page["hasNextPage"]:
        return None
    if not page.get("endCursor"):
        raise GitHubLookupError("GitHub pagination is incomplete")
    return str(page["endCursor"])


def link_commit(
    repo: str, sha: str, message: str, client: GitHubClient
) -> tuple[list[IssueLink], list[Issue]]:
    """Resolve both paths, preserving provenance even when both point at the same issue."""
    links: list[IssueLink] = []
    issues: dict[IssueRef, Issue] = {}
    for ref in direct_references(message, repo):
        issue = client.issue(ref)
        if issue:
            issues[ref] = issue
            links.append(IssueLink(repo, sha, ref.repo, ref.number, "direct"))
    for pr, ref in client.closing_issues(repo, sha):
        issue = issues.get(ref) or client.issue(ref)
        if issue:
            issues[ref] = issue
            links.append(IssueLink(repo, sha, ref.repo, ref.number, "via_pr", pr))
    return list(dict.fromkeys(links)), list(issues.values())
