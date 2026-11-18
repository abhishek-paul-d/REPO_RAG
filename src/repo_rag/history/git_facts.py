"""Facts read from a local checkout for metadata.db: tags and author identities (Phase 5b).

Identity merging is evidence-based and names-only (emails are never read into results):
- `.mailmap`: git's own `%aN` (mailmap-aware name) against the raw `%an` for every commit;
- GitHub login links: the author of a squash/rebase merge commit (one parent) is, by
  construction, the PR author, so (commit author name, PR author login) is real evidence.
  Ordinary merge commits are skipped because their author is the merger.
Two names are never merged by similarity. A raw name whose mailmap result is ambiguous
(the same name mapped to different canonical names) stays unmerged and is counted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from git import GitCommandError, Repo

from repo_rag.history.github_sync import PullRequest

_SEP = "\x1f"
_CHUNK = 100  # SHAs per git call, to stay under Windows' command-line length limit


@dataclass(frozen=True)
class TagRecord:
    repo: str
    name: str
    sha: str  # the tagged commit (annotated tags are peeled)
    date: str  # creation date of the tag object, or the commit date for lightweight tags
    annotated: bool


@dataclass(frozen=True)
class IdentityMap:
    canonical: dict[str, str]  # raw author name -> canonical name (only names that change)
    ambiguous: tuple[str, ...]  # raw names mailmapped to more than one canonical name


@dataclass(frozen=True)
class LoginLink:
    repo: str
    canonical_name: str
    login: str
    evidence: int  # merged single-parent PRs showing this pairing
    conflicts: int  # PRs pairing the same name with a different login


def read_tags(root: Path, slug: str) -> list[TagRecord]:
    """All tags that point at a commit, oldest first."""
    with Repo(root) as repo:
        text = repo.git.for_each_ref(
            "--format=%(refname:short)%1f%(objecttype)%1f%(objectname)%1f%(*objectname)"
            "%1f%(*objecttype)%1f%(creatordate:iso-strict)",
            "--sort=creatordate",
            "refs/tags",
        )
    tags: list[TagRecord] = []
    for line in text.split("\n"):
        if _SEP not in line:
            continue
        name, kind, target, peeled, peeled_kind, date = line.split(_SEP, 5)
        if kind == "commit":
            tags.append(TagRecord(slug, name, target, date, False))
        elif kind == "tag" and peeled_kind == "commit":
            tags.append(TagRecord(slug, name, peeled, date, True))
    return tags


def read_identity_map(root: Path, *, ref: str = "HEAD", max_commits: int) -> IdentityMap:
    """Raw author name -> mailmap canonical name, over the same bounded history as the index."""
    with Repo(root) as repo:
        text = repo.git.log(
            "--no-merges",
            f"--max-count={max_commits}",
            "--use-mailmap",
            "--format=%an%x1f%aN",
            ref,
        )
    seen: dict[str, set[str]] = defaultdict(set)
    for line in text.split("\n"):
        if _SEP in line:
            raw, canonical = line.split(_SEP, 1)
            seen[raw].add(canonical)
    canonical = {raw: next(iter(c)) for raw, c in seen.items() if len(c) == 1 and raw not in c}
    ambiguous = tuple(sorted(raw for raw, c in seen.items() if len(c) > 1))
    return IdentityMap(canonical, ambiguous)


def _squash_authors(root: Path, shas: list[str]) -> dict[str, str]:
    """sha -> raw author name, for commits that exist locally and have exactly one parent."""
    result: dict[str, str] = {}
    with Repo(root) as repo:
        for start in range(0, len(shas), _CHUNK):
            batch = shas[start : start + _CHUNK]
            try:
                text = repo.git.log("--no-walk=unsorted", "--format=%H%x1f%P%x1f%an", *batch)
            except GitCommandError:
                # One unknown SHA fails the whole call: fall back to one at a time.
                text = ""
                for sha in batch:
                    try:
                        text += repo.git.log("--no-walk", "--format=%H%x1f%P%x1f%an", sha) + "\n"
                    except GitCommandError:
                        continue
            for line in text.split("\n"):
                if line.count(_SEP) == 2:
                    sha, parents, author = line.split(_SEP)
                    if len(parents.split()) == 1:
                        result[sha] = author
    return result


def read_login_links(
    root: Path, slug: str, prs: Iterable[PullRequest], identities: IdentityMap
) -> list[LoginLink]:
    """Pair canonical commit-author names with PR author logins; skip anything unproven."""
    merged = {p.merge_commit_sha: p.author for p in prs if p.merge_commit_sha and p.author}
    authors = _squash_authors(root, sorted(merged))
    tally: dict[str, Counter[str]] = defaultdict(Counter)
    for sha, raw in authors.items():
        tally[identities.canonical.get(raw, raw)][merged[sha]] += 1
    links = []
    for name, logins in sorted(tally.items()):
        login, evidence = logins.most_common(1)[0]
        links.append(LoginLink(slug, name, login, evidence, sum(logins.values()) - evidence))
    return links
