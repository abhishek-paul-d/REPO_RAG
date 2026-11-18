"""Parse GitHub URLs and keep local clones under `data/repos/`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from repo_rag.config import Settings, get_settings

_OWNER = r"(?P<owner>[A-Za-z0-9-]+)"
_NAME = r"(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?"
_URL_PATTERNS = (
    # Extra path segments (/tree/main, /pull/12, ...) are ignored.
    re.compile(rf"^(?:https?://)?(?:www\.)?github\.com/{_OWNER}/{_NAME}(?:/.*)?$"),
    re.compile(rf"^git@github\.com:{_OWNER}/{_NAME}$"),
    re.compile(rf"^{_OWNER}/{_NAME}$"),
)


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.slug}.git"

    @property
    def dir_name(self) -> str:
        return f"{self.owner}__{self.name}"


def parse_repo_url(url: str) -> RepoRef:
    """Accept `[https://]github.com/o/r[.git][/...]`, `git@github.com:o/r.git` or `o/r`."""
    text = url.strip()
    for pattern in _URL_PATTERNS:
        match = pattern.match(text)
        if match and match["name"] not in {"", ".", ".."}:
            return RepoRef(owner=match["owner"], name=match["name"])
    raise ValueError(f"Not a GitHub repository URL: {url!r}")


_COMMIT = re.compile(r"[0-9a-fA-F]{7,40}")


def clone_repo(
    source: str, dest: Path, *, with_history: bool = True, commit: str | None = None
) -> Path:
    """Clone `source` into `dest`, or fast-forward `dest` if it is already a clone.

    With history this is a blobless clone (`--filter=blob:none`): every commit is fetched, file
    contents only on demand. Without history it is a `--depth 1` clone. Local sources must be
    `file://` URLs for either option to take effect.

    `commit` (7 to 40 hex characters) checks that revision out, detached, instead of following the
    default branch; an existing clone is then not fast-forwarded. It needs the full history.
    """
    if commit is not None:
        if not _COMMIT.fullmatch(commit):
            raise ValueError("commit must be 7 to 40 hexadecimal characters")
        if not with_history:
            raise ValueError("Pinning a commit needs the full history, not a shallow clone")
    if dest.exists():
        try:
            repo = Repo(dest)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            raise FileExistsError(f"{dest} exists and is not a git repository") from exc
        if commit is None:
            repo.remotes.origin.pull(ff_only=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if with_history:
            Repo.clone_from(source, dest, multi_options=["--filter=blob:none"])
        else:
            Repo.clone_from(source, dest, depth=1)
    if commit is not None:
        repo = Repo(dest)
        if not _has_commit(repo, commit):
            # A reused clone can be shallow (an earlier `--no-history` run) or older than the
            # pin: fetch the missing history instead of failing at the checkout.
            fetch = ["origin", "--tags", "--filter=blob:none"]
            if (Path(repo.git_dir) / "shallow").exists():
                fetch.insert(1, "--unshallow")
            repo.git.fetch(*fetch)
        repo.git.checkout("--detach", commit)
    return dest


def _has_commit(repo: Repo, commit: str) -> bool:
    try:
        repo.git.cat_file("-e", f"{commit}^{{commit}}")
    except GitCommandError:
        return False
    return True


def clone(
    url: str,
    settings: Settings | None = None,
    *,
    with_history: bool = True,
    commit: str | None = None,
) -> tuple[RepoRef, Path]:
    """Clone a public GitHub repository into `settings.clones_dir` (optionally at `commit`)."""
    settings = settings or get_settings()
    ref = parse_repo_url(url)
    path = clone_repo(
        ref.clone_url,
        settings.clones_dir / ref.dir_name,
        with_history=with_history,
        commit=commit,
    )
    return ref, path


def head_commit(repo_dir: Path) -> str:
    return Repo(repo_dir).head.commit.hexsha
