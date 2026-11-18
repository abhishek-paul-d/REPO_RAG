"""A small JSON registry of indexed repos: what was indexed, at which commit, with which model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RepoEntry:
    slug: str
    commit_sha: str
    embedding_id: str
    indexed_at: str
    clone_path: str = ""
    history_sha: str = ""  # history and code may be indexed at different revisions
    files: dict[str, int] = field(default_factory=dict)  # by kind: code / doc / config
    chunks: dict[str, int] = field(default_factory=dict)  # by collection kind: code / docs


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def entries(self) -> dict[str, RepoEntry]:
        return {slug: RepoEntry(**data) for slug, data in self._read().items()}

    def get(self, slug: str) -> RepoEntry | None:
        return self.entries().get(slug)

    def put(self, entry: RepoEntry) -> None:
        data = self._read()
        data[entry.slug] = asdict(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def remove(self, slug: str) -> None:
        data = self._read()
        if data.pop(slug, None) is not None:
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
