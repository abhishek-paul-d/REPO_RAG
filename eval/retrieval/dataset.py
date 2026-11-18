"""Pinned questions and source evidence. Scoring never reads symbol metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from git import Repo
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoldRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    symbol: str  # human audit aid only; never used by the scorer

    @model_validator(mode="after")
    def valid_range(self) -> GoldRange:
        path = PurePosixPath(self.path)
        if self.end_line < self.start_line:
            raise ValueError("end_line precedes start_line")
        if path.is_absolute() or ".." in path.parts or "\\" in self.path or ":" in self.path:
            raise ValueError("evidence path must be repository-relative POSIX")
        return self


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    repo: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    question: str = Field(min_length=1)
    category: str
    split: str = Field(pattern=r"^(dev|test)$")
    group: str
    provenance: str
    gold: list[GoldRange] = Field(min_length=1)
    supporting_gold: list[GoldRange] = Field(default_factory=list)
    schema_version: int = 1
    cohort: Literal["legacy_regression", "fresh"] = "legacy_regression"
    target_type: Literal["implementation", "test", "example"] = "implementation"
    review_status: Literal["author_checked", "independently_reviewed"] = "author_checked"
    gold_review: str = ""


def read_questions(path: Path) -> list[Question]:
    rows = [
        Question.model_validate_json(s)
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if not rows:
        raise ValueError(f"empty question dataset: {path}")
    if len({q.id for q in rows}) != len(rows):
        raise ValueError("duplicate question ids")
    if len({(q.repo, q.commit_sha) for q in rows}) != 1:
        raise ValueError("one dataset must describe exactly one repository revision")
    for group in {q.group for q in rows}:
        if len({q.split for q in rows if q.group == group}) > 1:
            raise ValueError(f"evidence group crosses dev/test: {group}")
    # Even differently labelled groups must not share evidence across splits.
    for left in rows:
        for right in rows:
            if left.split != right.split and any(
                overlaps(a, b)
                for a in [*left.gold, *left.supporting_gold]
                for b in [*right.gold, *right.supporting_gold]
            ):
                raise ValueError(f"overlapping evidence crosses dev/test: {left.id}, {right.id}")
    return rows


def overlaps(a: GoldRange, b: GoldRange) -> bool:
    return a.path == b.path and a.start_line <= b.end_line and b.start_line <= a.end_line


def evidence_hash(source: str, start: int, end: int) -> str:
    return hashlib.sha256(
        "".join(source.splitlines(keepends=True)[start - 1 : end]).encode()
    ).hexdigest()


def validate_checkout(root: Path, questions: list[Question]) -> None:
    repo = Repo(root)
    if repo.head.commit.hexsha != questions[0].commit_sha:
        raise ValueError(
            f"wrong revision for {questions[0].repo}: expected {questions[0].commit_sha}"
        )
    if repo.is_dirty(untracked_files=False):
        raise ValueError("evaluation requires a clean tracked checkout")
    for q in questions:
        for gold in [*q.gold, *q.supporting_gold]:
            source = (root / gold.path).read_text(encoding="utf-8")
            if gold.end_line > len(source.splitlines()):
                raise ValueError(f"out-of-file gold range: {q.id}")
            if evidence_hash(source, gold.start_line, gold.end_line) != gold.source_sha256:
                raise ValueError(f"source evidence changed: {q.id} {gold.path}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
