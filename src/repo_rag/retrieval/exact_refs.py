"""Detect exact references in a question: `#N`, SHAs, tags, `@handles` and file paths.

Pure text analysis, no repository access. The agent's pre-step uses the hits to route a
question to the exact-answer tools (git, metadata.db) instead of similarity search. A SHA or
tag is only a candidate here; the tool that looks it up confirms that it exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RefKind = Literal["issue_or_pr", "sha", "tag", "handle", "path"]

_NUMBER = re.compile(r"(?<![\w&])(?:#|(?:PR|pr|issue|Issue|pull request)\s*#?)(\d{1,7})\b")
_SHA = re.compile(r"(?<![\w/.-])([0-9a-f]{7,40})(?![\w-])")
_TAG = re.compile(
    r"(?<![\w/.-])(v?\d+\.\d+(?:\.\d+)*(?:[-.]?(?:a|b|rc|alpha|beta|dev)\d*)?)(?![\w-])"
)
_HANDLE = re.compile(r"(?<![\w@.])@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))(?![\w@-])")
_PATH = re.compile(
    r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,8}|[\w-]+\.(?:py|js|jsx|ts|tsx|go|java|md|rst|toml|ya?ml|json|cfg|ini|txt))(?![\w/-])"
)


@dataclass(frozen=True)
class ExactRef:
    kind: RefKind
    value: str
    start: int
    end: int


def _is_sha(token: str) -> bool:
    # Plain words such as "defaced" or numbers such as "2024001" are not SHAs: require both.
    return any(c.isdigit() for c in token) and any(c in "abcdef" for c in token)


def find_exact_refs(text: str) -> list[ExactRef]:
    """All exact references in order of appearance; a span is reported under one kind only."""
    found: list[ExactRef] = []
    taken: list[tuple[int, int]] = []

    def add(kind: RefKind, match: re.Match[str], value: str) -> None:
        start, end = match.span(1)
        if any(start < b and a < end for a, b in taken):
            return
        taken.append((start, end))
        found.append(ExactRef(kind, value, start, end))

    for match in _NUMBER.finditer(text):
        add("issue_or_pr", match, match.group(1))
    for match in _PATH.finditer(text):
        add("path", match, match.group(1))
    for match in _TAG.finditer(text):
        add("tag", match, match.group(1))
    for match in _SHA.finditer(text):
        if _is_sha(match.group(1)):
            add("sha", match, match.group(1))
    for match in _HANDLE.finditer(text):
        add("handle", match, match.group(1))
    return sorted(found, key=lambda ref: ref.start)
