"""Deterministic citation checks for answers built from retrieved code chunks.

The answer model sometimes writes a wrong `path:start-end` (an invented range, `start > end`, a
test name where a path belongs, a range from another file). Every such reference can be checked
against the chunks that were actually retrieved, so the agent validates before it answers.

`check_citations` reports each reference as valid or not. `repair_citations` strips the range from
invalid references, and when no valid reference remains it attaches `path:start-end` for retrieved
chunks the answer itself names. It never adds support the answer does not mention, and it never
touches an abstention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PATH_REF = re.compile(r"(?P<path>[\w./-]+):(?P<start>\d+)-(?P<end>\d+)")
MIN_SYMBOL_CHARS = 4


@dataclass(frozen=True)
class Chunk:
    path: str
    start_line: int
    end_line: int
    symbol: str = ""

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class Reference:
    text: str
    valid: bool


@dataclass
class Repair:
    answer: str
    removed: list[str] = field(default_factory=list)
    attached: list[str] = field(default_factory=list)
    uncited: bool = False


def chunks_from_results(rows: list[dict[str, Any]]) -> list[Chunk]:
    """Chunks from `search_code` result rows (each has a `metadata` dict)."""
    return [
        Chunk(
            str(row["metadata"]["path"]),
            int(row["metadata"]["start_line"]),
            int(row["metadata"]["end_line"]),
            str(row["metadata"].get("symbol", "")),
        )
        for row in rows
    ]


def _valid(path: str, start: int, end: int, chunks: list[Chunk]) -> bool:
    return start <= end and any(
        c.path == path and start <= c.end_line and end >= c.start_line for c in chunks
    )


def check_citations(answer: str, chunks: list[Chunk]) -> list[Reference]:
    clean = answer.replace("`", "")
    return [
        Reference(m.group(), _valid(m["path"], int(m["start"]), int(m["end"]), chunks))
        for m in PATH_REF.finditer(clean)
    ]


def named_chunks(answer: str, chunks: list[Chunk]) -> list[Chunk]:
    """Retrieved chunks whose file path, or whose symbol's last component, the answer names."""
    clean = answer.replace("`", "")
    found: list[Chunk] = []
    for chunk in chunks:
        name = chunk.symbol.split(".")[-1]
        hit = chunk.path in clean or (
            len(name) >= MIN_SYMBOL_CHARS
            and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", clean) is not None
        )
        if hit and chunk.ref not in [c.ref for c in found]:
            found.append(chunk)
    return found


def repair_citations(answer: str, chunks: list[Chunk], abstained: bool = False) -> Repair:
    result = Repair(answer)
    if abstained:
        return result
    clean = answer.replace("`", "")
    removed: list[str] = []

    def strip(match: re.Match[str]) -> str:
        if _valid(match["path"], int(match["start"]), int(match["end"]), chunks):
            return match.group()
        removed.append(match.group())
        return match["path"]  # keep the name, drop the unsupported range

    text = PATH_REF.sub(strip, clean)
    result.removed = removed
    if not any(r.valid for r in check_citations(text, chunks)):
        for chunk in named_chunks(text, chunks):
            result.attached.append(chunk.ref)
        if result.attached:
            text = f"{text.rstrip()} ({', '.join(result.attached)})"
        else:
            result.uncited = True
    result.answer = text if (removed or result.attached) else answer
    return result
