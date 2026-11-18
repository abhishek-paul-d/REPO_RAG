"""Evidence expansion: add the chunks around a hit so a multi-chunk answer fits the result list.

Phase 4's remaining weak spot was questions whose answer spans several chunks ("trace X to Y").
Two cheap, source-derived expansions, applied after ranking and never reordering the top hits:

* `adjacent`: the chunk just before and just after a seed in the same file.
* `callee`: chunks that define a name the seed calls, when that name is defined in only a few
  chunks of the repository (so `render` is skipped when ten classes define it). This is a
  name-matching heuristic, not a call graph; real call edges belong to the Phase 6 symbol table.

The result list keeps its length: the first `keep` hits stay put, expansions follow, and the
remaining original hits fill what is left.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from langchain_core.documents import Document

from repo_rag.chunking.common import split_header

Mode = Literal["none", "adjacent", "callee", "both"]

_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_NOT_CALLS = frozenset(
    [
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "function",
        "def",
        "class",
        "return",
        "await",
        "async",
        "super",
        "print",
        "len",
        "range",
        "isinstance",
        "getattr",
        "hasattr",
        "str",
        "int",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "object",
    ]
)


def _id(doc: Document) -> str:
    return str(doc.metadata["chunk_id"])


class ChunkIndex:
    """Lookup structures over the chunks of one repository."""

    def __init__(self, docs: Iterable[Document], max_definers: int = 3) -> None:
        self.max_definers = max_definers
        self.by_path: dict[str, list[Document]] = defaultdict(list)
        self.definers: dict[str, list[Document]] = defaultdict(list)
        for doc in docs:
            self.by_path[str(doc.metadata["path"])].append(doc)
            for qualname in str(doc.metadata.get("symbols", "")).split(","):
                if qualname:
                    self.definers[re.split(r"[.:]", qualname)[-1]].append(doc)
        for chunks in self.by_path.values():
            chunks.sort(key=lambda d: int(d.metadata["start_line"]))

    def neighbours(self, doc: Document) -> list[Document]:
        chunks = self.by_path[str(doc.metadata["path"])]
        position = next((i for i, c in enumerate(chunks) if _id(c) == _id(doc)), None)
        if position is None:
            return []
        return [chunks[i] for i in (position - 1, position + 1) if 0 <= i < len(chunks)]

    def callees(self, doc: Document) -> list[Document]:
        """Chunks defining a rarely defined name that `doc` calls, in order of first call."""
        found: list[Document] = []
        for name in dict.fromkeys(_CALL.findall(split_header(doc.page_content)[1])):
            if name in _NOT_CALLS:
                continue
            definers = self.definers.get(name, [])
            if 0 < len(definers) <= self.max_definers:
                found += [d for d in definers if _id(d) != _id(doc)]
        return found


def expand(
    ranked: Sequence[Document],
    index: ChunkIndex,
    *,
    mode: Mode,
    keep: int = 5,
    seeds: int = 3,
    per_seed: int = 2,
    size: int | None = None,
) -> list[Document]:
    """`ranked` best first. Returns at most `size` (default `len(ranked)`) distinct chunks."""
    size = len(ranked) if size is None else size
    if mode == "none" or not ranked:
        return list(ranked[:size])
    result = list(ranked[:keep])
    seen = {_id(d) for d in ranked}  # an expansion never duplicates a chunk already ranked
    for seed in ranked[:seeds]:
        added = 0
        candidates: list[Document] = []
        if mode in ("adjacent", "both"):
            candidates += index.neighbours(seed)
        if mode in ("callee", "both"):
            candidates += index.callees(seed)
        for doc in candidates:
            if added >= per_seed:
                break
            if _id(doc) in seen:
                continue
            seen.add(_id(doc))
            result.append(doc)
            added += 1
    result += list(ranked[keep:])
    return result[:size]
