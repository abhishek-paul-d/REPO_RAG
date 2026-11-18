"""Shared chunk construction: the Document shape and metadata every chunker emits."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from langchain_core.documents import Document

SymbolKind = str  # "function" | "method" | "class" | "module" | "block"


def splitter_sizes(max_chunk_chars: int, overlap: int) -> tuple[int, int]:
    """(chunk_size, chunk_overlap) for LangChain splitters, which reject overlap > size."""
    return max_chunk_chars, min(overlap, max_chunk_chars // 4)


def make_chunk_id(repo: str, path: str, symbol: str, start_line: int) -> str:
    return hashlib.sha1(f"{repo}:{path}:{symbol}:{start_line}".encode()).hexdigest()


def contextual_header(
    path: str,
    parent_symbol: str = "",
    parent_kind: str = "",
    signature: str = "",
    section: str = "",
    defines: Sequence[str] = (),
) -> str:
    """The lines prepended to every chunk before embedding (and shown to the LLM).

    `defines` lists the definitions of a chunk that merges several small ones; without it such a
    chunk is described only by its path and its first line's unrelated name.
    """
    lines = [f"# path: {path}"]
    if parent_symbol:
        label = "in class" if parent_kind == "class" else "in function"
        lines.append(f"# {label}: {parent_symbol}")
    if signature:
        lines.append(f"# signature: {signature}")
    if defines:
        lines.append(f"# defines: {', '.join(defines)}")
    if section:
        lines.append(f"# section: {section}")
    return "\n".join(lines) + "\n"


_HEADER_LINE = re.compile(r"^# (path|in class|in function|signature|defines|section): ")


def split_header(page_content: str) -> tuple[str, str]:
    """(contextual header, verbatim source lines) of a chunk's `page_content`."""
    lines = page_content.split("\n")
    n = 0
    while n < len(lines) and _HEADER_LINE.match(lines[n]):
        n += 1
    return "\n".join(lines[:n]), "\n".join(lines[n:])


def make_document(
    code: str,
    *,
    repo: str,
    commit_sha: str,
    path: str,
    language: str,
    start_line: int,
    end_line: int,
    symbol: str,
    symbol_kind: SymbolKind,
    chunker: str,
    parent_symbol: str = "",
    parent_kind: str = "",
    signature: str = "",
    symbols: Sequence[str] = (),
    id_key: str | None = None,
    section: str = "",
    defines: Sequence[str] = (),
    extra: Mapping[str, str | int] | None = None,
) -> Document:
    """A chunk as a LangChain Document. Metadata values are flat scalars (Chroma-compatible).

    `id_key` replaces `symbol` in the chunk id when the symbol alone isn't unique. `section`
    is a doc heading path; `extra` adds chunker-specific scalar metadata (e.g. `doc_kind`).
    """
    header = contextual_header(path, parent_symbol, parent_kind, signature, section, defines)
    return Document(
        page_content=header + code,
        metadata={
            **(extra or {}),
            "repo": repo,
            "commit_sha": commit_sha,
            "path": path,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "symbol": symbol,
            "symbol_kind": symbol_kind,
            "parent_symbol": parent_symbol,
            "signature": signature,
            "symbols": ",".join(symbols),
            "chunk_id": make_chunk_id(repo, path, id_key or symbol, start_line),
            "chunker": chunker,
        },
    )


def locate_line_spans(source: str, pieces: Sequence[str]) -> list[tuple[int, int]]:
    """1-based (start_line, end_line) of each piece of `source`, in order.

    For splitters that return text but not positions. Pieces are searched for in order, from
    where the previous piece started (overlapping pieces are allowed).
    """
    spans = []
    cursor = 0
    for piece in pieces:
        index = source.find(piece, cursor)
        if index < 0:
            index = source.find(piece)
        if index < 0:  # the splitter altered whitespace; anchor on the first non-blank line
            first = next((ln for ln in piece.splitlines() if ln.strip()), piece)
            index = max(source.find(first, cursor), 0)
        start = source.count("\n", 0, index) + 1
        spans.append((start, start + piece.count("\n")))
        cursor = index + 1
    return spans
