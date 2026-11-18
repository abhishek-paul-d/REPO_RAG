"""Text-splitter chunking for languages without an AST config, or files tree-sitter can't parse."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from repo_rag.chunking.common import locate_line_spans, make_document, splitter_sizes
from repo_rag.chunking.languages import langchain_language
from repo_rag.config import Settings, get_settings


def _splitter(language: str | None, settings: Settings) -> RecursiveCharacterTextSplitter:
    # chunk_size counts all characters, unlike the AST budget (non-whitespace), so fallback
    # chunks come out somewhat smaller than AST chunks. That's fine for a fallback.
    lc = langchain_language(language)
    size, overlap = splitter_sizes(settings.max_chunk_chars, settings.fallback_chunk_overlap)
    if lc is None:
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return RecursiveCharacterTextSplitter.from_language(
        Language(lc), chunk_size=size, chunk_overlap=overlap
    )


def fallback_chunks(
    source: str,
    *,
    repo: str,
    commit_sha: str,
    path: str,
    language: str | None,
    settings: Settings | None = None,
) -> list[Document]:
    settings = settings or get_settings()
    if not source.strip():
        return []
    pieces = _splitter(language, settings).split_text(source)
    return [
        make_document(
            piece,
            repo=repo,
            commit_sha=commit_sha,
            path=path,
            language=language or "",
            start_line=start,
            end_line=end,
            symbol="",
            symbol_kind="block",
            chunker="fallback",
            id_key=f"<fallback:{i}>",  # pieces of one long line can share a start line
        )
        for i, (piece, (start, end)) in enumerate(
            zip(pieces, locate_line_spans(source, pieces), strict=True)
        )
    ]
