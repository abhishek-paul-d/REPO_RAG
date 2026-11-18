"""Adapters for the same tracked code corpus, with exact source locations.

The header experiment applies the SAME independently derived context to every strategy.
The reference library retains its own chunk text (including its whitespace reconstruction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from eval.retrieval.anchors import definitions
from repo_rag.chunking.ast_chunker import AstChunker
from repo_rag.chunking.common import make_document, split_header
from repo_rag.chunking.languages import langchain_language
from repo_rag.config import Settings
from repo_rag.ingest.file_walker import walk_repo

STRATEGIES = ("char", "lc_lang", "astchunk", "ast")
REFERENCE_LANGUAGES = {"python", "java", "csharp", "typescript", "tsx"}


class SplitterOptions(TypedDict):
    chunk_size: int
    chunk_overlap: int
    add_start_index: bool
    strip_whitespace: bool


class UnsupportedStrategy(ValueError):
    """A reference cannot cover this corpus; report N/A, never silently substitute."""


def pieces(
    source: str, path: str, language: str, strategy: str, repo: str, sha: str, settings: Settings
) -> list[tuple[str, int, int]]:
    if strategy == "ast":
        chunks = AstChunker(settings).chunk(
            source, repo=repo, commit_sha=sha, path=path, language=language
        )
        return [
            (
                split_header(d.page_content)[1],
                int(d.metadata["start_line"]),
                int(d.metadata["end_line"]),
            )
            for d in chunks
        ]
    if strategy == "char":
        size, overlap = settings.max_chunk_chars, settings.eval_char_overlap
        if overlap >= size:
            raise ValueError("evaluation overlap must be smaller than chunk size")
        result = []
        for start in range(0, len(source), size - overlap):
            end = min(start + size, len(source))
            text = source[start:end]
            if text.strip():
                result.append(
                    (text, source.count("\n", 0, start) + 1, source.count("\n", 0, end - 1) + 1)
                )
            if end == len(source):
                break
        return result
    if strategy == "lc_lang":
        lc = langchain_language(language)
        kwargs: SplitterOptions = {
            "chunk_size": settings.max_chunk_chars,
            "chunk_overlap": settings.eval_char_overlap,
            "add_start_index": True,
            "strip_whitespace": False,
        }
        splitter = (
            RecursiveCharacterTextSplitter.from_language(Language(lc), **kwargs)
            if lc
            else RecursiveCharacterTextSplitter(**kwargs)
        )
        result = []
        for doc in splitter.create_documents([source]):
            start = doc.metadata["start_index"]
            if start < 0 or source[start : start + len(doc.page_content)] != doc.page_content:
                raise ValueError(f"splitter lost source location: {path}")
            if doc.page_content.strip():
                result.append(
                    (
                        doc.page_content,
                        source.count("\n", 0, start) + 1,
                        source.count("\n", 0, start + len(doc.page_content) - 1) + 1,
                    )
                )
        return result
    if strategy == "astchunk":
        if language not in REFERENCE_LANGUAGES:
            raise UnsupportedStrategy(f"astchunk 0.1.0 does not support {language} ({path})")
        try:
            from astchunk import ASTChunkBuilder
            from astchunk.astnode import ASTNode
        except ImportError as exc:
            raise RuntimeError("reference dependency missing: run uv sync --group dev") from exc
        builder = ASTChunkBuilder(
            max_chunk_size=settings.max_chunk_chars,
            language="typescript" if language == "tsx" else language,
            metadata_template="default",
        )

        class CoordinateNode(ASTNode):
            # tree-sitter 0.26 on this CPython 3.13 Windows build crashes after
            # repeated temporary Point.row/column access. Tuple access matches
            # our production parser and leaves the reference algorithm intact.
            @property
            def start_line(self):
                return self.node.start_point[0]

            @property
            def end_line(self):
                return self.node.end_point[0]

            @property
            def start_col(self):
                return self.node.start_point[1]

            @property
            def end_col(self):
                return self.node.end_point[1]

        # Keep library chunk boundaries and content. Its coordinates are zero-based,
        # with an exclusive end point. Access the nodes to handle end-column zero.
        encoded = source.encode()
        tree = builder.parser.parse(encoded)
        with patch("astchunk.astchunk_builder.ASTNode", CoordinateNode):
            windows = list(builder.assign_tree_to_windows(source, tree.root_node))
            chunks = builder.convert_windows_to_chunks(windows, {}, False)
        return [
            (c.chunk_text, c.start_line + 1, c.end_line + (c.ast_window[-1].end_col > 0))
            for c in chunks
            if c.chunk_text.strip()
        ]
    raise ValueError(f"unknown strategy: {strategy}")


def chunk_corpus(
    root: Path, repo: str, sha: str, strategy: str, headers: bool, settings: Settings
) -> list[Document]:
    docs = []
    for file in walk_repo(root, max_file_bytes=settings.max_file_bytes):
        if file.kind != "code":
            continue
        source = file.abs_path.read_text(encoding="utf-8", errors="replace")
        if not source.strip():
            continue
        language = file.language or ""
        try:
            anchors = definitions(source, language)
        except SyntaxError:
            anchors = []
        for i, (text, start, end) in enumerate(
            pieces(source, file.path, language, strategy, repo, sha, settings)
        ):
            containers = [a for a in anchors if a.start <= start and a.end >= end]
            context = min(containers, key=lambda a: a.end - a.start) if containers else None
            doc = make_document(
                text,
                repo=repo,
                commit_sha=sha,
                path=file.path,
                language=language,
                start_line=start,
                end_line=end,
                symbol="",
                symbol_kind="block",
                chunker=strategy,
                id_key=f"<eval:{strategy}:{i}>",
                parent_symbol=context.parent if context else "",
                parent_kind=context.parent_kind if context else "",
                signature=context.signature if context else "",
            )
            if not headers:
                doc.page_content = text
            doc.metadata["eval_body_chars"] = len(text)
            if strategy == "astchunk":
                original = source.splitlines()[start - 1 : end]
                reconstructed = text.splitlines()
                if len(original) != len(reconstructed):
                    raise ValueError(
                        f"reference line mapping lost alignment: {file.path}:{start}-{end}"
                    )
                # Upstream can drop oversized leaf text but pad its envelope with
                # newlines. Exclude those missing nonblank source lines from evidence.
                represented = [
                    line
                    for line, (raw, actual) in enumerate(
                        zip(original, reconstructed, strict=True), start
                    )
                    if actual.strip() or not raw.strip()
                ]
                spans: list[list[int]] = []
                for line in represented:
                    if spans and line == spans[-1][1] + 1:
                        spans[-1][1] = line
                    else:
                        spans.append([line, line])
                doc.metadata["eval_source_spans"] = json.dumps(spans)
            # Character chunks may cover partial lines; recorded spans are inclusive.
            docs.append(doc)
    if not docs:
        raise ValueError("no code chunks in corpus")
    if len({d.metadata["chunk_id"] for d in docs}) != len(docs):
        raise ValueError("duplicate evaluation chunk ids")
    return docs
