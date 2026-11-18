"""Chunking for docs and config files (plan.md §3.9).

Markdown and reStructuredText are split at headings, keeping the source text verbatim so line
numbers and citations stay exact. (LangChain's MarkdownHeaderTextSplitter rewrites whitespace,
so its output can't be located in the file.) The heading path goes into metadata and the
contextual header. Oversized sections, and config files that don't fit, are split further by
the text splitter.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from repo_rag.chunking.common import locate_line_spans, make_document, splitter_sizes
from repo_rag.config import Settings, get_settings

_MARKDOWN_SUFFIXES = {".md", ".mdx"}
_MAX_HEADING_LEVEL = 3
_MIN_PIECE_CHARS = 80  # smaller split pieces (e.g. a lone RST title line) join the next piece
_MD_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_RST_ADORNMENT = re.compile(r"^([=\-~^\"'`#*+:.])\1{2,}\s*$")
_MANIFESTS = {
    "pyproject.toml",
    "setup.cfg",
    "pipfile",
    "package.json",
    "go.mod",
    "cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "gemfile",
    "composer.json",
}


def doc_kind(path: str) -> str:
    """readme | changelog | contributing | docs | manifest | ci | other."""
    pure = PurePosixPath(path)
    name = pure.name.lower()
    if name.startswith("readme"):
        return "readme"
    if name.startswith(("changelog", "changes", "history", "news")):
        return "changelog"
    if name.startswith("contributing"):
        return "contributing"
    if name in _MANIFESTS or name.startswith("requirements"):
        return "manifest"
    if path.startswith((".github/workflows/", ".circleci/")) or name in {
        ".gitlab-ci.yml",
        ".travis.yml",
        "azure-pipelines.yml",
    }:
        return "ci"
    if path.startswith("docs/") or pure.suffix.lower() in _MARKDOWN_SUFFIXES | {".rst", ".adoc"}:
        return "docs"
    return "other"


def _language(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix == ".rst":
        return "rst"
    return suffix.lstrip(".") or "text"


def markdown_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line index, level, title) of `#` headings outside fenced code blocks."""
    out, in_fence = [], False
    for i, line in enumerate(lines):
        if _MD_FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and (m := _MD_HEADING.match(line)):
            out.append((i, len(m.group(1)), m.group(2)))
    return out


def rst_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line index, level, title) of underlined (optionally overlined) RST titles.

    RST assigns levels by the order adornment styles first appear in the file.
    """
    out: list[tuple[int, int, str]] = []
    styles: list[tuple[str, bool]] = []
    for i in range(1, len(lines)):
        title, under = lines[i - 1].strip(), lines[i].strip()
        if not title or _RST_ADORNMENT.match(title) or not _RST_ADORNMENT.match(under):
            continue
        if len(under) < len(title):
            continue
        over = i >= 2 and lines[i - 2].strip() == under
        style = (under[0], over)
        if style not in styles:
            styles.append(style)
        out.append((i - 2 if over else i - 1, styles.index(style) + 1, title))
    return out


def _splitter(language: str, settings: Settings) -> RecursiveCharacterTextSplitter:
    size, overlap = splitter_sizes(settings.max_chunk_chars, settings.fallback_chunk_overlap)
    if language in {"markdown", "rst"}:
        return RecursiveCharacterTextSplitter.from_language(
            Language(language), chunk_size=size, chunk_overlap=overlap
        )
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def _sections(source: str, language: str) -> list[tuple[str, int, int]]:
    """(heading path, start line idx, end line idx exclusive) covering the whole file."""
    lines = source.split("\n")
    if language == "markdown":
        headings = markdown_headings(lines)
    elif language == "rst":
        headings = rst_headings(lines)
    else:
        headings = []
    bounds = [h[0] for h in headings if h[1] <= _MAX_HEADING_LEVEL]
    if not bounds or bounds[0] != 0:
        bounds.insert(0, 0)
    titles = {i: (level, title) for i, level, title in headings if level <= _MAX_HEADING_LEVEL}

    sections: list[tuple[str, int, int]] = []
    stack: list[tuple[int, str]] = []  # (level, title) of the current heading path
    for n, start in enumerate(bounds):
        if start in titles:
            level, title = titles[start]
            stack = [s for s in stack if s[0] < level] + [(level, title)]
        end = bounds[n + 1] if n + 1 < len(bounds) else len(lines)
        sections.append((" > ".join(t for _, t in stack), start, end))

    # Fold sections that are only a heading (or an RST anchor) into the next one.
    merged: list[tuple[str, int, int]] = []
    carry_start: int | None = None
    for i, (heading, start, end) in enumerate(sections):
        start = start if carry_start is None else carry_start
        if _is_bare(lines[start:end]) and i + 1 < len(sections):
            carry_start = start
            continue
        carry_start = None
        merged.append((heading, start, end))
    return merged


def _is_bare(block: list[str]) -> bool:
    """True if every non-blank line is a heading, an RST adornment/title or an RST label."""
    for i, line in enumerate(block):
        stripped = line.strip()
        if not stripped or _MD_HEADING.match(line) or _RST_ADORNMENT.match(stripped):
            continue
        if stripped.startswith(".. _"):
            continue
        if i + 1 < len(block) and _RST_ADORNMENT.match(block[i + 1].strip()):
            continue  # an RST title line
        return False
    return True


def chunk_doc_file(
    source: str,
    *,
    repo: str,
    commit_sha: str,
    path: str,
    settings: Settings | None = None,
) -> list[Document]:
    settings = settings or get_settings()
    if not source.strip():
        return []
    language = _language(path)
    kind = doc_kind(path)
    lines = source.split("\n")
    splitter = _splitter(language, settings)

    docs: list[Document] = []
    for heading, start, end in _sections(source, language):
        text = "\n".join(lines[start:end])
        if not text.strip():
            continue
        if len(text) <= settings.max_chunk_chars:
            first = start + next(i for i, ln in enumerate(lines[start:end]) if ln.strip())
            last = start + max(i for i, ln in enumerate(lines[start:end]) if ln.strip())
            pieces = [("\n".join(lines[first : last + 1]), first + 1, last + 1)]
        else:
            parts = splitter.split_text(text)
            spans = [(start + s, start + e) for s, e in locate_line_spans(text, parts)]
            # The RST splitter can cut between a title and its underline; fold tiny pieces
            # into the next one. Text is rebuilt from source lines so it stays verbatim.
            folded: list[tuple[int, int]] = []
            carry: int | None = None
            for n, (s, e) in enumerate(spans):
                s = s if carry is None else carry
                piece_text = "\n".join(lines[s - 1 : e]).strip()
                if len(piece_text) < _MIN_PIECE_CHARS and n + 1 < len(spans):
                    carry = s
                    continue
                carry = None
                folded.append((s, e))
            pieces = [("\n".join(lines[s - 1 : e]), s, e) for s, e in folded]
        for piece, s, e in pieces:
            docs.append(
                make_document(
                    piece,
                    repo=repo,
                    commit_sha=commit_sha,
                    path=path,
                    language=language,
                    start_line=s,
                    end_line=e,
                    symbol=heading or PurePosixPath(path).name,
                    symbol_kind="section",
                    chunker="docs",
                    section=heading,
                    id_key=f"<doc:{len(docs)}>",
                    extra={"doc_kind": kind},
                )
            )
    return docs
