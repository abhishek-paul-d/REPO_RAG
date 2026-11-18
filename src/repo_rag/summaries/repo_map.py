"""Repo map: one searchable summary per file, directory and repository (plan §3.4).

Chunk-level retrieval answers "where is X" but not "how is this project organised". These
summaries are built from the checkout and the symbol extractor, with no model: a file summary
lists its module docstring, its classes (with their methods) and functions, and what it imports;
a directory summary lists its files with one line each; the repository summary adds the README
and the top-level layout. A `Summarizer` (any local model) may add a prose line on top; its output
is cached by content hash so re-indexing never repeats a call. Everything is capped, and the
text is plain so BM25 and the embedder see the same thing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from langchain_core.documents import Document

from repo_rag.chunking.languages import AST_LANGUAGES
from repo_rag.chunking.symbols import FileSymbols, extract_symbols
from repo_rag.config import Settings
from repo_rag.ingest.file_walker import RepoFile, walk_repo
from repo_rag.summaries.facts import FactContext

log = logging.getLogger(__name__)

KINDS = ("file_summary", "dir_summary", "repo_summary")
_MAX_METHODS = 12
_MAX_ITEMS = 25
_MAX_IMPORTS = 15
_DOC_LINE_CHARS = 160
_README_CHARS = 1500
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "*/")


class Summarizer(Protocol):
    """Anything that turns summary facts into one short prose paragraph (a local model)."""

    name: str

    def summarize(self, kind: str, path: str, facts: str) -> str: ...


@dataclass(frozen=True)
class SummaryResult:
    files: int
    directories: int
    documents: int
    summarized: int  # prose paragraphs produced by the model (not served from the cache)


_RULE = re.compile(r"^[\s~=\-^*#_.]+$")


def _first_line(text: str, limit: int = _DOC_LINE_CHARS) -> str:
    """First line that says something: not blank, an underline, or a bare `pkg.module` title."""
    lines = [line.strip().strip("\"'`# ").strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line and not _RULE.match(line)]
    if len(lines) > 1 and " " not in lines[0]:
        lines = lines[1:]
    return lines[0][:limit] if lines else ""


def _python_docs(source: str, *, every_definition: bool = False) -> tuple[str, dict[str, str]]:
    """(module docstring, {qualified definition name: first docstring line}); ({}) if unparsable."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return "", {}
    docs: dict[str, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                doc = ast.get_docstring(child)
                if doc:
                    docs[name] = _first_line(doc)
                visit(child, f"{name}.")
            elif every_definition:
                visit(child, prefix)

    visit(tree, "")
    return _first_line(ast.get_docstring(tree) or "", 300), docs


def _leading_comment(source: str) -> str:
    """The first comment block of a non-Python file, as one line of text."""
    lines: list[str] = []
    for raw in source.splitlines()[:40]:
        line = raw.strip()
        if not line and not lines:
            continue
        if not line.startswith(_COMMENT_PREFIXES):
            break
        lines.append(line.lstrip("#/* ").rstrip("*/ ").strip())
    return _first_line(" ".join(part for part in lines if part), 300)


def _local_names(files: Sequence[RepoFile]) -> set[str]:
    names: set[str] = set()
    for f in files:
        parts = PurePosixPath(f.path).parts
        names.update(parts[:-1])
        names.add(PurePosixPath(f.path).stem)
    return names


def _split_imports(symbols: FileSymbols, local: set[str]) -> tuple[list[str], list[str]]:
    internal: set[str] = set()
    external: set[str] = set()
    for i in symbols.imports:
        module = i.module.strip()
        head = module.lstrip("./").replace("\\", "/").split("/")[0].split(".")[0]
        relative = module.startswith(".")
        (internal if relative or head in local else external).add(module or i.bound)
    return sorted(internal)[:_MAX_IMPORTS], sorted(external)[:_MAX_IMPORTS]


def _definitions_text(symbols: FileSymbols, docs: dict[str, str]) -> tuple[str, str, list[str]]:
    classes: list[str] = []
    functions: list[str] = []
    top_level: list[str] = []
    methods: dict[str, list[str]] = defaultdict(list)
    for d in symbols.definitions:
        if d.kind == "method":
            methods[d.parent].append(d.name)
    for d in symbols.definitions:
        if d.kind == "class" and "." not in d.qualname:
            doc = f": {docs[d.qualname]}" if d.qualname in docs else ""
            members = methods.get(d.qualname, [])
            shown = ", ".join(members[:_MAX_METHODS])
            more = f", +{len(members) - _MAX_METHODS} more" if len(members) > _MAX_METHODS else ""
            classes.append(f"{d.name} ({shown}{more}){doc}" if members else f"{d.name}{doc}")
            top_level.append(d.name)
        elif d.kind == "function" and "." not in d.qualname:
            doc = f": {docs[d.qualname]}" if d.qualname in docs else ""
            functions.append(f"{d.name}{doc}")
            top_level.append(d.name)
    return "; ".join(classes[:_MAX_ITEMS]), "; ".join(functions[:_MAX_ITEMS]), top_level


@dataclass(frozen=True)
class _FileInfo:
    path: str
    language: str
    lines: int
    module_doc: str
    text: str  # the file summary facts
    top_level: tuple[str, ...]

    @property
    def one_line(self) -> str:
        if self.module_doc:
            return self.module_doc
        return ", ".join(self.top_level[:6])


def _file_info(
    f: RepoFile, source: str, local: set[str], settings: Settings, facts: FactContext
) -> _FileInfo:
    language = f.language or "text"
    if language == "python":
        module_doc, docs = _python_docs(
            source, every_definition="docstrings" in settings.summary_fact_features
        )
    else:
        module_doc, docs = _leading_comment(source), {}
    symbols = (
        extract_symbols(source, f.language, settings)
        if f.language in AST_LANGUAGES
        else FileSymbols((), ())
    )
    classes, functions, top_level = _definitions_text(symbols, docs)
    internal, external = _split_imports(symbols, local)
    lines = source.count("\n") + 1
    parts = [f"File: {f.path} ({language}, {lines} lines)"]
    if module_doc:
        parts.append(f"Purpose: {module_doc}")
    if classes:
        parts.append(f"Classes: {classes}")
    if functions:
        parts.append(f"Functions: {functions}")
    if internal:
        parts.append(f"Imports from this project: {', '.join(internal)}")
    if external:
        parts.append(f"Imports from libraries: {', '.join(external)}")
    parts.extend(facts.extra(f.path, source, docs))
    return _FileInfo(f.path, language, lines, module_doc, "\n".join(parts), tuple(top_level))


def _meta(repo: str, kind: str, path: str, language: str, text: str, symbols: str) -> dict:
    chunk_id = hashlib.sha1(f"{repo}:{kind}:{path}".encode()).hexdigest()
    return {
        "repo": repo,
        "chunk_id": chunk_id,
        "path": path,
        "language": language,
        "start_line": 1,
        "end_line": max(1, len(text.splitlines())),
        "symbol": PurePosixPath(path).name if path else "",
        "symbol_kind": kind,
        "parent_symbol": str(PurePosixPath(path).parent) if "/" in path else "",
        "signature": "",
        "symbols": symbols,
        "chunker": "summary",
    }


class _Cache:
    """Model prose by content hash: `sha256(model, kind, path, facts)` -> paragraph."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def key(model: str, kind: str, path: str, facts: str) -> str:
        return hashlib.sha256(f"{model}\0{kind}\0{path}\0{facts}".encode()).hexdigest()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, sort_keys=True), encoding="utf-8")


_FENCE = re.compile(r"```.*?```", re.S)
_IMAGE = re.compile(r"\[?!\[[^\]]*\]\([^)]*\)(\]\([^)]*\))?")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")


def _readme(files: Sequence[RepoFile]) -> str:
    candidates = [f for f in files if PurePosixPath(f.path).name.lower().startswith("readme")]
    if not candidates:
        return ""
    top = min(candidates, key=lambda f: (f.path.count("/"), f.path))
    text = top.abs_path.read_text(encoding="utf-8", errors="replace")
    text = _FENCE.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML.sub(" ", text)
    return " ".join(text.split())[:_README_CHARS]


def build_summaries(
    slug: str,
    checkout: Path,
    settings: Settings,
    *,
    summarizer: Summarizer | None = None,
) -> tuple[list[Document], SummaryResult]:
    """Summary Documents for every code file, directory and the repository (no embedding)."""
    files = list(walk_repo(checkout, max_file_bytes=settings.max_file_bytes))
    code = [f for f in files if f.kind == "code"]
    local = _local_names(code)
    infos: list[_FileInfo] = []
    facts = FactContext(slug, checkout, files, settings)
    try:
        for f in code:
            source = f.abs_path.read_text(encoding="utf-8", errors="replace")
            if source.strip():
                infos.append(_file_info(f, source, local, settings, facts))
    finally:
        facts.close()
    by_dir: dict[str, list[_FileInfo]] = defaultdict(list)
    for info in infos:
        by_dir[str(PurePosixPath(info.path).parent).removeprefix(".")].append(info)
    children: dict[str, set[str]] = defaultdict(set)
    for directory in list(by_dir):
        node = directory
        while node:
            parent = str(PurePosixPath(node).parent).removeprefix(".")
            children[parent].add(node)
            node = parent

    cache = _Cache(settings.data_dir / "summaries_cache" / f"{slug.replace('/', '__')}.json")
    produced = 0

    def prose(kind: str, path: str, facts: str) -> str:
        nonlocal produced
        if summarizer is None:
            return ""
        key = cache.key(summarizer.name, kind, path, facts)
        if key not in cache.data:
            cache.data[key] = " ".join(summarizer.summarize(kind, path, facts).split())
            produced += 1
        raw = cache.data[key]
        validate = getattr(summarizer, "validate", None)
        return validate(kind, path, facts, raw) if validate else raw

    cap = (
        settings.summary_fact_max_chars
        if settings.summary_fact_features
        else settings.summary_max_chars
    )
    docs: list[Document] = []

    def add(kind: str, path: str, language: str, facts: str, symbols: str = "") -> None:
        text = facts
        summary = prose(kind, path, facts)
        if summary:
            text = f"Summary: {summary}\n{facts}"
        truncated = len(text) > cap
        text = text[:cap].rstrip()
        metadata = _meta(slug, kind, path, language, text, symbols)
        metadata.update(facts_chars=len(facts), truncated=truncated, prose_chars=len(summary))
        docs.append(Document(page_content=text, metadata=metadata))

    for info in infos:
        add("file_summary", info.path, info.language, info.text, ",".join(info.top_level))

    def dir_facts(directory: str) -> str:
        shown = directory or "(repository root)"
        own = sorted(by_dir.get(directory, []), key=lambda i: i.path)
        subs = sorted(children.get(directory, set()))
        parts = [f"Directory: {shown} ({len(own)} files, {len(subs)} subdirectories)"]
        for info in own[:_MAX_ITEMS]:
            line = info.one_line
            parts.append(
                f"- {PurePosixPath(info.path).name}: {line}"
                if line
                else f"- {PurePosixPath(info.path).name}"
            )
        if subs:
            parts.append(
                "Subdirectories: " + ", ".join(PurePosixPath(s).name for s in subs[:_MAX_ITEMS])
            )
        return "\n".join(parts)

    for directory in sorted(set(by_dir) | set(children)):
        if directory:
            add("dir_summary", directory, "text", dir_facts(directory))

    top = sorted(children.get("", set()) | {d for d in by_dir if d and "/" not in d})
    overview = [f"Repository: {slug}"]
    readme = _readme(files)
    if readme:
        overview.append(f"README: {readme}")
    overview.append(f"Code files: {len(infos)} in {len([d for d in by_dir if d])} directories")
    if top:
        overview.append("Top-level directories:")
        for directory in top:
            tree = [i for i in infos if i.path.startswith(directory + "/")]
            subdirs = sorted(PurePosixPath(c).name for c in children.get(directory, set()))
            inits = [i for i in tree if PurePosixPath(i.path).name == "__init__.py" and i.one_line]
            inits.sort(key=lambda i: i.path.count("/"))
            hint = next(iter(inits), None) or next((i for i in tree if i.one_line), None)
            line = f"- {directory}/ ({len(tree)} files)"
            if subdirs:
                line += f", subdirectories: {', '.join(subdirs[:_MAX_ITEMS])}"
            overview.append(f"{line}: {hint.one_line}" if hint else line)
    if by_dir.get(""):
        names = ", ".join(
            PurePosixPath(i.path).name
            for i in sorted(by_dir[""], key=lambda i: i.path)[:_MAX_ITEMS]
        )
        overview.append(f"Top-level files: {names}")
    add("repo_summary", "", "text", "\n".join(overview))

    if summarizer is not None:
        cache.save()
    directories = len([d for d in by_dir if d])
    return docs, SummaryResult(len(infos), directories, len(docs), produced)
