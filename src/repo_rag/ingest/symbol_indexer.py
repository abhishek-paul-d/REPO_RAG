"""Extract definitions and call sites from a checkout into the symbol table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repo_rag.chunking.languages import AST_LANGUAGES
from repo_rag.chunking.symbols import FileSymbols, extract_symbols
from repo_rag.config import Settings
from repo_rag.ingest.file_walker import walk_repo
from repo_rag.store.symbol_table import SymbolTable


@dataclass(frozen=True)
class SymbolIndexResult:
    files: int
    definitions: int
    calls: int


def index_symbols(slug: str, checkout: Path, settings: Settings) -> SymbolIndexResult:
    """Replace this repo's symbol rows with the ones in `checkout` (idempotent)."""
    files: dict[str, tuple[str, FileSymbols]] = {}
    for f in walk_repo(checkout, max_file_bytes=settings.max_file_bytes):
        if f.kind != "code" or f.language not in AST_LANGUAGES:
            continue
        source = f.abs_path.read_text(encoding="utf-8", errors="replace")
        symbols = extract_symbols(source, f.language, settings)
        if symbols.definitions or symbols.calls or symbols.imports:
            files[f.path] = (str(f.language), symbols)
    with SymbolTable(settings.metadata_db_path, readonly=False) as table:
        table.replace_repo(slug, files)
        definitions, calls = table.counts(slug)
    return SymbolIndexResult(len(files), definitions, calls)
