"""Symbol table for one file: its definitions and the calls made inside them.

Definitions reuse the AST chunker's rules, so a symbol here has the same qualified name as in the
chunk metadata. Calls are found with tree-sitter and recorded by name only: the callee's last
identifier and whatever qualifier text precedes it (`self.client`, `requests`, ``). Resolving a
call to one definition is done at query time (`MetadataDB.find_callers`) and is name-based, so
it can over-report; every result says how it matched.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from tree_sitter import Node

from repo_rag.chunking.ast_chunker import AstChunker, SymbolRecord, _error_ratio, _parser
from repo_rag.chunking.languages import AST_LANGUAGES
from repo_rag.config import Settings

log = logging.getLogger(__name__)

MODULE_SCOPE = "<module>"
_CALLEE_MAX_CHARS = 200
_NAME_AT_END = re.compile(
    r"^(?P<qualifier>.*?)(?:\?\.|\.|::|->)?(?P<name>[A-Za-z_$][\w$]*)\s*$", re.S
)

# Call node type -> field naming the callee expression (python, JS/TS and Go split it at the
# last dot). Java's method_invocation keeps the method in `name` and the receiver in `object`.
_CALL_FIELDS: dict[str, dict[str, str]] = {
    "python": {"call": "function"},
    "javascript": {"call_expression": "function", "new_expression": "constructor"},
    "typescript": {"call_expression": "function", "new_expression": "constructor"},
    "tsx": {"call_expression": "function", "new_expression": "constructor"},
    "go": {"call_expression": "function"},
    "java": {"method_invocation": "name", "object_creation_expression": "type"},
}


@dataclass(frozen=True)
class CallSite:
    caller: str  # qualified name of the innermost enclosing definition, or MODULE_SCOPE
    callee: str  # last identifier of the called expression
    qualifier: str  # text before it: "self", "self.client", "requests", "" for a bare call
    line: int


@dataclass(frozen=True)
class ImportBinding:
    """A name an import creates in a file: `from .utils import f as g` is (g, .utils, f)."""

    bound: str  # the local name
    module: str  # the module text as written (".utils", "numpy", "./api")
    original: str  # the imported name, or "" when the import binds the module itself


@dataclass(frozen=True)
class FileSymbols:
    definitions: tuple[SymbolRecord, ...]
    calls: tuple[CallSite, ...]
    imports: tuple[ImportBinding, ...] = ()


def _text(node: Node) -> str:
    return (node.text or b"").decode("utf-8", "replace")


def _split_callee(text: str) -> tuple[str, str] | None:
    text = " ".join(text.split())
    if not text or len(text) > _CALLEE_MAX_CHARS:
        return None
    match = _NAME_AT_END.match(text)
    if not match:
        return None  # `f()()`, subscripts, lambdas: no stable name
    return match["qualifier"].strip(), match["name"]


def _python_imports(node: Node) -> list[ImportBinding]:
    found: list[ImportBinding] = []
    module = node.child_by_field_name("module_name")
    source = _text(module).strip() if module is not None else ""
    for item in node.children_by_field_name("name"):
        if item.type == "aliased_import":
            name, alias = item.child_by_field_name("name"), item.child_by_field_name("alias")
            if name is not None and alias is not None:
                original = _text(name).strip()
                bound = _text(alias).strip()
                found.append(ImportBinding(bound, source or original, original if source else ""))
        elif item.type == "dotted_name":
            imported = _text(item).strip()
            if source:  # from pkg import name
                found.append(ImportBinding(imported, source, imported))
            elif imported:  # import pkg.sub binds `pkg`
                found.append(ImportBinding(imported.split(".")[0], imported, ""))
    return found


def _js_imports(node: Node) -> list[ImportBinding]:
    found: list[ImportBinding] = []
    source_node = node.child_by_field_name("source")
    source = _text(source_node).strip("'\"` ") if source_node is not None else ""
    stack = list(node.children)
    while stack:
        item = stack.pop()
        if item.type == "import_specifier":
            name, alias = item.child_by_field_name("name"), item.child_by_field_name("alias")
            if name is not None:
                original = _text(name).strip()
                bound = _text(alias).strip() if alias is not None else original
                found.append(ImportBinding(bound, source, original))
        elif item.type == "namespace_import":
            identifier = next((c for c in item.children if c.type == "identifier"), None)
            if identifier is not None:
                found.append(ImportBinding(_text(identifier).strip(), source, ""))
        elif item.type == "identifier" and item.parent and item.parent.type == "import_clause":
            found.append(ImportBinding(_text(item).strip(), source, "default"))
        else:
            stack.extend(item.children)
    return found


_IMPORT_READERS = {
    "python": {"import_from_statement": _python_imports, "import_statement": _python_imports},
    "javascript": {"import_statement": _js_imports},
    "typescript": {"import_statement": _js_imports},
    "tsx": {"import_statement": _js_imports},
}


def _innermost(definitions: list[SymbolRecord], line: int) -> str:
    best: SymbolRecord | None = None
    for d in definitions:
        if d.start_line <= line <= d.end_line and (
            best is None or d.end_line - d.start_line <= best.end_line - best.start_line
        ):
            best = d
    return best.qualname if best else MODULE_SCOPE


def extract_symbols(source: str, language: str | None, settings: Settings) -> FileSymbols:
    """Definitions and call sites of one file. Never raises; unparsable input gives nothing."""
    fields = _CALL_FIELDS.get(language or "")
    if fields is None or language not in AST_LANGUAGES or not source.strip():
        return FileSymbols((), ())
    definitions = AstChunker(settings).symbol_records(source, language)
    try:
        data = source.encode("utf-8")
        root = _parser(str(language)).parse(data).root_node
        if _error_ratio(root, len(data)) > settings.ast_max_error_ratio:
            return FileSymbols((), ())
        calls: set[CallSite] = set()
        imports: set[ImportBinding] = set()
        readers = _IMPORT_READERS.get(str(language), {})
        stack = [root]
        while stack:
            node = stack.pop()
            reader = readers.get(node.type)
            if reader is not None:
                imports.update(reader(node))
            field = fields.get(node.type)
            if field is not None:
                callee = node.child_by_field_name(field)
                parts = _split_callee(_text(callee)) if callee is not None else None
                if parts and node.type == "method_invocation":  # Java: receiver is its own field
                    receiver = node.child_by_field_name("object")
                    parts = (_text(receiver).strip() if receiver is not None else "", parts[1])
                if parts and parts[1]:
                    line = node.start_point[0] + 1
                    calls.add(CallSite(_innermost(definitions, line), parts[1], parts[0], line))
            stack.extend(node.children)
        ordered = tuple(sorted(calls, key=lambda c: (c.line, c.callee, c.qualifier, c.caller)))
        bound = tuple(sorted(imports, key=lambda i: (i.bound, i.module, i.original)))
        return FileSymbols(tuple(definitions), ordered, bound)
    except Exception:
        log.exception("could not extract calls for %s", language)
        return FileSymbols(tuple(definitions), ())
