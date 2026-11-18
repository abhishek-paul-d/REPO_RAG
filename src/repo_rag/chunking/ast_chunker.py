"""AST chunking with tree-sitter: split-then-merge (cAST, Zhang et al. 2025) plus symbol metadata.

Algorithm, per file:

1. Parse with tree-sitter. Too many ERROR nodes, an unknown language, or any exception means we
   fall back to the text splitter. This chunker never raises on bad input.
2. **Split:** walk the tree top-down. A node within the size budget is one unit. An oversized
   node is replaced by its children, recursively. Size = non-whitespace characters.
3. **Merge:** greedily join adjacent units while the budget allows, but only units with the
   same enclosing definition. So a class header stays with its first methods, and a method
   never merges with the next top-level function. Comments and decorators stick to the unit
   that follows them.
4. Each chunk gets its symbol from the definitions it contains, or the definition it is a piece
   of. Qualified names (`Session.send`, Go `S.Method`), kind, parent and signature come with it.

A definition that fits the budget is therefore never split.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from tree_sitter import Node, Parser

from repo_rag.chunking.common import make_document
from repo_rag.chunking.fallback import fallback_chunks
from repo_rag.chunking.languages import AST_LANGUAGES, LanguageConfig
from repo_rag.config import Settings, get_settings

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(rb"\s+")
_SIGNATURE_MAX_CHARS = 200


@lru_cache
def _parser(language: str) -> Parser:
    from tree_sitter_language_pack import get_parser

    return get_parser(language)


@dataclass(frozen=True)
class DefinitionSpan:
    """An exact AST definition span, independent of chunk merging (1-based, inclusive)."""

    symbol: str
    kind: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SymbolRecord:
    """One definition with the names the symbol table needs (1-based, inclusive lines)."""

    name: str
    qualname: str
    kind: str  # function | method | class
    parent: str  # qualified name of the enclosing class/function, "" at top level
    signature: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _Def:
    """A definition: what a chunk is about, or what it is part of."""

    inner_id: int  # node id of the definition proper (inside any wrapper)
    name: str
    qualname: str
    kind: str  # function | method | class
    signature: str
    parent_qual: str
    parent_kind: str
    start_row: int
    end_row: int
    value_id: int | None = None  # for `name = class {...}`: the bound value, not a second def


@dataclass
class _Unit:
    start_row: int
    end_row: int
    size: int
    ctx: tuple[_Def, ...]  # enclosing definitions, outermost first (not including itself)
    definition: _Def | None  # set when the unit is itself a whole definition
    node_type: str
    contained: tuple[str, ...] = ()  # qualnames of every definition inside, itself included

    @property
    def ctx_key(self) -> tuple[int, ...]:
        return tuple(d.inner_id for d in self.ctx)


def _size(node: Node) -> int:
    return len(_WHITESPACE.sub(b"", node.text or b""))


def _end_row(node: Node) -> int:
    row, col = node.end_point
    return row - 1 if col == 0 and row > node.start_point[0] else row


def _error_ratio(root: Node, total_bytes: int) -> float:
    if not root.has_error:
        return 0.0
    error_bytes = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            error_bytes += node.end_byte - node.start_byte
        elif node.has_error:
            stack.extend(node.children)
    return error_bytes / max(total_bytes, 1)


class AstChunker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.max_size = self.settings.max_chunk_chars

    # ---------------------------------------------------------------- public

    def definitions(self, source: str, language: str | None) -> list[DefinitionSpan]:
        """Use the chunker's definition rules for history hunk attribution.

        Unsupported or badly parsed input has no reliable symbols; never invent them from
        diff headers. This does not change chunking or require a Phase 6 call graph.
        """
        config = AST_LANGUAGES.get(language or "")
        if config is None or not source.strip():
            return []
        try:
            data = source.encode("utf-8")
            root = _parser(str(language)).parse(data).root_node
            if _error_ratio(root, len(data)) > self.settings.ast_max_error_ratio:
                return []
            spans: list[DefinitionSpan] = []

            def visit(node: Node, ctx: tuple[_Def, ...]) -> None:
                definition = self._definition(node, ctx, config, data)
                if definition:
                    spans.append(
                        DefinitionSpan(
                            definition.qualname,
                            definition.kind,
                            definition.start_row + 1,
                            definition.end_row + 1,
                        )
                    )
                child_ctx = (*ctx, definition) if definition else ctx
                for child in node.named_children:
                    visit(child, child_ctx)

            visit(root, ())
            return spans
        except Exception:
            log.exception("could not extract history symbols for %s", language)
            return []

    def symbol_records(self, source: str, language: str | None) -> list[SymbolRecord]:
        """Every definition in a file with the chunker's own naming rules; [] if unparsable."""
        config = AST_LANGUAGES.get(language or "")
        if config is None or not source.strip():
            return []
        try:
            data = source.encode("utf-8")
            root = _parser(str(language)).parse(data).root_node
            if _error_ratio(root, len(data)) > self.settings.ast_max_error_ratio:
                return []
            records: list[SymbolRecord] = []

            def visit(node: Node, ctx: tuple[_Def, ...]) -> None:
                definition = self._definition(node, ctx, config, data)
                if definition:
                    records.append(
                        SymbolRecord(
                            definition.name,
                            definition.qualname,
                            definition.kind,
                            definition.parent_qual,
                            definition.signature,
                            definition.start_row + 1,
                            definition.end_row + 1,
                        )
                    )
                child_ctx = (*ctx, definition) if definition else ctx
                for child in node.named_children:
                    visit(child, child_ctx)

            visit(root, ())
            return records
        except Exception:
            log.exception("could not extract symbols for %s", language)
            return []

    def chunk(
        self, source: str, *, repo: str, commit_sha: str, path: str, language: str | None
    ) -> list[Document]:
        """Chunk one file. Never raises; falls back to the text splitter instead."""
        if not source.strip():
            return []
        config = AST_LANGUAGES.get(language or "")
        if config is None:
            return self._fallback(source, repo, commit_sha, path, language)
        assert language is not None
        try:
            data = source.encode("utf-8")
            tree = _parser(language).parse(data)
            ratio = _error_ratio(tree.root_node, len(data))
            if ratio > self.settings.ast_max_error_ratio:
                log.info("parse errors in %s (%.0f%% of bytes); using fallback", path, ratio * 100)
                return self._fallback(source, repo, commit_sha, path, language)
            return self._chunk_tree(tree.root_node, data, config, repo, commit_sha, path, language)
        except Exception:
            log.exception("AST chunking failed for %s; using fallback", path)
            return self._fallback(source, repo, commit_sha, path, language)

    # ---------------------------------------------------------------- definitions

    def _match(self, node: Node, config: LanguageConfig) -> tuple[Node, str] | None:
        """(definition node, kind) if `node` is or wraps exactly one definition."""
        if not node.is_named:  # e.g. the `class` keyword token shares a type name with defs
            return None
        if node.type in config.definitions:
            return node, config.definitions[node.type]
        holder = config.function_holders.get(node.type)
        if holder is not None:
            value = node.child_by_field_name(holder[1])
            if value is None:
                return None
            if value.type in config.function_values:
                return node, "function"
            if value.is_named and config.definitions.get(value.type) == "class":
                return node, "class"  # const Foo = class {...} / { ctor: class {...} }
            return None
        if node.type in config.wrappers:
            inner = [m for c in node.named_children if (m := self._match(c, config))]
            return inner[0] if len(inner) == 1 else None
        return None

    def _signature(self, outer: Node, inner: Node, config: LanguageConfig, data: bytes) -> str:
        body_owner = inner
        holder = config.function_holders.get(inner.type)
        if holder is not None:  # const f = (...) => {...}: the body belongs to the value
            body_owner = inner.child_by_field_name(holder[1]) or inner
        body = next(
            (b for f in config.body_fields if (b := body_owner.child_by_field_name(f))), None
        )
        end = body.start_byte if body is not None else outer.end_byte
        header = data[outer.start_byte : end]
        if body is None:
            header = header.split(b"\n", 1)[0]
        text = _WHITESPACE.sub(b" ", header).decode("utf-8", "replace").strip()
        return text[:_SIGNATURE_MAX_CHARS]

    def _receiver_type(self, inner: Node, config: LanguageConfig) -> str:
        if not config.receiver_field:
            return ""
        receiver = inner.child_by_field_name(config.receiver_field)
        if receiver is None:
            return ""
        stack = [receiver]
        while stack:
            node = stack.pop(0)
            if node.type == "type_identifier":
                return (node.text or b"").decode("utf-8", "replace")
            stack.extend(node.named_children)
        return ""

    def _definition(
        self, node: Node, ctx: tuple[_Def, ...], config: LanguageConfig, data: bytes
    ) -> _Def | None:
        match = self._match(node, config)
        if match is None:
            return None
        inner, kind = match
        if ctx and inner.id in (ctx[-1].inner_id, ctx[-1].value_id):
            return None  # the def proper reached through its wrapper, or its bound class value
        holder = config.function_holders.get(inner.type)
        value = inner.child_by_field_name(holder[1]) if holder else None
        name_node = inner.child_by_field_name(holder[0] if holder else "name")
        name = (name_node.text or b"").decode("utf-8", "replace") if name_node else "<anonymous>"
        name = _WHITESPACE.sub(b" ", name.encode()).decode().strip()  # `a.\n  b = ...`
        parent = ctx[-1] if ctx else None
        parent_qual, parent_kind = (parent.qualname, parent.kind) if parent else ("", "")
        receiver = self._receiver_type(inner, config)
        if receiver:  # Go: func (s *S) Method()
            parent_qual, parent_kind = receiver, "class"
        if kind == "function" and parent_kind == "class":
            kind = "method"
        qualname = f"{parent_qual}.{name}" if parent_qual else name
        return _Def(
            inner_id=inner.id,
            name=name,
            qualname=qualname,
            kind=kind,
            signature=self._signature(node, inner, config, data),
            parent_qual=parent_qual,
            parent_kind=parent_kind,
            start_row=node.start_point[0],
            end_row=_end_row(node),
            value_id=value.id if value is not None else None,
        )

    def _contained(
        self, node: Node, ctx: tuple[_Def, ...], config: LanguageConfig, data: bytes
    ) -> list[str]:
        """Qualnames of all definitions in `node`'s subtree, so lookups by symbol can find
        e.g. a method inside a class that fits in one chunk."""
        definition = self._definition(node, ctx, config, data)
        names = [definition.qualname] if definition else []
        child_ctx = (*ctx, definition) if definition else ctx
        for child in node.named_children:
            names += self._contained(child, child_ctx, config, data)
        return names

    # ---------------------------------------------------------------- split + merge

    def _units(
        self, node: Node, ctx: tuple[_Def, ...], config: LanguageConfig, data: bytes
    ) -> Iterator[_Unit]:
        definition = self._definition(node, ctx, config, data)
        size = _size(node)
        is_root = node.parent is None  # always split the file into its top-level statements
        if node.child_count == 0 and size > self.max_size and _end_row(node) > node.start_point[0]:
            # An unsplittable multi-line leaf (huge comment, string template): split by line.
            yield from self._line_units(node, ctx)
            return
        if (size <= self.max_size or node.child_count == 0) and not is_root:
            if size or node.is_named:
                yield _Unit(
                    node.start_point[0],
                    _end_row(node),
                    size,
                    ctx,
                    definition,
                    node.type,
                    tuple(self._contained(node, ctx, config, data)),
                )
            return
        child_ctx = (*ctx, definition) if definition else ctx
        for child in node.children:
            yield from self._units(child, child_ctx, config, data)

    @staticmethod
    def _line_units(node: Node, ctx: tuple[_Def, ...]) -> Iterator[_Unit]:
        start = node.start_point[0]
        lines = (node.text or b"").split(b"\n")[: _end_row(node) - start + 1]
        for offset, line in enumerate(lines):
            size = len(_WHITESPACE.sub(b"", line))
            yield _Unit(start + offset, start + offset, size, ctx, None, node.type)

    def _attach_forward(self, units: list[_Unit]) -> list[_Unit]:
        """Glue comments/decorators to the unit after them (same context only).

        Only as much as fits the budget together with that unit: the lines nearest the
        definition stick, the rest of a huge doc comment stands on its own.
        """
        out: list[_Unit] = []
        pending: list[_Unit] = []
        for unit in units:
            leading = unit.node_type.endswith("comment") or unit.node_type == "decorator"
            if pending and pending[-1].ctx_key != unit.ctx_key:
                out.extend(pending)
                pending = []
            if leading and unit.definition is None:
                pending.append(unit)
                continue
            if pending:
                keep: list[_Unit] = []
                total = unit.size
                for p in reversed(pending):
                    if total + p.size > self.max_size:
                        break
                    keep.insert(0, p)
                    total += p.size
                out.extend(pending[: len(pending) - len(keep)])
                if keep:
                    unit = _Unit(
                        keep[0].start_row,
                        unit.end_row,
                        total,
                        unit.ctx,
                        unit.definition,
                        unit.node_type,
                        unit.contained,
                    )
                pending = []
            out.append(unit)
        return out + pending

    def _merge(self, units: list[_Unit]) -> list[list[_Unit]]:
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        size = 0
        for unit in units:
            if current and not (
                unit.ctx_key == current[-1].ctx_key and size + unit.size <= self.max_size
            ):
                if unit.start_row > current[-1].end_row:
                    groups.append(current)
                    current, size = [], 0
                else:
                    # Chunks are whole lines, so a unit can't start a new chunk on the current
                    # chunk's last line. If the units starting on that line can move along
                    # (`it('x', function () {...})`), start the new chunk at the line;
                    # otherwise (the line is shared with an earlier multi-line unit) join.
                    k = len(current)
                    while k > 0 and current[k - 1].start_row >= unit.start_row:
                        k -= 1
                    if k > 0 and current[k - 1].end_row < unit.start_row:
                        groups.append(current[:k])
                        current = current[k:]
                        size = sum(u.size for u in current)
            current.append(unit)
            size += unit.size
        if current:
            groups.append(current)
        return groups

    def _chunk_tree(
        self,
        root: Node,
        data: bytes,
        config: LanguageConfig,
        repo: str,
        commit_sha: str,
        path: str,
        language: str,
    ) -> list[Document]:
        lines = data.decode("utf-8", "replace").split("\n")
        units = self._attach_forward(list(self._units(root, (), config, data)))
        docs = []
        for group in self._merge(units):
            start, end = group[0].start_row, max(u.end_row for u in group)
            code = "\n".join(lines[start : end + 1])
            if not code.strip():
                continue
            defs = [u.definition for u in group if u.definition]
            contained = [name for u in group for name in u.contained]
            ctx = group[0].ctx
            if ctx and start <= ctx[-1].start_row:
                # This chunk opens a split definition (e.g. `class X:` + first methods), so a
                # lookup for X itself should land here.
                contained.insert(0, ctx[-1].qualname)
            if defs:
                primary = defs[0]
                symbol, kind = primary.qualname, primary.kind
                parent, parent_kind = primary.parent_qual, primary.parent_kind
                # Several small defs merged into one chunk: one def's signature would mislead.
                signature = primary.signature if len(defs) == 1 else ""
            elif ctx:  # a piece of a definition too big for one chunk
                owner = ctx[-1]
                # If the pieces all landed in this one chunk, it is the whole definition.
                whole = start <= owner.start_row and end >= owner.end_row
                symbol, kind = owner.qualname, owner.kind if whole else "block"
                if whole:
                    contained.insert(0, owner.qualname)
                parent, parent_kind = owner.parent_qual, owner.parent_kind
                signature = owner.signature
            else:
                symbol, kind, parent, parent_kind, signature = "<module>", "module", "", "", ""
            symbols = list(dict.fromkeys(contained))
            defines = (
                self._defines(symbols) if self.settings.header_defines and len(defs) > 1 else []
            )
            docs.append(
                make_document(
                    code,
                    repo=repo,
                    commit_sha=commit_sha,
                    path=path,
                    language=language,
                    start_line=start + 1,
                    end_line=end + 1,
                    symbol=symbol,
                    symbol_kind=kind,
                    chunker="ast",
                    parent_symbol=parent,
                    parent_kind=parent_kind,
                    signature=signature,
                    symbols=symbols,
                    defines=defines,
                )
            )
        return docs

    def _defines(self, symbols: list[str]) -> list[str]:
        """Names for the `# defines:` header line, cut so a big merge stays one short line."""
        budget = self.settings.header_max_defines_chars
        kept: list[str] = []
        used = 0
        for name in symbols:
            used += len(name) + 2
            if used > budget and kept:
                return [*kept, "..."]
            kept.append(name)
        return kept

    def _fallback(
        self, source: str, repo: str, commit_sha: str, path: str, language: str | None
    ) -> list[Document]:
        return fallback_chunks(
            source,
            repo=repo,
            commit_sha=commit_sha,
            path=path,
            language=language,
            settings=self.settings,
        )
