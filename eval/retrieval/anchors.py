"""Source definitions for authoring gold and uniform headers, independent of our chunker.

Python uses the stdlib AST. JavaScript uses top-level tree-sitter statements, including
Express's prototype assignments and defineGetter calls. No retrieval result is used here.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache


@lru_cache
def javascript_parser():
    from tree_sitter_language_pack import get_parser

    return get_parser("javascript")


@dataclass(frozen=True)
class Anchor:
    symbol: str
    start: int
    end: int
    signature: str
    parent: str = ""
    kind: str = "function"
    parent_kind: str = ""


def definitions(source: str, language: str) -> list[Anchor]:
    lines = source.splitlines()
    result: list[Anchor] = []
    if language == "python":

        def visit(node: ast.AST, parents: tuple[str, ...] = (), parent_kind: str = "") -> None:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    (isinstance(d, ast.Name) and d.id == "overload")
                    or (isinstance(d, ast.Attribute) and d.attr == "overload")
                    for d in node.decorator_list
                ):
                    return  # type-only declarations are not the implementation gold
                name = ".".join((*parents, node.name))
                header_end = max(node.lineno, node.body[0].lineno - 1)
                signature = " ".join(line.strip() for line in lines[node.lineno - 1 : header_end])
                result.append(
                    Anchor(
                        name,
                        node.lineno,
                        node.end_lineno or node.lineno,
                        signature,
                        ".".join(parents),
                        "class" if isinstance(node, ast.ClassDef) else "function",
                        parent_kind,
                    )
                )
                parents = (*parents, node.name)
                parent_kind = "class" if isinstance(node, ast.ClassDef) else "function"
            for child in ast.iter_child_nodes(node):
                visit(child, parents, parent_kind)

        visit(ast.parse(source))
    elif language == "javascript":
        encoded = source.encode()
        tree = javascript_parser().parse(encoded)
        for node in tree.root_node.named_children:
            text = encoded[node.start_byte : node.end_byte].decode()
            # Restrict assignment detection to the declaration prefix, before the body.
            prefix = text.split("{", 1)[0]
            names = re.findall(r"([\w.]+)\s*=\s*(?=(?:[\w.]+\s*=\s*)?function\b)", prefix)
            direct = re.match(r"function\s+(\w+)\s*\(", text)
            getter = re.match(r"defineGetter\(req,\s*['\"](\w+)['\"]", text)
            if direct:
                names.append(direct[1])
            if getter:
                names.append(f"req.{getter[1]}")
            for name in dict.fromkeys(names):
                end_row, end_column = node.end_point
                start_row, _ = node.start_point
                end = end_row + (end_column > 0)
                result.append(Anchor(name, start_row + 1, end, " ".join(prefix.split()) + " {"))
    return result
