"""Language detection and per-language tree-sitter node-type config for the AST chunker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

# Extension → tree-sitter-language-pack language name.
# Languages with an entry in AST_LANGUAGES get AST chunking; the rest are indexed as code but
# chunked by the fallback splitter until they get a node-type config.
EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
}

DefKind = Literal["function", "class"]


@dataclass(frozen=True)
class LanguageConfig:
    """Which tree-sitter node types are definitions, and how to find their names.

    `definitions` maps a node type to "function" or "class". A function whose nearest enclosing
    definition is a class is reported as a method. "class" covers every container that groups
    members: classes, interfaces, enums, records, structs, TS namespaces.
    """

    definitions: dict[str, DefKind]
    # Nodes that wrap one definition (decorators, `export`), so the wrapper counts as the def.
    wrappers: frozenset[str] = frozenset()
    # Nodes that bind a name to a value: node type -> (name field, value field). When the value
    # is one of `function_values`, the node is a function definition with that name:
    #   const f = () => {}          variable_declarator   (name, value)
    #   res.send = function () {}   assignment_expression (left, right)  -> "res.send"
    #   { handler: () => {} }       pair                  (key, value)
    function_holders: dict[str, tuple[str, str]] = field(default_factory=dict)
    function_values: frozenset[str] = frozenset()
    # Field that holds a definition's body; the signature is everything before it.
    body_fields: tuple[str, ...] = ("body",)
    # Go-style receivers: (receiver field name) whose type names the method's parent.
    receiver_field: str | None = None
    langchain_language: str | None = None


_JS_DEFS: dict[str, DefKind] = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "method_definition": "function",
    "class_declaration": "class",
    "class": "class",  # class expressions; the chunker ignores the unnamed `class` keyword token
}


def _js_like(definitions: dict[str, DefKind], langchain: str) -> LanguageConfig:
    return LanguageConfig(
        definitions=definitions,
        # expression_statement wraps `res.send = function () {}`; the declarations wrap
        # `const f = () => {}`. Each counts only when it holds exactly one definition.
        wrappers=frozenset(
            {
                "export_statement",
                "expression_statement",
                "lexical_declaration",
                "variable_declaration",
            }
        ),
        function_holders={
            "variable_declarator": ("name", "value"),
            "assignment_expression": ("left", "right"),
            "pair": ("key", "value"),
        },
        function_values=frozenset({"arrow_function", "function_expression", "generator_function"}),
        langchain_language=langchain,
    )


_TS_DEFS: dict[str, DefKind] = {
    **_JS_DEFS,
    "abstract_class_declaration": "class",
    "interface_declaration": "class",
    "type_alias_declaration": "class",
    "enum_declaration": "class",
    "internal_module": "class",  # namespace NS { ... }
    "function_signature": "function",  # overload signatures
    "method_signature": "function",
    "abstract_method_signature": "function",
}

AST_LANGUAGES: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        definitions={"function_definition": "function", "class_definition": "class"},
        wrappers=frozenset({"decorated_definition"}),
        langchain_language="python",
    ),
    "javascript": _js_like(_JS_DEFS, "js"),
    "typescript": _js_like(_TS_DEFS, "ts"),
    "tsx": _js_like(_TS_DEFS, "ts"),
    "java": LanguageConfig(
        definitions={
            "class_declaration": "class",
            "interface_declaration": "class",
            "enum_declaration": "class",
            "record_declaration": "class",
            "annotation_type_declaration": "class",
            "method_declaration": "function",
            "constructor_declaration": "function",
        },
        langchain_language="java",
    ),
    "go": LanguageConfig(
        definitions={
            "function_declaration": "function",
            "method_declaration": "function",
            "type_spec": "class",
        },
        wrappers=frozenset({"type_declaration"}),
        receiver_field="receiver",
        langchain_language="go",
    ),
}

# LangChain `Language` enum values for fallback splitting of languages without AST config.
FALLBACK_LANGCHAIN_LANGUAGES: dict[str, str] = {
    "rust": "rust",
    "c": "cpp",
    "cpp": "cpp",
    "csharp": "csharp",
    "ruby": "ruby",
    "php": "php",
    "kotlin": "kotlin",
    "swift": "swift",
    "scala": "scala",
}


def language_for_path(path: str) -> str | None:
    """Return the language for a repo-relative POSIX path, or None if it isn't source code."""
    return EXTENSION_LANGUAGES.get(PurePosixPath(path).suffix.lower())


def langchain_language(language: str | None) -> str | None:
    """The LangChain splitter language for a tree-sitter language name, if any."""
    if language is None:
        return None
    if language in AST_LANGUAGES:
        return AST_LANGUAGES[language].langchain_language
    return FALLBACK_LANGCHAIN_LANGUAGES.get(language)
