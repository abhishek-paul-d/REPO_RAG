"""Fixed schemas and typed bindings to existing tools; never accept model SQL or repo ids."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repo_rag.agent.bakeoff import redact, validate_arguments
from repo_rag.config import Settings
from repo_rag.history.git_tools import blame, releases_containing
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.retrieval.retrievers import HybridRetriever
from repo_rag.store.symbol_table import SymbolTable


def _schema(name: str, description: str, fields: dict[str, str], required: list[str]) -> dict:
    properties: dict[str, dict] = {}
    for field, kind in fields.items():
        properties[field] = {"type": kind}
        if kind == "integer":
            properties[field]["minimum"] = 1
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def tool_schemas() -> dict[str, dict]:
    dates = {"since": "string", "until": "string"}
    definitions = [
        (
            "get_pr",
            "Get the full PR record, including merge_commit_sha.",
            {"number": "integer"},
            ["number"],
        ),
        (
            "get_issue_meta",
            "Get the full issue metadata record.",
            {"number": "integer"},
            ["number"],
        ),
        ("prs_closing_issue", "Get PRs closing an issue.", {"number": "integer"}, ["number"]),
        (
            "find_prs",
            "Filter PR records, newest first.",
            {
                "state": "string",
                "author": "string",
                "merged_by": "string",
                "label": "string",
                "title_contains": "string",
                "limit": "integer",
            },
            [],
        ),
        (
            "count_commits",
            "Count indexed non-merge commits with filters; ISO date bounds.",
            {
                "author": "string",
                "path": "string",
                "symbol": "string",
                "is_fix": "boolean",
                **dates,
            },
            [],
        ),
        (
            "top_contributors",
            "Authors ranked by commits, then additions/name. Default limit 3.",
            {"path": "string", "limit": "integer", **dates},
            [],
        ),
        (
            "change_frequency",
            "Files ranked by distinct commits. Default limit 3.",
            {"path_prefix": "string", "limit": "integer", **dates},
            [],
        ),
        (
            "releases_containing",
            "Release tags containing a commit SHA, earliest first.",
            {"sha": "string"},
            ["sha"],
        ),
        (
            "blame",
            "Commit provenance for an inclusive line range at the pinned HEAD.",
            {"path": "string", "start": "integer", "end": "integer"},
            ["path", "start", "end"],
        ),
        (
            "get_tag",
            "Get tag metadata including the full commit SHA.",
            {"name": "string"},
            ["name"],
        ),
        (
            "search_code",
            "Hybrid code search, fixed top 5 whole chunks with source ranges.",
            {"query": "string"},
            ["query"],
        ),
        (
            "get_symbol",
            "Find definitions, qualified or bare name. Default limit 20.",
            {"symbol": "string", "path": "string", "limit": "integer"},
            ["symbol"],
        ),
        (
            "find_callers",
            "Find plausible callers; name-based matches are not certain. Default limit 50.",
            {"symbol": "string", "path": "string", "limit": "integer"},
            ["symbol"],
        ),
    ]
    return {
        name: _schema(name, description, fields, required)
        for name, description, fields, required in definitions
    }


BODY_MAX_CHARS = 2000
BODY_TOOLS = ("get_pr", "get_issue_meta", "find_prs", "prs_closing_issue")


def bound_bodies(value: Any) -> Any:
    """Cut each long `body` string (protocol v3 `tools.record_projection`); nothing else."""
    if isinstance(value, list):
        return [bound_bodies(item) for item in value]
    if not isinstance(value, dict):
        return value
    bounded = {key: bound_bodies(item) for key, item in value.items()}
    body = bounded.get("body")
    if isinstance(body, str) and len(body) > BODY_MAX_CHARS:
        cut = len(body) - BODY_MAX_CHARS
        bounded["body"] = f"{body[:BODY_MAX_CHARS]}...[body truncated: {cut} more characters]"
    return bounded


class BoundTools:
    def __init__(
        self,
        repo: str,
        checkout: Path,
        pin: str,
        settings: Settings,
        retriever: HybridRetriever,
        db: MetadataDB,
        symbols: SymbolTable,
    ) -> None:
        self.repo, self.checkout, self.pin = repo, checkout, pin
        self.settings, self.retriever, self.db, self.symbols = settings, retriever, db, symbols

    def execute(self, name: str, args: dict) -> Any:
        schemas = tool_schemas()
        if name not in schemas:
            raise ValueError("Unknown tool")
        validate_arguments(args, schemas[name]["parameters"])
        if name == "search_code":
            self.retriever.forget_cached_indexes()
            hits = self.retriever.search(self.repo, args["query"], kind="code", k=5, rerank=False)
            return [
                {"text": h.document.page_content, "metadata": h.document.metadata} for h in hits
            ]
        if name == "blame":
            return redact(blame(self.checkout, **args, ref=self.pin))
        if name == "releases_containing":
            return redact(releases_containing(self.checkout, **args, settings=self.settings))
        if name in ("get_symbol", "find_callers"):
            return redact(getattr(self.symbols, name)(self.repo, **args))
        if name in ("top_contributors", "change_frequency"):
            args = {"limit": 3, **args}
        result = redact(getattr(self.db, name)(self.repo, **args))
        return bound_bodies(result) if name in BODY_TOOLS else result
