"""Production tools: the evaluated tool set plus docs search, commit search and PR/issue links.

`agent.bakeoff_tools` stays exactly as it was when the answer model was measured. This module
adds four read-only tools on top and is used when `Settings.agent_toolset == "production"`:

    search_docs     hybrid search over README, docs, CI and dependency files (same chunk shape
                    as search_code, so the citation check covers it)
    search_commits  hybrid search over indexed commits, for "has this bug been fixed before"
    get_references  written #N mentions of a PR or issue, in both directions
    find_callees    what a function calls, in source order (follow a call chain)

The added tools have not been through the Phase 7 bake-off; see docs/PROJECT_LOG.md.
"""

from __future__ import annotations

from typing import Any

from repo_rag.agent.bakeoff import redact, validate_arguments
from repo_rag.agent.bakeoff_tools import BoundTools, _schema, tool_schemas
from repo_rag.history.git_tools import GitToolError, file_history, pr_commits
from repo_rag.retrieval.intent import detect_intent
from repo_rag.store.manifest import Manifest
from repo_rag.store.symbol_table import is_test_path

SEARCH_K = 5
EXTRA_POOL_K = 40  # candidates searched for the extra files; the top 5 chunks come from the model
COMMIT_TEXT_CHARS = 1200
CALLEE_LIMIT = 25
PR_ISSUE_K = 4  # per collection: top PRs and top issues are shown separately
PR_TEXT_CHARS = 700
PR_COMMITS_LIMIT = 5  # a PR's own commits listed next to its merge commit by get_pr
ADDED_TOOLS = ("search_docs", "search_commits", "search_prs", "get_references", "find_callees")
HISTORY_TOOLS = ("file_history",)  # offered only by the `history_grounded` prompt variant
HISTORY_LIMIT = 8  # newest changes returned by file_history unless the model asks for fewer


def production_schemas(*, search_prs: bool = True, file_history: bool = False) -> dict[str, dict]:
    """The evaluated tools plus the added ones. `search_prs=False` leaves that one tool out of what
    the model is offered (prompt variant `no_search_prs`)."""
    schemas = tool_schemas()
    for name, description, fields, required in (
        (
            "search_docs",
            "Hybrid search over README, documentation, CI and dependency files; top 5 whole "
            "chunks with source ranges.",
            {"query": "string"},
            ["query"],
        ),
        (
            "search_commits",
            "Hybrid search over indexed commits (message, files, diff); top 5 with 12-character "
            "SHAs. For finding earlier fixes of a bug.",
            {"query": "string"},
            ["query"],
        ),
        (
            "search_prs",
            "Hybrid search over the text of pull requests and issues (title, labels, body) by "
            "topic: top 4 PRs and top 4 issues with their numbers. Use get_pr or get_issue_meta "
            "for exact facts about one number.",
            {"query": "string"},
            ["query"],
        ),
        (
            "get_references",
            "PRs and issues that a PR or issue mentions, and those that mention it, as written "
            "#N references in synced titles and bodies, plus the issues a PR closes.",
            {"number": "integer"},
            ["number"],
        ),
        (
            "find_callees",
            "Calls made inside a definition, in source order; name-based matches are not certain "
            "and library calls are marked external. Default limit 25.",
            {"symbol": "string", "path": "string", "limit": "integer"},
            ["symbol"],
        ),
    ):
        schemas[name] = _schema(name, description, fields, required)
    if not search_prs:
        del schemas["search_prs"]
    if file_history:
        schemas["file_history"] = _schema(
            "file_history",
            "Newest-first commits that changed a file (following renames) or one function: sha, "
            "author, date, subject. Give the path, or just a symbol name when you do not know the "
            "path (it is looked up). Default limit 8.",
            {"path": "string", "symbol": "string", "limit": "integer"},
            [],
        )
    return schemas


class ProductionTools(BoundTools):
    def _indexed(self, kind: str) -> None:
        entry = Manifest(self.settings.manifest_path).get(self.repo)
        if entry is None or not entry.chunks.get(kind):
            raise LookupError(f"{kind} are not indexed for {self.repo}")

    def _search(self, kind: str, query: str) -> list:
        """Commit search re-scores its fused top 50 on the model server when one is attached
        (`reranker_mode == "on"`); if the server fails, fall back to the fused order."""
        filters = (
            {"is_fix": "true"}
            if kind == "commits" and self.settings.agent_commit_fix_only
            else None
        )
        if kind == "commits" and self.settings.reranker_mode == "on":
            try:
                return self.retriever.search(
                    self.repo, query, kind=kind, k=SEARCH_K, filters=filters, rerank=True
                )
            except RuntimeError:
                self.rerank_failures += 1
        return self.retriever.search(
            self.repo, query, kind=kind, k=SEARCH_K, filters=filters, rerank=False
        )

    rerank_failures = 0

    def _file_history(
        self, path: str | None = None, symbol: str | None = None, limit: int = HISTORY_LIMIT
    ) -> list[dict[str, Any]]:
        """Changes to a file, or to one function in it. A symbol name alone is resolved through
        the symbol table; an ambiguous or unknown name is an error that lists what exists, so the
        model can pick a path instead of guessing one."""
        function: str | None = None
        if symbol:
            found = self.symbols.get_symbol(self.repo, symbol, path=path, limit=10)
            if not found:
                raise GitToolError(
                    f"no definition named {symbol!r}; use search_code or get_symbol to find it"
                )
            paths = sorted({row.path for row in found})
            if len(paths) > 1:
                raise GitToolError(
                    f"{symbol!r} is defined in {len(paths)} files ({', '.join(paths[:5])}); "
                    "call again with the path"
                )
            path, function = paths[0], found[0].qualname.rpartition(".")[2]
        if not path:
            raise GitToolError("give a path or a symbol")
        entries = file_history(
            self.checkout,
            path,
            function=function,
            limit=min(limit, HISTORY_LIMIT),
            ref=self.pin,
            settings=self.settings,
        )
        return redact(
            [
                {
                    "sha": e.sha[:12],
                    "author": e.author_name,
                    "date": e.author_date,
                    "subject": e.subject,
                    "pr": e.pr_number,
                    "path": path,
                    "function": function,
                }
                for e in entries
            ]
        )

    def _with_pr_commits(self, record: Any) -> Any:
        """Add the PR's own commits next to its merge commit, which holds no change itself."""
        merge = record.get("merge_commit_sha") if isinstance(record, dict) else None
        if not merge:
            return record
        try:
            entries = pr_commits(self.checkout, str(merge), limit=PR_COMMITS_LIMIT)
        except GitToolError:
            return record
        if not entries:
            return record
        own = [{"sha": e.sha[:12], "subject": e.subject} for e in entries]
        return {**record, "pr_commits": redact(own)}

    def _search_prs(self, query: str) -> list[dict[str, Any]]:
        """Top PRs and top issues for a topic, each labelled; a kind not indexed is skipped."""
        entry = Manifest(self.settings.manifest_path).get(self.repo)
        present = [k for k in ("pull_requests", "issues") if entry and entry.chunks.get(k)]
        if not present:
            raise LookupError(f"pull requests and issues are not indexed for {self.repo}")
        self.retriever.forget_cached_indexes()
        rows: list[dict[str, Any]] = []
        for kind in present:
            for hit in self.retriever.search(
                self.repo, query, kind=kind, k=PR_ISSUE_K, rerank=False
            ):
                meta = hit.document.metadata
                rows.append(
                    redact(
                        {
                            "kind": "pull request" if kind == "pull_requests" else "issue",
                            "number": meta["number"],
                            "state": meta.get("state", ""),
                            "labels": meta.get("labels", ""),
                            "text": hit.document.page_content[:PR_TEXT_CHARS],
                        }
                    )
                )
        return rows

    def _extra_files(
        self, question: str, shown: set[str], skip_tests: bool = False
    ) -> list[dict[str, Any]]:
        """Up to `agent_code_extra_files` more chunks for a broad question: the best chunk of each
        file the question's own search ranks high that the model's query did not show. Excerpts."""
        wanted = self.settings.agent_code_extra_files
        hits = self.retriever.search(self.repo, question, kind="code", k=EXTRA_POOL_K, rerank=False)
        rows: list[dict[str, Any]] = []
        for hit in hits:
            path = str(hit.document.metadata["path"])
            if path in shown or (skip_tests and is_test_path(path)):
                continue
            shown.add(path)
            text = hit.document.page_content[: self.settings.agent_extra_chunk_chars]
            rows.append({"text": text, "metadata": hit.document.metadata})
            if len(rows) == wanted:
                break
        return rows

    def execute(self, name: str, args: dict) -> Any:
        if name == "search_code":
            # `_question` is added by the agent graph, never by the model; it is not a schema field.
            question = args.get("_question")
            args = {k: v for k, v in args.items() if k != "_question"}
            ask = question if isinstance(question, str) else ""
            skip_tests = bool(
                self.settings.agent_code_skip_tests
                and detect_intent(ask) != "test"
                and detect_intent(str(args.get("query", ""))) != "test"
            )
            rows: list[dict[str, Any]]
            if skip_tests:
                validate_arguments(args, production_schemas()["search_code"]["parameters"])
                self.retriever.forget_cached_indexes()
                pool = self.retriever.search(
                    self.repo, args["query"], kind="code", k=EXTRA_POOL_K, rerank=False
                )
                kept = [h for h in pool if not is_test_path(str(h.document.metadata["path"]))]
                rows = [
                    {"text": h.document.page_content, "metadata": h.document.metadata}
                    for h in kept[:SEARCH_K]
                ]
            else:
                rows = list(super().execute(name, args))
            if self.settings.agent_code_extra_files and ask:
                self.retriever.forget_cached_indexes()
                shown = {str(row["metadata"]["path"]) for row in rows}
                rows = rows + self._extra_files(ask, shown, skip_tests)
            return rows
        if name == "get_pr":
            return self._with_pr_commits(super().execute(name, args))
        if name not in (*ADDED_TOOLS, *HISTORY_TOOLS):
            return super().execute(name, args)
        validate_arguments(args, production_schemas(file_history=True)[name]["parameters"])
        if name == "file_history":
            return self._file_history(**args)
        if name == "get_references":
            return redact(self.db.references(self.repo, **args))
        if name == "find_callees":
            return redact(self.symbols.find_callees(self.repo, **{"limit": CALLEE_LIMIT, **args}))
        if name == "search_prs":
            return self._search_prs(args["query"])
        kind = "docs" if name == "search_docs" else "commits"
        self._indexed(kind)
        self.retriever.forget_cached_indexes()
        hits = self._search(kind, args["query"])
        if kind == "docs":
            return [
                {"text": h.document.page_content, "metadata": h.document.metadata} for h in hits
            ]
        return [commit_row(h.document) for h in hits]


def commit_row(document: Any) -> dict[str, Any]:
    """A commit hit as the model sees it: short SHA, message and files, a bounded diff excerpt."""
    meta = document.metadata
    return redact(
        {
            "sha": str(meta["commit_sha"])[:12],
            "is_fix": meta.get("is_fix", "false"),
            "files_changed": str(meta.get("files_changed", ""))[:300],
            "text": document.page_content[:COMMIT_TEXT_CHARS],
        }
    )
