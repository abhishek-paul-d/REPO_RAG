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

import re
from typing import Any

from repo_rag.agent.bakeoff import redact, validate_arguments
from repo_rag.agent.bakeoff_tools import BoundTools, _schema, tool_schemas
from repo_rag.agent.prompts import HISTORY_TOOL_VARIANTS
from repo_rag.history.git_tools import (
    GitToolError,
    commits_between,
    exact_renames,
    file_history,
    pr_commits,
    releases_containing,
)
from repo_rag.history.issue_linker import strip_issue_references
from repo_rag.history.metadata_db import names_of
from repo_rag.retrieval.intent import detect_intent
from repo_rag.retrieval.retrievers import weighted_rrf
from repo_rag.store.manifest import Manifest
from repo_rag.store.symbol_table import is_test_path

SEARCH_K = 5
EXTRA_POOL_K = 40  # candidates searched for the extra files; the top 5 chunks come from the model
COMMIT_TEXT_CHARS = 1200
COMMIT_BRIEF_SUBJECT = 160  # a shortlist row past the first SEARCH_K: subject, files, symbols
COMMIT_BRIEF_FIELD = 150
COMMIT_AUTHOR_CHARS = 60
CALLEE_LIMIT = 25
PR_ISSUE_K = 4  # per collection: top PRs and top issues are shown separately
PR_TEXT_CHARS = 700
PR_COMMITS_LIMIT = 5  # a PR's own commits listed next to its merge commit by get_pr
ADDED_TOOLS = ("search_docs", "search_commits", "search_prs", "get_references", "find_callees")
# Offered only by the `commit_verify` prompt variant.
COMMIT_TOOLS = ("get_commit",)
# Offered only by the `history_grounded` prompt variant.
HISTORY_TOOLS = ("file_history", "compare_refs", "find_issues")
RENAME_TOOLS = ("count_commits", "top_contributors", "change_frequency")  # follow renames there
ISSUES_SHOWN = 10  # newest issues find_issues lists; the total is always given
COMPARE_LIMIT = 15  # newest commits compare_refs lists; the total is always given
HISTORY_LIMIT = 8  # newest changes returned by file_history unless the model asks for fewer
_FRAMING_HINT = re.compile(
    r"\b(?:fixed\s+before|this\s+repository|which\s+commit|earlier\s+fix|past\s+commit|bug\s+below)\b",
    re.IGNORECASE,
)


def strip_question_framing(question: str) -> str:
    """Drop a short conversational lead-in before a pasted bug report, never a real title."""
    text = question.strip()
    if "\n\n" not in text:
        return text
    head, rest = text.split("\n\n", 1)
    head_clean, rest_clean = head.strip(), rest.strip()
    if (
        head_clean.endswith("?")
        and len(head_clean) <= 240
        and len(rest_clean) >= 20
        and _FRAMING_HINT.search(head_clean)
    ):
        return rest_clean
    return text


def production_schemas(
    *, search_prs: bool = True, file_history: bool = False, get_commit: bool = False
) -> dict[str, dict]:
    """The evaluated tools plus the added ones. `search_prs=False` leaves that one tool out of what
    the model is offered (prompt variant `no_search_prs`); `get_commit=True` adds the commit reader
    (prompt variant `commit_verify`)."""
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
            "Hybrid search over indexed commits (message, files, diff); top 5 in full plus up to "
            "10 brief candidates with 12-character SHAs. For finding earlier fixes of a bug.",
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
    if get_commit:
        schemas["get_commit"] = _schema(
            "get_commit",
            "Read one indexed commit: its full message, each changed file with added and removed "
            "line counts, and the start of the diff. Give the 12-character SHA from a "
            "search_commits result; use it to check a candidate before naming it.",
            {"sha": "string"},
            ["sha"],
        )
    if file_history:
        schemas["file_history"] = _schema(
            "file_history",
            "Newest-first commits that changed a file (following renames) or one function: sha, "
            "author, date, subject. Give the path, or just a symbol name when you do not know the "
            "path (it is looked up). Default limit 8.",
            {"path": "string", "symbol": "string", "limit": "integer"},
            [],
        )
        schemas["compare_refs"] = _schema(
            "compare_refs",
            "What changed between two tags, releases or commits: how many commits ref_b has that "
            "ref_a lacks, and the newest ones (sha, author, date, subject, PR number). Give tag "
            "names exactly as the repository writes them; an unknown name returns an error. "
            "Default limit 15.",
            {"ref_a": "string", "ref_b": "string", "limit": "integer"},
            ["ref_a", "ref_b"],
        )
        schemas["find_issues"] = _schema(
            "find_issues",
            "Filter issues by state (OPEN or CLOSED), label or a word in the title: the total that "
            "match and the newest ones (number, title, state, labels, dates). Use it for 'are "
            "there open bugs about X' and 'how many issues ...'; search_prs only samples.",
            {"state": "string", "label": "string", "title_contains": "string", "limit": "integer"},
            [],
        )
    return schemas


class ProductionTools(BoundTools):
    _rename_cache: dict[str, str] | None = None

    def _renames(self) -> dict[str, str]:
        """Exact renames over the pinned history, read from git once per agent."""
        if self._rename_cache is None:
            self._rename_cache = exact_renames(self.checkout, self.pin)
        return self._rename_cache

    def _following(self, name: str, args: dict) -> Any:
        """count_commits, top_contributors and change_frequency across file renames. Without it a
        file moved in 2023 (`requests/auth.py` to `src/requests/auth.py`) shows only its commits
        since the move: the 2026-11-18 section 1 check named the wrong top author for it."""
        validate_arguments(args, tool_schemas()[name]["parameters"])
        if name in ("top_contributors", "change_frequency"):
            args = {"limit": 3, **args}
        renames = self._renames()
        if name == "change_frequency":
            return redact(self.db.change_frequency(self.repo, **args, renames=renames))
        path = args.get("path")
        alt = names_of(path, renames)[1:] if isinstance(path, str) else []
        return redact(getattr(self.db, name)(self.repo, **args, alt_paths=alt))

    def _find_issues(
        self,
        state: str | None = None,
        label: str | None = None,
        title_contains: str | None = None,
        limit: int = ISSUES_SHOWN,
    ) -> dict[str, Any]:
        """All issues matching the filters counted, the newest few listed (no bodies)."""
        found = self.db.find_issues(
            self.repo,
            state=state.upper() if state else None,
            label=label,
            title_contains=title_contains,
            limit=100_000,
        )
        return redact(
            {
                "total": len(found),
                "shown": [
                    {
                        "number": i.number,
                        "title": i.title,
                        "state": i.state,
                        "labels": list(i.labels),
                        "created_at": i.created_at,
                        "closed_at": i.closed_at,
                    }
                    for i in found[: min(limit, ISSUES_SHOWN)]
                ],
            }
        )

    def _indexed(self, kind: str) -> None:
        entry = Manifest(self.settings.manifest_path).get(self.repo)
        if entry is None or not entry.chunks.get(kind):
            raise LookupError(f"{kind} are not indexed for {self.repo}")

    def _fused_search(self, query: str, question: str) -> list:
        """`agent_commit_query == "fused"`: the model's query and the user's question are both
        searched and the two rankings merged by weighted RRF, so a good model query is kept and
        the details of a pasted bug report are not lost."""
        shortlist = max(SEARCH_K, self.settings.agent_commit_k)
        depth = max(self.settings.retrieval_k, shortlist)
        if self.settings.reranker_mode == "on":
            depth = min(depth, self.settings.rerank_candidate_depth)
        asked_query = strip_question_framing(question)
        hits = self._search("commits", query, k=depth)
        asked = self._search("commits", asked_query, k=depth)
        by_id = {h.chunk_id: h for h in asked} | {h.chunk_id: h for h in hits}
        fused = weighted_rrf(
            {"model": [h.chunk_id for h in hits], "question": [h.chunk_id for h in asked]},
            {"question": self.settings.agent_commit_fuse_question_weight},
            self.settings.rrf_k,
        )
        return [by_id[cid] for cid, _ in fused][:shortlist]

    def _search(self, kind: str, query: str, *, k: int | None = None) -> list:
        """Commit search re-scores its fused top 50 on the model server when one is attached
        (`reranker_mode == "on"`); if the server fails, fall back to the fused order."""
        filters = (
            {"is_fix": "true"}
            if kind == "commits" and self.settings.agent_commit_fix_only
            else None
        )
        limit = (
            k
            if k is not None
            else (max(SEARCH_K, self.settings.agent_commit_k) if kind == "commits" else SEARCH_K)
        )
        if kind == "commits" and self.settings.reranker_mode == "on":
            try:
                return self.retriever.search(
                    self.repo, query, kind=kind, k=limit, filters=filters, rerank=True
                )
            except RuntimeError:
                self.rerank_failures += 1
        return self.retriever.search(
            self.repo, query, kind=kind, k=limit, filters=filters, rerank=False
        )

    rerank_failures = 0

    def _compare_refs(self, ref_a: str, ref_b: str, limit: int = COMPARE_LIMIT) -> dict[str, Any]:
        """Commits between two refs, newest first, with the total. Authors by name, never email."""
        total, entries = commits_between(
            self.checkout, ref_a, ref_b, limit=min(limit, COMPARE_LIMIT), settings=self.settings
        )
        return redact(
            {
                "from": ref_a,
                "to": ref_b,
                "total_commits": total,
                "shown": [
                    {
                        "sha": e.sha[:12],
                        "author": e.author_name,
                        "date": e.author_date,
                        "subject": e.subject,
                        "pr": e.pr_number,
                    }
                    for e in entries
                ],
            }
        )

    def _file_history(
        self, path: str | None = None, symbol: str | None = None, limit: int = HISTORY_LIMIT
    ) -> list[dict[str, Any]]:
        """Changes to a file, or to one function in it. A symbol name alone is resolved through
        the symbol table; an ambiguous or unknown name is an error that lists what exists, so the
        model can pick a path instead of guessing one."""
        function: str | None = None
        lines: tuple[int, int] | None = None
        shown = min(limit, HISTORY_LIMIT)
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
            # By line range, not name: `git log -L :name:` cannot find an indented method.
            lines = (found[0].start_line, found[0].end_line)
        if not path:
            raise GitToolError("give a path or a symbol")
        entries = file_history(
            self.checkout,
            path,
            function=function,
            lines=lines,
            limit=shown + 1,  # one more than shown: tells a partial list from a whole one
            ref=self.pin,
            settings=self.settings,
        )
        partial = len(entries) > shown
        rows = redact(
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
                for e in entries[:shown]
            ]
        )
        if partial:
            note = f"only the newest {shown} changes are shown; older ones exist: say it is partial"
            rows.append({"note": note})
        return rows

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
        if name in RENAME_TOOLS and self.settings.agent_prompt_variant in HISTORY_TOOL_VARIANTS:
            return self._following(name, args)
        if name not in (*ADDED_TOOLS, *HISTORY_TOOLS, *COMMIT_TOOLS):
            return super().execute(name, args)
        # `_question` is added by the agent graph, never by the model; it is not a schema field.
        question = args.get("_question") if name == "search_commits" else None
        args = {k: v for k, v in args.items() if k != "_question"}
        schemas = production_schemas(file_history=True, get_commit=True)
        validate_arguments(args, schemas[name]["parameters"])
        if name == "get_commit":
            return self._get_commit(args["sha"])
        if name == "file_history":
            return self._file_history(**args)
        if name == "compare_refs":
            return self._compare_refs(**args)
        if name == "find_issues":
            return self._find_issues(**args)
        if name == "get_references":
            return redact(self.db.references(self.repo, **args))
        if name == "find_callees":
            return redact(self.symbols.find_callees(self.repo, **{"limit": CALLEE_LIMIT, **args}))
        if name == "search_prs":
            return self._search_prs(args["query"])
        kind = "docs" if name == "search_docs" else "commits"
        self._indexed(kind)
        self.retriever.forget_cached_indexes()
        if (
            kind == "commits"
            and isinstance(question, str)
            and question.strip()
            and self.settings.agent_commit_query == "fused"
        ):
            hits = self._fused_search(args["query"], question)
        else:
            hits = self._search(kind, args["query"])
        if kind == "docs":
            return [
                {"text": h.document.page_content, "metadata": h.document.metadata} for h in hits
            ]
        full = hits[:SEARCH_K]
        hide = self.settings.agent_commit_hide_summary
        rows = [commit_row(h.document, hide_summary=hide) for h in full]
        if self.settings.agent_prompt_variant in HISTORY_TOOL_VARIANTS:
            for row, hit in zip(rows, full, strict=True):
                row["first_release"] = self._first_release(str(hit.document.metadata["commit_sha"]))
        # A wider shortlist (agent_commit_k > 5): the rest are one-liners, so 15 to 20 candidates
        # cost about what 5 full ones do.
        return rows + [commit_brief(h.document, hide_summary=hide) for h in hits[SEARCH_K:]]

    def _get_commit(self, sha: str) -> dict[str, Any]:
        """One commit as the model reads it before naming it: the whole message and file list, and
        the diff cut to `agent_commit_read_chars`. Issue numbers are removed from the message, as
        in search results, so the tool shows what search matched on."""
        record = self.db.find_commit(self.repo, sha)
        diff = "\n".join(f"--- {f.old_path}\n+++ {f.path}\n{f.patch}" for f in record.files)
        limit = self.settings.agent_commit_read_chars
        return redact(
            {
                "sha": record.sha[:12],
                "date": record.author_date[:10],
                "author": record.author_name[:COMMIT_AUTHOR_CHARS],
                "message": strip_issue_references(record.message),
                "files": [
                    {"path": f.path, "added": f.additions, "removed": f.deletions}
                    for f in record.files
                ],
                "diff": strip_issue_references(diff)[:limit],
                "diff_truncated": len(diff) > limit,
            }
        )

    def _first_release(self, sha: str) -> str:
        """The earliest tag containing the commit, or "unreleased". A hit's release is what tells
        'introduced in v2.3' from 'was already in v2.1': section 1 bug-3 named a commit that had
        shipped two releases earlier. Empty when git cannot say."""
        try:
            tags = releases_containing(self.checkout, sha, settings=self.settings)
        except Exception:
            return ""
        return tags[0].tag if tags else "unreleased"


def without_summary(text: str) -> str:
    """The commit text minus a model-written `Summary:` line (see `agent_commit_hide_summary`)."""
    return "\n".join(line for line in text.split("\n") if not line.startswith("Summary: "))


def _brief_subject(content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    first = lines[0]
    if not first.startswith(("Files: ", "Symbols: ", "--- ", "+++ ", "@@ ")):
        return first[:COMMIT_BRIEF_SUBJECT]
    changes = [
        line.strip()
        for line in lines
        if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- "))
    ]
    if not changes:
        return ""
    return ("(no message) " + " | ".join(changes))[:COMMIT_BRIEF_SUBJECT]


def commit_brief(document: Any, *, hide_summary: bool = False) -> dict[str, Any]:
    """A shortlist row: short SHA, the commit's subject, and the files and symbols it touches."""
    meta = document.metadata
    content = without_summary(document.page_content) if hide_summary else document.page_content
    symbols = str(meta.get("symbols") or "")
    if not symbols:
        symbols = next(
            (
                line[len("Symbols: ") :]
                for line in content.splitlines()
                if line.startswith("Symbols: ")
            ),
            "",
        )
    return redact(
        {
            "sha": str(meta["commit_sha"])[:12],
            "date": str(meta.get("author_date", ""))[:10],
            "author": str(meta.get("author_name", ""))[:COMMIT_AUTHOR_CHARS],
            "subject": _brief_subject(content),
            "files_changed": str(meta.get("files_changed", ""))[:COMMIT_BRIEF_FIELD],
            "symbols": symbols[:COMMIT_BRIEF_FIELD],
        }
    )


def commit_row(document: Any, *, hide_summary: bool = False) -> dict[str, Any]:
    """A commit hit as the model sees it: short SHA, message and files, a bounded diff excerpt."""
    meta = document.metadata
    content = without_summary(document.page_content) if hide_summary else document.page_content
    return redact(
        {
            "sha": str(meta["commit_sha"])[:12],
            "is_fix": meta.get("is_fix", "false"),
            "date": str(meta.get("author_date", ""))[:10],
            "author": str(meta.get("author_name", ""))[:COMMIT_AUTHOR_CHARS],
            "files_changed": str(meta.get("files_changed", ""))[:300],
            "text": content[:COMMIT_TEXT_CHARS],
        }
    )
