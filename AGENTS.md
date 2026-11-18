# AGENTS.md — Code-Aware GitHub Repository RAG Assistant

## What this project is

A developer companion that indexes a GitHub repository and answers questions about it:

1. **Architecture questions**: "How does auth flow from the API layer to the DB?"
2. **Explaining code**: "What does `resolve_dependencies()` do and who calls it?"
3. **Bug report → past fix**: given a new bug report, find past commits that fixed similar bugs.
4. **History, people, PRs, issues, releases**: "Who merged #1234?", "What's PR #1234 about?",
   "What changed between v2.30 and v2.31?", "Who knows the caching code best?"
5. **Onboarding**: answers from the README, docs, CI and dependency files.

There are two kinds of question. **Questions about meaning** go to hybrid vector + BM25 search.
**Exact questions** (who, when, how many, `#N`, SHAs, tags) go to `metadata.db` and git tools.
See plan.md §3.7–3.9.

**What sets it apart:** chunking is **AST-based (tree-sitter)**, not character-based. Functions,
methods and classes stay whole, which makes code retrieval much more precise. Every claim
about this must be backed by the eval harness in `eval/`. Don't quote improvement numbers we
haven't measured.

The full roadmap is in [plan.md](plan.md). Read it before starting a new phase.

## Tech stack (decided, don't swap without asking)

| Concern | Choice |
|---|---|
| Language / env | Python 3.11+, managed with **uv** (`pyproject.toml`, `uv.lock`) |
| Orchestration | **LangChain** (`langchain-core`, `langchain`) + **LangGraph** for the agent/router |
| LLM | `langchain-anthropic` → `ChatAnthropic(model="Codex-sonnet-5-5")` by default; provider is configurable in `config.py` |
| AST parsing | `tree-sitter` + `tree-sitter-language-pack` (NOT the unmaintained `tree_sitter_languages`) |
| Fallback splitter | `RecursiveCharacterTextSplitter.from_language(...)` from `langchain-text-splitters` for unsupported languages / unparsable files |
| Vector store | **ChromaDB** via `langchain-chroma` (persistent, local). Keep it behind `store/vectorstore.py` so we can swap in LanceDB later |
| Lexical search | `BM25Retriever` (`rank_bm25`) fused with vector search with `EnsembleRetriever` (RRF) |
| Embeddings | Configurable. Default `voyage-code-3` (`langchain-voyageai`); local fallback `jinaai/jina-code-embeddings-0.5b` via `langchain-huggingface` |
| Reranker | Cross-encoder rerank of top-k (`BAAI/bge-reranker-v2-m3` local, or Voyage/Cohere rerank API) |
| Git history | `GitPython` on a local clone (cheap, no rate limits) |
| GitHub metadata | GitHub **GraphQL** via `httpx` for bulk PR/issue/review sync. `PyGithub` for one-off REST calls. Token from `GITHUB_TOKEN` |
| Exact facts | `metadata.db`: stdlib `sqlite3` (commits, files, symbols, PRs, reviews, issues, links, tags, dependencies) |
| API / UI | `FastAPI` backend + `Streamlit` chat UI; `Typer` CLI |
| Tests / quality | `pytest`, `ruff` (lint + format), `mypy` (lenient) |

## Repository layout (target)

```
src/repo_rag/
  config.py            # pydantic-settings; all env vars + defaults live here
  ingest/
    github_client.py   # clone repo, fetch issues/PRs via PyGithub
    file_walker.py     # walk tree, .gitignore-aware, skip vendored/binary/generated; tags code|doc|config
  chunking/
    languages.py       # extension → tree-sitter language + node-type config
    ast_chunker.py     # cAST-style split-then-merge chunker → LangChain Documents
    fallback.py        # RecursiveCharacterTextSplitter.from_language wrapper
    docs_chunker.py    # Markdown-by-heading for docs; whole-file for small configs
    symbols.py         # symbol table: defs, imports, call edges
  history/
    commit_indexer.py  # commits → message + diff hunks + touched symbols
    issue_linker.py    # direct "fixes #N" + via-PR (closingIssuesReferences) linking
    github_sync.py     # GraphQL sync of PRs, reviews and issues; cached, incremental
    metadata_db.py     # SQLite schema + loaders (plan.md §3.8), identity merging
    git_tools.py       # blame, file_history (log --follow / log -L), diff_between, releases_containing
  summaries/
    repo_map.py        # file/dir/module summaries for architecture questions
  store/
    vectorstore.py     # Chroma collections: code_chunks, docs, commits, pull_requests, issues, summaries
    bm25.py
  retrieval/
    exact_refs.py      # detect #N, SHAs, tags, @handles, paths in the question
    retrievers.py      # hybrid + rerank per collection
    query_rewrite.py   # HyDE / multi-query for NL→code
  agent/
    graph.py           # LangGraph: exact-ref pre-step → router (5 routes) → tool-using answerer
    tools.py           # semantic search tools + exact tools (get_pr, query_commits, blame, …)
    prompts.py
  api/server.py        # FastAPI
  ui/app.py            # Streamlit
  cli.py               # `repo-rag index <url>`, `repo-rag ask "<q>"`
eval/
  datasets/            # golden Q&A + issue→fix-commit pairs
  run_retrieval_eval.py
tests/
```

## Commands

```bash
uv sync                                   # install deps
uv run repo-rag index https://github.com/<owner>/<repo>
uv run repo-rag ask "How is the config loaded?"
uv run uvicorn repo_rag.api.server:app --reload
uv run streamlit run src/repo_rag/ui/app.py
uv run pytest -q                          # offline; network tests deselected
uv run pytest -m network                  # tests that clone from GitHub
uv run ruff check . && uv run ruff format .
uv run python eval/run_retrieval_eval.py --repo <name> --strategies char,lc_lang,ast
```

## Environment variables

`ANTHROPIC_API_KEY`, `GITHUB_TOKEN`, `VOYAGE_API_KEY` (optional), `REPO_RAG_DATA_DIR` (default `./data`;
clones and the Chroma DB go here). Load from `.env` via pydantic-settings. **Never commit `.env`
or anything under `data/`.**

## Conventions and rules

- **Every chunk is a LangChain `Document`** with this metadata (flat, Chroma-compatible scalars only):
  `repo, commit_sha, path, language, start_line, end_line, symbol, symbol_kind`
  (`function|method|class|module|block`), `parent_symbol, signature, chunk_id, chunker` (`ast|fallback`).
  `chunk_id = sha1(repo:path:symbol:start_line)`. Re-indexing must be idempotent.
- **Embed a contextual header, store raw code.** The text we embed starts with
  `# path: …\n# in class: …\n# signature: …` followed by the code. `page_content` returned to the
  LLM includes the header too, so answers can cite locations.
- **Never split a function body if it fits within `max_chunk_chars`.** Oversized nodes are split
  recursively at child boundaries. Small adjacent siblings are merged. This is the cAST
  algorithm; see plan.md §3.
- The AST chunker must **never raise on bad input**: on a parse error or unknown language, log it
  and fall back to `fallback.py`. Record `chunker="fallback"` in metadata.
- **Answers must cite sources** as `path:start-end` (code) or short SHA (commits). The agent
  prompt enforces this. If retrieval finds nothing relevant, say so rather than guess.
- **Names, dates and counts come from tools only.** For who/when/how-many questions the answer
  must quote tool output (`metadata.db` or git). The model must never estimate them.
- **The LLM never writes SQL.** Expose `metadata.db` only through typed, parametrized tool
  functions, and open the DB read-only at query time.
- **"Who pushed"** isn't in git. Report author, committer and `merged_by`, and say so.
- **Show names and GitHub handles, never author emails**, in answers and in the UI.
- Whole-repo scan answers ("any dead code?") must say they're based on the top results and
  aren't exhaustive.
- Keep LLM prompts in `agent/prompts.py`, not inline.
- Config values (chunk size, top-k, model names) come from `config.py`. No magic numbers in modules.
- Write tests for the chunker using small fixture files in `tests/fixtures/<lang>/`. Each test
  asserts that function boundaries are preserved.
- Windows dev machine: use `pathlib`, never hard-code `/` separators. Store `path` metadata in
  POSIX form (`as_posix()`).

## Working agreements for Codex

- **Keep [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) up to date.** After any work that changes
  code, data, decisions or measured numbers, add a dated entry to §2. Add decisions to the §3
  table, new tools to §4, and update §1 status and §5 open items. Copy numbers from real
  command output.
- Work phase by phase per plan.md. Tick the checkbox in plan.md when a phase's acceptance
  criteria pass.
- Before adding a dependency not listed above, ask.
- When changing chunking or retrieval, re-run the retrieval eval and paste the before/after table
  into the PR or commit message.
