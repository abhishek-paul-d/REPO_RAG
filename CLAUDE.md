# CLAUDE.md — Code-Aware GitHub Repository RAG Assistant

> **New session? Read [docs/HANDOFF.md](docs/HANDOFF.md) first.** It has the current status,
> what's next, the measured evidence behind every decision, and the pitfalls already hit.
> (`AGENTS.md` in the repo root is a stale copy of an old version of this file: ignore it.)

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
| LLM | **Open-source local models only** (user decision 2026-10-25). `llm_provider=local`; answer model `cyankiwi/Qwen3.8-27B-AWQ-INT4` served by vLLM from Colab with CUDA graphs + MTP (user decision 2026-11-18; met the Phase 6 gates, see `docs/PROJECT_LOG.md`), no reranker next to it; the previous provisional 14B (`Qwen/Qwen2.5-Coder-14B-Instruct`, prompt arm 2) is kept as `LOCAL_14B_*` in `config.py` and not re-measured under the new limits; summary model unset. No Anthropic key required |
| AST parsing | `tree-sitter` + `tree-sitter-language-pack` (NOT the unmaintained `tree_sitter_languages`) |
| Fallback splitter | `RecursiveCharacterTextSplitter.from_language(...)` from `langchain-text-splitters` for unsupported languages / unparsable files |
| Vector store | **ChromaDB** via `langchain-chroma` (persistent, local). Keep it behind `store/vectorstore.py` so we can swap in LanceDB later |
| Lexical search | `rank_bm25` behind our own `store/bm25.py` (code-aware tokenizer, per-repo pickle). Fused with dense search by **weighted RRF** in `retrieval/retrievers.py`, not `EnsembleRetriever`: we need metadata filters on both signals, a persisted index, and weights tuned on our eval (equal weights measurably hurt) |
| Embeddings | **`Qwen/Qwen3-Embedding-0.6B` @ 512 dims**, local `sentence-transformers` wrapped as a LangChain `Embeddings` (`store/embeddings.py`: query instruction, left padding, 2,048-token cap, Matryoshka truncation, exactly as benchmarked), one query instruction for all collections. Chosen by `eval/embedding_bench` (see `eval/RESULTS.md`); `voyage-code-3` remains an untested paid option |
| Reranker | **`Qwen/Qwen3-Reranker-8B`**, opt-in and **GPU only** (`reranker_mode="auto"`). Code comparison measured; Phase 5 compares 4B/8B on commits at fixed depth 50 before claiming a commit gain |
| Git history | `GitPython` on a local clone (cheap, no rate limits) |
| GitHub metadata | GitHub **GraphQL**: current Phase 5 cached linker uses stdlib `urllib`; Phase 5b bulk PR/issue/review sync targets `httpx`. `PyGithub` for one-off REST calls. Token from `GITHUB_TOKEN` |
| Exact facts | `metadata.db`: stdlib `sqlite3` (commits, files, symbols, PRs, reviews, issues, links, tags, dependencies) |
| API / UI | `FastAPI` backend + `Streamlit` chat UI; `Typer` CLI |
| Tests / quality | `pytest`, `ruff` (lint + format), `mypy` (lenient) |

## Repository layout (target)

```
src/repo_rag/
  config.py            # pydantic-settings; all env vars + defaults live here
  ingest/
    github_client.py   # clone repo (blobless), parse URLs
    file_walker.py     # walk tree, .gitignore-aware, skip vendored/binary/generated; tags code|doc|config
    indexer.py         # walk → chunk → embed changed chunks → BM25 → manifest (idempotent)
    updater.py         # `repo-rag update`: plan (dry run), embed only changed text, refresh derived indexes
  chunking/
    languages.py       # extension → tree-sitter language + node-type config
    common.py          # chunk Document shape, metadata, ids, contextual header
    ast_chunker.py     # cAST-style split-then-merge chunker → LangChain Documents
    fallback.py        # RecursiveCharacterTextSplitter.from_language wrapper
    docs_chunker.py    # Markdown-by-heading for docs; whole-file for small configs
    symbols.py         # symbol table: defs, imports, call edges (calls resolved by name)
  history/
    commit_indexer.py  # commits → message + diff hunks + touched symbols
    issue_linker.py    # direct "fixes #N" + via-PR (closingIssuesReferences) linking
    github_sync.py     # GraphQL sync of PRs, reviews and issues; cached, incremental
    metadata_db.py     # SQLite schema + loaders (plan.md §3.8), identity merging
    git_tools.py       # blame, file_history (log --follow / log -L), diff_between, releases_containing
  summaries/
    repo_map.py        # file/dir/module summaries for architecture questions
  store/
    embeddings.py      # LocalEmbeddings: Qwen3 via sentence-transformers as a LangChain Embeddings
    vectorstore.py     # Chroma collections: code_chunks, docs, (later) commits, pull_requests, issues, summaries
    bm25.py            # code-aware tokenizer + persisted BM25 index
    manifest.py        # data/manifest.json: which repos are indexed, at which commit/embedding
    transfer.py        # export/import an index as plain data (jsonl + npy in a zip), not Chroma files
  retrieval/
    exact_refs.py      # detect #N, SHAs, tags, @handles, paths in the question
    retrievers.py      # HybridRetriever: dense + BM25 weighted RRF, filters, optional rerank
    rerank.py          # Qwen3-Reranker (model-card scoring); GPU-only in auto mode
    query_rewrite.py   # HyDE / multi-query for NL→code
  agent/
    graph.py           # LangGraph: exact-ref pre-step → router (5 routes) → tool-using answerer
    remote_llm.py      # serve the answer model over HTTP from a Colab GPU; RemoteChatModel client (a Model)
    tools.py           # production tools on top of the evaluated set: search_docs, search_commits, get_references, find_callees
    prompts.py
  api/server.py        # FastAPI
  ui/app.py            # Streamlit
  cli.py               # `repo-rag index <url>`, `repo-rag ask "<q>"`
notebooks/
  build_colab_indexer.py   # generates index_on_colab.ipynb (embeds src/repo_rag + uv.lock pins)
  index_on_colab.ipynb     # GENERATED. Index repos on a Colab GPU, download a zip, import locally
eval/
  datasets/            # golden Q&A + issue→fix-commit pairs
  run_retrieval_eval.py
tests/
```

## Commands

```bash
uv sync                                   # install deps
uv run repo-rag index https://github.com/<owner>/<repo>
uv run repo-rag search "where are retries configured" --repo psf/requests   # no LLM; add --kind docs, --language, --path
uv run repo-rag update <owner/repo> [--dry-run] [--checkout <clone>]   # latest commit: plan, embed only changed text, refresh symbols/summaries/history
uv run repo-rag index-symbols <owner/repo> --checkout <clone>   # symbol table, CPU only
uv run repo-rag symbol Session.send --repo <owner/repo>        # where it is defined
uv run repo-rag callers Session.send --repo <owner/repo>       # who calls it, with match type
uv run repo-rag callees Session.send --repo <owner/repo>       # what it calls (add --external for library calls)
uv run repo-rag index-summaries <owner/repo> --checkout <clone>   # repo map; search --kind summaries
uv run repo-rag export-index idx.zip [--repo owner/name]   # pack an index: text + metadata + vectors
uv run repo-rag import-index idx.zip                       # merge into this machine's data; re-embeds nothing
uv run python notebooks/build_colab_indexer.py             # regenerate the Colab GPU-indexing notebook
uv run repo-rag ask "How is the config loaded?"
uv run uvicorn repo_rag.api.server:app --reload
uv run streamlit run src/repo_rag/ui/app.py        # chat UI; paste the URL + token from notebooks/serve_model_on_colab.ipynb
uv run repo-rag sync-github <owner/repo> --max-prs 100000 --max-issues 100000   # PRs + issues into metadata.db (needs GITHUB_TOKEN; the default cap is 500)
uv run python eval/agent_eval.py build|run --url <colab url> --token <token>      # exploratory 93-question check of the agent, per question type
uv run python eval/commit_failure_sort.py out.json        # where the gold commit ranked for every missed commit question (A/B/C/D buckets)
uv run python eval/commit_lenient_score.py RESULT.json    # strict vs lenient scoring (sibling commits naming the same issue count)
uv run python eval/commit_session.py --url <colab url> --token <token> --workers 8   # commit experiments in one Colab session: control x2, fused, verify, fix-only, hidden summaries
uv run pytest -q                          # offline; network tests deselected
uv run pytest -m network                  # tests that clone from GitHub
uv run ruff check . && uv run ruff format .
uv run python eval/run_retrieval_eval.py --repo <name> --strategies char,lc_lang,ast
uv run python eval/chunker_check.py psf/requests=data/repos/psf__requests   # chunker invariants on real repos
```

## Environment variables

`ANTHROPIC_API_KEY`, `GITHUB_TOKEN`, `VOYAGE_API_KEY` (optional), `REPO_RAG_DATA_DIR` (default `./data`;
clones, the Chroma DB, BM25 files and `manifest.json` go here), `REPO_RAG_EMBEDDING_DTYPE`
(`auto` | `float32` | `bfloat16`; use `float32` when building on a GPU for CPU-side queries). Load from `.env` via pydantic-settings. **Never commit `.env`
or anything under `data/`.**

## Conventions and rules

- **Every chunk is a LangChain `Document`** with this metadata (flat, Chroma-compatible scalars only):
  `repo, commit_sha, path, language, start_line, end_line, symbol, symbol_kind`
  (`function|method|class|module|block` for code, `section` for docs), `parent_symbol, signature,
  symbols` (comma-joined qualnames of every definition in the chunk), `chunk_id, chunker`
  (`ast|fallback|docs`). Docs chunks also carry `doc_kind`.
  `chunk_id = sha1(repo:path:symbol:start_line)`; fallback and docs pieces use an index instead of
  the symbol, so ids stay unique. Re-indexing must be idempotent. All of this is built in
  `chunking/common.py`; don't construct chunk Documents anywhere else.
- Qualified names: `Class.method`, `Outer.Inner.m`, `func.inner`; Go methods use the receiver
  type (`Store.Get`); JS/TS `const f = () => …` is a function named `f`.
- **Embed a contextual header, store raw code.** The text we embed starts with
  `# path: …\n# in class: …\n# signature: …` followed by the code. `page_content` returned to the
  LLM includes the header too, so answers can cite locations.
- **Never split a function body if it fits within `max_chunk_chars`.** Oversized nodes are split
  recursively at child boundaries. Small adjacent siblings are merged. This is the cAST
  algorithm; see plan.md §3.
- The AST chunker must **never raise on bad input**: on a parse error or unknown language, log it
  and fall back to `fallback.py`. Record `chunker="fallback"` in metadata.
- Chunks are **whole source lines, verbatim**. The chunk body equals `lines[start-1:end]`, and no
  two chunks share a line. Don't use splitters that rewrite whitespace (that's why the docs
  chunker has its own heading scanner instead of `MarkdownHeaderTextSplitter`).
- tree-sitter grammars are downloaded on first use into a per-user cache
  (`tree_sitter_language_pack.cache_dir()`). A fresh machine or CI needs network once, or
  `tree_sitter_language_pack.download([...])` beforehand.
- `tests/fixtures/` is excluded from ruff. Fixtures are deliberately odd code and must stay
  byte-for-byte.
- **Vectors from different embedding spaces never mix.** Chroma collections are named
  `<kind>__<embedding_id>` (model + dims), and the manifest records each repo's embedding id;
  `search` refuses a repo indexed under another one. Changing the model or dims means
  re-running `repo-rag index`.
- **Indexing large repos happens on a GPU.** CPU embedding is ~0.16 chunks/s on the dev laptop
  (requests took 36 min). Build the index in Colab with `notebooks/index_on_colab.ipynb`
  (float32, so vectors match laptop queries), then `repo-rag import-index`. The notebook
  embeds our source and the uv.lock versions, so **regenerate it after any change under
  `src/repo_rag`, `pyproject.toml` or `uv.lock`**; `tests/test_colab_notebook.py` fails if it's
  stale. Notebooks are excluded from ruff (it rewrites them).
- Index transfers carry plain data, never Chroma's files. Import replaces only the named
  repos, rebuilds BM25 from the text, and refuses a different embedding id.
- **Chroma collections are created with `hnsw.sync_threshold=2`** (`chroma_sync_threshold`),
  and existing ones are switched on open. Chroma's default (1000) keeps new vectors in memory
  only, and reads (vector queries and `get` with embeddings) then intermittently failed with
  `Error creating hnsw segment reader: Nothing found on disk`: a Chroma 1.5.9 defect,
  reproduced with raw `chromadb` (20 failures in 5,760 reads; 0 at 2) and fixed here (3
  full-suite runs, 0 failures). Don't raise the threshold. If that error ever returns, the
  cause is not "flaky": re-measure with a raw-chromadb loop before adding any retry.
- **Retrieval always filters by repo** (`repo` metadata). Measured: it adds +0.13 to +0.18
  nDCG@10 for every model. Reranking is opt-in per call and has only been measured on bug →
  fix commits; don't switch it on for code questions until Phase 3/4 measures it.
- Tests that need a vector store use `tests/fakes.py` (`HashEmbeddings` etc.). Never download a
  model in the offline suite.
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
  asserts that function boundaries are preserved. After any chunker change, also run
  `eval/chunker_check.py` on the real repos listed in `eval/chunker_check_results.md`.
  "Split: 0" must hold.
- Windows dev machine: use `pathlib`, never hard-code `/` separators. Store `path` metadata in
  POSIX form (`as_posix()`).

## Working agreements for Claude

- **Keep [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) up to date.** After any work that changes
  code, data, decisions or measured numbers, add a dated entry to §2. Add decisions to the §3
  table, new tools to §4, and update §1 status and §5 open items. Copy numbers from real
  command output.
- **Keep [docs/HANDOFF.md](docs/HANDOFF.md) current** when a phase finishes, a decision changes
  or a pitfall is found (its §3 status, §9 pitfalls, §11 next steps). It is what a new session
  reads first.
- Work phase by phase per plan.md. Tick the checkbox in plan.md when a phase's acceptance
  criteria pass.
- Before adding a dependency not listed above, ask.
- When changing chunking or retrieval, re-run the retrieval eval and paste the before/after table
  into the PR or commit message.
