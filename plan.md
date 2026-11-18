# plan.md — Building the Code-Aware GitHub Repository RAG Assistant

## 0. Goal and success criteria

Build a LangChain/LangGraph assistant that takes a GitHub repo URL and lets a developer:

- ask **architecture** questions,
- get **explanations of functions or classes**, including their callers and callees,
- **match a new bug report to past fix commits**,
- ask **history, people, PR, issue and release questions**: who changed what, what a PR was
  about, what shipped in a version, who the top contributors are,
- get **onboarding answers** from the README, docs and CI/config files.

We are done when:

1. `repo-rag index <url>` indexes a mid-size repo (~50k LOC) in under 10 minutes **on a GPU**
   (Colab A100: 311 chunks/s measured). **Not met on CPU:** this laptop embeds 0.16 code
   chunks/s with Qwen3-0.6B, so psf/requests (556 chunks) took 36.5 min and a 50k-LOC repo
   (~2,000 chunks) would take hours. Querying is fine on CPU (0.6 s per search). **Answer:**
   build the index on a Colab GPU with `notebooks/index_on_colab.ipynb`, download the zip and
   run `repo-rag import-index` (built and run once on 2026-10-24: the imported index matched the
   CPU-built one exactly in search results; the Colab run wasn't timed, so the 10-minute
   criterion is still unmeasured on a GPU).
2. The answer to each sample question below cites real `path:line` ranges or commit SHAs.
3. The eval shows **AST chunking beats character chunking on Recall@5 / MRR**, with the
   numbers recorded in `eval/RESULTS.md`. That table backs the resume claim.
4. Bug→commit matching reaches a measured Hit@5 on a held-out set of real issue→fix pairs.
5. **Exact questions** ("who merged #1234?", "commits by X since June", "top contributors to
   `auth/`") are answered from tool output, never from similarity search alone. They pass an
   auto-generated check set whose answers come straight from git and the GitHub API.

---

## 1. Example user questions (design targets)

These drive the retrieval design and become the first golden eval set.

**Architecture / big picture**
- "Give me a high-level overview of this repo's architecture. What are the main modules and how do they talk to each other?"
- "How does a request flow from the HTTP router to the database layer?"
- "Where is authentication handled, and what does middleware do before a handler runs?"
- "What design pattern does the plugin system use?"

**Explain legacy code**
- "Explain what `parse_legacy_config()` does, step by step, and why it might be written this way."
- "Who calls `RetryPolicy.backoff()`, and what happens if it throws?"
- "What's the difference between `Session.send` and `Session.request`?"
- "This function has no docstring. Write one based on how it's used."

**Bug report → past fix**
- "Users report `KeyError: 'charset'` when uploading files with no Content-Type. Has something like this been fixed before?"
- "Timeouts are ignored when proxies are set. Find commits that touched proxy and timeout handling."
- "Which commit introduced the `redirect` behaviour change in v2.3?"

**Navigational**
- "Where is `MAX_POOL_SIZE` defined and where is it overridden?"
- "Show me all the places we write to the cache."

**Impact / dependencies**
- "What breaks if I change the signature of `Session.send`?"
- "Which third-party libraries do we depend on, and which versions are pinned?"

**Change history**
- "When was retry logic added, and why?"
- "Show me the history of `parse_url`: who changed it and what each change did."
- "What changed in the `adapters/` folder in the last month?"

**People / ownership**
- "Who wrote most of `auth.py`?" / "Who last touched lines 40–80 of `sessions.py`?"
- "Who knows the caching code best?" (most recent and most frequent authors)
- "What did `@alice` work on last quarter?"
- "Who opened, who reviewed and who merged PR #1234?"

**Pull requests**
- "What's PR #1234 about? Give me its description."
- "Which PR introduced streaming uploads?"
- "What did reviewers push back on in #1234?"
- "List open PRs that touch the auth code."

**Issues**
- "Are there open bugs about timeouts?"
- "Has this crash already been reported?" / "Find duplicates of #900."

**Releases / versions**
- "What changed between v2.30.0 and v2.31.0?"
- "Which release first contained the fix for #123?"
- "When did the `json=` parameter ship?"

**Stats / aggregates**
- "Who are the top 5 contributors this year?"
- "Which files change most often?" / "How many fix commits landed last month?"

**Onboarding / how-to**
- "How do I run the tests locally?" / "How is CI set up?"
- "How do I add a new transport adapter? Show me an existing one to copy."

**Whole-repo scans (limited; the answer must say so)**
- "Is there dead code?" / "Where might user input reach a SQL query unescaped?"

**Out of scope for v1:** live state ("is CI green right now?", "what's deployed?").

---

## 2. Prior art (references we build on)

| Project / resource | What we take from it |
|---|---|
| **cAST paper** (Zhang et al., EMNLP Findings 2025) — [arXiv 2506.15655](https://arxiv.org/abs/2506.15655), code: [yilinjz/astchunk](https://github.com/yilinjz/astchunk) | The core chunking algorithm: parse to AST, recursively split oversized nodes, greedily merge small siblings under a size budget. Reports +4.3 Recall@5 on RepoEval vs line-based chunking. Size is measured in non-whitespace chars. |
| **codebase-qa-agent** — [NikithaDineshKumar/codebase-qa-agent](https://github.com/nikithadineshkumar/codebase-qa-agent) | Closest end-to-end match: tree-sitter function/class chunking, FAISS + BM25 fused with RRF, answers that cite file paths and line numbers. Not LangChain, but the same pipeline shape. |
| **PR Reviewer (zod-bird)** — [awhvish.is-a.dev/projects/zod-bird](https://awhvish.is-a.dev/projects/zod-bird) | Proves the tree-sitter + ChromaDB + LangChain combo works. Chunks on function/class boundaries. |
| **RepoRAGX** — [docs](https://www.mintlify.com/AnmolTutejaGitHub/RepoRAGX) | LangChain `GithubFileLoader` + ChromaDB baseline. Uses `RecursiveCharacterTextSplitter.from_language`, so it's a good **baseline to beat**. |
| **LanceDB "Building RAG on codebases"** — [Part 1](https://www.lancedb.com/blog/building-rag-on-codebases-part-1) | Syntax-level chunking with tree-sitter, embedding with metadata, "Cursor-style" codebase chat. |
| **LangChain Source Code loader** — [`LanguageParser`](https://python.langchain.com/docs/integrations/document_loaders/source_code), [code splitter](https://docs.langchain.com/oss/python/integrations/splitters/code_splitter) | `LanguageParser` gives top-level function/class splitting, but it's shallow and only covers some languages. We use `from_language` splitting as the **fallback** and as an eval baseline. |
| **astsnowballsplitter** — [PyPI](https://pypi.org/project/astsnowballsplitter) | Tree-sitter splitter that emits LangChain `Document`s. Shows how to plug into LangChain. |
| **SZZ algorithm / "fixes #N" conventions** | For bug→commit: link fix commits to issues by regex on commit messages, then use those links as ground truth and as enrichment. |

**Why write our own chunker instead of using `astchunk` directly?** `astchunk` covers only
Python/Java/C#/TS and doesn't emit symbol names or parent-class metadata, which we need for
citations and for `get_symbol`/`find_callers`. We'll reimplement the same algorithm, adding
`tree-sitter-language-pack` (hundreds of grammars) and richer metadata. We'll still run
`astchunk` in the eval as a reference implementation.

---

## 3. Architecture

```
            ┌─────────────── INDEXING ────────────────┐
GitHub URL ─► clone (GitPython) ─► file walker ─┬─► AST chunker ─┬─► code_chunks (Chroma + BM25)
            │                                  │  └ symbol table ┴─► symbols (defs, call edges)
            │                                  └─► docs/config chunker ─► docs (Chroma + BM25)
            │ git log -p ─► commit indexer ─► issue linker ─► commits (Chroma + BM25)
            │ GitHub API ─► PR indexer + issue indexer ─► pull_requests, issues (Chroma + BM25)
            │ git log / tags + PRs + issues ─► metadata.db (SQLite: exact facts)
            │ chunks ─► LLM file/dir summaries ─► summaries (Chroma)
            └─────────────────────────────────────────┘

            ┌─────────────── QUERY ───────────────────┐
question ─► exact-ref detector (#1234, SHA, v2.31.0, @user, path) ─► direct lookups
         ─► LangGraph router ─► {architecture | explain | bug_match | navigate | history}
            │   each route = ReAct-style answerer with its own tool subset (§3.6)
            │   semantic: (vector ∪ BM25) ─RRF─► cross-encoder rerank ─► top-k
            │   exact:    metadata.db queries + git tools (blame, log -L, diff, tags)
            └─► answer with citations (path:start-end, commit SHA, PR/issue #, tag)
```

### 3.1 AST chunker (the main feature)

Per file:
1. Detect language from extension → `tree_sitter_language_pack.get_parser(lang)`.
2. Parse. If parsing fails or the tree has `has_error` above a threshold, use the fallback splitter.
3. **Split-then-merge (cAST):**
   - `chunk(node)`: if `size(node) <= MAX` → emit a single unit. Otherwise recurse into the
     named children.
   - Walk siblings left to right and greedily merge adjacent small units while
     `size(merged) <= MAX`.
   - Size = non-whitespace characters (as in the paper). Default `MAX ≈ 1500` (tune in eval).
4. For each emitted chunk, find the **enclosing definition** by walking up to the nearest
   `function_definition` / `class_definition` / `method_declaration` node type (configured per
   language in `languages.py`). That gives `symbol`, `parent_symbol`, `signature`, `symbol_kind`.
5. Module-level leftovers (imports, constants) become a `module` chunk. Imports are also kept on
   every chunk in that file as a short `imports` string, which helps with "where is X used".
6. Build the **contextual header** (path, class, signature, docstring first line) and prepend it
   for embedding.

Languages for v1: Python, JavaScript/TypeScript, Java, Go. Adding a language is just a node-type
config entry.

### 3.2 Symbol table and call graph (lightweight)

We use tree-sitter queries (`.scm`) per language to capture definitions and call sites, then
resolve calls by name (same file first, then imported modules, then global by name). It isn't
perfect, but it's enough for "who calls X". Stored as JSON or SQLite next to Chroma. Each symbol
maps to its `chunk_id`s.

### 3.3 Commit / bug-fix index

For each commit (`git log --no-merges`, capped at N most recent, configurable):
- `message`, `author`, `date`, `sha`, `files_changed`
- per-file **diff hunks**, trimmed. Map each hunk's line range onto the AST of the post-change
  file to get **touched symbols** (e.g. `requests/adapters.py::HTTPAdapter.send`).
- `is_fix` heuristic: the message matches `fix|bug|issue|regression|crash|error|#\d+`.
- **Issue linking** has two paths, and we need both:
  1. **Direct:** regex for `(fix(es|ed)?|close[sd]?|resolve[sd]?):? #(\d+)` in the commit
     message. This is how older repos like requests and flask work.
  2. **Via the PR:** commit → PR → issue. Squash-merged repos (e.g. fastapi) put only
     `(#1234)` in the commit; the "Fixes #N" lives in the PR body. Find the PR from the
     `(#N)` suffix or `GET /repos/{o}/{r}/commits/{sha}/pulls`, then read the PR's
     `closingIssuesReferences` (GraphQL) to get the linked issues.
  Then fetch each issue's title and body with PyGithub and attach them. Cache every API
  response on disk under `data/github_cache/`.
- Current measured format = message + changed files + AST touched symbols + 800 characters of
  scrubbed diff (`symbols_diff800`). Linked issue title/body stay in metadata.db, never in
  searchable commit documents. Remove closing-reference lines and other issue/PR IDs from
  message/diff in production and eval so the same formatting path cannot leak query text.

The bug-match route searches `commits` with filter `is_fix=True`, reranks, then pulls the current
code of the touched symbols so the answer can say "this was fixed in `abc123` by changing
`X.send`; the current code looks like …".

### 3.4 Repo map / summaries (for architecture questions)

Chunk-level retrieval is weak for "explain the architecture". At index time:
- per-file summary (LLM, cheap model, using the file's chunk headers + docstrings),
- per-directory summary (rolls up file summaries),
- one repo-level overview plus the directory tree.

These go in the `summaries` collection. The architecture route searches these first, then
drills into code.

### 3.5 Retrieval

- **Exact references first** (`retrieval/exact_refs.py`): regexes pull `#1234`, 7–40 char hex
  SHAs, version tags that exist in the repo, `@handles`, and paths that exist in the tree out of
  the question. Each one is fetched directly (`get_pr`, `get_issue`, `get_commit`, …) and put in
  the context before any similarity search runs. A question like "what's #1234 about?" then
  never depends on embeddings.
- **Hybrid:** Chroma vector search + BM25, fused with `EnsembleRetriever` (RRF), k≈30.
- **Rerank:** cross-encoder to top 8.
- **Query rewriting** for NL→code: HyDE ("write the code that would answer this") and
  multi-query. Toggle in config and evaluate; don't assume it helps.
- **Metadata filters:** path prefix, language, symbol_kind. The agent can set these through tool
  args.
- **Expansion:** after retrieving a method, optionally add its class header and direct callees
  (from the symbol table). Budget is limited by tokens.

### 3.6 Agent (LangGraph)

- `router` node: a structured-output classifier picks one of **5 routes**. Bug reports are
  detected by stack traces or error strings. "Who / when / which PR / which release / how many"
  questions go to `history`.
- Each route is a tool-calling node with a route-specific system prompt and tool subset, at most
  ~5 tool calls (`history` may use ~8, since it chains exact lookups):

  | Route | Tools |
  |---|---|
  | `architecture` | `search_summaries`, `search_docs`, `search_code`, `read_file` |
  | `explain` | `search_code`, `get_symbol`, `find_callers`, `read_file`, `file_history` |
  | `bug_match` | `search_commits(is_fix)`, `search_issues`, `get_commit`, `get_symbol` |
  | `navigate` | `get_symbol`, `search_code` (BM25-weighted), `search_docs` |
  | `history` | `get_pr`, `search_prs`, `get_issue`, `search_issues`, `query_commits`, `blame`, `file_history`, `diff_between`, `releases_containing`, `top_contributors`, `change_frequency` |

- **Numbers and names in answers must come from tool output.** The `history` prompt forbids the
  model from estimating counts, dates or authors itself.
- `answer` node: forces the citation format, says "not found" when evidence is weak.
- Conversation memory: LangGraph checkpointer (in-memory/SQLite), so follow-ups like "and who
  calls *that*?" work.

### 3.7 Pull request and issue index

PRs and issues become first-class documents, not just enrichment for commits.

- **Fetch** through the GitHub GraphQL API (paginated, 100 per page) to stay within rate limits.
  Store raw responses in `data/github_cache/`. Re-sync incrementally using `updatedAt`.
- **`pull_requests` collection.** One document per PR: title + body + labels + a summary of
  review comments (cheap model, only when there are more than a few comments) + titles of linked
  issues. Metadata: `number, state, author, merged_by, created_at, merged_at, base, head,
  files_changed (comma-joined), linked_issues, merge_commit_sha`.
- **`issues` collection.** One document per issue, open or closed: title + body + the first N
  comments. Metadata: `number, state, author, labels, created_at, closed_at, closed_by_pr`.
- PRs, issues and commits cross-link by number and SHA, so any one leads to the others.
- **Who "pushed" what:** git records an author and a committer, and GitHub records the PR author
  and `merged_by`. The real pusher of a direct push is only in GitHub's Events API (about 90 days).
  Answers to "who pushed X" report author + merger and say so plainly.

### 3.8 Metadata DB and git tools (the exact-answer layer)

Vector search can't answer "who", "when", "how many" or "#1234" reliably, so these go to a
structured store and fixed tools.

- **`metadata.db` (SQLite, stdlib `sqlite3`)**, with tables:
  - `commits(sha, author_name, author_login, author_date, committer_name, message, is_fix, pr_number)`
  - `commit_files(sha, path, additions, deletions)`
  - `commit_symbols(sha, path, symbol)`
  - `pull_requests(number, title, state, author, merged_by, created_at, merged_at, merge_commit_sha)`
  - `pr_reviews(pr_number, reviewer, state, submitted_at)`
  - `issues(number, title, state, author, labels, created_at, closed_at)`
  - `issue_links(issue_number, pr_number, commit_sha, link_type)`
  - `tags(name, sha, date)`

  Author identities are merged across email, name and GitHub login using `.mailmap` plus the
  API's commit-author login.
- **Tools.** The LLM calls these with typed parameters. It **never writes SQL**.
  - `get_pr(number)`, `get_issue(number)`, `get_commit(sha)`: full record plus cross-links.
  - `search_prs(query, state?, author?)`, `search_issues(query, state?, label?)`: hybrid search.
  - `query_commits(author?, path?, symbol?, since?, until?, is_fix?, limit)`: filtered log.
  - `top_contributors(path?, since?, until?, limit)`, `change_frequency(path_prefix?, since?, limit)`.
  - `blame(path, start, end)`: `git blame --porcelain`, grouped by author and commit.
  - `file_history(path | symbol)`: `git log --follow` for files, and `git log -L :func:path` for
    functions, with PR numbers attached.
  - `diff_between(ref_a, ref_b, path?)`: commits plus a summary of changed files between two tags
    or SHAs.
  - `releases_containing(sha | pr | issue)`: `git tag --contains`, earliest tag first.
- Every tool result carries its source (SHA, PR #, tag) so the answer can cite it.

### 3.9 Docs and config files

- The file walker also yields `README*`, `docs/**/*.md|rst`, `CHANGELOG*`, `CONTRIBUTING*`,
  dependency manifests (`pyproject.toml`, `requirements*.txt`, `package.json`, `go.mod`, …) and
  CI config (`.github/workflows/*.yml`).
- Markdown is split by headings with `MarkdownHeaderTextSplitter`, which keeps the heading path
  in metadata. Small config files stay whole. Large ones fall back to the recursive splitter.
- They go into a `docs` collection with `doc_kind = readme|docs|changelog|manifest|ci|other`.
- Manifests are also parsed into a `dependencies(name, version_spec, source_file)` table in
  `metadata.db`, so "which version of X do we pin?" is an exact lookup.

---

## 4. Phases

### Phase 0 — Scaffolding  ☑
- `uv init`, `pyproject.toml`, `src/repo_rag` layout from CLAUDE.md, `config.py` with
  pydantic-settings, `.env.example`, `.gitignore` (`data/`, `.env`), ruff + pytest set up, `git init`.
- **Done when:** `uv run pytest` passes on an empty test, `uv run repo-rag --help` works.

### Phase 1 — Ingestion  ☑
- `github_client.clone(url) -> Path`. Use a **blobless clone** (`--filter=blob:none`): it has the
  full commit history but downloads file contents only when needed, so history is cheap even on
  huge repos. Use `--depth 1` only when commit indexing is turned off.
- `file_walker`: respects `.gitignore`, skips binaries, `node_modules`, `vendor`, lockfiles,
  minified/generated files, and files over 1 MB. Each file is tagged `kind = code|doc|config`
  (docs, manifests and CI files per §3.9).
- Tests run offline against a local fixture repo (see Tier 1 in Phase 5). Only one marked
  `@pytest.mark.network` test clones from GitHub.
- **Done when:** walking `psf/requests` yields the expected list of `.py` files.

### Phase 2 — AST chunker  ☑  *(main feature)*

*Done 2026-10-22.* Deviations from §3.1:
- Merging only joins units with the same enclosing definition.
- Merged multi-def chunks omit the signature header.
- The per-chunk `imports` string and the docstring line in the header are deferred to
  Phase 6 (symbol table).
- Unsplittable nodes (a single giant line, Go table-test elements that share lines) may
  exceed the budget: 31 of 5,425 chunks (0.6%) across 7 real repos in 5 languages.
  Verified by `eval/chunker_check.py`: 0 of 12,309 definitions split.
- `languages.py` config for py/js/ts/java/go.
- `ast_chunker.py` implementing §3.1, returning `list[Document]`.
- `fallback.py` wrapping `RecursiveCharacterTextSplitter.from_language`.
- `docs_chunker.py` for docs and config files (§3.9).
- Tests with fixtures: nested classes, decorators, a huge function (must split at statement
  boundaries), a file with syntax errors (must fall back), an empty file.
- **Done when:** no fixture function is split unless it's larger than `MAX`, and every chunk's
  `start_line/end_line` matches the source.

### Phase 3 — Store and baseline retrieval  ☑

*Done 2026-10-22.* The acceptance query passes on psf/requests (see the project log for the
numbers). Deviations from the bullets below:
- Fusion is our own **weighted** RRF (`retrieval/retrievers.py`), not `EnsembleRetriever`, and BM25
  is `store/bm25.py`: the benchmark showed equal weights hurt, and we need filters on both
  signals. **The BM25 weight (0.3) is provisional;** Phase 4 tunes it.
- Collections are named after the embedding space (`code_chunks__Qwen3-Embedding-0.6B_512`) so
  vectors from different models can't mix. Re-indexing embeds only changed chunks.
- Reranking exists (`retrieval/rerank.py`, Qwen3-Reranker, GPU-only in auto mode) but is **off by
  default for code search**: it has only been measured on bug → fix commits. Phase 4 measures
  it on code questions, and top 20 vs top 50, before it's switched on.
- Only the local Qwen3 embedder is wired. `voyage-code-3` raises `NotImplementedError`.

Original scope:
- Chroma persistent client with collections `code_chunks`, `docs`, `commits`, `pull_requests`,
  `issues`, `summaries`. Phase 3 only fills `code_chunks` and `docs`.
- Embedding provider abstraction (Voyage / HF local). **The model is chosen by the benchmark in
  `eval/embedding_bench/`** (CodeRankEmbed vs Qwen3-Embedding-0.6B vs nomic-embed-code, plus
  BM25 hybrids, on a Colab A100). Record the decision in `eval/RESULTS.md`. The current local
  fallback, `jina-code-embeddings-0.5b`, is CC-BY-NC, so it gets replaced either way.
- BM25 index persisted (pickle) per repo.
- `search_code(query, k, filters)` = hybrid + rerank.
- CLI: `repo-rag index <url>`, `repo-rag search "<q>"`.
- **Done when:** `search "where are retries configured"` on `requests` returns `adapters.py` /
  `Retry` code in the top 3.

### Phase 4 — Evaluation harness (before more features)  ☑

*Expansion outcome (2026-10-25):* the returned A100 run is valid; callee is rejected by
the frozen rule. Fresh proof 20/30 -> 17/30, paired delta -0.10 [-0.2424,+0.0345]; the required
>=0.05 gain and positive lower bound fail. Old strata and text-growth guards pass. Keep
expansion none and proceed to Phase 5 without further tuning on these labels. All 450 outputs,
source/protocol/label hashes and all 1,787 production chunks reproduce. Reports are in
eval/retrieval_results/phase4_expansion_full/ (additional imported copy: phase4_expansion_run/).
Reranking stays opt-in. Independent review/deployment assessment remain open. Phase 5 is next,
not yet implemented; docs/PHASE5_HANDOFF.md describes local history work and the token-dependent
GitHub linking steps. Preparation checkpoints below describe earlier states.

*Current follow-up (2026-10-25):* the 90-case run and120-case policy/header assessments are
recorded in PROJECT_LOG/RESULTS. Prepare the fixed fair expansion test from
docs/CODEX_HANDOFF_EXPANSION.md:30 new multi-block questions (12 on new repo Click),120
unchanged regression cases, plain rerank20 versus callee as the primary and both secondary. Labels
and success rule are frozen/hashed before query retrieval, no commit. Store-backed production
expansion is explicit and off by default; it does not affect unreranked search. BM25 and
default-off replay pass; new A100 execution and independent review are pending. Adopt only
if the fixed proof/regression/text-budget rule passes; otherwise move to Phase 5 without
iterating. Earlier preparation checkpoints below are historical. No Phase 5 work has started.

*Evaluation complete, rerank default pending (2026-10-25):* audited the returned fixed A100 follow-up across Requests, Express
and Flask: 129 questions (49 fresh), 1,032 scored rows and 1,548 timed calls; all 38 source
hashes per run and dataset hashes match. Fresh MRR hybrid → rerank20 is 0.701 → 0.966,
paired +0.265 [0.175, 0.355], with pooled test/example numerical guards passing. Independent
label review is pending; warm rerank20 p50 is 889–976 ms versus hybrid 61–62 ms. **Decision:
keep code reranking opt-in, current weights unchanged, no blanket test penalty or BM25 quota.**
Production now exposes a separate rerank_candidate_depth (default 50) and --rerank-depth
20/50. The alternative-evidence production-path comparison now has 90 frozen questions,
30 per repo (75 additions and 15 retained), with 36 cue-free ambiguous, 18 multi-block,
18 test, 12 implementation and 6 example cases. Use phase4_comparison_on_colab.ipynb and its
matching separate Drive ZIP. Independently execute hybrid/rerank20/rerank50, report each
stratum/retained/addition view, direct paired 50-20 intervals and per-repo p50/p95. Expanded
BM25 preflight and original15 regression pass; GPU execution and independent label review are
pending. All 24 original fresh test
questions say test/tested. A100 is experiment-only; deployment hardware remains undecided.
Timings on chosen deployment hardware are required before promotion. Results, limitations and the decision
are in `eval/RESULTS.md` and `eval/retrieval_results/phase4_v2_full/`. The evaluation phase is
complete; broader AST causal claims, ANN scale and generated-answer quality remain unproven.
Phase 5 is next and has not started.

*Historical checkpoints below describe the work before the v2 assessment.*

*Started 2026-10-25.* The runner, 40-question pinned datasets for Requests and Express,
dev/test evidence grouping, all four adapters, common-header ablation, offline correctness
tests and a compact Colab notebook with a separate Drive source ZIP now exist. Real BM25-only tables are recorded in
`eval/RESULTS.md`; these do **not** establish the embedding pipeline's AST advantage.
The first real A100 full run and separate code-reranker runs are now validated for both repos;
tables/audit are in `eval/RESULTS.md` and `eval/retrieval_results/colab_full/`. The comparison
remains open: add about 10 reviewed test/example-target questions per repo, preserve the original
dataset, and rerun by target type before adopting a test penalty. HNSW on 270–565 vectors is
only a plumbing check; matched-budget sensitivity and scale experiments remain open.
Post-hoc research is in `eval/PHASE4_FAILURE_ANALYSIS.md`: first add candidate/score/token
logging and task-specific code-reranker instruction parity, then reviewed mixed-target/minimal
evidence labels, a focused rerank/candidate experiment, and matched-budget context evaluation.
Keep original questions frozen as regression evidence; obtain fresh held-out assessment for
new policies. No production settings were changed by this research.
`astchunk==0.1.0` is dev-only and does not support JavaScript; its Express result is N/A.

- `eval/datasets/<repo>_qa.jsonl`: about 40 questions → gold `path` + line ranges. Start from the
  sample questions in §1, then write the rest by hand.
- Strategies to compare: `char` (fixed 1500 chars, 200 overlap), `lc_lang`
  (`from_language`), `astchunk` (reference), `ast` (ours). Also with and without contextual
  headers.
- Metrics: Recall@1/5/10, MRR, plus avg chunk count and size.
- Review follow-up: source-line union coverage and ≥90%-complete gold ranges; paired evidence-group
  bootstrap intervals. Fixed comparisons report all questions plus dev/test; selected settings
  retain held-out test assessment. Character/non-whitespace budget matching and independent
  question review remain unresolved limitations.
- Output a markdown table to `eval/RESULTS.md`.
- **Done when:** the table exists for at least 2 repos. This is the resume metric.

### Phase 5 — Commit history and bug→fix matching  ☑

*Implementation checkpoint (2026-10-25):* commit ingestion, exact old/new AST hunk attribution,
both verified/cached linking paths, repository-scoped typed SQLite history/link storage,
idempotent commits/BM25 indexing, commit search and `ask --bug` SHA citations are implemented.
Real inputs are frozen before retrieval: 141 Requests, 170 Flask and 18 FastAPI questions
(329 total, 319 connected groups; 263 test questions). Real BM25 results and unchanged120-code
regression are in `eval/retrieval_results/phase5_preflight/`. The notebook/Drive ZIP compares
BM25/dense/hybrid/4B/8B at fixed depth 50, with paired intervals and raw traces. The original
GPU run and CLI evidence are now audited; the manual scale tier was measured on pandas-dev/pandas on 2026-11-02 for the CPU stages (`eval/scale_results/README.md`; embedding not run).
**Phase 5 ticked 2026-11-02:** the three "done when" items pass (Hit@5 for requests 0.8584 and flask 0.9779 in
`eval/RESULTS.md`; fastapi via-PR links found; `ask --bug` sample cites real commit `f69934fb24a9`). The frozen
depth run kept depth 50 / weight 0.3 (`eval/commit_depth_results/`). Open caveats, not blockers: independent
labels and usefulness of similar fixes for new bugs are unmeasured.
Defaults stay unchanged. The next CLI sample records the exact bug report beside its evidence.
Review correction (2026-10-26): use frozen `phase5_bugfix_v2` and its matching v2 notebook/ZIP.
V1 remains historical; v2 retains descriptions, adds same-file localization and query-budget
sensitivity, and treats <30-group repo/link-path guards as descriptive. The task is own-fix
lookup, not demonstrated useful similar fixes for new bugs.
Review entry point: `docs/PHASE5_BUILD_REVIEW.md`. The user requires open-source answer models;
Phase 7 will select one, with no Anthropic key needed. No commits are authorized.

- `commit_indexer.py` + `issue_linker.py` per §3.3, covering both linking paths (direct and
  via the PR).
- **Test repositories.** Counts measured on 2026-10-21 from blobless clones:

  | Repo | Commits (excl. merges) | Mention fix/bug | Reference `#N` | Explicitly close an issue | Since | Role |
  |---|---|---|---|---|---|---|
  | psf/requests | 4,883 | 722 | 579 | 182 | 2011 | Main dev and eval repo |
  | pallets/flask | 3,829 | 911 | 476 | 215 | 2010 | Second eval repo |
  | fastapi/fastapi | 7,903 | 627 | 3,672 | 8 | 2018 | Tests the via-PR linking path |
  | langchain-ai/langchain | 16,924 | 5,378 | 16,546 | 491 | 2022 | AI showcase/demo repo; scale test; both linking paths |

- **Three testing tiers:**
  1. **Unit (offline, in CI):** a pytest fixture builds a small throwaway git repo with
     GitPython: a few files, a "Fix #1: …" commit, a squash-style "Title (#2)" commit and a
     non-fix commit. GitHub responses are faked from JSON fixtures in
     `tests/fixtures/github/`. This covers hunk → symbol mapping, `is_fix`, and both linking
     paths.
  2. **Dev and eval (real repos):** requests, then flask, indexed in full. About 200
     issue → fix-commit pairs each come straight from the direct links. fastapi pairs come
     from the via-PR path. Pairs are stored in `eval/datasets/<repo>_bugfix.jsonl`.
  3. **Scale (manual, not in CI):** a very large repo (pandas, django or cpython) with
     `max_commits` capped and a blobless clone. Here we measure indexing time, memory and
     store size, not accuracy. Django links Trac tickets rather than GitHub issues, so use it
     only for scale.
- **Eval method:** for each pair, the query is the **issue title and body**. During eval, the
  commit is indexed **without** its "Fixes #N" phrase or linked issue text, preserving the rest of the description, so the answer isn't
  sitting in the index. Measure Hit@1/5/10 and MRR over commits where `is_fix=True`. Report
  direct-path and via-PR pairs separately.
- **Done when:** Hit@5 for requests and flask is in `eval/RESULTS.md`, the via-PR path finds
  links for fastapi, and `repo-rag ask` on a sample bug report cites a real past commit.

### Phase 5b — PRs, issues, people and releases  ☑
- `history/github_sync.py`: GraphQL sync of PRs (with reviews) and issues, cached, incremental
  by `updatedAt`, capped by `max_prs` / `max_issues` in config.
- `history/metadata_db.py`: the SQLite schema from §3.8, filled from git (commits, files,
  symbols, tags), the GitHub sync, `.mailmap` identity merging and parsed manifests.
- `pull_requests` and `issues` collections (§3.7).
- `history/git_tools.py`: `blame`, `file_history` (file and function level), `diff_between`,
  `releases_containing`.
- The tool functions from §3.8 as plain Python first, wrapped as LangChain tools in Phase 7.
- `retrieval/exact_refs.py`: detect `#N`, SHAs, tags, `@handles` and paths in a question.
- **Exact-answer check set (auto-generated, no hand labelling):** sample about 50 facts from
  `metadata.db` and the GitHub cache, then turn each into a question with a known answer:
  - "Who merged #N?" → `merged_by`
  - "Which release first included the fix for #N?" → earliest tag containing the fix commit
  - "Who last changed line L of F?" → blame
  - "How many commits touched P in month M?" → count

  Store as `eval/datasets/<repo>_facts.jsonl`. Tier 1 unit tests use the fixture repo with
  fake PR, review and tag data.
- **Done when:** every tool returns correct results on the fixture repo, the exact-ref detector
  catches the reference types above, and the fact set passes at least 95% through the tools
  alone (no LLM). It's re-run end to end through the agent in Phase 7.

### Phase 6 — Symbol table and repo summaries  ☑
*Slice 1 done (2026-10-31): symbol table, `get_symbol`, `find_callers`, hand-checked on 7
symbols of psf/requests (`eval/symbol_check_results.md`). Slice 2 (2026-10-31): model-free repo
map and `summaries` collection built; summaries do not help code-question file lookup (BM25,
40 questions) so they remain separate from default code search.*

*Architecture follow-up (2026-11-01): 30 source-grounded questions are frozen in
`eval/datasets/architecture_v1/`, labels **unreviewed by a person**. E1/E2 are run and audited;
the dev-selected router lowers pooled test Recall@5 from 0.8292 to 0.7042 and code MRR@10
from 0.8375 to 0.7667. No richer-fact variant qualifies under the frozen rule. Default search
stays unchanged. `docs/SUMMARIES_BUILD_REVIEW.md` records all tables and limitations. E3's
Colab notebook is ready with F1 facts selected by the frozen dev tie order; the proposed
Qwen3-4B-Instruct-2507 needs user approval before any run. Human review, model prose and the
coherent architecture-answer acceptance check remained open, so Phase 6 stayed unchecked at that point (ticked 2026-11-18, see below).
*Answer check (2026-11-10): 15 requests architecture questions through the agent, gate 12/15 set by the user
beforehand, **result 5/15, not met** (one search per question; answers name one module of several).*
*Ticked by the user's decision, 2026-11-18. The architecture-answer gate (12/15, plus a fresh 10 with gate 8/10)
is met with **Qwen3.8-27B** (vLLM, 4-bit AWQ, CUDA graphs + MTP) and `agent_code_extra_files=3`,
`agent_code_skip_tests=True`, 24,000-character results and a 12,288-token input cap, which are now the defaults:
**13/15 and 8/10**, with code 39/40, docs 11/12, prsearch 10/10, probes 5/5. **Caveats, stated plainly:** the
metric is a coverage proxy (the answer names at least half the gold files), the 30 labels were never reviewed
by a person, the fresh set passes with no margin, single deterministic runs on one repository, and the 14B
fallback was not re-measured under these defaults. Human label review stays open (docs/PROJECT_LOG.md).*
`find_callees` (CLI `callees`) was added 2026-11-02 and hand-checked on `Session.send`
(`eval/symbol_check_results.md`); import aliases are now resolved in callers and callees.*
- Tree-sitter queries for defs/calls, `get_symbol`, `find_callers`.
- File/dir/repo summaries (§3.4) using a cheaper model; cached by file content hash so
  re-indexing doesn't redo them.
- **Done when:** "who calls X" returns correct callers for 5 hand-checked symbols, and the
  architecture question gets a coherent module-level answer.

### Phase 7 — LangGraph agent  ☐
- Exact-ref pre-step + router + 5 routes + tools (§3.6), citation-enforcing answer prompt,
  checkpointer memory.
- An end-to-end answer-quality check on the golden set with an LLM judge (faithfulness to cited
  context and correctness), kept small and cheap.
- Run the Phase 5b fact set through the full agent. The checks compare answers against the known
  values exactly (names, numbers, tags), with no LLM judge.
- **Done when:** all §1 sample questions (except "out of scope") produce cited, sensible answers
  on the demo repo, and the fact set scores at least 90% end to end.

### Phase 8 — API and UI  ☑
*Status (2026-11-10): ticked by the user's decision. Built — `GET /repos`, streaming `POST /ask`, `POST /repos/import`, the sources
panel with GitHub permalinks, and the Streamlit UI as a client of the API. **"Done when" reworded by
the user on 2026-11-09:** the original criterion ("paste a URL, wait for indexing, and chat") needed
`POST /repos` to index a URL, which does not exist because indexing needs a GPU (about 0.16 chunks/s
on a laptop CPU, so a repo like requests takes ~36 minutes). The supported route is to build the
index in `notebooks/index_on_colab.ipynb` and upload the zip to `POST /repos/import`. Asking while an
import runs now returns 409 for the repos being replaced (tested offline, not yet in a live Colab
session; a stream already in flight is not interrupted).*
- FastAPI: `GET /repos`, `POST /repos/import` (a zip built in Colab), `POST /ask` (streaming).
  Not built, by decision: `POST /repos` that clones and indexes a URL.
- Streamlit: repo picker, chat, a sources panel with clickable GitHub permalinks
  (`https://github.com/o/r/blob/<sha>/<path>#L<a>-L<b>`).
- **Done when:** a fresh user can index a repo on a Colab GPU with the provided notebook, import the
  zip, and chat with it in the UI, with answers that link to the source at the indexed commit.

### Phase 9 — Polish and stretch goals  ☐
- Incremental re-index: **built 2026-11-10 as `repo-rag update`**, differently from the original wording:
  measured chunking of the whole repo is 0.5 to 0.9 s, so unchanged files are not skipped; instead vectors of
  chunks with unchanged text are reused (even after they moved), the plan is shown before embedding, and symbols,
  summaries and history are refreshed. See PROJECT_LOG 2026-11-10.
- GitHub webhook or Action to re-index on push.
- LanceDB backend behind the same interface (native hybrid FTS); compare it with Chroma.
- MCP server exposing `search_code` / `search_commits`, so it can be used from Claude Code or an
  IDE.
- Live GitHub tools for out-of-scope v1 questions: CI status, open PR checks, deployments.
- README with an architecture diagram, eval table, and demo GIF.

---

## 5. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| tree-sitter grammar/version mismatches (esp. on Windows) | Pin `tree-sitter` and `tree-sitter-language-pack` versions together. Cover parsing for every v1 language in CI tests. |
| Huge repos and slow embedding | File filters, a cap on commit count, batching embeddings, caching by content hash. |
| GitHub API rate limits | Use a local clone for code and history. Use the API only for issues/PRs, with a token and an on-disk cache. Batch the via-PR lookups through GraphQL. |
| Squash-merge repos hide issue links in PR bodies, so regex on commit messages finds almost none (fastapi: 8 out of 7,903) | Use the via-PR linking path (§3.3). Report direct and via-PR pairs separately in the eval. |
| Eval leakage: the fix commit's text contains the issue it fixes | Remove "Fixes #N" lines and linked issue text from indexed commits in eval runs. Query with the issue text only. |
| Huge histories (100k+ commits) make indexing slow | Blobless clone, a `max_commits` cap, and only commits that touch files in supported languages. Run the Tier 3 scale test before claiming large-repo support. |
| Tests depend on the network or on GitHub data changing over time | Tier 1 runs on a generated fixture repo with faked GitHub responses. Real-repo eval runs pin a commit SHA and the cached API data. |
| AST chunks too small (one-line getters) and noisy | The cAST merge step. Also tune a minimum size in the eval. |
| Name-based call resolution is wrong | Label `find_callers` results as "likely callers". Language-server-backed resolution is a stretch goal. |
| LLM hallucinating code that isn't in context | Citation-required prompt, a "not found" path, and an eval faithfulness check. |
| LLM invents names, dates or counts for "who / when / how many" questions | These questions go to the exact-answer layer (§3.8). The prompt says numbers and names must come from tool output. The fact set checks this exactly. |
| LLM-written SQL is wrong or unsafe | No raw SQL from the model. Only typed, parametrized tools. The DB connection is opened read-only at query time. |
| One person appears under several names or emails, splitting their stats | `.mailmap` plus the GitHub login from the API, merged into one identity in `metadata.db`. |
| Big repos have tens of thousands of PRs and issues (API cost) | GraphQL pagination, `max_prs` / `max_issues` caps (most recent first), incremental sync by `updatedAt`, on-disk cache. |
| "Who pushed" can't be answered from git | Report author, committer and `merged_by`, and say clearly that the pusher isn't recorded. |
| Author emails are personal data | Store them for identity merging, but show names and GitHub handles in answers, never emails. |
| Whole-repo scans ("any dead code?") are presented as complete | The answer states it's based on the top retrieved results and isn't exhaustive. |

---

## 6. Initial dependency list

```
langchain langchain-core langchain-community langchain-text-splitters langgraph
langchain-anthropic langchain-chroma langchain-voyageai langchain-huggingface
chromadb rank-bm25 sentence-transformers
tree-sitter tree-sitter-language-pack
GitPython PyGithub httpx   # httpx for GitHub GraphQL; sqlite3 is stdlib
pydantic-settings typer rich fastapi uvicorn streamlit
# dev
pytest ruff mypy astchunk   # astchunk only as an eval reference
```

Demo/eval repos: `psf/requests` (Python, rich issue history), `pallets/flask`, `fastapi/fastapi`
(via-PR linking), and `expressjs/express` (JS) for a second language. Scale test: pandas, django
or cpython (see Phase 5).
