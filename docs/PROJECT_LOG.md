# Project log: Code-Aware GitHub Repository RAG Assistant

A running record of what was built, how, with what, and why. Newest entries go at the bottom of
§2. [plan.md](../plan.md) is the roadmap (what we *will* do); this file is the history (what we
*did*). [CLAUDE.md](../CLAUDE.md) holds the working rules.

**How to maintain:** every piece of work that changes code, data, decisions or measured numbers
adds an entry to §2. New decisions also go in the §3 table, and new tools in §4. Numbers are
copied from real command output, never estimated.

---

## 1. Status at a glance

| Phase | Status | Notes |
|---|---|---|
| 0. Scaffolding | Done | uv project, config, CLI, tests |
| 1. Ingestion | Done | Blobless clone + file walker; proven on psf/requests |
| Embedding + rerank choice | **Qwen3-Embedding-0.6B @512d + Qwen3-Reranker-8B (GPU, opt-in)** | 8B code comparison measured; Original Phase 5 4B/8B comparison audited; depth/weight extension pending |
| 2. AST chunker | Done | 6 languages + docs chunker; 0 of 12,309 defs split on 7 real repos (5 languages) |
| 3. Store and baseline retrieval | Done | Acceptance query passes on psf/requests; CPU indexing is slow (0.16 chunks/s), see log |
| 4. Eval harness | Evaluation complete; rerank default pending | Real expansion A100 matrix audited/rejected by frozen rule; keep none, no further tuning on these cases; Phase 5 next, independent review/deployment remain open |
| 5 | **Done 2026-11-02** (ticked): keep depth 50, weight 0.3; Hit@5 requests 0.8584, flask 0.9779; Pandas scale tier measured; independent labels and similar-fix usefulness open as caveats | Frozen 50/100/200 shared scores, dev-only weight selection and separate timing; docs/COMMIT_DEPTH_BUILD_REVIEW.md |
| 5b | Done | Git tools, GitHub sync, `metadata.db`, identities, exact commit tools, PR/issue collections; 60/60 auto-generated facts pass through the tools on httpx; issue->PR Hit@5 BM25 0.600, hybrid 0.800 (45 pairs, GPU run) |
| 6 | **Done 2026-11-18** (ticked by the user's decision) | Architecture-answer gate met with the 27B (vLLM, CUDA graphs + MTP): 13/15 and fresh 8/10, code 39/40, docs 11/12; proxy metric, labels reviewed by the owner (provenance unverified; flask labels not applied), fresh set has no margin; E1/E2 evaluation audited earlier, no summary change qualifies |
| 7 | **Closed with known gaps 2026-11-18** (ticked by the user's decision) | 27B: facts 59/60 (98.3%), section 1 1 of 32 flagged by the automatic checks (pr-3 data limit), regression held (code 40/40, docs 11/12, commit 5/10, probes 5/5); manual check of the 19 exact answers found 5 wrong, 4 fixed, bug-3 still weak; judge not run on 27B answers, no human labels, guards/snapping/`history_grounded` not default |
| 8–9 | Not started | See plan.md |
| Commit-summary pilot | Replicated on Flask 2026-11-20 | Summary index: requests 82 and 84 vs controls 76 to 78 of 113; Flask 106 and 107 vs 100 to 101 of 136. Not significant alone, not shipped by default. Default prompt is now `commit_explain`. |

Git: repo initialized on `main`, **nothing committed yet**.

---

## 2. Log

### 2026-10-21: Research and planning

**Goal set:** a RAG chatbot over a GitHub repo that answers architecture questions, explains
code and matches bug reports to past fix commits. What sets it apart is AST-based (tree-sitter)
chunking instead of character chunking.

**Prior art found** (web search). Full table in plan.md §2.
- *cAST* paper (EMNLP Findings 2025, arXiv 2506.15655) plus its code, `astchunk`: the
  split-then-merge AST chunking algorithm we reimplement. It reports +4.3 Recall@5 over
  line-based chunking.
- *codebase-qa-agent*: closest end-to-end project (tree-sitter + FAISS + BM25 + RRF, cited
  answers).
- *RepoRAGX*: LangChain + Chroma using `RecursiveCharacterTextSplitter.from_language`. This is
  our baseline to beat.
- LangChain's `LanguageParser` is shallow, so we use it only as a fallback and baseline.
- `tree_sitter_languages` is unmaintained; we use `tree-sitter-language-pack` instead.
- No published embedding-based bug→commit matcher turned up. SZZ and "fixes #N" conventions
  inform our approach.

**Created:** `CLAUDE.md` (stack, layout, rules) and `plan.md` (architecture and phases).

### 2026-10-21: Phase 0, scaffolding

- `uv init --package --name repo-rag`. Supports Python ≥3.11; 3.13 pinned locally via
  `.python-version`.
- `src/repo_rag/config.py`: pydantic-settings. Tunables use the `REPO_RAG_` prefix; API keys
  use their usual names (`ANTHROPIC_API_KEY`, `GITHUB_TOKEN`, `VOYAGE_API_KEY`) as `SecretStr`.
- `src/repo_rag/cli.py`: Typer app with `index`, `search`, `ask`, `config` and `--version`.
- ruff (lint + format), mypy, pytest. 13 tests to start.
- Environment note: the agent sandbox blocks the system temp folder, so test runs here pass
  `--basetemp=<scratchpad>`. A plain `uv run pytest` works on a normal machine.

### 2026-10-21: How we got git history, and choosing test repos

**Question:** how can we test bug→fix matching? We need repos with long, well-linked histories.

**Method:** blobless clones, which fetch every commit and tree but no file contents. Then
count commits from their messages:
```bash
git clone --filter=blob:none --no-checkout https://github.com/<owner>/<repo>.git
git rev-list --count --no-merges HEAD                      # commits
git log --format='%s%n%b%x00' | ... | grep -ciE '(fix(e[sd])?|close[sd]?|resolve[sd]?):? +#[0-9]+'
```

**Measured:**

| Repo | Commits (excl. merges) | Mention fix/bug | Reference `#N` | Explicitly close an issue | Since |
|---|---|---|---|---|---|
| psf/requests | 4,883 | 722 | 579 | 182 | 2011 |
| pallets/flask | 3,829 | 911 | 476 | 215 | 2010 |
| fastapi/fastapi | 7,903 | 627 | 3,672 | 8 | 2018 |

**Key insight:** squash-merge repos like fastapi put only `(#PR)` in the commit, and the
"Fixes #N" lives in the PR body. So issue linking needs two paths: direct regex, and
commit → PR → `closingIssuesReferences` (GraphQL).

**Decisions:**
- requests is the main dev/eval repo, flask the second, and fastapi tests the via-PR path.
- Three testing tiers: an offline fixture repo, real repos, and a manual scale test.
- Eval leakage rule: fix commits are indexed without their issue refs, and queries use issue
  text only.

### 2026-10-21: Question taxonomy and the exact-answer layer

**Question asked:** can it answer "who pushed what" and give a PR's description?
**Answer:** not in the first plan, so the plan was extended (plan.md §3.7–3.9, Phase 5b).
- 12 question categories, including people, PRs, issues, releases, stats and onboarding.
- **Questions about meaning** go to hybrid vector + BM25 search. **Exact questions** (who,
  when, how many, `#N`) go to a SQLite `metadata.db` and git tools (`blame`, `log -L`,
  `tag --contains`). The LLM never writes SQL.
- "Who pushed" isn't in git: we report author, committer and `merged_by`.
- A 5th agent route, `history`. Docs and config files get indexed too.

### 2026-10-21: Phase 1, ingestion

- `ingest/github_client.py`: parses every common GitHub URL form, makes a blobless clone into
  `data/repos/<owner>__<repo>`, and fast-forwards existing clones. `--no-history` gives
  depth 1. The token is not put in the clone URL, because git would save it in plain text.
- `ingest/file_walker.py`: uses `git ls-files -s -z`, so `.gitignore` is respected and
  symlinks and submodules are skipped. It skips vendored and build folders, lockfiles,
  minified and generated files (by name or a "DO NOT EDIT" banner), binaries and files over
  1 MB. Kept files are tagged `code` (with language), `doc` or `config`. `CODEOWNERS` was
  added for ownership questions.
- `chunking/languages.py`: maps extensions to tree-sitter language names.
- Tests: a generated fixture repo with 23 tricky files, plus one `@pytest.mark.network` test.
  53 offline tests in total.

**Result on psf/requests:** 6,495 commits cloned in 8.5 s (6.8 MB `.git`). 87 files kept
(37 Python, 29 doc, 21 config).

### 2026-10-21: Embedding model benchmark (built, smoke-tested, not run on GPU yet)

**Why:** the embedding model drives retrieval quality, and the choice should be measured.

**Research findings:**
- `jina-code-embeddings` is **CC-BY-NC** (no commercial use), so it's excluded, and our
  config's current local fallback needs replacing.
- A 7B model can't embed queries on a laptop CPU.

**Candidates:**
- `nomic-ai/CodeRankEmbed` (137M, MIT)
- `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, Matryoshka dims)
- `nomic-ai/nomic-embed-code` (7B, Apache-2.0)

References: BM25, each model + BM25 (RRF), and optionally `voyage-code-3`.

**Design** (`eval/embedding_bench/bench.py`, notebook generated by `build_notebook.py`):
- **Corpus:** about 2.9k chunks from a fixed stdlib-`ast` chunker over requests, flask and
  httpx, pinned to fixed SHAs. httpx's look-alike code makes hard negatives.
- **T1 docstring → code:** about 500 auto-generated queries.
- **T2 developer questions:** 36 hand-written, with gold symbols checked against the source.
- **T3 bug report → fix commit:** 147 linked issues and 4,882 commits. Needs `GITHUB_TOKEN`.
- **Metrics:** Hit@k, MRR@10, nDCG@10, throughput, GPU/CPU query latency, peak memory,
  truncation, index size and license.

**Bugs the CPU smoke test caught before any GPU time was spent:**
1. A pathspec on `git log` turned on history simplification and dropped 1,216 of 6,495
   commits, including PR-branch fix commits.
2. requests' history has diffs with raw NUL bytes, so NUL-separated parsing broke into 16,026
   pieces. It now uses text markers.
3. Truncation was checked against the global cap instead of each model's own limit.

### 2026-10-21: vLLM, deferred

vLLM officially supports Qwen3-Embedding. nomic-embed-code and CodeRankEmbed are probably
supported but would need a check that their embeddings match sentence-transformers.
**Decision:** don't use it in the first benchmark run. It only affects indexing speed, not the
model choice. It risks breaking Colab's torch install. And it only matters if the 7B model
wins. Revisit if nomic-embed-code beats the next model by more than ~0.03 mean nDCG@10 (hybrid).

### 2026-10-21: AI repos as test candidates

**Question:** why test on requests, flask and httpx? Why not AI repos?
**Clarification:** FastAPI and httpx are *plumbing in our stack* (the API server and GraphQL
calls). requests, flask and httpx are *test repos* chosen for stable, well-linked history. The
AI parts of our stack are LangChain, LangGraph, Claude, Chroma, tree-sitter and the embedding
models.

**Measured** (blobless clones, same method as above):

| Repo | Commits (all) | Excl. merges | Mention fix/bug | Reference `#N` | Explicitly close an issue | Since |
|---|---|---|---|---|---|---|
| langchain-ai/langchain | 17,038 | 16,924 | 5,378 | 16,546 | 491 | 2022 |
| run-llama/llama_index | 7,962 | 7,962 | 2,437 | 7,917 | 56 | 2022 |
| huggingface/transformers | 24,450 | 23,929 | 12,567 | 20,359 | 316 | 2018 |

**Decision:** add **langchain-ai/langchain** as the AI showcase and scale repo:
- It has the most direct links (491), and almost every commit references a PR, which exercises
  the via-PR path.
- It's 17k commits, a good scale test.
- We build on LangChain ourselves.

requests stays the controlled eval repo, because it's small, stable and its gold set is
verified. llama_index has 56 direct links, so it's only useful with the via-PR path.
transformers is very large; keep it for a later scale test.

### 2026-10-21: Faster Colab setup

The notebook installs packages with `uv pip install --system` instead of `pip install`. uv is
installed first if Colab doesn't already have it.

### 2026-10-21: First Colab run crashed on CodeRankEmbed, fixed

**Symptom:** `AttributeError: 'NomicBertModel' object has no attribute
'get_extended_attention_mask'`, raised on the first encode.

**Cause:** CodeRankEmbed ships its own `modeling_hf_nomic_bert.py` (`trust_remote_code`), which
calls a transformers helper that **transformers 5 removed**. Our `transformers>=4.51` let Colab
install 5.x.

**Fix:**
1. The notebook pins `transformers>=4.51,<5`.
2. `bench.py` attaches an equivalent `get_extended_attention_mask` to NomicBert modules when
   it's missing, as a fallback.
3. CodeRankEmbed's remote code is pinned to revision `3c4b608…`. Unpinned remote code can
   change under us, which is also a security concern.
4. The dimension API deprecation is handled.

**Verified locally on CPU:** the crash reproduced on transformers 5.20.2 and is fixed by the
patch. 4.57.6 works unpatched. Embeddings from both paths are identical (cosine 1.000000 on
all test texts).

**Also seen:** `GITHUB_TOKEN` wasn't set in Colab, so T3 was skipped. Add it in Colab
Secrets for the full run.

### 2026-10-21: Embedding benchmark run 1, results

Colab A100 run completed for all three models. T3 was skipped again because `GITHUB_TOKEN`
didn't reach the run. Full tables and findings are in [eval/RESULTS.md](../eval/RESULTS.md).

- **T2 (developer questions, nDCG@10):** Qwen3-0.6B 0.558, BM25 0.418, CodeRankEmbed 0.290,
  nomic-embed-code 0.277. The nomic models win only T1 (docstring→code: 0.892 / 0.842 vs
  Qwen3 0.813).
- Equal-weight RRF with BM25 *lowered* T1 for every model (about 0.84 → 0.59), so production
  fusion must be weighted.
- Qwen3 @ 512 dims matches full size at half the index.
- Qwen3 indexed fastest (311 docs/s). CPU query p50 was 269 ms on Colab's CPU.

**Changes made from these results:**
- `config.py` defaults to `Qwen/Qwen3-Embedding-0.6B` (`embedding_provider="huggingface"`),
  replacing the CC-BY-NC jina fallback. Tests updated.
- New **T2r** task: the T2 questions restricted to requests' chunks. 9 T2 questions were
  missed by every model, mostly because httpx has look-alike implementations and the
  questions don't name the repo. CPU smoke test (MiniLM): 0.49 → 0.65 nDCG@10 with the filter.
- The notebook's secrets cell now prints *why* a secret wasn't read and warns loudly that T3
  will be skipped.
- Decision is **provisional** until run 2 includes T3.

### 2026-10-21: Embedding benchmark run 2, decision final

T3 ran (144 issue → fix pairs over 4,882 commits) along with the new T2r task. T1 and T2
reproduced run 1 exactly. Full write-up in [eval/RESULTS.md](../eval/RESULTS.md).

- **T2r (questions, repo-filtered):** Qwen3 0.736, nomic-embed-code 0.431, CodeRankEmbed
  0.416, BM25 0.522.
- **T3 (bug → fix):** nomic-embed-code **0.596**, Qwen3 0.488, CodeRankEmbed 0.436, BM25
  0.347. The 7B model really does win here (+22 of 144 at Hit@5).
- **Mean of the two real tasks:** Qwen3 0.612 > 7B+BM25 0.550 > 7B 0.513.
- **Decision (final for v1):** Qwen3-Embedding-0.6B at 512 dims for all collections. The 7B
  model is rejected for query-time GPU need, index size and indexing speed. The T3 gap goes to
  follow-ups: tune Qwen3's T3 instruction, improve the commit doc format, and try
  Qwen3-Embedding-4B for commits only.
- Repo filtering adds 0.13–0.18 nDCG@10 for every model, which confirms the plan's metadata
  filter is essential.
- Run 1 results moved to `eval/embedding_bench/results_run1/`; run 2 is in `results/`.

### 2026-10-21: T3 tuning experiment (built, not run on GPU yet)

**Goal:** close Qwen3's bug → fix gap (0.488 vs 0.596 for the 7B model) cheaply, before
trying a bigger model.

**Built:**
- `eval/embedding_bench/t3_tuning.py`, plus the generated `t3_tuning.ipynb`. It runs Qwen3
  only, in about 5 minutes on Colab.
- 5 query instructions × 4 commit document formats, each scored at 512 dims (shipped) and
  full size. Formats:
  - `diff2000`: the run-2 baseline
  - `symbols_diff800`: message + files + touched symbols + short diff
  - `symbols`: message + files + touched symbols
  - `message`: message only
- **Touched symbols** come from git's hunk headers (`@@ … @@ def foo`) plus `def` / `class`
  in changed lines. It's a cheap stand-in for Phase 5's AST mapping, and fills in for 2,160
  of 4,882 commits.
- **No overfitting:** the 144 issues are split 50/50 into dev and test with a fixed seed. The
  winner is picked on dev and reported on test, with a paired bootstrap 95% CI against the
  baseline. The split's issue numbers are saved in `t3_split.json`.
- `bench.py` refactor: `commit_corpus_formats` builds every format from one `git log` pass,
  and `bug_queries` is split out of `task_bug_to_fix`. Behaviour of the main benchmark is
  unchanged.

**Smoke test (CPU, MiniLM, 600 commits, stand-in issues):** all outputs produced. It caught
a mislabelled-dims edge case for models under 512 dims, now fixed.

**Lesson (tooling):** shell heredocs mangled `\n` escapes twice. File content with
backslashes goes through the Edit or Write tools.

### 2026-10-22: T3 tuning results: no significant gain

Qwen3-only grid run on Colab (5 instructions × 4 commit doc formats, 72 dev / 72 test
issues). Details are in [eval/RESULTS.md](../eval/RESULTS.md).

- **The best combination on dev gains +0.017 nDCG@10 on test**, with a 95% CI of
  [-0.017, +0.050]. That's not significant. The gap to the 7B model (~0.1) remains.
- The diff carries the signal (message-only: -0.15). 800 chars matches 2,000. The
  instruction wording barely matters (±0.02).
- The baseline reproduced run 2 exactly (0.488 at full dims).

**Adopted for simplicity and speed:**
- One query instruction for all collections: `embedding_query_instruction`.
- Commit docs with touched symbols + 800 chars of diff: `commit_diff_chars`, about 1.4×
  faster indexing.
- Both are in `config.py`, with tests.

**Next lever:** a cross-encoder reranker over the top 50, or Qwen3-Embedding-4B for the
commits collection.

### 2026-10-22: T3 rerank experiment (built, not run on GPU yet)

**Goal:** test the two remaining levers for the bug → fix gap. These are a bigger embedder
(Qwen3-Embedding-4B) and a reranker over the top 50: Qwen3-Reranker-0.6B, Qwen3-Reranker-4B,
or bge-reranker-v2-m3 (the one currently in `config.py`).

**Built:** `eval/embedding_bench/t3_rerank.py` plus the generated `t3_rerank.ipynb`.
- **First stage** (3 options): Qwen3-0.6B @512 (production baseline), Qwen3-4B @512, and
  Qwen3-4B @2560. These use the settings adopted after T3 tuning (`symbols_diff800` docs,
  the shared instruction).
- **Reranking** of each first stage's top 50 by each of the 3 rerankers. Pairs shared across
  first stages are scored once.
- **Metrics:** recall@50 (the ceiling for any reranker), nDCG@10 and Hit@5 on all 144 and on
  the t3_tuning test half, with paired bootstrap CIs against the baseline.
- **Speed:** rerank pairs/s and latency per query.
- The Qwen3-Reranker yes/no scoring is implemented exactly as on its model card (prompt
  strings copied verbatim).
- `bench.py` gains a `qwen3-4b` model spec.

**Smoke test (CPU, this laptop):** the scoring sanity check passed (relevant commit 0.986 vs
irrelevant 0.002), and the full pipeline produced every output. **Deployment finding:**
Qwen3-Reranker-0.6B on this laptop's CPU ran at **0.2 pairs/s (23 s for 5 candidates)**, so
reranking a top-50 locally would take minutes per question. Any reranker win only applies
where there's a GPU or a hosted endpoint, the same constraint that ruled out the 7B embedder.

### 2026-10-22: T3 rerank results: gap closed

Colab A100 run of `t3_rerank.ipynb`. Full tables are in [eval/RESULTS.md](../eval/RESULTS.md).

- **Qwen3-Reranker-4B on production's Qwen3-0.6B candidates (top 50): 0.687 nDCG@10**, up
  from 0.494 (+0.192, CI [+0.133, +0.251]). That beats nomic-embed-code 7B (0.596).
- A bigger first stage barely matters after reranking: Qwen3-4B adds +0.004. Recall@50 is
  0.86 for 0.6B and 0.91 for 4B.
- Qwen3-Reranker-0.6B gives only +0.062. **bge-reranker-v2-m3 hurts (−0.121)**, and it was
  our config default.
- Speed on the A100: 4B reranker 1.83 s per query, 0.6B 0.58 s, bge 0.16 s. On a laptop CPU,
  reranking isn't viable.

**Changes:** in `config.py`:
- `reranker_model = "Qwen/Qwen3-Reranker-4B"`, replacing bge.
- `reranker_mode = "auto"` (GPU only).
- `rerank_instruction` added.
- `retrieval_k` goes from 30 to 50.

Tests updated. The embedder stays Qwen3-0.6B @512.

### 2026-10-22: Phase 2, AST chunker (done)

**Built** (`src/repo_rag/chunking/`):
- `languages.py`: tree-sitter node-type config for Python, JavaScript, TypeScript, TSX, Java
  and Go. It covers definitions, wrapper nodes (decorators, `export`, Go `type`),
  `const f = () => …` declarators and Go receivers.
- `ast_chunker.py`: cAST split-then-merge.
  - **Split:** oversized nodes are replaced by their children. Size = non-whitespace characters,
    budget 1,500. The file root is always split.
  - **Merge:** greedy, but only between units with the same enclosing definition, so a class
    header stays with its first methods and a method never merges with the next top-level
    function.
  - Comments and decorators stick to the next definition. A unit starting on the current
    chunk's last line always joins it, so chunks never share a line.
  - **Metadata:** qualified `symbol`, `symbol_kind`, `parent_symbol`, `signature`, and
    `symbols` (every definition inside the chunk). The chunk that opens a split class is also
    named by that class.
  - Falls back to the text splitter when more than 10% of bytes are ERROR nodes, for unknown
    languages, or on any exception. **It never raises.**
- `fallback.py`: `RecursiveCharacterTextSplitter.from_language`, with unique ids even for
  pieces of one long line.
- `docs_chunker.py`: its own Markdown/RST heading scanner. It keeps text verbatim and skips `#`
  inside fenced code. RST levels follow the order adornments appear. Bare headings and RST
  anchors fold into the next section. **LangChain's `MarkdownHeaderTextSplitter` was rejected
  because it rewrites whitespace**, which broke line numbers and citations.
- `common.py`: the single place chunk Documents, ids and contextual headers are built.
- New deps: `tree-sitter` 0.26.0, `tree-sitter-language-pack` 1.22.1, `langchain-core` 1.6.13,
  `langchain-text-splitters` 1.1.3. Grammars download on first use into a per-user cache.
- Config: `ast_max_error_ratio` (0.1) and `fallback_chunk_overlap` (100, clamped to a quarter
  of the chunk size).

**Tests:** 127 new, 180 in total, all passing.
- Properties checked on 7 fixtures across 6 languages at budgets 1,500 and 120:
  - text equals the source lines
  - no line is lost
  - **no within-budget definition is split**
  - no overlap
  - metadata schema
  - stable, unique ids
- Plus qualified names per language, huge-function splitting, syntax-error fallback, empty and
  unknown files, never-raises inputs (binary, a 50k-char line, CRLF and unicode), and docs
  headings, verbatim text, `doc_kind` and fallback ids.

**Real repos** (requests, flask, httpx at the benchmark SHAs):

| Repo | Files | Code chunks | Doc chunks | Time | Within-budget defs | Split |
|---|---|---|---|---|---|---|
| psf/requests | 88 | 274 | 282 | 0.3 s | 775 | **0** |
| pallets/flask | 179 | 422 | 653 | 0.4 s | 1,572 | **0** |
| encode/httpx | 93 | 386 | 397 | 0.4 s | 1,206 | **0** |

- All code chunks used the AST path; none fell back.
- Median chunk is about 1,200 non-whitespace chars.
- 16 of 1,082 chunks (1.5%) exceed the budget, up to 3,636. Each is an unsplittable docstring,
  decorator stack or dict literal.
- **57/57** of the embedding benchmark's T2 gold symbols land in a chunk that names them.

**Bugs found and fixed while building:**
- The root node fitting the budget became one symbol-less chunk.
- A split class header was labelled with its first method.
- `class X extends Y {` was split across two chunks on the same line.
- Splitters crashed when the overlap exceeded the chunk size.
- `MarkdownHeaderTextSplitter` rewrote whitespace.
- Ruff was about to reformat test fixtures; `tests/fixtures/` is now excluded.

**Not measured yet:** whether AST chunks retrieve better than character chunks. That's Phase 4,
and per CLAUDE.md no improvement number gets quoted until then.

### 2026-10-22: Chunker checked on real JS, TS, Java and Go repos

**Why:** Phase 2's real-repo check covered only Python. The other five languages had only
small hand-written fixtures.

**Built:** `eval/chunker_check.py`, a reusable check for any checkout. Per language it reports:
- the chunker mix and fallback reasons
- chunk text vs source lines, lost lines, overlaps, duplicate ids
- **within-budget definitions that were split** (re-parsed independently)
- sizes and over-budget chunks
- anonymous symbols

The final report is in `eval/chunker_check_results.md`.

**Repos (shallow clones, pinned):**
- expressjs/express `0dee2c52`
- trpc/trpc `1c9bf51e`
- google/gson `e2292f9c`
- spf13/cobra `adbc8813`

Plus requests, flask and httpx at the benchmark SHAs.

**First run:** 0 split definitions, 0 fallbacks, 0 text, lost-line, overlap or duplicate-id
issues. Four problems found:
1. **The `class` keyword token was counted as an anonymous class definition** (7 `<anonymous>`
   symbols in TS). It shares its type name with JS/TS class expressions. Fix: only named nodes
   can be definitions.
2. **Express-style `res.status = function status() {}` wasn't recognised.** Most of Express's
   API is written this way. Fix: assignments, object pairs (`{ handler: () => {} }`) and bound
   class expressions (`const Foo = class {}`) are definitions named by their binding.
   `lib/response.js` went from 0 to 32 named functions.
3. **Huge doc comments were glued onto the definition after them**, making Java/JS chunks up to
   5,292 chars. Fix: comments attach forward only as far as fits the budget; the lines
   nearest the definition stick.
4. **Unsplittable leaves and same-line cascades.** Giant comments and Go string templates
   reached 11,481 chars. In callback-heavy code (`it('x', function () {…})`), the "never share a
   line" rule forced whole callbacks into already-full chunks. Fixes:
   - multi-line leaves over budget are split by line
   - the merge step **backtracks**: the units starting on the shared line move to the new
     chunk, and a join is forced only when the line is shared with an earlier multi-line unit

**After the fixes (all 7 repos, 5 languages):**

| Repo | Language | Chunks | Defs checked | Split | Over budget | Max size |
|---|---|---|---|---|---|---|
| psf/requests | python | 274 | 775 | **0** | 1 | 2,579 |
| pallets/flask | python | 429 | 1,572 | **0** | 0 | 1,500 |
| encode/httpx | python | 391 | 1,206 | **0** | 1 | 1,503 |
| expressjs/express | javascript | 384 | 291 | **0** | 1 | 1,515 |
| trpc/trpc | typescript / tsx / js | 1,510 / 563 / 38 | 2,880 / 980 / 15 | **0** | 7 / 9 / 0 | 2,331 / 4,584 / 1,485 |
| google/gson | java | 1,478 | 4,021 | **0** | 8 | 2,889 |
| spf13/cobra | go | 358 | 569 | **0** | 4 | 4,933 |

- **12,309 definitions checked, 0 split.**
- Over-budget chunks fell from 116 to **31 of 5,425 (0.6%)**.
- 0 fallbacks, 0 anonymous symbols, 0 text, lost-line, overlap or duplicate-id issues.
- What remains over budget is genuinely unsplittable: single giant lines (an SVG path in TSX),
  Go table tests whose elements share lines (`}, {`), and a few long literals.
- The checker now recognises definitions with the chunker's own matcher, so bindings are
  verified too: Express went from 123 to 291 definitions checked.

**Tests:** 6 new regression tests, 186 in total, all passing:
- giant Javadoc
- giant Go string
- JS assignment/object functions
- `class` keyword token
- callback moved off a shared line
- class expression naming

### 2026-10-22: Phase 3, store and baseline retrieval (done)

**Built:**
- `store/embeddings.py`: `LocalEmbeddings`, a LangChain `Embeddings` over sentence-transformers.
  It reproduces the benchmarked setup: Qwen3-Embedding-0.6B, query instruction, left padding,
  2,048-token cap, Matryoshka truncation to 512 dims and re-normalisation. It loads lazily.
- `store/vectorstore.py`: langchain-chroma, one collection per kind, **named after the embedding
  space** (`code_chunks__Qwen3-Embedding-0.6B_512`) so vectors from different models can't mix.
  `sync()` embeds only new or changed chunks, rewrites metadata without re-embedding (a new
  commit sha), and deletes vanished chunks.
- `store/bm25.py`: a persisted BM25 index with the benchmark's code-aware tokenizer.
- `retrieval/retrievers.py`: `HybridRetriever`, with dense and BM25 fused by weighted RRF,
  metadata filters (language, symbol_kind, doc_kind, path prefix) and an optional reranker.
- `retrieval/rerank.py`: Qwen3-Reranker per the model card. GPU-only in `auto` mode and opt-in.
- `ingest/indexer.py` and `store/manifest.py`: walk, chunk, sync, BM25, and record each repo's
  commit and embedding id. `repo-rag search` refuses a repo indexed under another embedding.
- CLI: `repo-rag index` (clone + chunk + embed + progress) and `repo-rag search` (`--repo`,
  `--kind`, `--language`, `--symbol-kind`, `--path`, `--rerank`).
- New deps: chromadb 1.5.9, langchain-chroma, sentence-transformers 5.7.0, rank-bm25, torch
  2.14.1 (CPU build), transformers pinned `<5`.

**Tests:** 75 new, 261 in total, all passing offline (`tests/fakes.py` holds the hash-based fake
embedder). They cover idempotent re-indexing (a repeat run embeds 0 chunks; an edit re-embeds
only its chunks), deletions, per-repo isolation, filters, fusion math, rerank gating and the
CLI end to end.

**Real run, psf/requests** (commit `611c6162`, real Qwen3-Embedding-0.6B on this laptop's CPU):
- 556 chunks: 274 code (37 files, all AST) and 282 docs (51 files). Wall-clock **36.5 min**,
  including the one-off 1.2 GB model download.
- **Acceptance query** "where are retries configured": rank 1 `tests/test_requests.py`
  (`test_urllib3_retries`), rank 2 `adapters.py` `HTTPAdapter.__init__`, rank 3 `adapters.py`
  `HTTPAdapter`. **Passes** (adapters.py Retry code is in the top 3).
- **Sanity check on the 36 benchmark questions** through the shipped pipeline, top 5:

  | Fusion | Hit@1 | Hit@5 |
  |---|---|---|
  | dense only | 0.667 | 0.944 (34/36) |
  | hybrid, BM25 weight 0.3 (default) | 0.556 | 0.972 (35/36) |
  | hybrid, equal weights | 0.500 | 0.972 (35/36) |

  The benchmark's repo-filtered reference was 0.889 dense and 0.917 equal-weight hybrid. It is
  **not like-for-like**: our chunks merge small definitions, so one chunk can hold several gold
  functions. BM25 trades top-1 precision for top-5 recall (4 questions vs 1), too few to tune
  on, so **the 0.3 weight stays provisional for Phase 4**.
- Verified: left padding is applied (alone-vs-batched vector cosine 1.0), and a fresh embedding
  matches the stored vector exactly (cosine 1.0).

**Finding: CPU indexing is impractically slow.** Measured with the model cached and nothing else
running: **0.16 code chunks/s** (about 6 s per chunk), 0.33 doc chunks/s, 13 s model load and
**609 ms per query**. The A100 did 311 docs/s. Searching on CPU is fine, but a ~50k-LOC repo
(~2,000 chunks) would take hours, so the plan's "under 10 minutes on a laptop" criterion only
holds on a GPU (plan.md updated). The design already allows the answer: the embedding space is
named in the collection and the manifest, so an index built on Colab can be copied into `data/`
and searched locally.

**Other changes and lessons:**
- **Intermittent Chroma error.** One test run failed with "Error creating hnsw segment
  reader: Nothing found on disk". It couldn't be reproduced in 11 reruns or 60 stress rounds, so
  vector queries now retry that one error up to 3 times with a logged warning (tested).
- sentence-transformers 5.7 renamed `tokenizer_kwargs` to `processor_kwargs` (fixed, verified).
- BM25 on tiny corpora: a term in 1 of 2 documents has IDF exactly 0, which matters only for
  test fixtures (they now include filler chunks).
- Settings are cached per process, which tests had to account for.

**Deliberately not done:** reranking is off by default for code search (only measured on bug →
fix commits); `voyage-code-3` isn't wired; no eval harness yet (Phase 4).

### 2026-10-24: Colab GPU indexing workflow, and the Chroma defect found on the way

**Why:** CPU indexing runs at 0.16 chunks/s, so large repos must be indexed on a GPU and
searched locally (Phase 3 finding).

**Built:**
- `store/transfer.py` and the CLI commands `repo-rag export-index` / `import-index`. The zip
  holds plain data: `export.json`, plus per repo and collection a `.jsonl` (id, text, metadata)
  and a float32 `.npy`. It never carries Chroma's own files (HNSW binaries and pickles can
  differ across OS and library versions).
- **Import** replaces only the named repos' chunks, re-embeds nothing, rebuilds BM25 from the
  text, merges the manifest, and **refuses a different embedding id** (model + dims).
- `notebooks/build_colab_indexer.py` generates `notebooks/index_on_colab.ipynb`. It embeds the
  `repo_rag` source (a base64 zip) and installs dependencies at the exact `uv.lock` versions, and
  runs the real `repo-rag index` / `export-index` commands, so there is no second implementation.
  `tests/test_colab_notebook.py` fails if the notebook is stale, if a pin violates
  `pyproject.toml`, or if a pinned package has no Linux x86_64 wheel for CPython 3.12/3.13 (all
  have one; the language pack needs glibc 2.34, which Colab's Ubuntu 22.04 has).
- `embedding_dtype` setting (`auto` | `float32` | `bfloat16`). The notebook indexes in float32,
  matching a CPU laptop's queries.

**Measured:**
- **Real round trip** (the 556-chunk requests index, real model): export 1.2 MB. Import into an
  empty data folder: text, metadata and ids identical; the same 36 questions return the same
  chunks in the same order with the same scores (36/36); vectors equal to float32 rounding (max
  1.5e-8 per component, because Chroma's cosine index re-normalises on insert).
- **float32 vs bfloat16** (CPU stand-in; no GPU here): query cosine >= 0.9998; bf16 queries
  against the fp32 index returned the same top-1 chunk 36/36, the same top-5 set 36/36, the same
  top-5 order 35/36. So the mismatch is small and bf16 is a safe speed-up; float32 is just the
  conservative default.

**Defect found: Chroma 1.5.9 intermittently fails reads right after writes.** The test suite
began failing with `Error creating hnsw segment reader: Nothing found on disk`: 4 occurrences in
different tests, then 2 of 3 full-suite runs. I first treated it as a rare transient and added a
retry; the retry then failed too, so it wasn't transient. Debugging, in order:
1. Plain raw-Chroma stress with one collection: 0 failures (the first attempts were too simple).
2. **Two collections in one database with several interleaved writes and reads: reproduced with
   raw `chromadb`, none of our code.**
3. Only reads needing the vector index fail (`get` with embeddings, vector queries); `get` of
   ids/documents/metadata never did. So the vector index wasn't on disk yet.
4. Chroma's default `hnsw.sync_threshold` is 1000, so new vectors live only in memory until then.
   **Default: 20 failures in 5,760 reads. `sync_threshold=2`: 0** (alone, at creation, or applied
   afterwards with `modify()`). Cost: none measurable (2,000 x 512-d vectors in 1.2 s vs 1.0 s).

**Fix:** `chroma_sync_threshold = 2` (a setting, min 2), used when creating collections and
applied to existing ones when they are opened (the real requests index's `code_chunks` flipped
from 1000 to 2 on one ordinary search). **The retry was removed**: with the fix, 3 full-suite runs
showed 296 passed, 0 hnsw errors and 0 retries fired, so it was dead code hiding a solved problem.

**Mistakes made along the way (kept for the record):**
- Called the failure "rare" and "transient" after too little evidence; the evidence that it wasn't
  came from the failing retry and a second look at which calls failed.
- A Windows-sleep gap (1.5 days) made one test run look hung; it wasn't.
- I wrote a script to `D:\Users\...` by mistake (a `..\..` in a path). It holds one script file in
  empty folders I created; the sandbox refused my cleanup, so `D:\Users` is for the user to delete.
- `ruff format` rewrote the generated notebook once, and once ran after generation; the staleness
  test caught both. Notebooks are now excluded from ruff, and the notebook is generated last.
- Wording claimed "a free T4 works" before anything had run on a GPU; it now says "should".

**First real Colab run (2026-10-24, same day): worked.** `repo_rag_index.zip` (1.24 MB) was
built by the notebook for psf/requests at commit `611c6162`, the same commit as the local CPU
index, and imported with `repo-rag import-index` (the local copy was backed up first). Nothing
Colab-specific went wrong. Compared with the CPU-built index of the same commit:
- ids, texts and metadata identical for all 556 chunks (274 code, 282 docs);
- vectors: cosine (GPU float32, CPU float32) >= 0.9999998 for every chunk, worst single
  component difference about 1e-6;
- search on the 36 benchmark questions: same top-8 chunks in the same order 36/36, and the gold
  symbol in the top 5 for 35/36 on both.

So a GPU-built index is interchangeable with a CPU-built one. **Not recorded:** which GPU it ran
on and how long indexing took (the archive stores only the finish time, 18:08 UTC), so the "under
10 minutes on a GPU" criterion is still unmeasured end to end.

### 2026-10-25: Project handoff file

Wrote `docs/HANDOFF.md`: a self-contained briefing for a new session (what the project is, status,
architecture and code map, the evidence behind each decision, performance facts, environment and
Windows quirks, pitfalls, how the user likes to work, a Phase 4 brief, pending decisions, a glossary,
and commands that check the file's own claims). Every claim was checked against the repo before
finishing (git state, manifest, acceptance query, 292 passing tests, config defaults).

Findings while writing it:
- **`AGENTS.md` in the repo root is stale and wrong.** Something other than our sessions created it
  as an old copy of `CLAUDE.md` with a bad find-and-replace (the LLM is written `Codex-sonnet-5-5`).
  It still names `voyage-code-3`, the non-commercial jina model, `bge-reranker-v2-m3` (measured
  harmful) and `EnsembleRetriever`. An agent that reads `AGENTS.md` by default would be misled.
  `CLAUDE.md` now says to ignore it; the user decides whether to delete it or point it at `CLAUDE.md`.
- `main` still has no commits (24 staged from an early snapshot, 10 modified, 77 untracked).
- The Qwen3-Reranker-4B weights are not in the local Hugging Face cache (GPU-only anyway).

`CLAUDE.md` now points to the handoff at the top and requires keeping it current.

### 2026-10-25: Phase 4 evaluation harness started

**Built:** `eval/run_retrieval_eval.py` with pinned-checkout validation, lexical-only and full
embedding modes; `eval/retrieval/` adapters, line-grounded metrics, isolated Chroma storage,
portable vector caches, exact/HNSW diagnostics, weighted-RRF/test-file-penalty grids and
development-only selection. Optional CUDA reranking compares top 20 and 50 with a code-specific
instruction. Production chunking/retrieval defaults and the existing index were not changed.

**Gold:** 40 manually authored Questions per repo, Requests `611c6162cbc4` and Express
`0dee2c52c7c8`; each has **16 dev / 24 test**. Gold ranges are independently resolved from
reviewed source anchors and hashed. Overlapping evidence stays in the same split: **32 evidence
groups for Requests, 40 for Express**. Requests reuses 36 earlier benchmark questions plus four
new questions; Express has 40 new questions. The Requests split is not held out from earlier
embedding model selection. `eval/README.md` documents this and other interpretation limits.

**Compared locally:** all supported chunkers (char, lc_lang, reference astchunk, ours), header
off/on, with real BM25 on both pinned corpora: **14 supported configurations**. Full table in
`eval/RESULTS.md`; raw summaries/settings/source hashes and per-question rankings in
`eval/retrieval_results/bm25/` (22 files, **1,198,927 bytes** copied from command outputs).
Results are mixed: with headers, Requests Recall@5 is **0.708 char / 0.729 ours**, while
Express is **0.833 char / 0.750 ours**. These are lexical-only, not evidence for the full
embedding pipeline's headline claim. No BM25 weights were selected or production defaults changed.

**Reference and parser findings:** added the already-planned `astchunk==0.1.0` as a dev-only
dependency (plus its transitive grammars/pyrsistent). It has no JavaScript support, so Express
reference rows are N/A; the corpus is never reduced to suit a strategy. Temporary tree-sitter
Point `.row/.column` access reproduced a native access violation on this Windows/CPython 3.13
combination. Tuple access (already used by the production chunker) worked. The eval adapter
substitutes only coordinate access in upstream ASTNode wrappers; reference split/merge and
reconstructed content remain unchanged. Its Requests comparison completed successfully.

**GPU workflow:** `notebooks/build_retrieval_eval.py` generates the self-contained
`notebooks/retrieval_eval_on_colab.ipynb`, embedding the exact source and datasets with locked
direct dependency versions. It checks out pinned public sources, runs the real runner and
downloads `phase4_results.zip`, excluding Chroma binaries. Code/dataset freshness and cell syntax
are checked offline. Regenerated the existing indexing notebook after the config change too.

**Validation:** final full offline suite **310 passed, 1 deselected in 121.89 s**. Ruff:
**All checks passed; 78 files already formatted**. Mypy: **no issues in 27 source files**.
The tests check scoring independence from symbol names, duplicate evidence, split leakage,
wrong revisions/hashes, Unicode/repeated-text coordinates, reference coordinate conversion,
header boundary invariance, vector-cache invalidation, actual Chroma queries and parity with
the production HybridRetriever using offline test vectors. Fake vectors are never published as
retrieval-quality results. The manifest still reports the imported **274 code + 282 docs** index
at the same Requests commit.

**Pending:** run the notebook on the user's Colab GPU, review its real embedding/hybrid/HNSW
and reranker tables, and then decide tuning. The full GPU workflow is not yet executed or timed;
Phase 4 remains unchecked. No commits, pushes, AGENTS.md changes or deletions were made.

---

### 2026-10-25: Phase 4 review follow-up — completeness and uncertainty

Added source-line union coverage and the fraction of gold ranges reaching **90% coverage**,
with nested/overlapping gold and repeated retrieved lines deduplicated. Added **10,000-resample,
95% pointwise evidence-group bootstrap intervals (seed 42)** and paired AST-minus-baseline
differences. Groups preserve shared-answer dependence; estimates remain question-macro averages.
Reports now lead with all **40** questions for fixed configurations, also show dev/test, and
keep tuned selections assessed using held-out test only. Selection remains development MRR
then Recall; the newly inspected completeness metric does not change the selection rule.

Re-ran BM25 for both pinned repositories. Header-on test completeness reproduces the review:
Requests **0.583 / 0.583 / 0.646**, Express **0.750 / 0.708 / 0.833** for char/lc_lang/AST.
Before/after comparison: **560 question/configuration pairs**, identical result ids/order/source
envelopes, **0 changes** to existing overlap/MRR values. Full tables and paired intervals are
in `eval/RESULTS.md` and `eval/retrieval_results/bm25_review/`. Original `bm25/` evidence remains
unchanged. Retained **24 compact JSON/Markdown files (2,097,486 bytes)** for the new run;
large chunks, vectors, Chroma databases and clones remain under ignored `data/`.

Some pointwise intervals exclude zero: for example, Requests test MRR AST-minus-char is
**+0.126 [ +0.005, +0.262 ]**, while all-question MRR is **+0.071 [ −0.019, +0.165 ]**.
Express all-question completeness AST-minus-char is **+0.175 [ +0.075, +0.300 ]**. These are
exploratory sample-specific associations, not a general AST-win claim. The metric was introduced
after inspection, budgets differ, questions share an author, some name the target symbol, and
Requests was used in earlier model selection. These limits are explicit in the reports.

The reference-span audit found **5 text/source mismatches among 270 Requests reference chunks**:
two partial source-line reconstructions and three chunks with omitted multiline leaf text.
The upstream library retains an enclosing line envelope while replacing oversized omitted text
with blank padding. Its text and algorithm stay unchanged; the adapter now exports represented
source spans, and missing nonblank source lines earn no hit/coverage credit. Regression cases
cover Unicode, missing trailing newlines, split definitions, dropped multiline leaves and gap
scoring. The original ranking and overlap numbers happened to remain unchanged on this dataset.

Fixed the **7 Phase 4 mypy errors** with typed splitter options, explicit result types, a
literal-value check and supported Settings construction. The earlier clean report covered
application code only; default mypy now includes the maintained Phase 4 package and entry points.
Legacy standalone `eval/embedding_bench/` scripts and `eval/chunker_check.py` are explicitly
excluded rather than implicitly claiming they are typed. Default check: **36 source files
clean**; `mypy eval`: **10 source files clean**. No runtime dependencies were added.

Regenerated both notebooks after changes. Final full offline suite: **323 passed, 1 deselected
in 91.84 s**; Ruff and formatting checks are clean. The GPU/full workflow is still
unexecuted. No production retrieval defaults, dataset wording, commits or staging were changed.

### 2026-10-25: Use uv in the Phase 4 Colab installer

Updated `notebooks/build_retrieval_eval.py` and regenerated its notebook. Pinned runtime and
reference dependencies now install through `python -m uv pip install --system --python` targeting
the notebook kernel's interpreter. Pip is used only to bootstrap uv when it is absent. Added
plain-language notebook/README text explaining that `ARCHIVE_B64` is a Base64-encoded ZIP of
the project source and question datasets, automatically decoded and extracted on Colab.

Validation: existing notebook freshness/payload/cell-syntax test **1 passed in 1.11 s**; builder
lint/format and `git diff --check` clean. The GPU notebook has still not been run on Colab;
no installation-speed measurement is claimed. Evaluation results and production code unchanged.

### 2026-10-25: Publish current metrics in the main saved-result folders

The review identified a discoverability gap: coverage/CIs were already saved in
`eval/retrieval_results/bm25_review/`, but the main `bm25/` path still held overlap-only reports.
Preserved that original snapshot in `bm25_legacy/`, re-ran BM25 on both pinned repositories,
and published current compact reports into `bm25/`. Kept the prior review snapshot as history
and added a results-directory README explaining which version is current. No files were deleted.

Published **42 summary rows** (14 configurations, each with all/dev/test views) across both repos,
with coverage, completeness, marginal confidence intervals, paired comparisons and raw rankings.
Current compact evidence is **24 files, 2,097,855 bytes**. Verification against the archived
original: **560 rankings and all original overlap/MRR values unchanged**. The current summary
rows all contain coverage/completeness and confidence intervals.

Updated `eval/RESULTS.md` to link the main reports directly and label the old table historical.
Express interpretation is specific: BM25 AST evidence is more complete on all 40 questions
(≥90% completeness delta **+0.175 [ +0.075, +0.300 ]** vs char); the held-out-test delta is
**+0.083 [ +0.000, +0.208 ]**, touching zero. MRR differences span zero. Header variants and
all/dev/test tables reuse questions, and historical folders are versions of one experiment,
not extra replications. Blank lines, unmatched budgets, author/wording bias and exploratory
unadjusted intervals remain disclosed limitations.

Validation: **14 affected reporting/statistics/notebook tests passed in 1.71 s**; Ruff clean;
mypy clean across **36 source files**. Regenerated the Phase 4 notebook to carry the clarified
report wording. No retrieval behavior, datasets, production defaults, staging or commits changed;
the real GPU/full run remains pending.

### 2026-10-25: Replace the Phase 4 Base64 payload with a Drive ZIP

At the user's request, `notebooks/build_retrieval_eval.py` now generates a compact
`retrieval_eval_on_colab.ipynb` (**10 KB**, rounded command output) plus a separate
`phase4_source.zip` (**68 KB**). The user uploads the ZIP to the top level of My Drive, opens
the notebook in Colab, and approves its Drive mount. `SOURCE_ZIP` is editable if they choose
another folder. The notebook checks the source ZIP's SHA-256 against its generated expected
digest before extracting the project Python source and question datasets into Colab storage.
There is no embedded Base64 source in the Phase 4 notebook. uv dependency installation remains.

Tests now validate the external ZIP against current source, notebook/ZIP pairing, small code
cells, Drive extraction with a mocked mount, and missing/mismatched archive failures. Both
notebook test modules: **17 passed in 1.93 s**; maintained mypy scope: **36 source files clean**.
The existing indexing notebook is unchanged. No Drive access or real GPU run occurred here;
the user will upload the supplied archive and run Colab. Retrieval/data/results are unchanged.

### 2026-10-25: Validate the first real Phase 4 GPU results

Imported the user's `phase4_results.zip` (**14,727,140 bytes**, **88 members**, **54,706,757
uncompressed bytes**) into ignored `data/phase4_colab/`. Treated archive contents as data and
executed only local trusted validation code. The archive contains **two main runs with
rerank=false** and **two separate runs under rerank_experiment with rerank=true**. Both code
reranking depths (20/50) were therefore measured; the main flag alone does not describe the
whole archive.

Hardware: **NVIDIA A100-SXM4-40GB**, Linux/Python **3.13.16**, Qwen3-Embedding-0.6B **512d,
float32**, Chroma **1.5.9**, sentence-transformers **5.7.0**, tree-sitter **0.26.0**, NumPy
**2.1.3**, torch **2.11.0+cu130**. Repo pins and the original 40 questions/repo are unchanged.
All **34 source hashes per run** match current local source after LF normalization. Dataset
hashes match after the CRLF→LF normalization used in source packaging.

Validation reproduced **16 variants and 8,480 question/method rows**: all exported chunks,
metrics, summary bootstrap intervals and strategy comparisons, exact dense top-10 from portable
vectors, and BM25 top-10. Vectors are finite, unit-normalized and 512-dimensional. This checks
the saved inference outputs; neural inference was not repeated locally. Full compact evidence
is under `eval/retrieval_results/colab_full/`: **55 files / 4,736,423 bytes** before its explanatory
README was added. Per-question JSON is losslessly compressed; large chunks/vectors stay ignored.
Input SHA-256: `4a533beaded0125220e8f377c8b7d718bebd760c17034a083a7ee71998c51a2e`.

**Result:** evidence completeness is the strongest AST observation. Exact dense/header-on
all-question ≥90% completeness is Requests **0.842 AST vs 0.654 char**, delta **+0.188
[+0.085, +0.292]**; Express **0.825 vs 0.725**, delta **+0.100 [+0.000, +0.225]**. On Express
test it is **0.958 vs 0.792**, delta **+0.167 [+0.042, +0.333]**. Reference astchunk matches our
Requests header-on completeness (**0.842**, paired difference **0.000 [−0.038, +0.038]**).
No consistent ranking advantage or implementation superiority is established. Unequal budgets
prevent a clean causal attribution to AST boundaries. Tables cover the predeclared header-off
comparison as well, not only the stronger-looking contextual variants.

**Retrieval settings:** Requests AST/header-on MRR is **0.840 dense vs 0.781 default hybrid**;
Express is **0.434 dense / 0.494 hybrid / 0.599 BM25**. The default fusion is not universally
best. The 0.8 test-file penalty gives large apparent gains, but every gold target is implementation
source, so the study cannot measure damage to test/example questions. **Do not adopt it as a
production default.** Every strategy/header group selected it on dev, which shares that bias.
Different BM25 weights win on 16-question dev sets; keep the current default provisional.

**HNSW:** reported neighbour recall@1/5/10 is **1.0** for all **560 main query/configuration
cases**, on only **270–565 vectors per index**. This is a small-index plumbing observation,
not evidence of ANN accuracy at scale or a best-index conclusion.

**Reranking actually ran:** AST/header-on default hybrid → rerank20 all-question MRR is Requests
**0.781 → 0.880**, Express **0.494 → 0.769**. Test: **0.781 → 0.849** and **0.592 → 0.740**.
Both test improvement intervals span zero. Depths 20 and 50 are close, using shared top-50
scores; distinct top-20 latency was not measured. Recorded max-depth scoring-call means were
**2.346 s/query** (Requests) and **2.894 s/query** (Express), potentially including first-load
cost. Keep code reranking opt-in; do not claim the open question is untested or automatically settled.

Updated `eval/RESULTS.md`, report navigation and handoff with these limits. All intervals are
pointwise/exploratory; splits/header variants/repeated runs are views of the same questions.
The comparison stays unresolved: add about **10 test/example-target questions per repo**, retain
the original dataset as frozen evidence, and rerun mixed-target assessment. Matched budgets,
independent labels and an ANN scale check remain open. No production code/defaults or original
questions changed, and no staging/commit occurred. Local audit assertions passed; no additional
model run was required to inspect the valid archive. After the documentation/notebook-intro
update, both notebook test modules passed (**17 passed in 2.15 s**), maintained mypy scope was
clean (**36 source files**), and the notebook builder passed Ruff lint/format checks.

### 2026-10-25: Deep Phase 4 failure research

Added `eval/analyze_retrieval_failures.py`, a model-free audit of the original ZIP's exported
chunks and portable vectors, and `eval/PHASE4_FAILURE_ANALYSIS.md` with source-grounded cases,
primary research links and an ordered experiment plan. Diagnostics are saved compactly under
`eval/retrieval_results/failure_analysis/`. Reconstructed **14 variants / 560 question-variant
cases**; exact dense/BM25 top10 rankings and original metrics reproduce. AST reconstructed
exact-based fusion top10 matches all **80** saved AST/header-on rankings. One reference
astchunk/header-off case has a hybrid-order mismatch; original ANN@50 lists were not exported,
so the cause is unresolved. Reconstructed deeper pools are labelled accordingly.

**Diagnosis, AST/header-on:** Requests default hybrid finds some gold in **40/40** top10s,
Express **38/40**. Rerank20 finds some gold in **40/40** each; all ranges are ≥90%-covered on
**36/40 Requests** and **40/40 Express** questions. Exact dense top50 contains ≥90% of every
gold range on **all 80**. This motivates ranking/evidence selection before a database/model swap;
it is not ANN@50 or generalization evidence. Default fusion improves/ties/worsens first-gold
MRR on **3/30/7 Requests** questions and **13/24/3 Express** questions.

Express has **288 test + 52 example + 44 other chunks**; dense/hybrid/rerank20 ranks tests
first on **28/ 26/13** questions. Relevant tests compete with the labelled implementations.
The old set cannot assess test-target harm from a blanket penalty. `requests-034` has the
TooManyRedirects check/raise at rank1 but zero whole-function completeness because gold spans
122 lines; minimal required evidence needs a reviewed, versioned distinction from whole-function
coverage. Original gold has not been edited.

**Parity gaps:** all **384 Express production body/span pairs** match exported AST chunks,
but signatures appear in **53 production versus 7 evaluation chunks** (**21 versus 3** of the
43 lib chunks). The common header policy often loses ownership at comments/merged definitions.
No quality gain for production headers was measured. Eval supplies a code rerank instruction;
production's default reranker supplies the bug-to-commit instruction. Record this as an open
integration fix, not a cause of the valid code-instruction experiment's failures.

Cached-tokenizer audit: **0/5,356** exported chunk texts exceed 2,048 tokens; max **1,419**.
Tokenizer provenance is a local cached revision, not recorded GPU telemetry. No missing-query
instruction/padding/normalization bug was found. Fixed top10 budgets still differ in returned
text; matched-token context and budgeted expansion remain experiments, not measured fixes.

**Next:** full candidate/score/token logging and task instruction parity; reviewed minimal and
test/example gold with fresh held-out assessment; focused rerank20/union comparison; bounded
symbol context and expansion; model/dimension and ANN scale studies only as separately justified.
Phase 4 remains unchecked. No production defaults/source, original datasets or GPU results were
changed. No new dependencies, staging or commits. Verification results are recorded below.

**Verification:** the completed offline audit and optional 384-chunk source/header check passed.
Both existing retrieval/scoring/statistics test modules passed (**30 tests**, count confirmed by
collection). Ruff lint and format checks passed for the analyzer; mypy passed the maintained
**36-file** scope and the new analyzer separately (**1 file**). `git diff --check` passed.
Sandboxed uv cache persistence failed for the tooling check, so lint/type checks used the same
existing uv-managed virtual environment directly; the audit and pytest ran via offline uv.

### 2026-10-25: Implement the ordered v2 follow-up and prepare the GPU run

**Step 1:** production `HybridRetriever` now selects code/docs/commit reranker instructions
and caches scorers by instruction. An injected-scorer factory regression test verifies code
does not receive the commit prompt and switching tasks preserves distinct instructions.
Code reranking remains opt-in; existing weights and model choices are unchanged. Added an
independent reranker token cap with the same **2,048** default, plus out-of-timer token auditing.

**Step 2:** built `eval/datasets/phase4_v2/` from pinned source: **53 Requests, 53 Express,
23 Flask** questions (**129 total**, **49 newly authored**). Each repo contributes 10 new
test/example targets and 3 explicit-symbol questions; Flask adds 10 other implementation
questions. Requests has 10 test targets; Express/Flask each have 7 test + 3 example targets.
Flask is pinned to **d086db856be187255b8ec61ef409357393020f32**. Labels are source-checked by
the author, not independently reviewed. Some symbol targets reuse previously assessed code.

**Step 3:** reviewed all **8 Requests + 2 Express** incomplete cases. Required/supporting gold
is split for 001/024/034 with disjoint ranges partitioning the original gold; 007's line iterator
and 018's registration method become supporting. Other decisions/rationale are in the v2
manifest. Original v1 files remain unchanged. Preserved the matching original source ZIP at
`eval/retrieval_results/colab_full/phase4_v1_source.zip`, SHA-256
**51399ee184dc5955196db91ce7bb978c9ca2713e8993c3c65d9f2b33500416ed**. Old source-match claims
refer to import-time validation; current runtime intentionally includes the instruction fix.
The v1 builder now refuses to overwrite existing labels. The failure auditor checks the frozen
source snapshot when present.

**Step 4:** added the fixed-matrix runner (`eval/run_followup_eval.py`), complete compressed
JSONL traces, raw dense/BM25/fusion/pair scores, candidates, repetitions, separate required and
supporting coverage, v1-regression rescoring, real query-to-result timings, cold first-pool
timing, CUDA allocation and token/model revision diagnostics. IDs 028/036 in **both** original
repos get extra top50 pair scores. Primary latency calls independently encode/retrieve/rerank;
three repetitions rotate method order and are not counted as independent questions.

**Preselected experiment:** dense HNSW, hybrid (1.0 / 0.3, RRF 60), rerank20, and rerank20 over
a deduplicated 20-candidate pool reserving the first 5 BM25 hits. Primary policies use actual
production chunks/headers. V1's common-header study stays separate. Test penalty and explicit
source-derived symbol routing are diagnostics. No weights selected from old dev or fresh
preflight outcomes. Per-repo and pooled checklists require fresh win/target-type safeguards,
label review and explicit p50/p95 review; no code enables a production default automatically.

**Measured before/after regression, v1 AST/common headers/BM25, all 40 per repo:**

| Repo | MRR@10 before → after | Gold ≥90%@10 before → after | Changed top10 rankings |
|---|---|---|---|
| Requests | 0.569 → 0.569 | 0.633 → 0.633 | 0/40 |
| Express | 0.599 → 0.599 | 0.900 → 0.900 | 0/40 |

All per-question overlap/coverage metrics match exactly. This is lexical regression, not a
measurement of the corrected GPU reranker. Actual v2 BM25-only preflight on fresh questions:
Requests MRR **0.365**, required ≥90% **0.769**; Express **0.567 / 0.846**; Flask **0.545 /
0.826**. Test-target MRR is **0.250 / 0.524 / 0.126**, respectively. Saved summaries/provenance
and compressed question rows are in `eval/retrieval_results/phase4_v2_preflight/`. No GPU
performance or algorithm improvement is inferred from the different labels/header policy.

**GPU package:** generated `notebooks/phase4_followup_on_colab.ipynb` (**7,698 bytes**) and
`notebooks/phase4_followup_source.zip` (**103,939 bytes**), SHA-256
**d0605b605af2ff7d30681a09304560aa4f55cc3bcb22b9b36d58cf6919ebfceb**. It mounts Drive, verifies
the matching source ZIP, installs pinned dependencies with uv, checks out three pinned repos,
runs the matrix and pooled reports, and downloads `phase4_followup_results.zip`. No Base64
payload. Regenerated the existing source-dependent notebooks too; original runtime is frozen
separately. ZIP content matches current source/data/matrix after LF normalization.

**Verification:** **335 passed, 1 network test deselected in 116.85 s**. The earlier full run
caught two stale archive/notebook pairs after a source edit; regeneration and the final full
suite fixed both. Ruff lint clean, **107 files** pass formatting, mypy clean across **41 source
files**, and `git diff --check` clean. Real final BM25 preflight hashes match the packaged
runtime. No new dependencies, staging or commits.

**Pending steps 5–7:** local torch reports **CUDA unavailable**, so the new A100 matrix,
fresh reranker results, p50/p95 and final production decision have not run. The user can run
the ready notebook/Drive ZIP and return results. Update measured write-ups and tick Phase 4
only after that assessment; keeping reranking opt-in on inconclusive evidence is valid.

### 2026-10-25: Audit the returned v2 GPU run and complete Phase 4

**Input:** `phase4_followup_results.zip`, **4,609,914 bytes / 50 members**; SHA-256
`6f15971e3675ea887b397739bb4939f974ba59453b64b0eb3ad32b2f77ea75f1`. Validated archive paths,
duplicates, symlinks and CRCs, then extracted into ignored `data/phase4_followup_colab/`.
No code/instructions from the archive were executed.

**Provenance/reproduction:** all **38 source hashes per run**, dataset hashes and fixed matrix
match current code and the matching preserved v2 source ZIP. Pinned clean checkouts validate
all required/supporting evidence. Local production chunking exactly matches **274 Requests,
384 Express, 429 Flask** chunks. All **1,032 scored rows / 129 questions**, required/supporting/
frozen-v1 metrics, group-bootstrap intervals, paired comparisons and pooled numerical decision
checks reproduce. All **1,548 primary calls** are complete; lexical/fusion/quota and rerank
ordering reproduce from saved lists/scores; warm latency quantiles recompute from raw traces.
No primary top10 changes across repetitions. Neural inference and elapsed GPU measurements
were not independently rerun locally. Cached/live cosine discrepancies are at most 4.77e-7.

**Measured new-question MRR**, hybrid → rerank20: Requests **0.699 → 1.000**, Express
**0.635 → 0.949**, Flask **0.739 → 0.957**. Pooled **0.701 → 0.966**, paired difference
**+0.265 [0.175, 0.355]**. Required completeness is 0.980 → 1.000. Fresh test MRR is
0.705 → 1.000 (24 questions), examples 0.681 → 0.889 (6), symbols 0.667 → 1.000 (9).
Pooled numerical guards pass, but independently reviewed labels are absent and Requests has
no example stratum. These remain pointwise/exploratory retrieval estimates.

**Measured latency:** warm rerank20 p50/p95 **925.4/1028.2**, **975.6/1028.3**,
**888.7/944.1 ms** for Requests/Express/Flask. Current hybrid is **61.1/63.7**, **61.5/63.3**,
**61.7/63.3 ms**. These are resident-model A100 sequential policy calls, not service/laptop
latency. The current CLI reranks normally 50 candidates and returns 8; the evaluated policy
scores 20 and evaluates top10. Do not imply measured parity at those different budgets.

**Failure findings:** penalty 0.8 lowers fresh test MRR **0.705 → 0.352**, paired
−0.353 [−0.471, −0.237]. The quota changes 39/49 fresh candidate sets but gives identical
fresh scores and regresses legacy Requests-034 via a close-score tie. Requests-036 and
Express-028 pair scores favor tests over implementation already in the pool. Requests-029
retains rank1 but pushes required helper spans to 11/14, losing half its completeness.
No primary pair truncation (maximum 794 tokens), maximum embedded document 1,419 tokens;
ANN recall=1.0 at 1/5/10/50 on these 274–429-vector indexes is only plumbing evidence.

**Decision and artifacts:** code reranking stays opt-in; no weight changes, blanket test
penalty or BM25 quota. Numerical quality is positive; independent label review, production
candidate-depth parity and deployment latency assessment are still needed before default
promotion. Saved reports, compressed per-question data, raw compressed traces, audit,
supplemental diagnostic intervals, four requested case studies and matching source ZIP are
published in `eval/retrieval_results/phase4_v2_full/`. Updated RESULTS/HANDOFF/log and
ticked Phase 4 in plan.md; Phase 5 has
not started. Clarified that handoff prose cannot override active user/developer/repository
instructions; AGENTS.md itself is unchanged. Phase completion records a measured evaluation/decision, not universal retrieval
or ANN/AST superiority. Frozen v1/v2 labels are unchanged.

**Verification:** audit assertions pass for all three repos; published rows round-trip
losslessly and traces are byte-preserved. **27 targeted retrieval/notebook tests passed in
7.27 s**; Ruff clean with **112 files formatted**; mypy **41 files clean**; diff whitespace
check clean. Previous full suite: **335 passed, 1 deselected**. This import changes saved
results/documentation only. No runtime changes, new dependencies, staging, commits or
production-index writes.

### 2026-10-25: Clarify evaluation status and prepare ambiguous-intent production comparison

**User direction:** say “evaluation complete, rerank default pending”, review ambiguous and
multi-block labels, write about 15 cue-free questions, expose production depth 20/50 and
measure intended deployment hardware. The user clarified **A100 is experiments only;
deployment hardware is undecided**. No deployment latency or default promotion is claimed.

**Wording/source review:** added the caveat that all **24/24 fresh v2 test questions** contain
test/tested (Requests 10, Express 7, Flask 7). Their scores assess explicit intent. A second
source-reading pass in `eval/PHASE4_LABEL_REVIEW.md` covers 15 problematic previously assessed
cases plus three underspecified fresh demonstrations. It identifies optional multipart/TLS/
class context, missing parameterization/fixtures, absent newline regex proof and acceptable
test alternatives. This review is by the same assistant, **not independent/blinded review**;
no v1/v2 labels, review flags, datasets or measured scores were changed.

**New stress set:** `eval/datasets/phase4_ambiguous/` freezes **15 questions, five per repo**,
without test/tested, tutorial/example cues or explicit symbol identifiers. They deliberately
reuse inspected behaviours, so are post-hoc stress questions, not an independent fresh holdout.
Each has two alternative sufficient evidence sets; ranges within a set are jointly required.
Strict any_complete_proof@10 requires ≥90% of every range in one alternative, preventing
partial evidence from different alternatives from being counted as one complete proof.
Shared Requests header fixtures join two questions into one evidence group.

**Production setting:** added positive `rerank_candidate_depth` (env
REPO_RAG_RERANK_CANDIDATE_DEPTH), compatibility default **50**, and CLI `--rerank-depth`.
First-stage retrieval remains independently configured (default 50); scoring may use 20/50.
When needed, retrieval fetches enough candidates for the requested scoring depth. Reject
reranked output counts larger than scoring depth; do not mix scored and unscored results.
Explicit `--rerank` remains required; laptop auto mode continues to skip the reranker.
GPU opt-in remains available, but deployment/default selection stays pending.

**Measurements/checks:** reran all **129 v2 BM25 rows**: all scores/top10 rankings unchanged.
Fresh MRR/completeness before → after: Requests **0.365/0.769 → 0.365/0.769**, Express
**0.567/0.846 → 0.567/0.846**, Flask **0.545/0.826 → 0.545/0.826**. Controlled replay with
logged dense lists, real BM25 and recorded rerank20 scores reproduces all **129 hybrid and
129 rerank20** top10 lists and exact candidate order through the production path. This is
wiring validation only; cached/live ANN lower-order replay was not assumed identical across
rebuilt indexes. No new GPU inference was run. Actual 15-question BM25 stress preflight is
saved under `eval/retrieval_results/ambiguous_preflight/`, with source/dataset hashes.

**GPU package:** `eval/run_ambiguous_eval.py` invokes actual production search for hybrid,
rerank20 and rerank50, with three rotated warm repetitions, full dense/lexical/fusion/pair
traces, strict alternative-proof scores, paired CIs, model/source/dataset provenance and
per-repo p50/p95. It does not tune policies or promote defaults. Generated compact
`phase4_ambiguous_on_colab.ipynb` **5,615 bytes**, separate Drive ZIP **88,802 bytes**, SHA-256
`f34e3a4b0ddf168753de8f86fa4e7cf786db498193ed0b22f64b7fb8cde411a2`; uv install, no Base64.
Regenerated the other source-dependent notebooks/ZIPs; previously evaluated v1/v2 source
snapshots remain frozen with their results.

**Verification:** **341 passed, 1 network deselected in 120.30 s**; Ruff clean and **123 files
formatted**; mypy clean across **44 maintained files**, including the new runner/builder.
The first test collection caught a missing root import path in the new module; fixed before
the passing full run. Independent review, new A100 hybrid/rerank measurements and eventual
chosen-hardware timings remain pending. No new dependencies, staging, commits or production
index writes. Phase 4 evaluation stays ticked; Phase 5 has not started.

### 2026-10-25: Compare the existing rerank20 and rerank50 results directly

The user requested the depth comparison. Located both depths in the **original v1 A100**
Requests/Express experiments; the new ambiguous set still has BM25 preflight only and no new
result ZIP is present. Verified all **34 source hashes per run** against the frozen v1 ZIP,
unchanged dataset hashes, exact scorer/statistics source parity, and rescored **160 method/
question rows**. Stored means and bootstrap intervals reproduce; added direct paired group
comparisons for all/dev/test plus descriptive pooled rows, without new neural inference.

Original all-question MRR20/MRR50: Requests **0.8800/0.8779**, Express **0.7692/0.7679**.
Gold completeness20/50@10: Requests **0.925/0.900**, Express **1.000/1.000**. Paired 50−20 MRR
is Requests **−0.00208 [−0.00610, 0.00000]**, Express **−0.00125 [−0.00375, 0.00000]**;
all intervals include zero. Held-out test MRR is Requests **0.8486/0.8486**, Express
**0.7403/0.7382**. Fifty gives no observed MRR gain in this sample, not proof of universal
20-candidate superiority. Whole-gold scope remains a limitation of these old labels.

The frozen original runner scored a fifty-candidate pool once, then derived both rankings by
restricting the same scores. **No independent rerank20 latency or speedup exists in that run.**
The new prepared notebook calls production at 20 and 50 independently, so can measure actual
warm timing when the user runs it on experimental A100 hardware. Deployment remains undecided.
Compatibility depth50 and opt-in enablement remain unchanged; rerank default stays pending.

Published `eval/retrieval_results/rerank_depth_comparison/{README.md,comparison.json}` and
updated RESULTS/HANDOFF. Reproduction assertions pass for all 160 rows and stored intervals;
no runtime, dataset, dependency, staging or commit changes. Existing full-suite result remains
341 passed, 1 network deselected; no runtime tests rerun for this derived-data/docs change.

### 2026-10-25: Expand the depth comparison from 15 to 90 questions

The user requested a larger, better sampled comparison. Added a frozen v3 diagnostic set:
**30 questions each** on Requests, Express and Flask, **90 total**. The original 15 questions
and accepted proofs are retained unchanged under new IDs with origin links; **75 additions**
were authored/source-reviewed before inspecting expanded retrieval scores. Original v1/v2/15
labels and evaluated source ZIPs remain frozen. This is same-assistant source review, not
independent review or a random/independent production holdout. Old-v2 evidence overlap is
recorded per question; newly authored wording is not independent/unseen code.

The sample has **36 ambiguous** queries without test/example/identifier cues, **18 multi-block**
workflows, **18 explicit test** targets, **12 implementation** queries and **6 example** targets.
Requests has no comparable indexed examples directory. Evidence covers **58 repo/file pairs**;
shared intervals in any alternative proof connect **28/27/28 groups**, **83 total**. Independent
source anchors include parameter decorators and behaviour-specific fixture/helper assertions.
Author review corrected cookie-path/domain wording, an outgoing redirect-method assertion,
integer-expiry type scope and previously handled exception/teardown semantics before freezing.
Each of the 90 cases has at least one complete accepted proof somewhere in the full production
chunk corpus. Hashes/line ranges validate against clean pinned checkouts. Full-corpus coverage
is an answerability check and does not claim top10 success.

The fixed protocol locks models, token caps, query/task instructions, AST budget, weights,
depths, proof rules, bootstrap settings and repetition policy. The shared runner accepts
--protocol, records its digest, and independently executes hybrid, rerank20 and rerank50.
Three rotated repetitions yield **810 timed calls**, but quality uses pre-chosen repetition
zero. The new summarizer validates label/protocol/source/settings parity, row identity,
actual 20/50 pair counts and repetition-zero quality. It reports all90/added75/retained15,
per-repo/stratum views, direct paired 50-minus-20 intervals and separate per-repo latency.
Repetitions, cohort views and strata are not independent replications. Intervals remain
exploratory and pointwise; a CI crossing zero is not equivalence.

**Measured BM25 preflight**, after labels/protocol were frozen:

| Repo | Questions | MRR@10 | Complete accepted proof@10 |
|---|---:|---:|---:|
| Requests | 30 | 0.5009 | 0.5667 |
| Express | 30 | 0.6346 | 0.7333 |
| Flask | 30 | 0.3479 | 0.5333 |

Re-ran original15: **0 changed rankings, 0 changed metrics** versus the stored preflight.
Production retrieval/defaults are unchanged. Compact results and regression assertions are
in eval/retrieval_results/comparison_v3_preflight/. Expanded GPU quality/latency is unmeasured;
no labels or weights were changed after the BM25 run.

New next-run pair: **notebooks/phase4_comparison_on_colab.ipynb**, **7,664 bytes**, and
**phase4_comparison_source.zip**, **118,440 bytes**, SHA-256
`aa47cbf96a17c445be55726a85138475aee6183434ae25c940d45b943b7322f9`.
Uses uv and a separate Drive ZIP, no encoded source cell. Returned results include label/protocol
snapshots, source ZIP, candidate/pair traces and aggregate reports. Regenerated live older
notebooks/ZIPs for shared eval-source parity; evaluated v1/v2 snapshots remain unchanged.

**Validation:** full offline suite **350 passed, 1 network deselected in 94.44 s**; Ruff check
clean, **135 files already formatted**; mypy **47 maintained files clean**. Nine new test cases
cover frozen sample/old-proof preservation, cue constraints, overlap grouping, overwrite
protection, notebook/ZIP parity, direct paired comparison units and rejection of mismatched
depth/protocol/labels/quality. No new dependencies, staging, commits or production-index writes.
Updated RESULTS, HANDOFF, eval README and plan. Phase 4 remains evaluation complete, rerank
default pending; Phase 5 unstarted. Independent review, the new GPU run and eventual deployment
hardware/timing remain open. A100 is experiment-only; reranking stays opt-in.

### 2026-10-25: Write a standalone handoff for the expanded comparison

At the user's request, created docs/phase4_comparison/HANDOFF.md and linked it from the
main project handoff. It explains the project context, earlier findings and limitations,
why the old depth timing and explicit test cues require this follow-up, the frozen90 sample,
alternative-proof metrics, fixed hybrid/rerank20/rerank50 matrix, matching Drive files,
pending GPU status and the returned-result audit/decision workflow. Includes existing measured
baseline/validation numbers and labels the 20–40 minute A100 estimate as extrapolated.
Independent review and deployment hardware remain open; reranking stays opt-in. No code,
labels, protocol, notebook/source ZIP, production default or phase-status changes. Existing
350-test result is referenced as prior validation; no runtime tests were rerun for this
documentation-only change. No staging or commits.

### 2026-10-25: Implement the fair expansion test from the Codex brief

Read docs/CODEX_HANDOFF_EXPANSION.md, the main handoff, roadmap and current implementation.
The prior callee gain was post-hoc on saved lists, not sufficient adoption evidence. Implemented
the brief's fresh evaluation while preserving off-by-default production behaviour and old
labels/results/source snapshots. No Phase 5 work has started; rejection of the fresh primary
must lead there without another expansion tuning loop.

**Fresh labels before retrieval:** authored 30 implementation flow questions from pinned source,
six each for Requests, Express and Flask and twelve for new repo pallets/click at
2247b35ea1c47c727d7a06e51fa280e12a863ff6. They form 26 connected evidence groups. Each has
one sufficient proof with jointly required hashed ranges. Source/corpus validation confirms
every proof exists and none fits in one indexed chunk; production corpus counts are
274 / 384 / 429 / 700. Parameter/helper/delegation scope is recorded; external dependency
internals are not claimed. Same-assistant author review is not independent review. The manifest
records source overlap with prior cases, so newly authored wording is not a fresh-code guarantee.
Copied all 120 original intent-policy cases byte-for-byte as regression guards.

The manifest and protocol were frozen/hashed **before any fresh query retrieval**:
protocol aed33f2b62b61d0eeec45680141ea3cb7eb953733553a09ca5338d3d2a64c34f;
manifest 8ed5db211f49854fa38eb968ae08c637a34083caabf541cb78a1e39cbdb9bd01.
The brief's request to commit preregistration conflicts with the user's repeated no-commit
instruction; followed the latter and recorded an on-disk freeze instead. No staging or commit.
The authoring script refuses to overwrite the frozen version.

**Production option:** Settings.evidence_expansion accepts none/adjacent/callee/both, default
none, with keep/seeds/per_seed/max_definers in config (5/3/2/3). HybridRetriever applies it
only after requested/enabled code reranking, on the returned top-k list. Kept hits, relative
order and result count are preserved; additions respect repo/language/path/metadata scope.
VectorStore.repo_documents reads text/metadata without embeddings; the chunk index is lazy
and cached with query scope. forget_cached_indexes invalidates lexical and expansion caches
after a reused retriever's repo is reindexed. The default path adds no corpus read. Added
SearchHits are marked with expansion_mode, have no rerank probability, and display as
unscored related evidence in the CLI (numeric score zero is a placeholder).

**Frozen quality matrix:** plain rerank20, rerank20+callee primary, rerank20+both secondary,
150 questions / 450 outputs. Fixed original headers, one code prompt, plain P(yes), depth50
per signal, weights1.0/0.3 and RRF60. It traverses actual production search/store-backed
expansion; exact identical model pairs share scores as in the earlier quality-only policy
runner. No latency is measured. Returned snapshots include raw chunks, dense/BM25 lists,
twenty fused candidate probabilities, pre-expansion/final order, scores, text counts and
inserted/displaced IDs. Analysis independently rebuilds fusion/expansion, rescores every
metric and rejects incomplete or inconsistent source/model/data/settings/probability traces.

The pre-chosen callee rule requires fresh proof gain >=0.05 with paired lower 95% bound >0,
no old pooled stratum losing >0.02 MRR or proof (all five, including multi-block), and text
growth <=10% separately on fresh and regression sets. Text includes headers. Group-bootstrap
uses 10,000 draws/seed42. Both cannot replace a rejected primary, no default changes
automatically, and independent review remains outstanding. Incomplete/invalid runs do not
constitute rejection. A100 is experiments only; deployment remains undecided.

**Real local BM25 preflight**, after freezing:

| New questions | Count | MRR@10 | Complete proof@10 |
|---|---:|---:|---:|
| Requests | 6 | 0.7917 | 0.3333 |
| Express | 6 | 0.5208 | 0.5000 |
| Flask | 6 | 0.5889 | 0.6667 |
| Click | 12 | 0.5444 | 0.2500 |
| Pooled fresh | 30 | 0.5981 | 0.4000 |

Old 120 BM25 is 0.4776 MRR / 0.6500 proof, matching the earlier rounded 0.478 / 0.650 baseline.
Holding prior header-off twenty-pair pools/probabilities fixed, current default-none production
also reproduces **all 120 rankings and metrics** with no expansion-corpus read:

| Repo | Questions | MRR before / after | Proof before / after |
|---|---:|---:|---:|
| Requests | 42 | 0.8889 / 0.8889 | 1.0000 / 1.0000 |
| Express | 39 | 0.9188 / 0.9188 | 0.8718 / 0.8718 |
| Flask | 39 | 0.8419 / 0.8419 | 0.9744 / 0.9744 |

Published eval/retrieval_results/phase4_expansion_preflight/ with compact rows, provenance,
replay validation and before/after table. This is lexical and fixed-input wiring validation,
not new neural/ANN inference or timing. The real A100 expansion matrix has **not** run.

New matching workflow: notebooks/phase4_expansion_on_colab.ipynb **7,317 bytes**, separate
phase4_expansion_source.zip **158,096 bytes**, SHA-256
4e772036c1e5e6c5df9ffc40054d786ccd37d6bc75e244518b39e8d49eef3402.
Uses uv and Drive; no encoded source cell. All eight live notebook builders were regenerated
after source/data formatting; evaluated old snapshots remain intact. The source ZIP matches
all four saved preflight source hashes. Return phase4_expansion_results.zip for actual audit.

**Checks:** full suite **410 passed, 1 network deselected in 111.76 seconds**; Ruff check clean,
**169 files already formatted**; mypy **54 maintained source files clean**. Eighteen added test
cases check production insertion/kept-hit/filter/k/disable behaviour, cache invalidation,
repo-scoped metadata-only reads, frozen old-label identity, every success-rule boundary,
notebook/source parity, and the complete 450-output runner/audit with offline test models.
Synthetic test-model scores are not benchmark results or adoption evidence. The audit also
rejects deliberately changed proof scores and returned character budgets.

Updated RESULTS, main HANDOFF, original brief status, eval README, historical comparison
handoff and plan. No new dependencies, user-index writes, staging, commits or default
enablement. The real GPU result and independent label review remain pending.

### 2026-10-25: Fully reproduce the returned expansion run and hand off to Phase 5

The uploaded phase4_expansion_results.zip is **5,297,614 bytes / 38 members**, SHA-256
f3e745d316d2e5a509366eb43f8380ea4be017769184430cf99ded5d685585a2. Imported with path validation
into ignored data/phase4_expansion_colab/, preserving original reports; a separate copy under
data/phase4_expansion_audit/ regenerated the analysis without overwriting incoming evidence.

The returned source ZIP is byte-identical to the prepared 158,096-byte package, SHA-256
4e772036c1e5e6c5df9ffc40054d786ccd37d6bc75e244518b39e8d49eef3402. All **45 source hashes per
run** match both that snapshot and local files at import. All four runs used **NVIDIA
A100-SXM4-40GB**, embedding revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 and reranker
revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5, with the frozen settings and labels.

**All 450 quality rows/traces and aggregate JSON reproduce exactly**, including first-stage
fusion, twenty-pair parity, expansion reconstruction, complete metrics, returned characters
and displacement. Gold, pinned revisions, manifest/protocol hashes validate against clean
checkouts. Rebuilding production corpus reproduces all **1,787 chunk texts and metadata**
(Requests 274, Express 384, Flask 429, Click 700). All fresh proofs remain answerable and none
fits in one chunk. Cached vectors are finite/unit-normalized float32 at 512 dimensions.
No GPU inference was repeated locally; this is deterministic audit/reproduction, not new
independent model or label evidence.

| Cohort | Method | MRR@10 | Complete proof@10 |
|---|---|---:|---:|
| Fresh 30 | plain | 0.7659 | 0.6667 (20/30) |
| Fresh 30 | callee | 0.7611 | 0.5667 (17/30) |
| Fresh 30 | both | 0.7648 | 0.5667 (17/30) |
| Old 120 | plain | 0.8833 | 0.9500 (114/120) |
| Old 120 | callee | 0.8833 | 0.9667 (116/120) |
| Old 120 | both | 0.8833 | 0.9417 (113/120) |

A separate direct NumPy calculation independently reproduces primary fresh proof delta
**-0.1000 [-0.2424,+0.0345]**, 30 questions / 26 whole evidence groups, one gain/four losses.
MRR delta is -0.00476 [-0.01587,0]. Frozen >=0.05 gain and positive-lower-bound requirements
fail; all five old-stratum guards and text limits pass. Returned text grows **0.332% fresh /
0.712% old**, separately. The proof CI includes zero: reject adoption under the fixed rule
without claiming a universal statistically significant harm. Both is secondary and also loses
three net fresh proofs; it cannot replace the primary.

Fresh traces show unscored additions displacing a mean 3.73 ranked tail hits. Required ranges
for proxy settings, context object handling, parser construction and digest challenge fields
are lost in Express-002, Click-003/-005 and Requests-004. Click-001 gains a parameter-memo
helper; Express-005 loses its only overlap at rank 7. Case-study JSON records actual source
coverage and inserted/displaced IDs, including merged-chunk symbols that alone can mislead.
This closes the experiment; it is not grounds for tuning another expansion budget or policy.

Published compact full evidence under eval/retrieval_results/phase4_expansion_full/ with
frozen source/labels, compressed chunks/rows/traces, validation, case studies and reports.
Another summary/import was added concurrently under phase4_expansion_run/; preserved it and
merged the stronger provenance/reproduction details into RESULTS/HANDOFF rather than replacing
the concurrent work. Their numerical conclusions agree.

**Decision:** keep evidence_expansion=none, unchanged opt-in reranking and no further expansion
tuning on these inspected labels. Phase 5 is next; created docs/PHASE5_HANDOFF.md covering
local commit indexing, direct/via-PR linking, fixtures and leak-resistant bug-to-fix evaluation.
GITHUB_TOKEN is unset; local git/fixture development can proceed, API-backed linking needs it.
Phase 5 implementation has not started and its checkbox remains open. Independent label review,
generated answers and deployment hardware/latency remain unmeasured/open.

No runtime, defaults, labels, dependencies, production indexes, staging or commits changed.
No new runtime suite was needed for this report/data-only audit; prior validation remains
**410 passed / 1 network deselected**, Ruff clean, mypy 54 maintained files clean. Analysis
and source/corpus/hash assertions above passed on the real uploaded data.

### 2026-10-25: Build Phase 5 and freeze the real commit-retrieval comparison

Followed `docs/CODEX_HANDOFF_PHASE5.md`. Implemented local capped non-merge commit records,
old/new diff ranges and shared AST-definition attribution, both direct/via-PR link paths,
cached read-only GraphQL, typed repo-scoped SQLite history/link tables, idempotent commit
vector/BM25 indexing and `ask --bug` evidence with SHA citations. General answering remains
Phase 7. History-only index transfers preserve existing code/docs and their revision.
No dependencies added, no commits/staging/push, no Phase 6 expansion or include-tests work.

The user authorized read-only GitHub calls and configured GITHUB_TOKEN locally. Verified
Requests `d568f4727849` -> issue6949 directly and FastAPI `ad03e117c001` -> PR15077 -> issue15401
through the production linker. No token is in a URL, source ZIP, run settings or notebook.
The user also explicitly required open-source models: `llm_provider=local`, answer/summary
models unset until Phase 7; no Anthropic key or paid LLM call is needed.

Frozen before retrieval: protocol
`314ab8b93072dac058583f8f4c3054543a62c316828cf2d9afdd129a4762089c`, 329 issues / 319 connected
groups. Requests 141 (28 dev / 113 test), Flask 170 (34/136), FastAPI 18 (4/14). Corpus: 13,712
non-merge history records and 4,724 fix candidates. Approximate 200/repo targets were not
fabricated; FastAPI discovery and the 5,000-commit cap leave a small via-PR sample. Full
selection/skips and decompressed hashes are preserved. `verify_corpus` rebuilds every
document and every gold/group/split; issue text is never appended to searchable commits.

Real BM25-only test baseline (no neural inference):

| Repo / path | n | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Requests direct | 113 | 0.2655 | 0.4867 | 0.5575 | 0.3498 |
| Flask direct | 136 | 0.5735 | 0.7868 | 0.8456 | 0.6656 |
| FastAPI via PR | 14 | 0.5000 | 0.7143 | 0.7143 | 0.5952 |
| Pooled test | 263 | 0.4373 | 0.6540 | 0.7148 | 0.5262 |

Existing120-code BM25 regression, before -> after:

| Repo | n | MRR@10 | Complete proof@10 | Changed rankings / metrics |
|---|---:|---:|---:|---:|
| Requests | 42 | 0.4564 -> 0.4564 | 0.5952 -> 0.5952 | 0 / 0 |
| Express | 39 | 0.5965 -> 0.5965 | 0.7692 -> 0.7692 | 0 / 0 |
| Flask | 39 | 0.3813 -> 0.3813 | 0.5897 -> 0.5897 | 0 / 0 |

Prepared a compact uv/Drive-ZIP A100 notebook with an explicit model-download cell and
separate BM25/dense/hybrid/4B/8B processes at fixed depth50. Inputs are bundled; a Colab
token is unnecessary for them. Raw candidate and pair scores, source/model/input hashes,
group-bootstrap intervals, a fixed test-only 8B rule and real CLI citation are required.
Analysis reconstructs rankings/metrics and rejects incomplete or mismatched runs. Offline
fixture tests exercise the whole matrix and reject corrupted metrics and issue-text leakage.

Fixed two issues found by real preparation: GitHub explicit NOT_FOUND issue responses and
Unicode separators inside JSON strings (JSONL must split on LF, not Python splitlines).
The first full suite caught runtime BM25 files entering the source ZIP; moved runtime
indexes outside labels, added a declared-file allowlist and tested actual ZIP extraction.
Frozen JSON/JSONL and gzip bytes are preserved, including Windows line endings.

Results/source snapshots: `eval/retrieval_results/phase5_preflight/`; frozen inputs:
`eval/datasets/phase5_bugfix_v1/`; review: `docs/PHASE5_BUILD_REVIEW.md`.
**Final validation:** 444 passed, 1 network test deselected in 611.20 s (10:11). Ruff lint and
format checks are clean; mypy checks 65 maintained files. Actual Drive-ZIP extraction and
all frozen input hashes pass; all 53 source hashes/run match the measured source ZIP and
local files. Raw final-suite XML is preserved in the preflight artifacts.
All notebooks were regenerated after source/config changes. The measured source snapshot
and raw per-question files allow independent reproduction. Older research scripts remain
outside the default mypy scope; the new maintained Phase 5 scripts are included.

**Not measured:** real dense/reranker inference, 4B versus 8B commit gain, generated answers,
manual scale memory/time, deployment latency or independent label review. Link-derived
gold/current issue bodies and possible old T3 overlap limit generalization. A100 is for
experiments only. Phase 5 remains unchecked until its remaining acceptance work passes.

### 2026-10-26: Correct Phase 5 after review and freeze v2 before GPU inference

Addressed `docs/PHASE5_REVIEW_FINDINGS.md` without touching frozen v1 or Phase 4 results.
The shared scrub now removes closing-reference phrase spans, retaining adjacent descriptions
in messages and diff lines. Added Issue N, Sphinx role/anchor handling and tests that ordinary
numeric code is retained. Email-like strings are scrubbed from searchable diff text too.

Re-froze `phase5_bugfix_v2` using hash-verified v1 records, labels, groups, splits and pins:
13,712 records, unchanged 4,724 candidate documents and 329 queries / 319 groups (263 test).
Queries, gold, splits, raw records, issue/link and selection evidence remain byte identical
across versions. No live GitHub calls. This corrects the benchmark; it is not a fresh holdout.
Every gold qualifies from the message heuristic alone, independently of links. Verification
also checks stored flags against the heuristic, so link promotion cannot hide in the corpus.
The heuristic itself remains broad and differing pool sizes accompany every report table.

The benchmark is explicitly own-linked-fix lookup. Added secondary File Hit@5 (any top-five
commit shares an old/new changed path with any gold, exact gold included). It is a localization
proxy, not proof of a useful similar fix. Added full-query versus 2,000-character BM25
sensitivity and query-length reports; neural token budgets remain different. Primary exact
Hit/MRR remains unchanged in definition. The fixed six-method GPU matrix includes these two
BM25 rows, dense, hybrid, 4B and 8B at the same depth 50, without retuning weights.

V2 keeps the pooled 8B MRR >=+0.02, positive paired CI lower bound and >=20 test-group gates.
Per-repo/link-path Hit@5 guards now require >=30 connected test groups. Smaller strata are
reported but do not veto and establish no safety conclusion (FastAPI has 14 test queries).
Protocol and inputs were hashed before v2 retrieval; no Phase 5 GPU result has been seen.

**Measured test BM25, full queries, v1 -> v2:**

| Repo | Candidate pool | n | Hit@5 before / after | MRR before / after | File Hit@5 after |
|---|---:|---:|---:|---:|---:|
| Requests | 1,101 | 113 | 0.4867 / 0.4867 | 0.3498 / 0.3683 | 0.8142 |
| Flask | 1,181 | 136 | 0.7868 / 0.8235 | 0.6656 / 0.6995 | 0.9412 |
| FastAPI | 2,442 | 14 | 0.7143 / 0.7143 | 0.5952 / 0.5595 | 0.7857 |

- all: MRR delta +0.0236, paired 95% CI [+0.0116, +0.0372] (263 questions, 254 groups).
- fastapi/fastapi: MRR delta -0.0357, paired 95% CI [-0.1071, +0.0000] (14 questions, 14 groups).
- pallets/flask: MRR delta +0.0339, paired 95% CI [+0.0159, +0.0555] (136 questions, 130 groups).
- psf/requests: MRR delta +0.0185, paired 95% CI [+0.0047, +0.0355] (113 questions, 110 groups).

This is the complete formatter change, including additional reference/email scrubbing, not
only the reviewer's phrase-only patch. FastAPI's ranking loss remains visible. Raw rows,
traces, source hashes and paired intervals live in `phase5_preflight_v2/`; older v1 results
remain under `phase5_preflight/`. The 120-question code BM25 check again has 0 changed rankings
and 0 changed metrics: Requests MRR 0.4564/proof 0.5952, Express 0.5965/0.7692, Flask
0.3813/0.5897. It is not a new neural measurement.

SQLite redacts nested commit/issue payload text on write and legacy typed reads. A real-history
check caught an implementation error: replacing addresses in serialized JSON can corrupt an
escaped newline directly before an address. Corrected this to walk text values before JSON
serialization and extended the regression test. Existing raw database files are not migrated
silently; reindex active rows or create a fresh DB. Frozen public source records remain raw.

Regenerated all notebook builders after the final source changes. Current artifacts are
`phase5_bugfix_v2_on_colab.ipynb` and matching `phase5_bugfix_v2_source.zip`; old v1 artifacts
remain unchanged. V2 verifies bundled frozen inputs before explicit model downloads, uses uv,
and needs no token or paid LLM key. Current code refuses v1 corpus reconstruction; use v1's
archived source snapshot. Documented stdlib urllib as the actual cached linker, httpx as the
Phase 5b bulk-sync target, and the >=one association lookup/commit cold API cost and cache reuse.

Final validation: **452 passed, 1 network deselected in 323.20 s (5:23)**; Ruff lint/format
clean (**204 files**), mypy **66 maintained files** clean. Final validation and package
hashes are recorded in `phase5_preflight_v2/validation.json`. All **54 source hashes**
match every measured run and both source ZIPs; **51 v1 artifact files** plus the old
notebook/ZIP remain unchanged. Real new DB writes and typed reads for all **13,712 commits**
contain zero email-pattern matches in decoded payloads. Final notebook ZIP SHA-256:
`6dbf77d83e9e5e17ff64a27258b1b5e05b645eecb084f6dfe65fa7f3b28a2da9`.
GPU inference, real GPU CLI sample, manual scale, independent labels, useful similar-fix
judgements and generated answer quality remain pending. No default change or Phase 5 tick;
no staging, commits or pushes by this work.

### 2026-10-26: Build the pre-registered commit depth/weight comparison

Implemented `docs/CODEX_HANDOFF_COMMIT_DEPTH.md`, keeping every original v1/v2 dataset and
result file unchanged. The original v2 results ZIP was located in Downloads and copied into
an ignored audit workspace; no GitHub calls or new models were needed locally. Reconstructed
all 329 x 6 historical rows, original BM25/RRF/pair scores, settings/model revisions and all
54 historical source hashes. The 8B rule passes narrowly: test MRR +0.0244885, paired 95% CI
[+0.0029874,+0.0461713], same Hit@5 as 4B. This is verification of an existing GPU run, not
new inference. The 208-source subset (103 Requests, 93 Flask, 12 FastAPI) reproduces; the
other 55 test questions have docs/changelog-only gold.

New `phase5_commit_depth_v1` protocol SHA-256:
`87181ddde8e33032ffa0d83f35f68d9c0f1925410116a271abbb0e163f693d17`.
Written and frozen before the new experiment. Freeze covers unchanged v2 input hashes,
source/timing cohorts and archived old 4B/8B controls. Historical source ZIP was compacted
without changing any of its 54 source file bytes; the original downloaded ZIP is untouched
and identified separately by SHA-256. No original gold, query, split or source record changed.

`eval/retrieval/commit_depth.py`, `eval/run_commit_depth_eval.py` and
`eval/analyze_commit_depth.py` implement separate first-stage weight selection and depth
quality. Depth uses actual production hybrid top200 per signal, weights 1.0 / 0.3, one
200-pair scoring call/query/judge, then stable probability-descending/fused-order prefixes
50/100/200. All first-stage and pair scores are stored, and every prefix/metric rebuilds.
8B is primary; 4B secondary. Historical depth50 has retrieval_k=50, so it is reported apart
from the new fixed-200-list prefix50 baseline. Pool recall is reported at all three depths.

Weights 0/0.1/0.3/0.5/1.0 use first-stage retrieval_k=50. Select from 66 dev queries only,
retain 0.3 unless the best alternative gains >=0.01 MRR. Historical saved-list CPU replay
selects **0.1**: dev MRR **0.756728** versus **0.733748** for 0.3, gain **+0.022980**.
This is not fresh GPU retrieval and does not change defaults. The notebook runs its own
first stage/selection; if nondefault, fresh 8B depth50 calls compare it with 0.3. Test views
are descriptive, never select. No joint new-weight/depth200 result is claimed.

The fixed depth rule follows the brief exactly: test Hit@5 >=+0.02 with positive paired
lower bound; MRR delta >=-0.005; source-subset Hit@5 delta >=0; no repo loss >0.02 with >=30
connected test groups. Prefer 100 if both pass unless 200-100 Hit@5 >=+0.01 with positive
lower bound. Small strata are descriptive; passing only makes a setting review-eligible.
Production config and retrieval are unchanged.

Timing is separate ordinary `HybridRetriever.search` calls (no tracing wrapper in the timed
path), CUDA-synchronized, 3 rotated warm repetitions over 60 frozen test IDs: Requests 23,
Flask 23, FastAPI 14. Round-robin order avoids omitting a repo. All timing depths retain
retrieval_k=200, matching quality prefixes; report per-repo p50/p95 and ratios, never average
percentiles. Those ratios are against prefix50 at retrieval_k=200, not historical production
retrieval_k=50. Shared quality scores are not independent depth latency measurements.

The CLI sample now saves the exact bug report as JSON and beside its retrieved evidence.
New Drive ZIP/notebook use uv, explicit model downloads/revision checks, offline model caches
after download, and per-completed-stage matching-source Drive checkpoints. Source ZIP is
28,592,786 bytes; notebook 13,228 bytes at generation. All notebook builders were regenerated
last after source/config changes. No new dependencies were added.

Eight new test cases cover score-prefix isolation/ties/ceilings, dev-only hysteresis and test
leakage, rule/source/repo guards, frozen byte preservation, exact CLI report capture, and a
full fake-model production/Chroma matrix with rotated timing calls and tampered-prefix
rejection. Focused suite: **7 passed in 43.64 s**. Fake scores/timings are software checks,
not neural measurements. Original120 code BM25 rerun again has **0 ranking/metric changes**:
Requests MRR/proof 0.4564/0.5952, Express 0.5965/0.7692, Flask 0.3813/0.5897.

Current review entry point: `docs/COMMIT_DEPTH_BUILD_REVIEW.md`. Historical reconstruction,
cohort tables, stored-list replay, code regression and final checks are saved under
`eval/retrieval_results/phase5_commit_depth_preflight/`. New GPU depths/weights/timings,
answer quality, useful similar historical fixes, independent labels, manual scale, laptop
latency and other repos/languages remain unmeasured. Phase 5 stays unchecked, with no default,
staging, commit or push by this work.

Final validation: **460 passed, 1 network deselected in 351.26 s (5:51)**; Ruff lint/format clean (**217 files**), mypy **69 source files** clean. All **57 current source hashes** match the new ZIP. All **113 original v1/v2 dataset/result files** are unchanged. Production config/retriever/CLI hashes are unchanged. Package hashes and checks: `eval/retrieval_results/phase5_commit_depth_preflight/validation.json`.

### 2026-10-26: Release the Colab runtime automatically after the depth run

At the user's request, the commit-depth notebook now ends with an automatic runtime-release
cell. It compares the local and Drive results ZIP hashes, flushes pending Drive writes with
a two-minute timeout, then calls Colab's `runtime.unassign()`. Save/hash/sync failures stop
shutdown so the local results remain recoverable. The ZIP remains available in My Drive even
if the browser download has not completed when the runtime disconnects. This runs only after
successful completion of the preceding cells; it is not a watchdog for earlier failures.

The notebook was regenerated after builder changes. The source ZIP, frozen protocol,
datasets, production defaults and phase status are unchanged. Checks execute the last cell
with a simulated Colab runtime, covering successful release, mismatched results and failed
Drive sync; no live Colab session was disconnected here.

Validation: **4 passed, 7 deselected in 45.27 s**; changed builder/test Ruff lint and
format checks clean. Generated notebook: **14,541 bytes**. Source ZIP remains **28,592,786
bytes**, SHA-256 `869432f8c4947f16bbc400ff7d73e90a7fcfe83fb27fef02ac54474338bcd826`.
Updated artifact hashes and targeted checks are in the preflight `validation.json`.

### 2026-10-31 - Architecture summary experiment: frozen inputs and implementation

- Implementing `docs/CODEX_HANDOFF_SUMMARIES.md` in order; no staging, commits, push or new dependency.
- Q0: `eval/datasets/architecture_v1/` contains 15 source-grounded architecture questions each for
  Requests and Flask, with 5 dev / 10 test questions each and repo-qualified topic groups kept in one split.
  Every gold file has an AST/source anchor, line range and SHA-256. Labels are **unreviewed by a person**.
  The protocol, both sets and a byte-identical copy of the 40 Requests code questions were frozen/hashed
  before architecture retrieval. Requests pin `611c6162`; Flask pin `d086db85` matches its existing dataset.
- E1/E2: `eval/run_architecture_eval.py` uses production hybrid retrieval, file-level recall and fixed
  paired group bootstrap (10,000 samples, seed 42). Router scope is a single union of paths applied to
  dense and lexical candidates, with bounded dense overfetch and directory-boundary matching.
  Choices use dev only. `eval/analyze_architecture.py` independently re-scores saved rankings and gates.
- F1/F2/F3/F4 are opt-in summary facts: all definition docstrings, static raises/constants/config names,
  ancestor-only commit/merged-PR titles through typed read-only SQLite, and matching documentation sections.
  Baseline cap remains 3,000; experimental facts use the protocol's 16,000-character cap. Both character
  and 2,048-token embedding truncation are audited. Default code search continues to exclude summaries.
- Existing cached vectors were verified against regenerated production chunk texts: 274 Requests and
  429 Flask code chunks. Frozen history has 4,883 / 3,829 commits respectively; the existing Flask PR
  snapshot has 500 PRs. Requests has no synced PR snapshot for this experiment. These counts describe
  available snapshots, not complete PR-to-file coverage; titles require an indexed merge SHA and checkout ancestry.
- Environment issue: sandbox parser cache was empty, so the first attempts fell back to text splitting;
  corpus validation stopped the run before scoring. A writable copy of the existing six grammar DLLs and
  offline Hugging Face assets resolves this without downloads. During implementation, all-definition
  traversal was restricted to F1 so the baseline's existing extraction stays unchanged; final runs must
  use that corrected baseline. The frozen labels/protocol were not changed.
- E3 scaffolding: facts-only prompt in `agent/prompts.py`, lazy transformers summarizer, pinned model
  proposal, strict whole-identifier check, raw output cache and per-paragraph rejection audit. Validation
  runs on cached prose too. This measures lexical grounding, not semantic entailment. The proposed
  Qwen3-4B-Instruct-2507 is 4.0B, Apache-2.0; estimated 8.05 GB bf16 weights, plan 12-16 GB VRAM.
  Model revision `cdbee75f17c01a7cc42f958dc650907174af0554`. User approval is pending; no prose model run.
  Source: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 . Final results/validation follow below.

### 2026-11-01 - Architecture E1/E2 complete: no adoption; E3 prepared and approval-gated

- **Frozen inputs unchanged:** 30 architecture questions (15 per repo, 5 dev / 10 test each),
  source anchors/hashes and the byte-identical 40-question Requests code regression set.
  Labels remain **unreviewed by a person**. `docs/ARCHITECTURE_LABEL_REVIEW.md` exposes every
  question and flags the Flask CLI/server evidence concern; v1 labels were not changed after retrieval.
- **E1 result:** dev selects the top-two-directory router. Pooled architecture test Recall@5
  falls from 0.8292 (chunks) to 0.7042, paired delta -0.1250 [-0.2540, +0.0079]. Per-repo
  deltas are Requests -0.0083 and Flask -0.2417. Code MRR@10 falls from 0.8375 to 0.7667
  (-0.0708). It fails every adoption guard; summaries remain outside default code search.
- **E2 result:** every fact variant has the same routing dev scores (Recall@5 0.825,
  MRR@10 1.000), so the frozen tie order selects **F1** for E3. Routing test/code scores
  are unchanged. Docstrings improve the file-summary arm's test Recall@5 from 0.6875 to
  0.7833. The largest fusion Recall@5 gain is only +0.0333. F1 reaches 0.8625, with paired
  delta +0.0333 [-0.0351, +0.0909] and code MRR 0.7875. F1+F2 reaches the same Recall@5,
  with paired delta +0.0333 [-0.0702, +0.1053] and code MRR 0.8146. Both miss the +0.05 gain, positive
  lower bound and <=0.01 code-drop requirements. **No variant qualifies.** Full standalone,
  cumulative, per-repo and pooled tables are in `docs/SUMMARIES_BUILD_REVIEW.md` and
  `eval/facts_results/architecture_v1_hybrid/`.
- **Availability and limits:** F3 has commit subjects for 36 Requests / 80 Flask files and
  merged-PR titles for 0 / 9 files; Requests has no synced PR snapshot for this experiment.
  The shallow Flask source checkout was left unchanged; ancestry comes from the existing
  full clone at the source pin, including all 3,829 indexed commits. The baseline character
  cap truncates one summary per repo; no rich variant reaches 16,000 characters. The
  all-four-facts variant exceeds the 2,048-token embedding limit in one summary per repo.
- **Reproduction:** both BM25 and hybrid reports' 32 result rows, dev choices, intervals and
  gates recompute exactly from saved rankings. A final hybrid replay against the completed
  source reproduces all 32 rows and the F1 selection. Current source hashes, frozen input
  hashes, clean source checkouts, capsule hashes and zero email matches in summary texts are
  checked in `eval/facts_results/architecture_v1_hybrid/validation.json`. Existing code vectors
  are reused (274 Requests, 429 Flask); Flask's vector cache identity is independently
  reconstructed from its original texts, model/settings and embedding source hash.
- **E3 prepared, not run:** `notebooks/summary_prose_on_colab.ipynb` is self-contained with
  `summary_prose_inputs.zip`, SHA-256
  `a9db3fd741694bc6822e1eeea3fc2d190a0f046dfb9aaca25c307ef1a4484c5e`.
  It uses dev-selected F1 facts, cached code/query vectors and the proposed pinned
  Qwen3-4B-Instruct-2507 (4.0B, Apache-2.0; estimated 12-16 GB VRAM). Approval is deliberately
  unset. Every identifier, including Unicode and qualified names, is checked mechanically;
  cached raw prose is rechecked, violations drop the whole paragraph, and the raw audit,
  cache hits and actual GPU/runtime details are saved. No model judge. Lexical grounding
  does not establish semantic entailment. No summary model has been downloaded or run.
- **Validation:** full offline suite **596 passed, 1 deselected in 703.96 s** (includes the
  new notebook and capsule checks); Ruff clean, **283 files already formatted**; mypy clean
  for **84 maintained files plus 5 new evaluation entrypoints**. The sandbox blocks Git's
  `file://` helper with an MSYS named-object error and invalid-handle cleanup; the 21 local
  Git fixture tests and then the entire suite passed in the normal Windows environment.
  No tests were skipped for that issue. All notebooks were regenerated; an in-place ZIP
  write rejected by Windows was replaced atomically within the notebook directory.
- **CPU diagnostic only:** one 3,995-character summary took 40.311 / 31.737 / 21.039 s with
  1 / 2 / 4 threads, respectively; retained four threads and cached document batches of four.
  This is not an indexing-throughput benchmark. No dependency added, staging, commit or push.
  Phase 6 stays unchecked: human review, approved E3 execution, `find_callees` and the coherent
  architecture-answer acceptance check remain open.

### 2026-11-01 - Phase 7 answer bakeoff: freeze protocol/inputs and propose models

Read HANDOFF, the Phase 7 bakeoff brief, plan §3.6–3.9/Phase 7 and the summaries review/model
decision entries. Kept the separate approved summary experiment untouched. Created the
stdlib-only `eval/prepare_answer_bakeoff.py`; it writes one immutable v1 freeze or verifies
input, original-dataset, preparer and locally available probe-source hashes. Verification
reports **130 questions**: facts **15/45**, code **16/24**, architecture **0/20**, probes
**2/8** dev/test. Fact types do not overlap; the original code group split is preserved.
Ten probes were authored from pinned source inspection before candidate research/model
loads. Probe and architecture labels remain **unreviewed by a person**. Facts retain the
existing raw GitHub/git labels. No hidden labels may enter model prompts.

`eval/datasets/phase7_bakeoff_v1/protocol.json` is frozen with five input files and four
original dataset hashes. It fixes strict JSON/no retry, four calls, common dev-only prompt
tuning, exact tool-trace scoring, citation range checks, abstention phrases, Wilson intervals,
test gates and tie order. Citation validity and lexical grounding do not prove semantic
correctness; missing citations remain a separately reported limitation, not an extra gate.
Review details in `docs/PHASE7_BAKEOFF_PROTOCOL_REVIEW.md`.

After this freeze, inspected public Hugging Face metadata/cards/configuration/templates
for eight models; fetched no weights or vocabulary files and loaded no model. Proposed
four exact-revision Apache-2.0 candidates in a separate frozen proposal: Qwen3-4B-Instruct-2507,
Olmo-3-7B-Instruct, Qwen3-14B, Qwen2.5-Coder-14B-Instruct. The proposal labels all VRAM as
analytical estimates, with measured fields null. Olmo adds a newer independent family with
documented Transformers >=4.57.0 support; installed version is **4.57.6**. Newer Qwen3.5
4B/9B and Ministral 3 8B were screened but need runtime/dependency changes beyond the current
setup; Qwen3-8B was omitted under the four-model cap. These are scope exclusions, not quality
results. Full IDs, parameter counts, revisions, licences and source/template hashes are saved.

**Approval boundary:** stop for user approval of exact revisions. No harness, fake-model
tests, Colab notebook, model download/run or answer-model selection yet. No new dependency,
`src/repo_rag` change, summary-experiment change, staging, commit or push. Notebook regeneration
is unnecessary without a source change. No model quality/latency/memory results exist at
this checkpoint.

**Validation:** `uv run --offline pytest -q` with the existing non-network default, writable
parser/uv caches and a fresh data-directory basetemp completed successfully. JUnit reports
**599 tests, 0 failures, 0 errors, 0 skipped, 706.988 s** (all 599 passed); raw report/log:
`data/phase7_protocol_pytest_20261101.xml` / `.log`. The Windows sandbox blocks the existing
local Git fixture helpers, so this offline run used approved execution outside the sandbox.
`uv run --offline ruff check .`: **All checks passed!** `uv run --offline mypy`:
**Success: no issues found in 85 source files**, including the new preparer now added to the
normal type-check scope. Preparer formatting passes; both input/proposal freezes verify,
all three checkout HEADs match pins, and attempting to re-freeze correctly refuses overwrite.
Changing mypy scope triggered uv's editable-project launcher rebuild; a sandbox permission
error was repaired by reinstalling only the existing local project outside the sandbox
(`uv sync --offline`), with no dependency additions/upgrades. Final checks above pass.

### 2026-11-01 - Phase 7 v2 answer bakeoff implementation (models not run)

First ran `uv run python eval/prepare_answer_bakeoff_v2.py verify`; after selecting the writable
uv cache it passed. Preserved both preparers and all v1/v2 frozen files. Continued under the
user's approval of all four exact revisions, with no new dependency or local model load/download.
V2 has **141 questions: 44 dev / 97 test; facts 26 / 45**. Its test questions, labels and gates
are unchanged from v1; the hard dev patterns are now covered with the disclosed entity overlap.

Built a question-string-only controller, one common prompt in `agent/prompts.py`, strict JSON
parser/schema validation, four-call cap and complete-record result budgets. Existing typed
read-only metadata/Git/symbol/hybrid tools are wrapped without exposing SQL or hidden labels.
Model output is redacted before logging/model exposure; redaction preserves escaped decorators
and malformed outer syntax rather than repairing parsing. Added offline scripted-model tests
for leakage, exact answers without traces, parse failures/no retry, call limits, budget/citation
scope, release chains, abstention, raw recomputation and frozen-input tampering.

Prepared `answer_bakeoff_inputs.zip`: **274 Requests / 429 Flask code chunks**, existing vectors,
symbol tables, raw input hashes and redacted metadata reconstructed from frozen GitHub + pinned
Git (**1,482 non-merge commits**, not expected labels). Dev-only typed-tool audit: **26/26** pass;
this is setup validation, not answer-model accuracy. Raw Git cannot be redacted while retaining
SHAs, so the capsule carries a three-repo pinned public clone/tag recipe; future tags are excluded
and only redacted typed outputs enter prompts. SQLite is rebuilt/vacuumed and its logical text
checked; serialized/binary false-positive email matches are not treated as real addresses.

**Negative setup result:** auditing the real bound tools through serialization and the frozen
result budget passes **25/26** dev facts. `issue_author:1428` has an oversized full issue record
that is omitted under the 12,000-character cap, making its required author field unavailable.
The direct typed-tool audit's 26/26 excludes that context budget. No model ran; no test labels
were scored to inspect analogous cases. Preserved v2 without trimming fields, removing labels,
increasing the budget or relaxing gates; this known dev miss remains visible in review.

Before final checks, tightened the data identity: the index manifest hashes metadata, frozen
Git/tag recipe and every code text/vector payload; runtime rejects a changed database even
if its capsule checksum is updated. The freeze also binds both input manifests. The focused
suite including this tampering case reports **39 tests, 0 failures/errors/skips, 18.290 s**
in `data/phase7_payload_binding_final.xml` (software checks only). Final full suite is rerun
after this guard; no scoring threshold or frozen dataset changed.

Generated `notebooks/answer_bakeoff_on_colab.ipynb`: one 14B dev-code question logs peak GPU
allocated/reserved memory before the matrix; greedy bf16 native non-thinking templates on one
A100 40 GB, no quantization/framework. User runs it. Dev supports the initial common prompt plus
at most two shared revisions, all four candidates required. The freezer recomputes dev scores,
records prompt/parser/source/index hashes before test, and rejects changed/incomplete artifacts.
`eval/analyze_answer_bakeoff.py` rebuilds dev choice and every test table from raw turns, with
exact tool traces, Wilson intervals, lexical/citation caveats and explicit failed gates. Cached
scores/finals are ignored. Test is locked until that freeze; no tuning after unlock.

Added the six evaluation modules to normal mypy scope. A local-variable rename in `run_facts.py`
fixes its newly reached type error without changing `matches()` or behavior. Regenerated all
**17 notebook builders**, including source ZIPs and the embedding benchmark, after source and
project-file changes. Portable tests unpack the actual capsule away from the workspace and
verify both frozen versions/import the CLIs without model downloads. Review details and commands:
`docs/PHASE7_BAKEOFF_BUILD_REVIEW.md`. Initial source identity is saved in
`eval/facts_results/answer_bakeoff_build/implementation.json`; final dev selection awaits Colab.

**Not measured:** model quality, A100 latency/throughput/VRAM or any winning model. No production
model/search default or separate summary experiment changed; full Phase 7 stays unchecked.
No project staging, commit or push.

**Final validation:** `uv run --offline pytest -q` with a fresh writable basetemp and JUnit
output reports **638 tests, 0 failures, 0 errors, 0 skipped, 827.627 s**; all 638 passed.
Network tests remain excluded by the existing configuration. Raw evidence:
`data/phase7_full_20261101_03.xml` / `.log`. Existing local Git fixtures require execution
outside the Windows sandbox; this approved run retained offline model/network settings.
`uv run --offline ruff check .`: **All checks passed!** `uv run --offline mypy`:
**Success: no issues found in 95 source files.** Final v2 verification passes, including v1
supersession and unchanged test/gates. The saved initial implementation identity and generated
notebook match current source. The pre-existing staged snapshot still has 24 files; no stage
operation was performed. These are software/setup checks, not model quality or GPU results.

### 2026-11-01: Phase 7 bake-off v3 (bounded PR/issue body), before any model run

Review of Codex's harness found a budget flaw. In the capsule database, **35 of 500** httpx
PR records and **10 of 500** issue records exceed the 12,000-character per-question tool-result
budget when returned whole; the excess is almost all the free-text `body`. The v2 context policy
then drops the whole record, so author, merged_by, merge_commit_sha and state disappear for
every model. Codex had already measured it on dev (bound-tool audit **25/26**, `issue_author:1428`).
The test gate allows four misses in 45 facts, prompt wording cannot fix it, and a larger budget
would hit the 8,192-token input cap. No registered fact reads a body.

At the user's request ("yea u do it") I added **protocol v3**: `eval/datasets/phase7_bakeoff_v3/`,
frozen by `eval/prepare_answer_bakeoff_v3.py` (freeze/verify; verifies v1 and v2 first).
Questions, labels, splits (byte-identical copies of v2), gates, scoring, generation limits, the
12,000-character budget, timing and the four approved models are unchanged. The only difference
is `tools.record_projection`: in `get_pr`, `get_issue_meta`, `find_prs` and `prs_closing_issue`
a `body` string over **2,000 characters** is cut and marked
(`...[body truncated: N more characters]`) before redaction and the budget
(`bakeoff_tools.bound_bodies`). Measured after the change: **0 of 500 PRs and 0 of 500 issues
omitted**; largest wrapped record 6,422 chars (PR) and 2,699 (issue). v1/v2 files and both
hash-checked preparers are untouched. `answer_bakeoff_io.py` now verifies v1, v2 and v3 and
reads v3; the capsule was rebuilt (metadata.db hash unchanged, index manifest changed because
it binds the protocol), the Colab notebook uses `repo_rag_answer_bakeoff_v3` on Drive, and
`pyproject.toml` mypy scope gained `eval.prepare_answer_bakeoff_v3`. All notebook builders were
regenerated. No answer model was loaded. Test labels were not opened for this change.

At the user's request the answer-bakeoff notebook no longer embeds base64 payloads (it fell from
5.5 MB to 8 KB). `notebooks/build_answer_bakeoff.py` now also writes
`notebooks/answer_bakeoff_bundle.zip` (source + capsule, deterministic); the user uploads it to Drive
folder `repo_rag_answer_bakeoff_inputs`, and the notebook mounts Drive, checks the bundle's sha256
against the value baked into the notebook, and extracts it. A freshness test covers both files.

### 2026-11-01: Phase 7 answer-model bake-off v3, first full run (A100 40 GB)

Results are in `eval/facts_results/answer_bakeoff_v3/` (raw transcripts, freeze, summary). I
re-ran `analyze_answer_bakeoff.py report` on the returned folder: the models table and the
selection are identical to the returned `summary.json`. Hardware A100-SXM4-40GB, transformers
4.57.6, 0 label decodes during execution; dev ran 16:08-16:22 UTC, the freeze at 16:23:05, test
started 16:23:12 (freeze before test, as registered). Total about 45 minutes, not the 2.5-3.5 h
estimated. 14B peak allocated 31-32 GiB, so it fits; smoke preflight passed.

**Test (97 questions). No candidate qualifies.**

| Model | Exact facts /45 | Probes /8 | Answerable abstention /44 | Citations | Parse failures /97 |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 0 | 0 | 41 | none | 93 |
| Olmo-3-7B-Instruct | 8 | 0 | 44 | none | 84 |
| Qwen3-14B | 35 | 8 | 22 | none | 2 |
| Qwen2.5-Coder-14B-Instruct | 31 | 1 | 44 | none | 47 |

Gates are 41/45, 7/8, at most 4/44 and a citation rate of at most 0.05. The gate failed for
every model on facts and answerable abstention. All four also failed the citation gate. No
model emitted a single citation, so the invalid-citation rate is defined as 1.0.

Cause: mostly the strict output format, not retrieval or tool use. 4B: 86 of 93 parse failures
are valid `final` JSON without the required `abstain` key. Coder: 47 first turns use
`{"type":"search_code",...}` instead of the tool_call envelope. Olmo: mixed (plain prose,
`parameters` instead of `arguments`, wrong shapes). Qwen3-14B almost always parsed (2 failures)
but 20 of its 22 answerable abstentions made zero tool calls, and none of its answers carried a
`path:start-end` citation. Its 10 fact misses: 4 blame (full SHA instead of the first 12
characters), 4 issue_closed_by (string instead of a list), 2 hot_file (see below).

Scoring-validity finding: the two hot_file facts have gold args `path_prefix: "httpx"`, but the
question says "under httpx/"; the model used "httpx/", the answer matched, and arguments_ok was
False. This costs every model up to 2 facts. Registered scoring was not changed.

Process gap: the notebook ran straight through, so the two registered dev prompt revisions were
never used, and the frozen prompt is revision 0 (dev fact accuracy 0.327 pooled). Test is now
unlocked and dev tuning is closed for this experiment. Any follow-up needs a new version
(v4) and must state that the test failures were seen.

### 2026-11-01: Phase 7 bake-off v4 built (not run)

Design: `docs/PHASE7_BAKEOFF_V4_DESIGN.md`; the user approved it ("Approve, build it"). No answer
model has run on v4. Built:

* `eval/build_v4_fresh_facts.py` + `eval/datasets/httpx_facts_v4_fresh.jsonl`: 45 httpx facts for
  entities and arguments not among the 71 v3 facts (seed 20261102, round robin by type). The typed
  tool audit on the capsule database (all 116 dev+test facts, with the 12,000-character budget)
  found one fresh fact with no defensible answer, `issue_closed_by:3621` (gold PR 3761, the tool
  also returns 3759; 3759 closed unmerged and 3761 is still open). It is listed in `EXCLUDED` with
  its reason and the rule re-run (my own un-run v4 files were regenerated; nothing had run). The
  audit now passes **116/116**.
* `eval/datasets/phase7_bakeoff_v4/` frozen by `eval/prepare_answer_bakeoff_v4.py` (verifies v1,
  v2, v3 first): 186 questions, **dev 89** (the 71 seen facts + 16 code + 2 probes), **test 97**
  (45 fresh facts + the 24 code, 20 architecture and 8 probe questions of v3 test, flagged as
  seen). Gates, tools, generation limits, timing and the four approved models are unchanged.
* `agent/prompts.py`: revision 0 now has worked tool-call and final envelopes (with `abstain`),
  search-before-answer for code questions, a citation example and answer-shape examples, all
  synthetic.
* `answer_bakeoff_scoring.py`: `fact_ok` (corrected) next to `fact_ok_strict` (unchanged v3 rule).
  Corrections: trailing slash on `path`/`path_prefix`, a full 40-hex SHA for a 12-character blame
  gold, a bare string for a one-member set. The analyzer table prints both.
* Notebook: `answer_bakeoff_v4_bundle.zip` (top level of My Drive; was a folder until 2026-11-02),
  results to `repo_rag_answer_bakeoff_v4`, and `CONFIRM_FREEZE = False`: Run all stops after dev
  until it is set to True.
* `tests/test_answer_bakeoff_v4.py` plus v4 additions to the existing bake-off tests.

### 2026-11-01: Phase 7 bake-off v4, dev revision 0 (A100) and revision 1 prepared

Raw dev transcripts: `eval/facts_results/answer_bakeoff_v4_dev/` (`dev_0`, `smoke_0`; smoke
passed). The dev run took about 30 minutes and Run all stopped at the `CONFIRM_FREEZE` guard as
designed. I scored it with the analyzer (`rebuild_dev`). Test was not run and not opened.

| Dev, revision 0 (71 facts, 16 code, 2 probes) | Facts corrected | Facts strict | Probes | Answerable abstention | Citations | Parse failures /89 |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-14B-Instruct | 71 | 66 | 2/2 | 9/16 | 0 | 2 |
| Qwen3-14B | 62 | 48 | 2/2 | 4/16 | 3 (0 invalid) | 1 |
| Qwen3-4B-Instruct-2507 | 51 | 40 | 1/2 | 3/16 | 0 | 3 |
| Olmo-3-7B-Instruct | 33 | 24 | 0/2 | 16/16 | 0 | 42 |

Format failures fell from 47-93 per model on v3 test to 1-3 for the Qwen models. Caveat: the
revision 0 prompt was written after the v3 failures, and the 45 v3 test facts are now dev, so the
fact scores partly reflect fitting to failures I had seen; the fresh test facts are the check.
Remaining gaps: (1) almost no citations (3 across all models), which fails the citation gate;
(2) Qwen3-14B and 4B lowercased state values (pr_state 0/4: "open" for "OPEN"); (3) Olmo often
answers in plain text outside the JSON; (4) Qwen2.5-Coder wraps two replies in a code fence and 4B
used a tool name as `type` three times; (5) Coder searched and then abstained on 7 of 16 code
questions.

Revision 1 (common prompt only, same for all four): tool name never in `type`; no plain text
outside JSON; copy states exactly (OPEN, MERGED, CLOSED); citations go inside the answer string
on one line with an example, and an answer without one is incomplete. Notebook default is now
`REVISION = 1`; the revision 0 bundle and notebook are kept in `data/answer_bakeoff_rev0/`.

At the user's request the v4 notebook now ends the Colab runtime once results are on Drive
(`AUTO_DISCONNECT = True`): after the dev stage unless `CONFIRM_FREEZE` is set, and at the very end
after saving `MyDrive/repo_rag_answer_bakeoff_v4_results.zip`. `run_once` skips finished stages, so
Run all in a fresh runtime resumes at the freeze; an incomplete stage folder raises instead of being
overwritten.

### 2026-11-02: Phase 7 bake-off v4, dev revision 1 (A100) scored

Returned from Colab as `dev_1` (89 questions x 4 models, complete, 19:00-19:31 UTC on 2026-11-01) and
`smoke_1` (Qwen3-14B passed; peak allocated 33.3 GB of 42.4 GB). Copied unchanged to
`eval/facts_results/answer_bakeoff_v4_dev/{dev_1,smoke_1}`; each model's raw `.jsonl` sha256 matches
`run.json` `raw_sha256`, and `index_manifest_sha256` equals dev_0's (`47b11997...`). Scored with
`rebuild_dev` (no local `index_manifest.json` matches that digest, so the equal-digest check across
the two revisions was done by hand). Test was not run and not opened. No freeze yet.

| Dev, 71 facts / 16 code / 2 probes | Facts corrected r0 -> r1 | Strict r0 -> r1 | Probes | Answerable abstention (lower is better) | Citation-complete /16 | Invalid cites | Parse fails /89 | Gates failed r1 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | 62 -> **67** | 48 -> 52 | 2/2 -> 2/2 | 4 -> 1 | 2 -> **14** | 0 | 1 -> 0 | **none** |
| Qwen2.5-Coder-14B | 71 -> 68 | 66 -> 54 | 2/2 -> 2/2 | 9 -> 2 | 0 -> 9 | 0 | 2 -> 0 | answerable_abstention |
| Qwen3-4B-Instruct-2507 | 51 -> 50 | 40 -> 37 | 1/2 -> 0/2 | 3 -> 0 | 0 -> 8 | 0 | 3 -> 0 | fact_exactness, probe_abstention |
| Olmo-3-7B-Instruct | 33 -> 51 | 24 -> 38 | 0/2 -> 0/2 | 16 -> 16 | 0 -> 0 | 0 | 42 -> 19 | fact_exactness, invalid_citations, probe_abstention, answerable_abstention |

Dev ranking tuple (facts corrected, citation-complete, citation validity, probes, -revision):
r0 (0.764, 0.031, 1.0, 0.625), r1 (0.831, 0.484, 1.0, 0.500). Revision 1 ranks first.

Read with care:

* Qwen3-14B is the only candidate with no failed gate on dev. Its four corrected fact misses are all
  `commits_in_month` (2) and `top_committer`/`hot_file` for 2024 (2); the `pr_state` lowercase failures
  (4 in r0) are gone. Dev is not a clean estimate: the 71 facts were seen when revision 0 was written.
  Only the fresh 45 test facts can show whether this holds.
* Qwen2.5-Coder lost three `commits_in_month` facts and 12 strict-only points (6 `issue_closed_by`,
  5 `blame_line`, 3 `hot_file` by type), while its citations rose from 0 to 9/16. It still abstains
  on 2/16 answerable code questions (`requests-032`, `requests-035`), which fails that gate.
* Olmo still writes plain text outside the JSON on 19/89 questions and cites nothing.
* Probe abstention is 2/2 only for the two 14B models. 4B dropped to 0/2.
* Citation validity is 1.0 for every model that cited, so the gain is more citations, not invented ones.
* Timing (pooled p50/p95): Qwen3-14B 3.9/8.8 s, Coder 5.2/8.2 s, 4B 3.5/9.5 s, Olmo 2.8/8.1 s.

Open: the `CONFIRM_FREEZE` decision. Options are freeze revision 1 as it stands, or write a revision 2
(prompt change, all four models again, new smoke). Qwen3-14B's remaining misses are tool-argument
issues in `commits_in_month`/2024 rankings, which a prompt revision might or might not fix.

### 2026-11-02: Phase 6 `find_callees`; import aliases in `find_callers`

`SymbolTable.find_callees` and `repo-rag callees` list the calls inside a definition (including
nested definitions, and a class's methods), matched to repo definitions by the same labels as
`find_callers`, with library and builtin calls marked `external`. Hand check on psf/requests
`Session.send`: all 22 call sites in the body are listed (see `eval/symbol_check_results.md`).
The check exposed a defect that also affected `find_callers`: `from m import f as g` then `g()`
was not matched to `f`. Fixed in both and covered by a test. Known limits unchanged: matching is
by name, no inheritance, no typed receivers (`kwargs.get` matches `api.get` as `name only`).
Tests, ruff and mypy pass. Phase 6 stays unchecked (see plan.md for what is still open).

### 2026-11-02: Phase 5 manual scale tier on pandas-dev/pandas

`eval/measure_scale.py` and `eval/scale_results/README.md`. 5,000 commits: full clone 316 s
(.git 428 MiB), reading commits with diffs 698 s (peak 165 MiB, 20,385 files, 31.6 M patch chars),
commit documents 6.9 s (8.9 M chars), BM25 2.1 s (6.5 MiB), history rows 7.2 s (77.3 MiB database),
symbol table 47 s (38,518 definitions, 250,612 calls, 169 MiB database); process peak 369 MiB.
Symbol queries take 2.4-2.9 s including `uv run` start-up. A blobless clone made commit reading
impractical (about 1.2 s per commit in a 12-commit sample; the 5,000-commit read did not finish in
about 45 minutes, so I stopped it). Embedding was not run (GPU needed; about 8.7 h at the CPU rate
quoted in CLAUDE.md). The first script version reported 0 MiB memory (missing ctypes argument
types); fixed. No tests or source changed.

### 2026-11-02: Phase 5 depth run can resume from the old checkpoint (resume notebook)

The user supplied `phase5_commit_depth_checkpoint_869432f8c494.zip`. Contents: weights, 4B and 8B
depth-200 and selected-weight (0.1, 0.3) stages for Requests, Flask and FastAPI, a root
`selection.json`, and **no timing stage for any repo**. The checkpoint embeds its own `source.zip`
(SHA-256 prefix 869432f8c494, 28,592,786 bytes), so the exact old payload was recoverable.
Check: unpacked, that payload's `hashes()` equals the recorded source hashes of all 15 runs (0
differ), and `verify_protocol` passes on the checkpoint. The current source ZIP cannot resume it:
9 files changed since (`ast_chunker`, `cli`, `config`, `issue_linker`, `metadata_db`,
`retrievers`, `bm25`, `vectorstore`, `pyproject.toml`) and 15 files were added.

New: `notebooks/build_commit_depth_resume.py` generates `phase5_commit_depth_resume_on_colab.ipynb`,
the depth notebook bound to `notebooks/phase5_commit_depth_resume_source.zip` (the vendored old
payload). The finished stages print "Already complete"; only the 8B timing for the three repos,
the frozen-rule analysis and the CLI sample remain. Estimate: about 2 h instead of 4-5 h. Test:
`tests/test_commit_depth_resume.py`.

### 2026-11-02: Phase 5 commit depth run finished (resume notebook)

Resumed from checkpoint `869432f8c494` with `phase5_commit_depth_resume_on_colab.ipynb` on an A100 40 GB.
Result ZIP: `phase5_commit_depth_checkpoint_869432f8c494 (1).zip` (the old checkpoint plus timing,
aggregate, `cli_sample*`); the small result files are kept in `eval/commit_depth_results/`
(the index ZIP and `metadata.db` stay in the ZIP).

- **Frozen rule outcome: `keep_50`, no default changed.** Depth 100 and 200 both fail `hit5_gain` and
  `hit5_lower`. Pooled Hit@5 gain over 50 for 8B: 263 questions, +0.0076 (95% CI 0.0 to 0.019) at depth 100.
- Source-path subset (208 questions), 8B: Hit@5 0.9183 / 0.9279 / 0.9279 and MRR@10 0.8542 / 0.8566 / 0.8573
  at depth 50 / 100 / 200; pool recall 0.9567 / 0.9663 / 0.9760. 4B: Hit@5 0.9087 / 0.9135 / 0.9087.
- Weight: dev-selected 0.1 (dev MRR 0.7567 vs 0.7337 for 0.3, 66 dev questions); default stays 0.3. Reranking
  lifts un-reranked Hit@5 0.7788 to 0.9183 on the source subset.
- A100 warm timing, p50 per repo, ratio to depth 50 (depth 100 / 200): requests 3.9 s (1.95x / 3.89x),
  flask 3.5 s (1.91x / 3.60x), fastapi 9.2 s (1.95x / 3.76x). Latency scales about linearly with depth.
- CLI sample recorded with its bug report (`psf/requests#1000`): candidate 1 `f69934fb24a9`
  ("Import oauthlib without the obsolete wrapper") is the right fix; candidate 2 is an unrelated docs commit.
- **Still not measured** (decision.json): similar-fix usefulness for a new bug, independent label review,
  answer quality, laptop latency, other repos and languages.

### 2026-11-02: Phase 7 v4 dev revision 1 scored

smoke_1 passed (Qwen3-14B peak allocated 33.3 GB on the A100 40 GB). Dev (89 questions, 71 facts, 16 answerable code,
2 probes), corrected facts / strict facts / probes / answerable abstentions / citation-complete (valid citations):

| Model | Rev 0 | Rev 1 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | 51 / 40 / 1 / 3 / 0 (0 cites) | 50 / 37 / 0 / 0 / 8 (13 cites) |
| Olmo-3-7B-Instruct | 33 / 24 / 0 / 16 / 0 | 51 / 38 / 0 / 16 / 0 |
| Qwen3-14B | 62 / 48 / 2 / 4 / 2 (3 cites) | 67 / 52 / 2 / 1 / 14 (18 cites) |
| Qwen2.5-Coder-14B | 71 / 66 / 2 / 9 / 0 | 68 / 54 / 2 / 2 / 9 (11 cites) |

Pooled corrected-fact rate 0.764 (rev 0) vs 0.831 (rev 1); no invalid citations in either. The registered ranking
selects revision 1. Olmo still abstains on all 16 answerable code questions; 4B misses both probes.
Caveat: rev 1 was written after seeing rev 0 failures; dev is tuning data, the fresh 45 test facts are untouched.
Raw dev_1 and smoke_1 are in `eval/facts_results/answer_bakeoff_v4_dev/`. Test not run.

**Freeze failure and fix (2026-11-02):** the first Colab freeze was refused ("Restore the dev-selected common
prompt/parser/source"): the bundle in use carried the later `find_callees` versions of `src/repo_rag/cli.py` and
`src/repo_rag/store/symbol_table.py`, while `dev_1` ran with the earlier ones (the bake-off imports `SymbolTable`). The
dev-time bytes were recovered from `data/answer_bakeoff_rev0/answer_bakeoff_v4_bundle.zip` (both hashes match the
`dev_1` record) and vendored in `notebooks/pinned_dev_sources/`; `build_answer_bakeoff.py` now bundles those two files
instead of the live ones. The rebuilt bundle (sha256 `6b01a1bb044f...`) matches all 64 source hashes recorded by `dev_1`.
No result changed; nothing was frozen or tested before this. Upload the rebuilt bundle and notebook together.

### 2026-11-02: Phase 7 v4 test attempt 1 failed on infrastructure

The freeze passed (selected revision 1) and `test_1` started. Qwen3-4B finished (97 rows); loading Olmo-3-7B then raised
`OSError` (`fatal_error_class` in `run.json`; models recorded: 4B only; no Olmo files written; GPU 0 MiB and 243 GB disk
free afterwards, so a transient Hugging Face download/load failure is the likely cause, not confirmed). No test labels
were decoded and no test scores or answers were looked at. The runner refuses to overwrite a run and any edit to it would
change the bound source hash, so the partial folder is renamed on Drive to `failed_test_1_oserror` (kept, not deleted) and
the frozen test is run again from scratch with identical source, prompt and greedy decoding. This deviates from "never rerun
over saved artifacts" for an infrastructure failure only; it must be stated in the final report.

### 2026-11-03: Phase 7 v4 test result: no candidate qualifies

Frozen revision 1; test rerun completed (all four models, `complete: true`); raw files, summary and the failed first attempt
in `eval/facts_results/answer_bakeoff_v4_test/`. The rerun reproduced the failed attempt's Qwen3-4B outputs exactly (hash of every
turn identical on 97 of 97 questions), so the rerun after the Olmo-load `OSError` is comparable. Gates: facts >=41/45,
probes >=7/8, answerable abstentions <=4/44, invalid citation rate <=0.05.

| Model | Facts (corrected / strict) | Probes | Answerable abstentions | Citations (invalid) | Answerable with no citation | Failed gates |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 34 / 26 | 4/8 | 3/44 | 40 (0) | 27/44 | facts, probes |
| Olmo-3-7B-Instruct | 34 / 26 | 1/8 | 41/44 | 0 | 44/44 | facts, probes, abstentions, citations |
| Qwen3-14B | 45 / 35 | 5/8 | 5/44 | 66 (0) | 6/44 | probes, abstentions |
| Qwen2.5-Coder-14B | 44 / 36 | 8/8 | 6/44 | 17 (0) | 30/44 | abstentions |

**No default or answer model is adopted; gates are not relaxed.** Qwen2.5-Coder-14B misses a single gate by two
questions (6 vs <=4 abstentions); Qwen3-14B passes facts and cites most often but misses probes (5 vs >=7) and abstentions
(5 vs <=4). Caveats: the code, architecture and probe test questions were seen in v3 (flagged); architecture and probe labels
are not reviewed by a person; citation validity does not prove correctness. A100 timings are experiment costs.

### 2026-11-03: Phase 7 v5 designed (not approved, nothing built)

Diagnosis of Coder-14B's v4 test behaviour: of 44 answerable questions it cited 14, named a file without a line range on 13,
named no file on 11 and abstained on 6; 21 of its 24 answered-but-uncited answers name a retrieved symbol or file; 5 of 6
abstentions are architecture questions after a single search (Qwen3-14B cited 38/44). `docs/PHASE7_BAKEOFF_V5_DESIGN.md` proposes a
fresh seeded question set (dev 16+6, test 48+16), a shared R2 prompt plus a deterministic citation check, Coder as subject and
Qwen3-14B as control, with gates scaled from v4. Awaiting the user's four decisions (sizes, treatments, label review, control).

### 2026-11-03: Phase 7 v5 built and frozen (no model run)

Built per `docs/PHASE7_BAKEOFF_V5_DESIGN.md` (section 10 lists the differences from the design).

- **Fresh questions:** `eval/datasets/phase7_bakeoff_v5/` has 127 rows: dev 8 code + 7 architecture + 6 probes, test 24 code + 21
  architecture + 16 probes, plus the 45 v4 test facts as a seen, test-only guard. Targets are drawn by `eval/build_v5_targets.py` (seed
  20261103); `eval/build_v5_questions.py` holds the question text, a per-target drop log and the probe absence audit (one term,
  `sqlalchemy`, was removed from a probe after the audit found it in Flask's source text). The frozen preparer checks disjointness from
  v3/v4 (question text, code gold file+symbol, architecture file pairs, probe anchor files).
- **Arms:** `BAKEOFF_ARMS` in `agent/prompts.py`: revision 0 is the unchanged v4 prompt (hash `317abdd017b1...`, equal to the `dev_1` prompt),
  1 is R2, 2 is R3; a fixed reminder is added as a user message only after a `search_code` result. `run_question`, the runner
  (`test-baseline` action, smoke on arm 1), `identity()` (`arms_sha256`) and the analyzer (citation-presence gate, subject-model dev rank,
  paired baseline table, checked-citation columns) were extended; `answer_bakeoff_scoring.attach_citations` is the deterministic answer
  check, reported beside the raw score and never a gate.
- **Bug caught by a test before any GPU time:** the scorer's independent transcript replay rejected every reminder-arm run, because it
  did not rebuild the reminder message; `replay`/`score_trace` now take the reminder. An end-to-end fake matrix (three dev arms, selection,
  frozen test, paired baseline, report) covers it in `tests/test_answer_bakeoff_v5.py`.
- **Notebook:** `notebooks/answer_bakeoff_on_colab.ipynb` + `notebooks/answer_bakeoff_v5_bundle.zip` (Drive top level), results to
  `MyDrive/repo_rag_answer_bakeoff_v5`. A failing stage now prints the subprocess stderr (v4 hid it). The capsule was rebuilt with the
  v4/v5 verification files; its six retrieval payload hashes are identical to v4's, only the manifest's protocol hash changed.
  Unpacking the bundle in an empty folder and running `prepare_answer_bakeoff_v5.py verify` passes.
- **Re-freeze note:** the v5 dataset was re-frozen twice before any run (once after `ruff format` touched the frozen preparer, once for
  mypy annotations); the authored question file was byte-identical each time.
- **Other notebooks:** editing `src/repo_rag` and `pyproject.toml` made 16 generated notebooks stale; all were regenerated with their builders
  (the Phase 5 resume notebook stays pinned to its old source on purpose). Full offline suite green (exit 0, no failures), ruff and mypy clean.
- **Next (user):** upload the bundle, run the notebook (smoke + dev for 3 arms x 2 models, about 15 to 25 minutes), send `dev_*`/`smoke_1`,
  review, then `CONFIRM_FREEZE = True`. Do not edit `src/` or `eval/*answer_bakeoff*.py` before the test run.

### 2026-11-03: Phase 7 v5 result: tuned Coder-14B fails one gate by one reference

Run completed on the A100 (smoke on arm 1, dev for arms 0/1/2 x 2 models, freeze, test for the selected arm and the baseline); raw
files, report and the freeze are in `eval/facts_results/answer_bakeoff_v5/`. The user ran the freeze and test straight after the dev
stop, so dev was not reviewed by me first; the arm is chosen by the registered rule alone. Note: the report prints "R{revision}", so
"R2" in the tables means **revision 2 = the R3 prompt** (stricter citation sentence, shorter reminder).

Dev, subject Coder (probes / answerable abstentions of 15 / citation-complete raw of 15 / invalid refs): revision 0 6/6, 1, 4, 0 of 7;
revision 1 6/6, 2, 11, 1 of 23; revision 2 6/6, 1, 13, 1 of 27. Rank put revision 2 first (citation-complete 0.867).

Test, 45 answerable + 16 probes + 45 seen facts (guard), gates: facts >= 43, invalid refs <= 5%, probes >= 14, abstentions <= 4,
citation-complete >= 34:

| Arm | Model | Facts | Probes | Abstentions | Citation complete (raw / checked) | Invalid refs | Failed gates |
|---|---|---|---|---|---|---|---|
| rev 2 (R3 prompt) | Qwen2.5-Coder-14B | 45 | 14/16 | 4/45 | 37 / 37 | 5/90 (0.056) | invalid_citations |
| rev 0 (v4 prompt) | Qwen2.5-Coder-14B | 44 | 14/16 | 2/45 | 16 / 34 | 0/21 | citation_complete |
| rev 2 | Qwen3-14B (control) | 44 | 5/16 | 4/45 | 40 / 40 | 1/84 | probe_abstention |
| rev 0 | Qwen3-14B (control) | 45 | 3/16 | 3/45 | 40 / 41 | 1/57 | probe_abstention |

Findings: the tuned prompt raised Coder's raw citation presence from 16 to 37 of 45 with probes unchanged (14/16) and facts 45/45. It
did not reduce abstentions (2 to 4); on these fresh questions the baseline abstained only twice, so v4's 6 of 44 did not reproduce
(different questions; not explained). The one failed gate is 5 invalid references of 90 against a limit of 4.5, i.e. one reference
over: a made-up range (flask cli.py 649-698), a malformed range (logging.py 67-55), two test function names written as paths, and
one file mixed up with another. The answer check cannot fix these. Qwen3-14B still answers most probes (5/16 and 3/16).
**Nothing is adopted and no gate is relaxed.** Caveats: Flask-heavy architecture split, related questions share functions, labels
unreviewed by a person, seen facts only a guard.

### 2026-11-03: Provisional answer model chosen; citation checker added

User decision ("Yes, provisional Coder"): `Settings.llm_model` = Qwen2.5-Coder-14B-Instruct, `llm_model_revision` = `aedcc2d42b62...` (the
approved revision), `llm_prompt_arm` = 2. This is **not** a qualified selection: the v5 run left one gate failed (invalid citations 5/90, limit
4). Qwen3-14B is not used because it answered most probes (5/16, 3/16). `src/repo_rag/agent/citations.py` (`check_citations`,
`repair_citations`, `chunks_from_results`) validates every `path:start-end` against the retrieved chunks, strips unsupported ranges, and
attaches references for chunks the answer names; it never touches an abstention or invents support (`tests/test_citations.py`). Offline
on the 41 answered tuned-Coder v5 outputs it removed all 5 invalid references and attached 2, leaving 41/41 answered questions with a
valid citation (checked with the scorer's own `references`). Exploratory only: written after seeing the failures, needs fresh questions.
The v5 run used the bundle with sha256 prefix `28e55e18fb55`; later `src/` additions (citations.py, config) mean the repo's rebuilt bundle
differs from it. All stale generated notebooks were regenerated again.

### 2026-11-03: Phase 7 agent core (LangGraph) built; no real model wired

- **Dependency:** `langgraph==1.2.19` added with `uv add` (listed in the decided tech stack; only additions to `uv.lock`: langgraph,
  langgraph-checkpoint 4.3.1, langgraph-prebuilt 1.1.1, langgraph-sdk 0.4.11, ormsgpack 1.13.0). Every generated notebook was regenerated again.
- **`src/repo_rag/agent/graph.py`:** `RepoAgent(model, execute, settings)` is a LangGraph `StateGraph`: pre-step (exact references via
  `retrieval/exact_refs.py`, returned to the caller only) -> answer (the evaluated strict-JSON tool loop `run_question`, prompt arm and
  reminder from `Settings.llm_prompt_arm`) -> citation check (`agent/citations.py`; stripped/attached references reported) -> remember
  (`MemorySaver`, per `thread_id`). The model sees only the arm prompt and the standalone question, so the measured behaviour is what runs;
  history is stored and returned but deliberately not fed to the model (follow-up rewriting would need its own measurement). A parse,
  budget or tool failure returns an explicit "No answer could be produced" abstention, never a guess.
- **Config:** `agent_max_tool_calls=4`, `agent_max_turns=5`, `agent_tool_result_chars=12000`, the limits the model was evaluated under.
- **Tests:** `tests/test_agent_graph.py` (6) with the scripted model: citation repair, exact-fact pass-through, abstention untouched, failure
  becomes explicit no-answer, model input is exactly prompt + question, per-thread memory.
- **Not done in Phase 7:** a real local model adapter (Coder-14B bf16 peaked at about 31 GB on the A100, so it needs a GPU, a quantized
  runtime (a new dependency, to be asked first) or a hosted GPU session), the 5-route router (only add routes with measured evidence;
  the evaluated config gave every question the full tool set), `repo-rag ask`, the golden-set faithfulness check, and the Phase 5b fact set
  run through the full agent (target 90%). Phase 7 stays unticked.

### 2026-11-03: Answer model runs on a Colab GPU; loader, runtime and notebook built

User decision: the answer model runs in a Colab GPU session (no tunnel, no new dependency).

- `src/repo_rag/agent/local_llm.py` `LocalChatModel`: bf16, SDPA, greedy, batch 1, the model's native chat template (refused if its hash differs
  from `Settings.llm_chat_template_sha256`), input cap 8192 and output cap 512 tokens from `Settings`; same generation code as the evaluated
  `HFAnswerModel`. Its pure helpers (`trim_terminal_eos`, `check_input_tokens`) are tested; the model load itself has **not** run (needs the GPU).
- `src/repo_rag/agent/runtime.py` `evaluated_settings` / `build_agent`: pins the retrieval setup the model was measured with (top-50 fused,
  dense 1.0 / BM25 0.3 / RRF 60, no reranker, no expansion, float32 queries) and builds `RepoAgent` over the manifest entry's commit and a
  git checkout. An offline integration test indexes the fixture repo with fake embeddings, runs the agent with a scripted model and gets a
  valid citation (`tests/test_agent_runtime.py`).
- `notebooks/ask_on_colab.ipynb` + `notebooks/ask_source.zip` (upload to Drive as `repo_rag_ask_source.zip`; the index as `repo_rag_index.zip` from
  `repo-rag export-index`): imports the index, clones the repo at the indexed commit, builds the symbol table (history optional via
  `BUILD_HISTORY`), loads the model, then `ask(...)` / an input loop. Source is hash-checked, no base64. `tests/test_ask_notebook.py` checks freshness.
- **Limits:** only psf/requests is indexed locally; commit history, PRs and issues are not indexed by default, so those question types have little
  data (PR/issue sync needs a GitHub token); the real model path is untested until a Colab run; the model stays provisional.
- Suite: pytest exit 0 with 0 FAILED lines on a run started after all notebooks were regenerated; ruff and mypy clean. Two later
  edits (notebook builder line lengths) were re-checked with the ask-notebook and agent tests only.

### 2026-11-04: First Colab run: model loads and answers; tool failures now say why

Evidence from the user's first run (psf/requests, Coder-14B, arm 2): "hi" answered with no tools; `get_pr` for #27 and #140 ended as
`ABSTAINED | tool` with the generic "reply was not valid" text (the tool raised; the reason was dropped); the `requests.get()` architecture
question abstained after one `search_code`, the failure already measured in v5 (5 of 6 Coder abstentions were architecture after one search).
The model load path therefore works on a GPU.

- `agent/graph.py`: the tool wrapper keeps the last tool exception (`name: Class: message`, 200 chars) and a failed run now answers
  "a tool failed (…). Commit, PR and issue data may not be indexed" with `result["error"]`. Results also carry `sources` (the retrieved
  `path:start-end symbol` list). The frozen `bakeoff.py` is untouched. Two tests added in `tests/test_agent_graph.py`.
- `notebooks/build_ask_colab.py`: `ask()` prints the error and retrieved sources; `peek(query)` prints the top-10 `search_code` would see, to
  separate retrieval from model failures on the abstaining architecture question (not yet run).
- Suite: pytest exit 0 (run with `--basetemp` in the scratchpad: the shared `pytest-of-user` temp folder gave WinError 5), ruff and mypy clean.

### 2026-11-04: Streamlit UI with the model served from Colab (user approved; adds streamlit)

User asked for a basic Streamlit UI instead of running the model in a notebook. Design: only token generation is remote.

- `agent/remote_llm.py`: `serve_model` (stdlib `ThreadingHTTPServer`; `GET /health`, `POST /generate`; bearer token compared with `hmac.compare_digest`; token of
  at least 16 characters required; request body capped at 1 MB; `generate` serialized behind a lock; request bodies are not logged) and `RemoteChatModel`, which is a
  `Model` so `RepoAgent` runs unchanged. Input-cap failures cross the wire as HTTP 413 and come back as `BudgetFailure`. No fastapi or uvicorn was needed.
- `agent/graph.py`: a model failure (for example an unreachable server) now answers "the answer model failed (reason)" with `result["error"]`, like tool failures.
- `ui/app.py`: sidebar for URL, token and repository (from `manifest.json`); chat with answer, a one-line citation summary, expanders for retrieved code and tool
  calls; connection errors shown, not raised; "New chat". Embeddings load once on the laptop CPU (float32, as evaluated). `describe()` says citations show location,
  not correctness.
- `notebooks/serve_model_on_colab.ipynb` (`notebooks/build_serve_colab.py`): same source ZIP as the ask notebook, loads `LocalChatModel`, serves, opens a
  Cloudflare quick tunnel (cloudflared 2026.9.1, SHA-256 pinned and checked before it runs; the digest was read from the GitHub releases API through a summarizing
  fetch, and a wrong digest makes the notebook refuse to run), prints URL and a random per-session token. Prompts and retrieved code cross a public tunnel: public repos only.
- Dependency: `streamlit>=1.66.1` via `uv add` (user-approved). `uv.lock` changed by additions plus resolution-marker rewrites; no existing package version moved.
  `UI_ONLY = {"streamlit"}` keeps it out of every Colab notebook's pins (its pinned pandas and pyarrow could clash with Colab's own).
- Tests: `tests/test_remote_llm.py` (7: round trip, auth, token length, 413 mapping, malformed requests, unreachable server, a full agent run over HTTP),
  `tests/test_ui_app.py` (4, Streamlit `AppTest` with a fake agent), `tests/test_serve_notebook.py` (2), plus graph tests for tool and model failure reasons.
- Found while preparing the first run: the imported index has no `clone_path` in the manifest, so `ui/app.py` now falls back to `data/repos/<owner>__<name>` (the clone is at the indexed commit `611c6162cbc4`); `data/metadata.db` did not exist, so `repo-rag index-symbols psf/requests --checkout data/repos/psf__requests` was run (807 definitions, 2660 call sites, 35 files). Related tests, ruff and mypy pass after this change; the full suite was not rerun.
- **Not verified:** the UI against a real Colab server and tunnel, and a real browser session. Streamlit tests use a fake agent.

### 2026-11-04: Broader agent: PR/issue data synced, four production tools, five-type evaluation (not yet run)

User asked for a system that answers all five question types. What was built, and what is and is not measured:

- **PR and issue data (real run, user's `GITHUB_TOKEN`, read-only):** `repo-rag sync-github psf/requests` stopped at the default first-sync cap (500 PRs, 500 issues,
  "newest only"), and a re-run only fetched changes because a watermark existed. The two cached state files were moved to the scratchpad (not deleted) and the sync re-run with
  `--max-prs 100000 --max-issues 100000`: **3,086 PRs and 4,186 issues, complete**. `#27` in psf/requests is an **issue**, not a PR (shared numbering), which is why `get_pr(27)` was empty.
- **`MetadataDB.references(repo, number)`:** written `#N` and same-repo `/pull/N` or `/issues/N` mentions in synced titles and bodies, outgoing and incoming, with kinds, plus `pr_closes`.
  Not GitHub timeline cross-reference events. Cross-repo mentions and longer numbers (`#270` for 27) are excluded. Typed and parametrized; no model SQL.
- **`agent/tools.py` (production toolset, `Settings.agent_toolset="production"` by default; `"evaluated"` is the bake-off setup exactly):** `search_docs` (README/docs/CI/dependency files, same chunk
  shape so the citation check covers it), `search_commits` (needs commits embedded; says so if not), `get_references`, `find_callees`. `PRODUCTION_PROMPT` is the arm-2 prompt plus guidance for these four
  and a rule to check `get_issue_meta` when `get_pr` finds nothing. `graph.py` flags 7-40 character hex strings in an answer that appear in no tool result (`unverified_shas`) and lists commits in `sources`.
  `bakeoff.run_question` gained a `reminder_tools` parameter (default unchanged).
- **Why architecture questions abstained (laptop retrieval, real output, not the model):** for "How does a call to requests.get() reach the network?" the top 10 held `api.get`, `Session.request`, `Session.get`,
  then redirect tests; `Session.send` and `HTTPAdapter.send` were not in the top 10, so the top 5 the model sees stop at the entry. Naming the symbols finds them (rank 1 and 4). Test files crowd results.
  Retrieval was left alone (measured; changing it needs the retrieval eval). The model-side response is `find_callees` plus the call-chain prompt rule, which is **unmeasured**.
- **`eval/agent_eval.py`:** 93 questions on psf/requests (code 40 from the seen golden set, docs 12 authored, history 11 and refs 10 and kind 5 derived from the synced data with seed 20261104,
  commit 10 from the seen Phase 5 pairs, probes 5 audited to be absent from the repo). `run` asks them through the real agent with the model on the Colab server and scores mechanically per type.
  Exploratory: no gates, no selection depends on it. Dataset sha256 prefix 8aadc486eb09. **Not run yet**: it needs the Colab model server.
- **Not done / not verified:** no model has run any of the new tools; commit search has no data until `index_on_colab.ipynb` is re-run with `SHALLOW_CLONE = False` and imported; PR/issue
  embeddings (`index-prs`) are not built and no tool searches them; answers to "who pushed" report author, committer and `merged_by` only; the production prompt and tools have no bake-off result.
- Tests added: `tests/test_references.py` (7), `tests/test_agent_eval.py` (8), graph tests for tool and model failure reasons. Full suite pytest exit 0 after the final notebook regeneration; ruff and mypy clean.
  Lessons: Python heredocs in this shell turn `
` and `` into real control characters (broke prompts.py and two regexes); write such text with the Write or Edit tools.

### 2026-11-04: `repo-rag index --commit` so the Colab run indexes the same revision

Before the Colab commit-history run: upstream psf/requests HEAD is now `3c7baaa05ab4` (`git ls-remote`), while the local clone and index are at `611c6162cbc4`.
`index_on_colab.ipynb` would have cloned the newer tip, which would move the code chunks, mismatch the local clone that the git tools read, and shift the line ranges the
40 golden code questions are scored against. `clone_repo`/`clone` and `repo-rag index` now take `--commit <sha>` (7 to 40 hex characters, needs full history, detached checkout, an
existing clone is not fast-forwarded), and the indexer notebook has `PIN_COMMIT` (use `"611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60"` with `SHALLOW_CLONE = False`). **Correction (same day):** my first instructions said `SHALLOW_CLONE = False` alone would embed commits. It does not: `repo-rag index` only controls clone depth and `index-history` is a separate command. The user's first Colab zip (`export.json`: commit `611c6162cbc4`, 274 code + 282 docs chunks, `history_sha` empty, no `commits/` folder) confirmed it. The notebook now has `INDEX_HISTORY = True`, which runs `index-history --offline` after `index`. Tests: pin and
validation in `tests/test_github_client.py`; three test stubs of `clone` gained the optional `commit` argument. `import-index` carries search collections only, so commit facts
(`history_commits`, used by `count_commits` and friends) still have to be written locally with `repo-rag index-history psf/requests --checkout data/repos/psf__requests --offline`;
the chunks should be unchanged, so it should embed nothing (unverified: stop it if it starts embedding).

### 2026-11-05: Commit-history indexing 13x faster per commit; GPU batch sizes exposed

User: the A100 sits idle in the indexing notebook. Cause (measured, not assumed): the GPU is only used by the last step, `store.sync`. Before it, `index_history` reads every
commit with GitPython, one `git diff` each, and reads both versions of each changed file. In a blobless clone each missing file version is fetched from GitHub on its own, in series.
Timing on a copy of the local psf/requests clone (60 commits at `611c6162cbc4`, 128 changed files, this laptop):

| | time | per commit |
|---|---|---|
| extraction, blobs fetched lazily one by one | 60.6 s | 1,011 ms |
| one batch fetch (124 missing blobs, 1.1 s), then extraction | 1.1 s + 3.6 s | 60 ms |

Extrapolated to the 5,000-commit cap that is about 84 minutes against about 5 minutes of extraction (an extrapolation of per-commit time, not a run; the Colab CPU and the real blob count
will differ). The embedding step itself is small: roughly 5,000 documents, and the earlier A100 benchmark measured 239 to 311 docs/s in bf16 (float32, the notebook default, is slower and was not timed).

- `history/commit_indexer.prefetch_blobs`: `git log --raw` lists the exact old and new file versions of the bounded history (it needs trees only, so it works in a blobless clone); the ones not local
  (from `cat-file --batch-all-objects`) are fetched with one `git fetch --stdin` request per 5,000 (fallback: ids as arguments for git without `fetch --stdin`; any failure returns early and reading
  fetches on demand, so correctness never depends on it). A second call requests 0. `index_history` calls it first and the CLI prints "Prefetched N file versions".
  First version listed "objects introduced in the window" and missed the parent-side versions of the oldest commit's files (85 of 124 blobs; 25.5 s for 60 commits); the `log --raw` version fixed it.
- `Settings.index_write_batch` (default 64, was a hard-coded 64) and `history_prefetch_chunk`; the Colab indexer notebook gains `EMBED_BATCH = 64` and `WRITE_BATCH = 256`
  (env `REPO_RAG_EMBEDDING_BATCH_SIZE`, `REPO_RAG_INDEX_WRITE_BATCH`; the laptop default batch size of 16 was far too small for an A100). Their effect on GPU time is **not measured**.
- Tests: `tests/test_prefetch.py` (4: whole history local after one batch, a full clone has nothing to fetch, the no-`--stdin` fallback, a failed fetch is not fatal).
- Not done: extraction is still one process (about 60 ms per commit); a process pool is the next step if the Colab run is still CPU-bound. TF32 for float32 matmuls on the A100 would speed embedding but
  changes vectors slightly and needs a GPU measurement first, so it was left off.
- Suite: the first full run after these changes had 2 failures (`test_commit_confirm::test_full_offline_matrix_rebuilds_and_rejects_tampered_rows`, `test_remote_llm::test_a_wrong_or_missing_token_is_refused`); only the last lines of its output were kept, so the messages are unknown. I was editing docs while it ran. Both pass alone, and a full rerun with nothing edited meanwhile passed (pytest exit 0). Cause not established; if either recurs, capture the full output.
- Lesson: this tool layer collapses doubled backslashes, so `"\n"` written into a Python string lands as a real newline; use raw strings with single backslashes or edit with the Edit tool.

### 2026-11-05: First agent evaluation with the real model; prompt fix for tool choice; commit-search diagnosis

Commit index imported (Colab, float32, pinned `611c6162cbc4`): psf/requests now has 274 code, 282 docs and 4,883 commit chunks locally. The first Colab run crashed with CUDA out of memory
while embedding commits (batch 64 x up to 2,048 tokens in float32 on a 40 GB A100); the indexer notebook gained `COMMIT_EMBED_BATCH = 8` for that step only (not tuned). The local
`index-history --offline` then embedded 0 (4,883 unchanged) and filled the commit facts in `metadata.db` (1,101 fix candidates; 0 links, because offline reuses only existing links).

`eval/agent_eval.py` run against `Qwen/Qwen2.5-Coder-14B-Instruct` on an A100 40 GB through the cloudflared tunnel (exploratory, 93 questions, no gates; the code and commit questions come
from sets seen before). Production toolset, first full run (`agent_eval_production_20261105T212416.json`):

| type | passed | of | abstained | failed |
|---|---|---|---|---|
| code | 26 | 40 | 2 | 2 |
| docs | 8 | 12 | 0 | 2 |
| history | 11 | 11 | 0 | 0 |
| refs | 10 | 10 | 0 | 0 |
| kind | 5 | 5 | 0 | 0 |
| commit | 4 | 10 | 1 | 0 |
| probe | 5 | 5 | 5 | 0 |

- **Finding:** in 10 of the 14 code misses the model called `search_docs` (7) or `search_commits` (3) first and cited docs or a commit. Same 40 code questions with `--toolset evaluated`
  (code search only): 35/40, 0 failed (`...T213239.json`). So the added tools misled the model. Cause in the prompt: "configuring, using ... call search_docs first" matched code questions.
- **Fix:** `_PRODUCTION_TOOLS` now says questions about how code works, what a function does, where something is implemented or how it is configured in the code are code questions: `search_code`
  first; `search_docs` only for README/docs/install/contributing/CI/dependency files; `search_commits` only for bug reports and "was this fixed before". After it: code 35/40 (1 abstained, 1 failed),
  docs 7/12, commit 3/10; history 11/11, refs 10/10, kind 5/5, probe 5/5. Single runs: the 8 to 7 and 4 to 3 changes are one question each and are not read as real; the code gain (26 to 35) is.
- **Invalid run, discarded:** a rerun at 21:35 scored code 6/40 because the Colab tunnel died mid-run (1 x 502, 33 x 530); `agent_eval_production_20261105T213507.json` is not a result. The
  server also refuses to load when the Drive zip differs from the notebook's embedded SHA-256: regenerate all notebooks and re-upload the zip as `repo_rag_ask_source.zip` after any change under `src/repo_rag`.
- **Commit questions (bug -> past fix), diagnosis.** Searching with the full question and no rerank, the gold commit ranks: 1 (x2: 082, 086), 2 (083, 084), 5 (081), 14 (088), 31 (085), 35 (087), 44 (080),
  not in top 50 (079). The model sees the top 5, so 5 of 10 can never be answered with this search; 2 more ranked 2nd and the model chose another commit. Answers are often a bare 12-character SHA
  with no explanation. Ranks use the full question text, not the model's own query (queries are not recorded in the result rows).
- **Tried and reverted (no clear gain):** 15 hits per `search_commits` call: commit 3/10 (same). Prompt telling the model to pass the bug report unchanged as the query: commit 4/10 against 3/10, which is
  within the 3 to 4 range of earlier runs. Not kept. Untested: reranking the top 50 with the reranker (Phase 5 measured it on commits) next to the answer model on one A100.
- **Parse failures: cause found and fixed (2026-11-05).** Captured the raw replies by running the five parse-failing questions 4 times each: code-004, code-013 and docs-045 passed 4/4 (their earlier
  failures were occasional); code-008 and docs-046 failed 4/4, deterministically. code-008: the model's first reply is a correct tool call wrapped in a ```json fence, which `parse_reply` rejects
  (no fences) before any search. docs-046: a valid, correctly cited final answer rejected because the answer text contains a ``` code example (the parser rejects ``` anywhere). Fix:
  `parse_reply(..., tolerant=True)` for the production toolset only (`Toolset.tolerant`, `run_question(tolerant=)`): accepts one outer fence and backticks inside the answer string; duplicate keys, extra
  keys, wrong arguments, reasoning tags and prose around the JSON still fail; the bake-off parser stays strict. Tests: `tests/test_tolerant_parse.py` (7). Result: code 36/40 (0 failed), docs 8/12 (0 failed).
- **Docs-vs-code prompt sentence, tried and reverted:** adding "license, or how a user should use the library -> search_docs" gave docs 8/12 and code 33/40 (from 7 and 35): one up, two down, so not kept.
  Remaining docs misses: docs-045/050/051 go to search_code; docs-049 (license) answers "Apache 2.0" citing docs/api.rst, which is right, but the scorer only accepts LICENSE, README.md or pyproject.toml
  (scorer left unchanged). Remaining code misses: 016 and 037 cite the wrong range with search_code; 032 and 033 are symptom-style questions that the prompt sends to search_commits.
- **Run-to-run variation, measured (2026-11-05, current code, same server).** Three full 93-question runs (`agent_eval_production_20261105T225555`, `...T230803`, `...T232023`): identical per-type
  scores (code 36/40, docs 8/12, history 11/11, refs 10/10, kind 5/5, commit 3/10, probe 5/5), no pass/fail flips, and all 93 answers word for word identical between runs 1 and 3 (and 1 and 2). So
  with this model, server and prompt the agent is effectively deterministic on this dataset, and the earlier one- and two-question differences between configurations were real effects of the changes, not noise.
  That corrects my earlier reading above of the 4/10 vs 3/10 commit result ("within noise"): it was a real one-question difference, from a single run per configuration and not confirmed by a second.
  Determinism covers repeats of one configuration; it says nothing about other questions, other repos or other models.
- **Reranker for commit search, built but NOT yet measured (2026-11-05).** Phase 5 measured reranking on bug -> fix commits (test MRR 0.677 -> 0.823 with the 4B, 0.847 with the 8B; first-stage pool
  ceiling 0.91 to 0.96), which is the 5-of-10 ranking gap seen above. The default `Qwen3-Reranker-8B` would not fit next to the 14B answer model on a 40 GB A100 (about 28 GB + 16 GB from parameter counts,
  an estimate, not measured), so the serve notebook loads `Qwen3-Reranker-4B` (`RERANKER_MODEL`, None to disable) and prints GPU memory; if it fails to load the server runs without it.
  `serve_model(..., reranker=)` adds `POST /rerank` ({query, texts<=100, instruction}), serialized with `/generate` behind one lock; `/health` reports `rerank`. `RemoteChatModel.reranker_factory(info)`
  plugs a `RemoteReranker` into `HybridRetriever`; `build_agent(..., reranker_factory=)` sets `reranker_mode=on` (production toolset only), and `ProductionTools._search` re-scores the fused top 50 for
  `search_commits` only, falling back to the fused order (counted in `rerank_failures`) if the server errors. `eval/agent_eval.py run --no-rerank` gives the A/B. Tests: `tests/test_remote_llm.py` (+3),
  `tests/test_commit_rerank.py` (3). To measure: new Colab session with the regenerated notebook and zip, then `--types commit` with and without `--no-rerank`.
- **PR/issue text search, built but NOT yet run (2026-11-05).** New production tool `search_prs` (topic search over the `pull_requests` and `issues` collections, top 4 of each, labelled by kind; a kind
  that is not indexed is skipped, none indexed raises `LookupError`); existing `find_prs` only matched PR titles by substring and never issues. `_PRODUCTION_TOOLS` says "Five more tools" and tells the model to
  use it for topic questions and get_pr / get_issue_meta for exact facts. The embeddings do not exist yet: 3,086 PRs + 4,186 issues are too slow for the laptop CPU, so the Colab indexer notebook has
  `INDEX_PRS` (you upload `data/metadata.db`, about 40 MB, which also holds commit and symbol tables; it runs `repo-rag index-prs` with `PR_EMBED_BATCH = 32`, untuned), and `export-index` / `import-index` already carry
  those collections. Eval: `agent_eval_v1.jsonl` now has 103 questions (first 93 byte-identical, checked), new type `prsearch` (5 PRs + 5 issues, "which one is about: <its own title>", pass if the number is in the answer);
  this is an easy lookup that checks the tool end to end, not a paraphrase test. Tests: `tests/test_search_prs.py` (2) and a not-indexed case in `tests/test_references.py`. Full suite, ruff and mypy pass. Not measured:
  whether adding a fifth tool changes tool choice on the other types (the fourth-tool experience says it can), so the whole 103-question set must be re-run once the index exists.
- **PR/issue index built and imported (2026-11-06).** Colab run on a free Tesla T4 (15 GB), not an A100; the notebook's batch of 64 ran out of memory there (`CUDA out of memory`, 14.56 GiB total), so
  the indexer notebook now reads the GPU size and caps batches below 30 GiB (code/docs 8, PRs 8, commits 4; a guess, untuned) and prints the GPU it found. Other failures met and fixed: Colab renames a repeat upload
  to `metadata (2).db` (the notebook now reuses an existing `/content/data/metadata.db` and accepts any single uploaded file); a leftover shallow clone made `git checkout --detach <sha>` fail with "reference is not
  a tree" (`clone_repo` now fetches the missing history, with `--unshallow` if needed, before checking out a pinned commit; test in `tests/test_github_client.py`); the notebook now puts the last 60 lines of a failed command into the
  exception text. The zip (pinned `611c6162cbc4`, 3,086 PRs, 4,186 issues, vector files sized exactly 3,086 / 4,186 x 512 float32) imported cleanly: manifest now code 274, docs 282, commits 4,883, pull_requests 3,086,
  issues 4,186, and `history_sha` was kept. A `--kind issues` search for "requests hangs forever without a timeout" returned #4846 ("Adding `timeout` changes behavior...") first. Not yet run with the model.
- Probe type name for `--types` is `probe` (not `probes`); with a wrong name the script runs nothing and exits cleanly.

### 2026-11-06: First run of the commit reranker, `search_prs` and the 103-question set (A100 40 GB)

Server: `Qwen/Qwen2.5-Coder-14B-Instruct` plus `Qwen/Qwen3-Reranker-4B` loaded together on an
A100-SXM4-40GB. One run per configuration, `psf/requests`, dataset sha256 prefix `d8bf41dcb3a6`.

| Type | Reranker on | Previous run (11-05, no reranker, 93 questions) |
|---|---|---|
| code | 34/40 | 36/40 |
| docs | 7/12 | 8/12 |
| history | 11/11 | 11/11 |
| refs | 10/10 | 10/10 |
| kind | 5/5 | 5/5 |
| commit | 1/10 | 3/10 |
| probe | 5/5 | 5/5 |
| prsearch (new, 10 questions) | 10/10 | n/a |

- `search_prs` works: all 10 new questions used it and passed, and it did not change tool choice on
  history, refs or kind. Code and docs each lost points (see below); not traced to the fifth tool.
- Commit questions again without the reranker (`--no-rerank`): 3/10 (081, 082, 086 pass). With it:
  1/10 (only 086). The model cited different commits on 081 and 082.
- Rank check with the full question text as the query (`rerank_diag`, scratch script): gold rank
  fused order → reranked was 080: 44 → 17, 081: 5 → 3, 082: 1 → 2, 083: 2 → 2, 084: 2 → 1,
  085: 31 → 6, 079: not found either way. The check then crashed on the next question (below), so
  086 to 088 are not measured.
- **Reranker out-of-memory on the server.** `/rerank` returned 500: `CUDA out of memory. Tried to
  allocate 1.74 GiB ... 279 MiB free` of 39.49 GiB. `ProductionTools` falls back to the fused
  order on this and counts `rerank_failures`, but the result rows did not record the counter. So the
  commit 1/10 above may include searches that were not reranked; how many is unknown. Intermittent:
  seven questions reranked fine in the same diagnostic.
- Likely cause (read from the code, not yet verified on the GPU): `QwenReranker.score` took
  `model(...).logits[:, -1, :]`, which builds logits for every position (batch 4 x 2,048 x 152k).
- code-032 took 300.3 s with no tool call and was scored a failure; looks like a server stall, not
  rerun. code-035, docs-046 and docs-047 are new misses; causes not checked.

Changes made (nothing measured on the GPU yet):
- `retrieval/rerank.py`: the forward pass uses `logits_to_keep=1`; the scorer only reads the last
  position, so scores should be unchanged. Needs a live check against the previous scores.
- `eval/agent_eval.py`: every row has `rerank_failures`, the result file has the total, and the run
  prints a warning when it is above zero.
- `notebooks/build_colab_indexer.py`: two long lines wrapped (no change in behaviour).
- All notebooks and `ask_source.zip` regenerated (SHA-256 starts `3128bc40`); full offline suite,
  ruff and mypy pass. The zip on Drive must be replaced before the serve notebook will run.

Next: a fresh A100 session with the new zip; rerun the 10 commit questions with the reranker and
check `rerank_failures` is 0 before comparing with the 3/10 without it.

### 2026-11-07: Reranker memory fix confirmed; the reranker still lowers the agent's commit score

Fresh A100 40 GB session, new zip (`3128bc40`), same 14B model and `Qwen3-Reranker-4B`.

- **Out-of-memory fixed.** The rank diagnostic that crashed on 2026-11-06 now finishes all 10 questions.
  The first seven ranks are identical to the earlier run, so `logits_to_keep=1` did not change the
  scores. `rerank_failures` was 0 across all 10 agent searches.
- **Rank with the full question text** (fused → reranked): 079 none, 080 44 → 17, 081 5 → 3,
  082 1 → 2, 083 2 → 2, 084 2 → 1, 085 31 → 6, 086 1 → 1, 087 35 → 1, 088 14 → 1.
- **Agent, commit questions, same session:** with the reranker 1/10 (only 086), without 3/10
  (081, 082, 086). The wrong SHAs for 081 and 082 are the same as in the 2026-11-06 reranked run.
- **The model sends short queries**, not the question text. Gold rank for the model's own query
  (plain → reranked; one fresh run of the diagnostic, so the queries may differ from the eval's):
  080 29 → 36, 081 1 → 3, 082 5 → 3, 083 4 → 2, 085 13 → 6, 086 1 → 1, 088 19 → 9; 079, 084 and 087
  not in the top 50 either way. Gold is in the top 5 for 4 of 10 questions both with and without the
  reranker (081, 082, 083, 086 in both cases), so reranking did not raise what the model can see.
  It changes the order, and the model then names a different commit on 081 and 082. Why it picks
  them is not established.
- **Conclusion (n = 10, two reruns agree):** at the evaluated settings the reranker does not help
  the agent's commit answers and costs 2 of 10. The Phase 5 gain (MRR 0.677 → 0.823) was measured on
  the full bug report as the query, which is not what the agent sends. `ui/app.py` still passes the
  reranker factory, so the UI reranks; this is not yet changed.
- Not explained: code-032 stall (300 s), code-035, docs-046, docs-047.

Decision (user, 2026-11-07): the UI no longer uses the reranker by default. New setting
`agent_commit_rerank` (default `False`, env `REPO_RAG_AGENT_COMMIT_RERANK`); `ui/app.py` passes the
reranker factory only when it is on. The reranker code, the server `/rerank` endpoint and the eval's
default (reranker used unless `--no-rerank`) are unchanged. New `tests/test_ui_rerank_default.py`.
All notebooks and `ask_source.zip` regenerated (SHA-256 now starts `2c0a2cce`; supersedes
`3128bc40`); the Drive zip must be replaced before the next serve session. Ruff, mypy and the
offline suite pass, with one exception: in one full run `tests/test_remote_llm.py::
test_a_wrong_or_missing_token_is_refused` failed with `[WinError 10053]` (connection aborted by the
local host); the file then passed 5 of 5 repeats. Cause not established, not touched by this change.

### 2026-11-07: Tracing the code and docs changes from the 103-question run (fresh A100 session)

Reran code and docs (52 questions, no reranker, which these types do not use). Code 35/40, docs 7/12.
Compared with the 2026-11-06 full run, only two answers differ: code-032 now passes and code-033's
wording changed (still a miss); the other 50 answers are word for word the same.

- **code-032 (300 s, no tool call) was a server stall.** The error was `Model server unreachable: The
  read operation timed out` (client read timeout); the same question now answers normally with
  `search_code`. Not a tool-choice or retrieval problem.
- **code-035 ("Requests ignores my NO_PROXY environment variable") is a model abstention, not a
  retrieval miss.** The model's query was `NO_PROXY environment variable`; for that query the gold
  chunks `should_bypass_proxies` and `get_environ_proxies` rank 2 and 4 locally, yet the model
  answered "No relevant implementation was found in the retrieved context." Same result in both runs.
- **docs-046 and docs-047 are the known routing weakness.** "How do I use HTTP basic authentication?"
  went to `search_code` with the query `HTTP basic authentication`; the answer describes
  `HTTPBasicAuth` in `src/requests/auth.py` instead of the docs page.
- **Compared with the 2026-11-05 run (code 36/40, docs 8/12): code-035, docs-046 and docs-047 are
  newly missed, and docs-050 newly passes.** The prompt gained `search_prs` between the two runs, so
  a link is possible; I did not test it (nothing is committed, so the old prompt would have to be
  rebuilt by hand). Still open: code-006, 016, 033, 037, docs-045, 049, 051.

### 2026-11-07: Phase 8 started: GitHub permalinks in the Streamlit sources panel

- `ui/app.py`: new `source_link(repo, sha, source)` turns a `path:start-end symbol` source into
  `https://github.com/<repo>/blob/<indexed commit>/<path>#L<start>-L<end>`; `show_result` takes the
  repo and the manifest's commit and shows links, falling back to plain text for a source that does
  not parse (for example a path with a space) or when no commit is recorded. No new dependency.
- Three tests added to `tests/test_ui_app.py`; the full offline suite, ruff and mypy pass.
  Not checked in a real browser against the live index.
- `ask_source.zip` regenerated, SHA-256 now starts `b9c6bd31` (supersedes `2c0a2cce`); replace the
  Drive copy before the next serve session.
- Still open in Phase 8 (plan.md §Phase 8, not ticked): FastAPI `GET /repos` and streaming
  `POST /ask` (adds fastapi and uvicorn to `pyproject.toml`); `POST /repos` indexing, which cannot run
  on a laptop CPU (about 0.16 chunks/s) and so needs a decision: a GPU indexing server, or an
  endpoint that only imports an index built on Colab.

### 2026-11-07: Phase 8, API backend: `GET /repos`, streaming `POST /ask`, `POST /repos/import`

User approved the plan (read-only endpoints, then import of a Colab-built index) and the new
dependencies `fastapi` and `uvicorn` (plus `httpx` in the dev group for the test client; it was
already in `uv.lock` through other packages).

- `api/server.py`: `create_app(settings, agent_factory=, store_factory=)`; `uvicorn
  repo_rag.api.server:app` builds the app on first use.
  - `GET /repos`: slug, commit, history commit, indexed_at, chunk and file counts, and `searchable`
    (false when the repo was indexed under another embedding). No clone path.
  - `POST /ask` {repo, question, thread_id?}: server-sent events `start`, `step` per graph node
    (prestep, answer, check, remember), then `result` or `error`. Not token streaming: the model
    answers in whole turns (`RepoAgent.stream` in `agent/graph.py` yields node updates). Errors before
    the stream: 404 not indexed, 409 other embedding, 503 model server not configured, 502 unreachable.
  - `POST /repos/import`: raw zip body (no multipart, so no `python-multipart`), optional `?repo=`;
    cap `api_max_import_bytes` (2 GiB, 413 beyond it), one import at a time (409), temp file removed
    always, temp path not shown in errors, cached agent for an imported repo dropped.
  - The model token stays server-side. No authentication: localhost only (stated in the module
    docstring). Not checked: an `/ask` that reaches a live model; asking while an import runs.
- 13 tests in `tests/test_api.py` (fake agent; the import tests build a real index zip with the hash
  embedder). Started real uvicorn on the live index: `/health`, `/repos` (code 274, docs 282, commits
  4,883, issues 4,186, PRs 3,086), `/ask` 503 without model settings and 404 for an unknown repo.
- Full offline suite, ruff and mypy pass. `ask_source.zip` regenerated, SHA-256 starts `78651e79`
  (supersedes `b9c6bd31`); replace the Drive copy before the next serve session.
- **Phase 8 stays unticked in plan.md.** Its "done when" (a fresh user pastes a URL, waits for
  indexing, chats) is not met: indexing needs a GPU (about 0.16 chunks/s on a laptop CPU), so there is
  no endpoint that indexes a URL. The path is: build the index in `notebooks/index_on_colab.ipynb`,
  then upload the zip to `POST /repos/import`. The Streamlit UI does not use the API yet.

### 2026-11-07: First live `/ask`; follow-ups did not see earlier turns, now they do

- Ran the API on localhost against the A100 server (14B + 4B reranker, reranker off in the agent).
  `GET /repos`, then `POST /ask` "How does requests follow redirects?": SSE `start`, four `step`
  events, `result`; 3 valid citations (`sessions.py:186-235`, `801-829`), 61 s (includes the first
  embedding-model load). Line ranges not checked against the file. "Who calls resolve_redirects?" on a
  new thread: `find_callers` with the right symbol, 7 callers listed, 10 s (no sources: `find_callers`
  is not a chunk tool).
- **Finding:** thread memory was recorded but never shown to the model. "Who calls the function you
  just mentioned?" produced the placeholder `function_name`; "And what does it return?" abstained
  with no tool call. (`graph._answer` passed only `state["question"]`; the old test
  `test_memory_is_per_thread_and_not_fed_to_the_model` asserted exactly that.)
- **Change (user decision: option 1):** `graph.with_history` prefixes the question with the last
  `agent_history_turns` (2) answered turns of the same thread, answers cut to
  `agent_history_answer_chars` (600), abstentions skipped. With no answered turn the model input is
  byte-for-byte the old one (tested), so the single-turn eval is unchanged by construction. It is not
  a measurement: follow-ups themselves have not been evaluated. Template in `prompts.HISTORY_BLOCK`.
- Tests: the old test replaced by five (per-thread memory; follow-up sees own thread only; off at 0;
  last N and cut; abstentions skipped). `ask_source.zip` regenerated, SHA-256 starts `145a814f`.
- **Live check of the change (one run, new A100 session, thread of 3 turns):** "How does requests
  follow redirects?" answered with 3 valid citations (104 s, cold start). "Who calls it?" now
  resolved: `find_callers` with `resolve_redirects`, 6 callers listed (before: placeholder
  `function_name`). "And what does it return?" failed: no tool call, `failure: parse` ("the model's
  reply was not valid within the limits"). So one of two vague follow-ups works; not a pass. The
  caller answer cites `sessions.py:804-821`, which no check covers (`find_callers` is not a chunk
  tool, so `citations.valid` is 0). An earlier first attempt failed to connect (server not up) and
  is not counted.
- **Why the third turn failed (what is known):** `failure: parse` is reported only when the output
  did not hit the token cap (a capped output is `budget`), so the model ended its reply by itself and
  the reply broke the strict format (`parse_reply`: exactly one JSON object, exact keys, known tool).
  The raw text is stored in the trace (`turns[].generation.text`) but was not returned, so the exact
  reply was never seen. **Hypothesis, not measured:** the history block showed earlier answers as
  plain prose with no format reminder, which can lead the model to answer in prose.
- **Fix (also unmeasured):** (1) the history block now says the reply must still be exactly one JSON
  object and that a tool must still be called before a final answer; (2) a failed run now returns
  `raw_reply` (last model text, 800 chars, redacted), so the next failure can be read. Successful
  runs are unchanged. Tests added for both. The prompt change does not apply to first questions.
- **Flaky test explained and fixed.** `test_remote_llm.py::test_a_wrong_or_missing_token_is_refused`
  failed again in the full suite (`AssertionError`, earlier `WinError 10053`; 8 of 8 pass alone). Cause:
  `do_POST` replied 401/404 without reading the request body, and closing with unread data resets the
  connection on Windows, so the client sometimes saw a reset, not the 401. Reproduced with a 200 KB body
  (new test `test_a_refused_request_with_a_large_body_still_gets_its_401` failed on the old code with
  `WinError 10053`). Fix: `_drain()` reads and drops the body (up to `MAX_REQUEST_BYTES`) before the
  401/404. This server also runs on Colab, so the notebooks and `ask_source.zip` were regenerated.
- Not yet done from the entry above: the live 3-turn rerun and the eval rerun (see the 2026-11-08
  entry for the plan).

### 2026-11-08: Offline preparation for one combined Colab session

User approved (a) the A/B prompt-switch design and (b) the E3 run of Qwen3-4B-Instruct-2507. Nothing
below has run on a GPU yet.

- **Phase 7 fact gate was misread earlier, corrected.** The Phase 5b fact set is
  `eval/datasets/httpx_facts.jsonl` (60 facts, **encode/httpx**), not psf/requests, so it cannot be
  scored from the existing requests runs. httpx has no vector index here and `data/metadata.db` holds
  only requests. Tools alone (`eval/run_facts.py` on `data/answer_bakeoff_capsule`, 500 httpx PRs):
  **60/60**. New `eval/agent_facts.py` asks the same 60 questions through the agent; target 90%.
  Deviation to keep in view: search tools raise "no vector index", so it tests routing among the
  exact tools only, and the answer check accepts the expected value among other text. Wiring verified
  offline with a scripted model on the real metadata (`tests/test_agent_facts.py`, 7 tests).
- **UI is now a thin client of the API.** `ui/api_client.py` (stdlib only, no new dependency) reads
  `GET /repos` and the SSE stream of `POST /ask`; `ui/app.py` holds no model token, lists only
  searchable repos, shows the step progress and, for a failed run, `raw_reply`. New setting
  `api_url` (default `http://127.0.0.1:8000`). Removed `build_remote_agent`; the reranker-default test
  moved to the API's `default_agent_factory` (`tests/test_api_rerank_default.py`). A real-uvicorn
  round trip is tested (`tests/test_ui_api_client.py`). Not checked in a browser.
- **A/B switch `agent_prompt_variant`** (`baseline` default | `no_search_prs` | `docs_hint`).
  `baseline` is pinned byte for byte (SHA-256 `cf955a40...`, verified identical to the prompt text of
  the 103-question runs). `no_search_prs` drops the tool and its paragraph; `docs_hint` adds a rule
  sending "how do I use X" questions to `search_docs` first. `eval/agent_eval.py run --variant ...`
  tags the result file. None is measured, so none is the default.
- **E3:** `notebooks/build_summary_prose_eval.py` now sets `APPROVED_MODEL` (revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`). It is a separate notebook and needs its own Colab run.
- Full offline suite, ruff and mypy pass; every notebook regenerated. `ask_source.zip` SHA-256 starts
  `5395e28b` (supersedes `fbd0c3f5`, `518a3b19`, `145a814f`, `78651e79`). Replace the Drive copy
  `repo_rag_ask_source.zip` with it before the serve session. The E3 test was updated: approval is
  now set, and the gate still refuses any other model. The frozen capsule file
  `model_proposal.json` still says "pending" (immutable); the approval is this entry and the chat.
- **Results of the combined session (2026-11-08, A100, 14B + 4B reranker; every eval run with
  `--no-rerank`, single runs, psf/requests):**
  - *Follow-up thread (3 turns)*: all three answered with the right tool (`find_callers`, then
    `get_symbol`); no placeholder, no parse failure. One run, one thread. The turn-3 answer cites
    `sessions.py:186-307` from a tool that returns no chunks, so no check covers that range and it was
    not verified.
  - *Baseline, 103 questions*: code 35/40, docs 7/12, history 11/11, refs 10/10, kind 5/5, commit
    3/10, probe 5/5, prsearch 10/10. Identical question by question to the 2026-11-07 code/docs rerun
    (same 10 misses; commit-081/082/086 pass), 0 parse/budget/rerank failures. So the history change and
    the other offline changes cost nothing on single-turn questions in this run.
  - *Phase 7 fact gate, `eval/agent_facts.py` (httpx, 60 facts)*: **59/60 (98.3%)**, target 90%.
    The miss is `blame_line:httpx/_config.py:114`: right commit, SHA shortened to 10 characters
    (`26cd4f54e8`) against the expected 12 (`26cd4f54e885`); left as a miss, the check was not loosened.
    The model used only exact tools, never a search tool, 5.6 minutes total. Deviations: no httpx vector
    index (search tools unavailable), 500 saved PRs, and an answer passes if it contains the expected
    value among other text. This scores routing among the exact tools, not retrieval.
  - *`docs_hint`, code+docs (52)*: docs 10/12 (baseline 7/12), code 34/40 (35/40); total 44 vs 42.
    Fixed docs-045/046/047, the questions the hint was written for (so not an independent test); lost
    code-011 and code-031 (code questions routed to `search_docs`); code-035 flipped to pass (it flips
    between runs). The hint is too broad. Not adopted; `baseline` stays the default.
    **Correction found afterwards:** the hint's three examples ("set a timeout", "basic
    authentication", "configure proxies") were the subjects of docs-045, 046 and 047 themselves, so
    the +3 docs result is contaminated (the prompt named the answer's route) and says little about
    unseen questions. The lost code questions are unaffected by that. The new `docs_hint_narrow`
    uses examples that no evaluated question mentions, enforced by a test.
  - *`no_search_prs`, code+docs+prsearch (62)*: code 36/40, docs 8/12, prsearch 4/10 (expected: the
    tool is gone, 2 of the 6 misses were parse failures with no tool call). Versus baseline on
    code+docs: gained code-006, code-035, docs-046, docs-047; lost code-032 (routed to
    `search_commits`) and docs-050. code-035, docs-046 and docs-047 passed in the 2026-11-05 run
    (before `search_prs` existed), missed in the 11-06, 11-07 and 11-08 baseline runs, and pass here.
    That is consistent with the `search_prs` addition having cost those three, but it is three
    questions in single runs and is not shown. The trade is visible: `search_prs` is worth 6 of the 10
    PR-search questions. Nothing adopted.
- **UI checked in a real browser (built-in browser pane, API and Streamlit on localhost):** the page
  loads, takes the API URL from `REPO_RAG_API_URL`, lists `psf/requests` as searchable and shows the
  chat box. With no model server configured, a question shows the API's 503 message under the
  answer box, not a traceback. Found and fixed: the status box kept saying "Working..." after a failed
  question; it now says "Failed" with an error icon. Not checked in the browser: a successful answer
  (needs a model server), the sources links, and "New chat".
- Full suite, ruff and mypy pass after these changes; `ask_source.zip` SHA-256 starts `6afc2787`
  (supersedes `fea67577`, `5395e28b` and the earlier ones). Upload it before the next serve session.
- **Stage 1 A/B results (2026-11-08, second GPU session, zip `6afc2787`; 3 interleaved passes per arm,
  62 questions each (code 40, docs 12, prsearch 10), `--no-rerank`, psf/requests):**

  | arm | code | docs | prsearch |
  |---|---|---|---|
  | baseline | 35/40 | 7/12 | 10/10 |
  | `docs_hint_narrow` | 36/40 | 10/12 | 10/10 |
  | `prs_narrow` | 35/40 | 7/12 | 6/10 |

  - **The three passes of every arm were identical** (same scores, same failing ids), so decoding is
    deterministic and repeats measure nothing about chance. This corrects the earlier remarks in this
    log that code-035 "flips between runs" (noise): those runs differed in prompt/code, and each
    difference is a real effect of the prompt on that exact question. What stays unknown is
    generalisation beyond these questions.
  - `docs_hint_narrow` gained docs-045, docs-046, docs-047 and code-035 and lost nothing; the broad
    hint's losses (code-011, code-031) did not recur. Its examples ("turn on debug logging", "send a
    compressed request body") are in no evaluated question, but the rule's "how do I / how can I"
    wording was written knowing the three docs questions start that way, so the gain on them is on
    questions that shaped the design. Not adopted yet: needs a fresh question set (below).
  - `prs_narrow` lost prsearch-099 to 102 in every pass. The raw replies (recorded from pass 2 on, 8 of
    8) all have the right tool and query but `"type": "search_prs"` instead of `"type": "tool_call"`:
    a format slip the strict parser rejects (`Unknown reply type`), triggered by the paragraph's
    "Call search_prs only when..." wording. Rejected. The same slip may explain the failed third turn
    of the follow-up thread earlier (raw reply was never captured) and two of the `no_search_prs`
    failures (not checked). `eval/agent_eval.py` rows now store `raw_reply`.
  - **Open decision (user):** add a production protection against this slip? (1) retry once with the
    parse error fed back, (2) accept `type` == a known tool name when `name` matches, or (3) keep the
    parser strict. Not built.
  - **UI live check (built-in browser, API + Streamlit, real model):** a question shows the step
    label, then "Done", the citation line and 4 source links to GitHub at the indexed commit
    (`611c6162...`, `#L186-L235` anchors); the follow-up "Who calls it?" resolved through the thread;
    "New chat" clears the conversation. Small unclear point: a callers answer that cites a line range
    shows "0 citation(s) match retrieved code" because `find_callers` returns no chunks to check
    against; the range itself (`sessions.py:804-821`) was not verified.
- **Parse-failure protection built (user chose option 1, retry with feedback); not yet measured live.**
  `run_question(parse_retries=...)` answers a reply that fails the format check once with a fixed
  user message (`prompts.PARSE_RETRY`) that names the reason (from the parser's own check, e.g.
  "Unknown reply type") and restates the two legal shapes, then continues. Rules: production
  toolset only (the bake-off/evaluated toolset stays strict, tested); a reply cut off by the output-
  token cap is never retried; a retry uses one generation turn, so `agent_max_turns` still bounds the
  run; the second bad reply fails as before and returns `raw_reply`. Setting `agent_parse_retries`
  (default 1, 0 turns it off). The result and `eval/agent_eval.py` rows carry `parse_retries`. It can
  only act on replies that would have failed anyway, so baseline runs (0 parse failures) should be
  unchanged; that is to be confirmed live. **The live test:** rerun `prs_narrow` (its 099 to 102
  failures are exactly this slip): if the feedback works they should pass with `parse_retries` 1.
  6 new tests in `tests/test_agent_graph.py`. Full suite, ruff and mypy pass; `ask_source.zip`
  SHA-256 starts `f0fde72b` (supersedes `6afc2787`).
- **Fresh question set built (offline, not yet run):** `eval/build_fresh_questions.py` writes
  `eval/datasets/agent_eval_fresh_v1.jsonl`: 10 usage questions ("How do I ...", scored by naming the
  covering doc file) and 10 code questions ("How does ...", scored by a cited range overlapping the
  gold function). Written after the hint, from the docs and the source at `611c6162`: usage topics
  were located in the docs by reading them; code gold ranges are `ast` `def` ranges; the build
  refuses any question or gold range that repeats `agent_eval_v1`. Not reviewed by a person. Purpose:
  baseline vs `docs_hint_narrow` on questions the rule was not written around (the code half checks
  the hint does not pull "how does" questions into `search_docs`). `agent_eval.py run --dataset`
  selects the file; `tests/test_fresh_questions.py` guards the set (4 tests).
- **Next GPU session, one serve session, zip `f0fde72b`:** (1) fresh set: baseline and
  `docs_hint_narrow`, one pass each (20 questions; decoding is deterministic so one pass is the
  measurement); (2) `prs_narrow` on code,docs,prsearch to see whether the parse retry rescues
  prsearch-099..102 (`parse_retries` in the rows); (3) baseline on code,docs,prsearch to confirm the
  retry changed nothing where there were no parse failures. About 30 to 35 minutes of GPU.
- **Prepared offline after the session, not yet run:** variants `docs_hint_narrow` (usage rule only
  for "how do I / how can I" phrasings, clean examples) and `prs_narrow` (the `search_prs` tool kept,
  its paragraph limited to questions that say pull request, PR or issue), and
  `agent_eval.py run --repeat N` (each repeat builds a fresh agent, so no thread carries over).
  `baseline` is still pinned by SHA-256.
- **Still to do after this session (now built, see above):** a narrower `docs_hint` ("How do I..." phrasings only, no
  symptom or "how does" wording) and a reworded `search_prs` paragraph, each measured against
  `baseline` the same way, ideally with repeated runs; E3 (separate notebook); human review of the
  architecture labels.
- (Superseded plan, kept for the record) **To do on a GPU (one serve session, then the E3 notebook separately):** the 3-turn thread
  (a failure now shows `raw_reply`); `eval/agent_eval.py run` for `baseline`, then `--types code,docs`
  for `docs_hint` and `no_search_prs` (and `--types prsearch` for `no_search_prs`); the facts run
  `eval/agent_facts.py`; the 103-question baseline to confirm single-turn scores did not drop. A single
  run per arm cannot separate a real effect from noise on 12 docs questions: report it as such.

### 2026-11-09: GPU session: fresh question set and parse retry (A100 40 GB, zip F0FDE72B, `--no-rerank`)

One deterministic pass per arm, so these are real effects on these exact questions only.

- **Fresh set (`agent_eval_fresh_v1`, 10 usage + 10 code, written after `docs_hint_narrow`):**
  baseline usage 1/10 (2 abstained, 2 failed; 8 of 10 went to `search_code`), code 8/10;
  `docs_hint_narrow` usage 9/10 (all 10 to `search_docs`), code 8/10. The same two code misses
  (code-f01, code-f10) under both. Remaining usage miss: docs-f08 (right tool, expected doc not named).
- **Parse retry (`prs_narrow`, code/docs/prsearch):** code 35/40, docs 7/12, prsearch 6/10.
  prsearch-099..102 still fail with `failure=parse`, `parse_retries=1`: after the feedback message the
  model repeated `{"type": "search_prs", "name": "search_prs", ...}`. **The retry does not rescue this
  slip.** (Retry is in the default path; it did not change baseline, below.)
- **Baseline re-run:** code 35/40, docs 7/12, prsearch 10/10, the same as before the retry existed.
- **Decision (user, 2026-11-09): `docs_hint_narrow` is now the default** (`Settings.agent_prompt_variant`).
  `baseline` stays pinned by SHA-256 and is still what `eval/agent_eval.py` runs unless `--variant` is
  given, so evals stay comparable. Notebooks regenerated; the Drive zip must be re-uploaded before
  the next serve session. `prs_narrow` stays rejected.
- **Serve notebook auto-release:** the keep-alive cell now stops the server and tunnel and calls
  `runtime.unassign()` after `IDLE_MINUTES` (10, changed from 20 at the user's request) without a generate or rerank request, or after
  `MAX_HOURS` (4). `serve_model(...)` returns a server with `idle_seconds()` (health pings do not
  count; a running request counts as activity). Test: `test_idle_seconds_counts_real_work_only...`.
  Full suite passes; new zip hash starts B8291C0D. Note: pytest's default temp folder
  (`%TEMP%\pytest-of-user`) returned `PermissionError: [WinError 5]` on this machine, so the suite was run
  with `--basetemp` pointing elsewhere; not a code failure.

### 2026-11-09: Commit questions at 3/10: offline diagnosis (laptop retrieval, no model)

Agent results for the 10 commit questions (last run, 20261108T204707): passes 081, 082, 086;
wrong SHA 079, 080, 083, 084, 088; abstained 085, 087. Every question used `search_commits` once.
Replayed retrieval on the local `psf/requests` commits index (4,883 commits), rank of the gold
commit in the top 50, using the bug title or the whole report, with and without `is_fix=true`
(the agent tool applies no filter and the model writes the query, which is not saved in old rows;
the title is only a stand-in for it):

| Query | gold in top 5 | top 10 | top 50 |
|---|---|---|---|
| whole report + is_fix | 6/10 | 7/10 | 10/10 |
| whole report | 5/10 | 5/10 | 8/10 |
| title + is_fix | 5/10 | 6/10 | 8/10 |
| title | 5/10 | 5/10 | 8/10 |

Reading (a diagnosis from stand-in queries, not a measurement of the agent's own queries):
- Ceiling: with the best of these the gold commit is in the top 5 for 6/10, and the agent got 3 of
  those 6 (081, 082, 086). It lost 083 (gold rank 2, answered a rank-3 commit that touches the same
  files), 084 (gold rank 1 for the title; the SHA it gave is not in the title's top 50, so its own
  query must have differed) and 085 (gold rank 3 with the report, yet it abstained with "no relevant
  implementation").
- Retrieval misses: 079 (best rank 28), 080 (13), 087 (11), 088 (9) are outside the top 5 under every query.
- Several wrong answers are near neighbours of the gold commit (083: four of the top five touch the
  same two sidebar templates), so part of the shortfall is a single-gold scoring effect.
- Correction, same day: the "whole report" rows above strip the question's boilerplate sentence ("Was
  something like this fixed before in this repository? ..."). The agent's `question` mode sends the
  question as typed, boilerplate included, and that replays worse: gold ranks 27, 10, 1, 1, 2, 1,
  20, 1, 24, 8 with `is_fix` (top 5: 5/10, top 10: 6/10) and None, 44, 5, 1, 2, 2, 31, 1, 35, 14
  without it (top 5: 5/10). So the expected retrieval gain of `--commit-query question` over the
  model's own query is about zero in the top 5; the switch exists to measure, not because it is
  expected to win. Built (default off): `Settings.agent_commit_query` (`model` | `question`),
  `Settings.agent_commit_fix_only`, and `eval/agent_eval.py --commit-query/--commit-fix-only`.
- **Measured on the agent (same day, A100, `--no-rerank`, baseline docs_hint_narrow prompt, 10 commit questions, one deterministic pass per arm):**
  model's own query 3/10 (2 abstained); `question` 4/10 (2 abstained); `question` + fix-only 4/10 (2 abstained);
  model query + fix-only 2/10 (1 abstained). The only gain is commit-084 (`question` and `question`+fix
  pass it); commit-080 moves from a wrong SHA to an abstention. 081, 082 (not under model+fix), 086 pass in
  every arm; 079, 083, 088 miss in every arm. The model's own queries are sensible (for example
  "fix sidebar layout after Gittip rename", "301 redirect pyopenssl SNI"): query drift is not the main cause.
  **Decision: defaults stay `model` / no fix filter**; +1 of 10 is not a result to act on, and fix-only
  with the model's query lost a question.
- **code-016 and code-037 are not citation-range bugs (checked against the source and the local index).**
  Gold ranges are right (`extract_cookies_to_jar` 135-150, `Session.close` 883-886, both verified in the
  checkout at the indexed commit 611c6162). code-016: the top `search_code` hit is the merged chunk
  cookies.py:114-161 whose label is `MockResponse` but whose `symbols` metadata and body include
  `extract_cookies_to_jar`; the model instead cited cookiejar_from_dict (579-625, rank 2), which is a real chunk
  range. A selection error. code-037: `Session.close` is inside the chunk sessions.py:831-886
  (label `Session.merge_environment_settings`, `symbols` includes `Session.close`) at rank 7; the tool
  returns 5, so the model never saw it and abstained. Not fixed: raising k is blocked because the evaluated
  `search_code` is frozen at k=5 and 8 chunks (~2,000 characters each) would exceed the 12,000-character
  tool-result cap. Correction: the "merged chunk's symbols in the embedded header" idea already exists
  (`Settings.header_defines`, a `# defines:` line) and was measured on the A100 on 2026-10-25: neutral
  (MRR@10 -0.001 [-0.038, +0.032] hybrid, complete proof -0.008), so it stays off (eval/RESULTS.md,
  "Phase 4 header run"). It would not have helped code-016 anyway: the right chunk was rank 1 and the model
  already sees its `symbols` metadata. Re-running it for code-037 alone would be a one-question anecdote.
- **UI citation wording fixed.** The citation report now carries `checked` (True only when the check ran
  on retrieved code/docs chunks and the model did not abstain). The UI said "0 citation(s) match
  retrieved code" for caller, commit, history and count answers, where nothing could be checked; it now
  says "Citations were not checked: ...". It also shows `unverified_shas` (commit ids in the answer that
  no tool returned), which the graph already computed but the UI never displayed. A server that does not
  send `checked` is read as checked. Tests: `test_describe_says_when_nothing_could_be_checked...`.
- **Decision D1 (user, 2026-11-09): Phase 8's "done when" is reworded.** A fresh user indexes a repo on a
  Colab GPU with the notebook, imports the zip, and chats in the UI with source links at the indexed
  commit. No `POST /repos` URL indexer will be built. plan.md Phase 8 and docs/FINISH_PLAN.md updated.
  Phase 8 stays unticked until the one remaining check passes: asking a question while an import runs.
- **Asking during an import, checked offline (2026-11-09; TestClient, scripted agent, real zip import
  held mid-flight from another thread).** Before the fix: `/health`, `/repos`, asks about other repos and
  the refusal of a second import (409) all worked, but an ask about the repo being replaced went through
  with a 200, while its index files were being rewritten one after another (a mix of old and new chunks
  was possible). Fixed: `POST /ask` now returns 409 "is being replaced by an import" for the repos in the
  zip (read up front by the new `transfer.archive_repos`, or the `?repo=` list), and the block is lifted
  when the import ends, including when it fails. Not covered: a question already streaming when an
  import starts is not interrupted. Tests: `test_asking_while_an_import_runs`,
  `test_a_failed_import_releases_the_repos_it_was_replacing`, `test_archive_repos_...`.
- **Phase 8 ticked in plan.md (user decision, 2026-11-10).** Basis: the reworded "done when" (D1), the
  API and UI built, and the check above. Full suite and ruff passed at zip hash 3A35E4E7. Still not
  tried: the new `/ask` guard in a live Colab session.
- Not tested: tool with a system-supplied whole-report query, `is_fix` filter in the tool, a prompt
  that stops abstaining on commit results. `eval/agent_eval.py` now stores each call's arguments
  (`calls`) so the next run shows the model's actual queries.

### 2026-11-10: Phase 7 answer-quality judge: D3 decided, check built (not yet run)

- **D3 (user, 2026-11-10): a quantized Qwen model as the judge.** The user wrote "Qwen3.7-28B"; no such model
  exists. Hugging Face lists **Qwen3.8-27B** (official `Qwen/Qwen3.8-27B`, Apache-2.0, 28B in the safetensors
  metadata) and treated as that. Official quantized release: `Qwen/Qwen3.8-27B-FP8`; third-party 4-bit:
  `cyankiwi/Qwen3.8-27B-AWQ-INT4`. Default FP8 (about 28 GB, tight on an A100 40 GB at a 16k context);
  fall back to the INT4 if it runs out of memory. The "reuse the Phase 4 judge" default did not apply: that
  judge is the Qwen3-Reranker, which scores query/document pairs and cannot grade an answer.
- **Evidence for the model choice is thin.** Qwen3.8-27B JudgeBench 74.3 / RewardBench 93.8 (Qwen3.5-27B 77.4 /
  90.5, Qwen3-32B 66.9 / 88.2) come from one vendor paper (JEV, arXiv 2609.26550) quoted in a GitHub issue; the
  paper itself was not opened. Nothing was tested on this project's answers. Self-preference is possible (judge
  and answer model are both Qwen) and is not tested.
- **What was built:** `eval/judge_answers.py` (prepare / judge / report), frozen protocol
  `eval/datasets/phase7_answer_judge_v1/protocol.json`, `notebooks/build_answer_judge.py` ->
  `notebooks/answer_judge_on_colab.ipynb` + `notebooks/answer_judge_bundle.zip`, `tests/test_answer_judge.py`
  (6 tests, offline, scripted judge). Scope: the 52 `code` and `docs` answers of
  `eval/agent_eval_results/agent_eval_production_20261109T013636.json` (50 judged, 2 abstained; Coder-14B,
  baseline prompt, no reranker). Other types are scored exactly by `agent_eval.py`. For each answer the judge
  grades faithfulness (against the cited lines, read with `git show` at commit 611c616) and correctness (against
  the gold chunk for code, the start of the reference files for docs). 96 negative controls (answers paired with
  another question's evidence) must be rejected at >= 90% or the judge is not trusted; unparsable replies <= 5%.
- Dry run of `prepare` on the laptop: 194 judge prompts (98 on answers, 96 controls), longest 15.4k characters;
  all citations resolved. Nothing has run on a GPU.
- Not measured: agreement with a person; docs correctness sees only the start of reference files.
- The notebook's last cell zips the results (also partial ones plus `RUN_ERROR.txt` when a step fails), copies the zip to My Drive, downloads it, waits 60 s and calls `runtime.unassign()` to release the A100. If an earlier cell (unpack, install, model download) fails the run stops before that cell: disconnect by hand. Not yet tried in a real session.
- **First Colab try failed in `prepare` (2026-11-10):** the bundle normalised CRLF to LF in every file, so the question file's sha256 no longer matched the one recorded in the answers file and `prepare` exited ("produced from a different question file"). Fixed: only `.py` files are normalised; a test checks the bundled hash; the notebook now prints each step's own output.
- **Second Colab try failed in `judge` (2026-11-10):** vLLM 0.33.0 had been installed into Colab's system Python with `--torch-backend=auto`; its torch (CUDA 13.2) clashed with Colab's preinstalled torchaudio (CUDA 13.0), so `import vllm` raised. Fix: vLLM now goes into its own venv (`/content/vllm_env`) with `vllm==0.33.0` and `--torch-backend=cu130` (`uv pip compile` resolves torch/torchaudio 2.14/2.11 +cu130 together; cu132 has no solution); the install cell imports vLLM (and torchaudio) in the venv so it fails early; `limit_mm_per_prompt` image/video 0 because Qwen3.8-27B-FP8 is a `Qwen3_5ForConditionalGeneration` with a vision tower. The `LLM.chat` and `LLM` arguments were checked against the v0.33.0 source. Judge plumbing run end to end with a fake vllm module (prepare, judge, report, quantization refusal); a real GPU run is still pending, as is FP8 block-quantized weights on an A100 (compute 8.0).
- **Third Colab try failed in `judge` after the model loaded (2026-11-10):** vLLM's engine died in its warm-up sampler: FlashInfer compiles a CUDA kernel on first use (`FileNotFoundError: ninja`; the venv's `bin` was not on PATH, and `nvcc` was never checked). Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` (set in the notebook env and, with PATH, inside `judge_answers.py`; greedy decoding does not need it) and `enforce_eager=True` (protocol `runtime`; no torch.compile or CUDA-graph build, fewer first-run failure points, slower decode that does not matter for 194 short replies). Checked offline: a fake `vllm` module that raises the same `ninja` error unless the sampler is off (test `test_judge_step_runs_against_a_fake_vllm_without_flashinfer_jit`, also shown to fail when the flag is forced on), and the generated notebook cell run end to end (clone, prepare, judge, report) against the fake. Real-GPU result still pending. vLLM source checked for v0.33.0: `LLM.chat` / `LLM(...)` arguments, `VLLM_USE_FLASHINFER_SAMPLER`.
- **Compatibility audit before the fourth try (2026-11-10).** (1) `uv pip compile` for vllm 0.33.0 on linux, Python 3.13, cu130: torch 2.14.0, torchaudio 2.11.0, flashinfer 0.7.2.post1, ninja 1.13.2, nvidia-cuda-nvcc 13.5.37, transformers 5.20.2, compressed-tensors 0.17.0, numpy 2.4.6; one consistent set. cu132 has no solution. The previous run reached vLLM's sampler warm-up on the GPU, so the cu130 torch works with Colab's driver. (2) **Default judge changed from the official FP8 to `cyankiwi/Qwen3.8-27B-AWQ-INT4`**: the FP8 release is block-wise (128x128) FP8 with dynamic activations, an A100 (sm80) has no FP8 hardware, vLLM's source did not show which kernel serves block FP8 there, and one community report says FP8 gave garbage on an A100; the 4-bit build is compressed-tensors W4A16, group size 32, about 21 GB, which vLLM runs with Marlin on Ampere (not run here). Vision tower and most linear-attention layers are left unquantized in it. (3) New preflight cell (fails fast, before the download): GPU capability >= 8.0, free GPU memory >= 0.92 x total, disk >= 45 GiB, RAM >= 24 GiB; warns on a missing `Python.h` or `gcc` (Triton builds a launcher with them). Tested with a fake torch (A100 passes; an sm75 GPU and a GPU with 20 GiB free fail). Still unverified on Colab: that Marlin serves this 4-bit checkpoint, memory at a 16k context, and `Python.h` in the venv.
- **No A100 available (2026-11-10): the CPU-only part was built.** `eval/review_sheet.py` turns the judge's prepared items into a human review sheet (`eval/review/answer_review_v1/review_sheet.md` and `.csv`: each of the 50 answers beside the exact cited lines and the reference lines; a person fills `faithful` and `correct`). `compare` then gives person-vs-judge agreement and `disagreements.md` (judge vs mechanical score, judge vs person) once the judge has run. A person's marks are the only evidence about whether the judge can be believed (it was listed as not measured). Tests: `tests/test_review_sheet.py` (3). The sheet alone is also a valid fallback for the Phase 7 answer-quality check if no GPU turns up: fill it and record the rates by type. Another session was editing `src/repo_rag/cli.py` and `ingest/updater.py` (incremental re-index) at the same time, so notebook-freshness tests failed in my full-suite runs until its notebooks are regenerated, and overlapping pytest runs produced Chroma "database is locked" errors; neither comes from this work.
- vLLM is installed unpinned in the notebook (the version is recorded in `judge_run.json`); whether FP8 weights
  load on an A100 with it is untested.

### 2026-11-10: README rewritten from measured results

- `README.md` replaced (it still said "early scaffolding"). It has a mermaid architecture diagram, the
  quickstart, and an eval section whose every number was copied from `eval/RESULTS.md` or this log:
  chunking completeness (Requests 0.654 / 0.625 / 0.842, Express 0.725 / 0.725 / 0.825 for char /
  language-aware / AST) with the "ranking not consistently better" and "astchunk ties us" caveats; the
  embedding table (0.738 / 0.482 for Qwen3-0.6B @512d); 8B vs 4B rerank (0.860 -> 0.926); the Phase 5
  commit table (263 questions, Hit@5 0.920); httpx facts 60/60 tools and 59/60 through the model; the
  103-question agent table; and a known-limits list. Commit questions are shown at 3/10 as the weak spot.
- The agent table mixes two prompts on purpose and says so: baseline for the full 103-question run,
  `docs_hint_narrow` (the default) only where it was measured (code 36/40, docs 10/12, prsearch 10/10).
  The default prompt has not been run on history, refs, kind, commit or probes.
- Not in the README because not measured or not done: a demo GIF, laptop latency, a second repository,
  the Phase 7 answer-quality judge result (check built, not run). Update the table when those exist.

### 2026-11-10: Phase 6 closing work: the "coherent architecture answer" check did not exist, now built

- **Finding:** `docs/FINISH_PLAN.md` Stage 4 called it "the frozen coherent-architecture acceptance check", but
  nothing in the repo defines one (searched md/json for "coherent": only plan.md and the plan notes). The only
  frozen architecture work is retrieval (E1/E2: file Recall@5), which is not the same as an answer. Corrected
  `FINISH_PLAN.md`.
- **Built (offline):** `eval/agent_eval.py build-architecture` writes `eval/datasets/agent_eval_architecture_v1.jsonl`
  (15 psf/requests questions taken unchanged from `architecture_v1/requests.jsonl`, sha256 prefix `ac3086e572fb`).
  New scoring type `architecture`: an answer passes when it is not an abstention or failure and names at least
  half of the question's gold files, rounded up (`need`). It is a coverage proxy for "module-level": it does not
  check that the explanation is right, so a person must still read the answers. `agent_eval_v1.jsonl` is byte
  identical to before (compared). Tests: `test_architecture_needs_half_the_gold_files_named`,
  `test_architecture_question_file_is_built_from_the_frozen_labels`; ruff clean.
- **Only requests can run:** the 15 Flask questions need a Flask index, and `data/repos` has no Flask clone and no
  Flask index here, so they are not in this file.
- **Still open for Phase 6 (nothing run yet):** (1) the pass gate for this check is **not decided**; proposed,
  to be confirmed by the user before any run: at least 10 of 15 pass mechanically and the user reads the 15
  answers (labels are unreviewed, so a pass is weak evidence); (2) D2, a person reviewing the 30 labels
  (`docs/ARCHITECTURE_LABEL_REVIEW.md`); (3) E3 prose run (`summary_prose_on_colab.ipynb`, approval cell, user's
  Colab). Run command once decided: `uv run python eval/agent_eval.py run --url ... --token ... --dataset
  eval/datasets/agent_eval_architecture_v1.jsonl --variant docs_hint_narrow`.

- **Gate decided (user, 2026-11-10): at least 12 of 15** architecture questions pass mechanically (the user wrote
  "12-13 of 15 is fine"; read as a floor of 12, since 12 already meets it), and the user reads the answers. Fixed
  before any run; the question file, the "half the gold files" rule and this gate must not change after seeing
  results. Phase 6 still also needs D2 (label review) and the E3 prose run, or a recorded "built, not adopted".

### 2026-11-10: the model-server source zip is built under its Drive name

- `notebooks/build_ask_colab.py` now writes `notebooks/repo_rag_ask_source.zip` (was `notebooks/ask_source.zip`),
  the exact name the ask and serve notebooks look for on Drive, so it is uploaded as built, with no renaming.
  The old `ask_source.zip` was deleted (generated file). The zip content is unchanged (hash starts `3a35e4e7`);
  only the serve notebook's instruction text changed. Earlier entries in this log that say `ask_source.zip`
  mean this file.

### 2026-11-10: Phase 9, incremental re-index: measure first, then `repo-rag update`

- **Finding that shaped the work:** re-running `repo-rag index` was already incremental where it costs
  (`VectorStore.sync` embeds only new or changed text). So the open question was what was still wasted. A
  chunking-only measurement on psf/requests (no model; the old tree exported with `git archive` into a scratch
  git repo; script in the session scratchpad, not in the repo) compared the stored chunks at an older commit
  with the chunks at 611c616 (HEAD):

  | window | files changed | code chunks to embed (id-based) | with same-text reuse | docs (id-based) | with reuse |
  |---|---:|---:|---:|---:|---:|
  | HEAD~3 | 3 | 0 | 0 | 3 | 3 |
  | HEAD~10 | 8 | 0 | 0 | 9 | 9 |
  | HEAD~30 | 25 | 93 | 45 | 82 | 21 |
  | HEAD~100 | 62 | 235 of 274 | 194 | 155 of 282 | 44 |

  Chunking the whole repo takes 0.5 to 0.9 s, so skipping unchanged files would save nothing measurable and was
  not built. The waste is elsewhere: a chunk id includes its start line, so code that only moved (a line added
  above it) gets a new id and was re-embedded although its text is identical. One repo, Python and docs only,
  counts of chunks (not embedding time); the table is a prediction of what the new sync does, checked by unit
  tests with fake embeddings, not by a real embedding run.
- **Built:** (1) `VectorStore.sync` copies the stored vector of a chunk with identical text instead of calling the
  model (`SyncStats.reused`; `embedded` = added + updated - reused; `force` turns it off) and has `dry_run=True`,
  which writes nothing and loads no model. (2) `src/repo_rag/ingest/updater.py` and `repo-rag update <owner/repo>`
  (`--dry-run`, `--no-pull`, `--yes`, `--checkout PATH` for an index imported from Colab, which records no
  clone): fast-forward the clone, print the plan (chunks to embed per collection, reused, removed, unchanged,
  files changed in git), refuse above `update_embed_limit` (200, a guard, not a measurement) unless `--yes`, then
  index and refresh the symbol table, summaries and commit history if each existed before. History is refreshed
  offline, so new commits are not linked to issues until `index-history --github`.
- **Not done:** no real embedding run through `update` (the dev index was imported from Colab; run
  `uv run repo-rag update psf/requests --checkout data/repos/psf__requests --no-pull --dry-run` to see its plan);
  PR/issue sync (`sync-github`) and `index-prs` are not part of `update`; the Phase 9 webhook/Action that would
  call `update` on push is not built.
- A bug caught on the way: my first edit left a helper function in the middle of the `VectorStore` class, which
  silently turned the methods after it (including `count`) into dead nested functions; the new updater tests
  found it. Fixed by moving the helper to module level.

### 2026-11-10: Phase 6 architecture-answer check run: 5/15, the gate (12/15) is not met

- Run (A100 40 GB, Coder-14B, 4B reranker on the server, prompt `docs_hint_narrow`, dataset sha256 prefix
  `ac3086e572fb`, single deterministic pass, psf/requests only):
  `eval/agent_eval_results/agent_eval_production_docs_hint_narrow_agent_eval_architecture_v1_20261110T224945.json`.
  **5 of 15 pass** (004, 009, 013, 014, 015); 1 abstained (001); 0 failures. Gate decided beforehand: at least 12.
- **Pattern, from the raw rows:** the model made exactly one `search_code` call per question (15 of 15; no
  `find_callees`, no second search) and then answered. Of the 9 answered misses, 9 name exactly 1 of the 2 files
  they needed (for example 002 and 003 name 1 of 3 and 1 of 4 gold files). So the answers are one-module
  answers to questions that span several. This matches the earlier observation (2026-11-04) that architecture
  questions stop after one search; the cause here is the number of searches and what 5 chunks cover, not the
  citation check.
- **Reading:** the criterion "a coherent module-level answer" is **not met** with this agent and prompt. The
  check is a coverage proxy on unreviewed labels, so it can understate quality (an answer can be right and name
  fewer files than the label lists); no answer was read by a person, so how many of the 10 misses are truly
  incomplete is unmeasured. Together with the E1/E2 retrieval result (router lowers Recall@5 0.8292 to 0.7042),
  Phase 6's summaries work stays "built, not adopted" and Phase 6 stays unticked.
- **Not tried, and not run after seeing this:** a prompt line that tells the model to search again or use
  `find_callees` for questions that span modules; a larger `search_code` k for architecture questions. Either
  would be a new arm tested on these same 15 questions, so it must be reported as tuned on them, with fresh
  questions for any claim. The gate and the question file are unchanged.
- The Colab URL and token were used in one command and are in no file.

### 2026-11-10: `architecture_hint` prompt arm: built and its test protocol fixed before any run

- **Variant:** `architecture_hint` = `docs_hint_narrow` plus one paragraph (search at least twice with a different
  query per part, then `find_callees`/`get_symbol`, name every file, do not answer from the first search). No
  library-specific examples. It was written after seeing the 5/15 result, so it is tuned on those 15 questions.
  Test: removing the block gives `docs_hint_narrow` exactly. The default stays `docs_hint_narrow`.
- **Fresh set:** `eval/datasets/agent_eval_architecture_fresh_v1.jsonl`, 10 psf/requests questions (sha256 prefix
  `98185a299eaf`) on topics not in architecture_v1: proxies, netrc, URL preparation, status errors, JSON decoding,
  digest auth, body length / chunking, closing, text encoding, default headers. Three gold files each, `need` 2.
  Written after the 5/15 result; each named function was checked to exist in the source by hand; no one else
  reviewed them. Build: `uv run python eval/agent_eval.py build-architecture-fresh`.
- **Protocol (fixed now, before any run; same scoring as before, "names at least half the gold files"):**
  one serve session, three runs: (a) `docs_hint_narrow` on the 10 fresh questions, (b) `architecture_hint` on the
  same 10, (c) `architecture_hint` on the original 15 (reported as tuned, descriptive only).
  - The arm counts as working for architecture questions only if (b) reaches **8 of 10** (the same 80% as the
    12/15 gate) and beats (a) on the same questions.
  - Phase 6 is not ticked on (c) alone. A pass on both (b) and (c) >= 12/15 is a recommendation to the user, who
    decides; the labels are still unreviewed.
  - It becomes the default prompt only if the 62-question code + docs + prsearch run does not lose against
    `docs_hint_narrow` (code 36/40, docs 10/12, prsearch 10/10); a loss means it stays an architecture-only
    option or is dropped. Single deterministic runs, one repo.
- Source changed (`prompts.py`, `config.py`): notebooks regenerated, `ask_source` zip hash now starts `57c2a57e`.

### 2026-11-11: `architecture_hint` result: not working, no change to the default

- **(a)** `docs_hint_narrow` on the 10 fresh questions: **4/10**. **(b)** `architecture_hint` on the same 10:
  **4/10** (gate was 8/10 and beating (a); neither met). **(c)** `architecture_hint` on the original 15: **6/15**
  (the baseline arm got 5/15; gate for Phase 6 is 12/15, not met).
- **What happened:** in all 35 questions of the three runs the model called `search_code` exactly once and then
  answered, so the paragraph asking for at least two searches did not change its behaviour. The differences
  (fresh set: 002, 008, 010 pass in both; 004 only with the hint, 009 only without) are within what one
  deterministic run can show; no effect is claimed.
- **Decision:** the default stays `docs_hint_narrow`; the 62-question regression run was not started because the
  arm did not pass the first gate. Phase 6 stays "built, not adopted". The variant stays in the code as a
  labelled negative result. A prompt rule does not make this 14B model search more than once; a code-side
  change (for example a forced second search for architecture-routed questions) would be a different arm and
  needs its own protocol.
- Result files: `eval/agent_eval_results/*architecture_fresh_v1_20261111T120312.json` (a), `...T120609.json` (b),
  `*architecture_v1_20261111T120939.json` (c).

### 2026-11-11: Codex reviews all 50 saved code/docs answers without a GPU

- **Requested:** fill every non-abstained answer in `eval/review/answer_review_v1/review_sheet.csv`
  and run the comparison. Read the complete cited and reference evidence for all 38 code and
  12 docs answers; verified the Markdown sheet matches the prepared evidence. All 50 CSV rows
  now have allowed faithfulness/correctness verdicts and evidence-specific notes. The two
  original abstentions, answers, evidence and mechanical labels were preserved.
- **Provenance:** these are AI judgments by Codex, not independent human labels. Every note
  identifies the reviewer. `ai_review.json` records the policies, verdicts and hashes;
  `review_summary.md` records the results and limitations. No additional model process or GPU
  was used. This is an answer-quality audit, not proof the proposed judge agrees with a person.
- **Counts from the saved CSV and comparison output:**

  | Group | n | Faithful / partial / unfaithful | Correct / partial / incorrect |
  |---|---:|---|---|
  | Code | 38 | 22 / 14 / 2 | 24 / 12 / 2 |
  | Docs | 12 | 7 / 3 / 2 | 6 / 6 / 0 |
  | All | 50 | 29 / 17 / 4 | 30 / 18 / 2 |

  Both top verdicts: 22/50. Mechanical pass/miss vs AI correctness: 32/50 (64%), with partial
  counted as a miss. Mechanical false negatives relative to this review: `code-006`,
  `docs-049`, `docs-051`; 18 total correctness disagreements are listed in the summary.
- **Evidence limits:** `docs-044` has no captured cited evidence because the preparer's regex
  misses extensionless `Makefile:1-27`, although the reference validates `make test`.
  `docs-047` and `docs-050` receive provisional partial correctness marks because prepared
  references omit the proxy/SSL sections; these are not proven answer errors. The timeout
  reference also omits its topic (`docs-045`), while cited code exposes the incorrect total
  timeout claim. Wrong pinpoint citations, response-cookie extraction, an invented
  `Response.text` helper call, and a bare-SHA answer are documented per case.
- **Command:** `.venv\Scripts\python.exe eval/review_sheet.py compare --out
  eval/review/answer_review_v1` ran successfully. No `judgments.jsonl` exists, so judged=0 and
  judge agreement cannot be computed. Existing output fields named `person` count the CSV's
  AI marks; `agreement.json` has an explicit provenance annotation (human reviews=0).
  Re-running the current comparison regenerates the statistics but drops that added JSON
  annotation; the CSV note prefixes and `ai_review.json` preserve provenance.
  The generated disagreements header was also clarified to distinguish 0 judged from 50
  eligible answers; it links to the mechanical-versus-AI cases in the review summary.
- **Validation:** 50 unique non-skipped IDs match the prepared items and CSV exactly; all
  verdicts are permitted and every row has an AI-attributed evidence note. No source code,
  retrieval, chunking, model default or acceptance gate was changed. Phase 7 remains open;
  human calibration and adequate reference sections are still required before a trust claim.
  `tests/test_review_sheet.py`: **5 passed**. Artifact checks verify the source/CSV/report
  hashes, preserved mechanical labels and evidence, and recompute the saved comparison exactly.

### 2026-11-11: why `architecture_hint` did nothing, and the `architecture_hint_reminder` arm (protocol fixed before any run)

- **Cause found:** `run_question` appends a fixed user message after every `search_code`/`search_docs` result
  (`PRODUCTION_REMINDER` = "Write the final JSON now unless one more search is needed..."). It is the last
  thing the model reads before replying, so it overrode the "search at least twice" rule in the system prompt:
  one search in 35 of 35 questions. The first arm tested the prompt paragraph with a contradicting reminder.
- **New variant:** `architecture_hint_reminder` = the same prompt as `architecture_hint`, plus `ARCH_REMINDER`
  ("If the question asks how several parts of the code work together, search again with different terms for each
  part you have not yet seen ... Otherwise write the final JSON now ..."). Fixed text, not derived from the
  question; `production_reminder(variant)` selects it; every other variant keeps `PRODUCTION_REMINDER` unchanged.
  `architecture_hint` stays as the recorded negative result.
- **Protocol (same gates as above, fixed now):** run `architecture_hint_reminder` on the 10 fresh questions
  (counts as working at >= 8/10, against 4/10 for the default) and the original 15 (descriptive; Phase 6 gate
  12/15 needs the user's decision). The fresh questions were scored under two earlier arms but nothing was tuned on
  them. If it passes, the 62-question code + docs + prsearch run must not lose against `docs_hint_narrow`
  (code 36/40, docs 10/12, prsearch 10/10) before it becomes the default. Also watch that non-architecture
  questions do not start searching twice for no reason (tools per question in that run).

- **Result (2026-11-11, same day): the reminder hypothesis is disproved.** `architecture_hint_reminder`: fresh
  **3/10** (gate 8/10; default 4/10), original 15 **5/15** (gate 12/15). Every question still made exactly one
  tool call (counted from the `calls` lists in the result files, 35 of 35 earlier and 25 of 25 now), with 2 abstentions and
  1 failed parse on the fresh set. A single-question probe printed the messages: the model did receive the new
  reminder as the last message and still replied with a final answer at once. So the shared reminder was not the
  cause, or not the only one; this model does not follow "search again" instructions in either place, and
  no prompt or reminder wording has been shown to change it. The default stays `docs_hint_narrow`; the 62-question run was
  not started. A code-side change (the loop itself runs a second search for architecture-routed questions)
  would be a different arm with its own protocol, and has not been built.

### 2026-11-11: where the architecture check fails (offline research) and the `--extra-files` arm (protocol fixed before any run)

- **Method:** for the 25 architecture questions (15 original + 10 fresh), measured offline with the production
  retriever settings (`evaluated_settings`, no model needed) how many gold files each retrieval choice puts in
  front of the model, "enough" meaning at least the `need` (half) of them. Scripts were scratch files; numbers are
  from their real output.
- **Finding 1, the ceiling is retrieval breadth.** `search_code` returns the top 5 chunks, which come from about
  2.6-3.0 distinct files on these questions. With the model's own (short, rewritten) query, enough gold files are
  in the top 5 for **13 of 25** questions; with the question text as the query, **16 of 25**; the top 20
  chunks, **22/25** (model query) and **24/25** (question). The model also rewrites the question into 3-6 keyword
  words, which loses information (e.g. "proxy request connection").
- **Finding 2, taking one chunk per file works, but costs code questions.** Best chunk per file from the top 40,
  first 5 files: 24/25; but on the 40 code questions the gold range was in the results for 37/40 instead of
  39/40. A cap of 2 chunks per file over 8 chunks keeps 39/40 and gives 24/25, but the result grows from about
  8.3k to 13.2k characters, past the 12,000-character per-question tool budget.
- **Finding 3, so keep the top 5 and add files.** Top 5 chunks unchanged + up to 3 chunks from files not already
  shown, found by searching with the whole question: **24/25** (2 extras: 23/25, 4 extras: 24/25). Extras taken
  from the model's own query pool only reach 21/25, so the question text is what helps. Code questions cannot lose
  a result because the first 5 are identical. Extras are cut to 1,000 characters (a chunk's metadata keeps its
  full line range), so the result stays under the budget.
- **Finding 4, the model also drops files it was shown.** Even where enough gold files were retrieved, some
  answers named too few (for example `architecture-013`: all 3 retrieved, 0 named). The earlier prompt and
  reminder changes did not move this. Not addressed by this arm.
- **Finding 5, summaries could not be tested here.** A search of the `summaries` collection on this machine
  returns 0 hits for `psf/requests`, so no conclusion about them is drawn; the numbers above use code chunks only.
- **Built (default off):** `agent_code_extra_files` (0..6, default 0 = every earlier evaluation) and
  `agent_extra_chunk_chars` (1000) in `config.py`; `ProductionTools._extra_files` in `agent/tools.py`; the graph
  passes the user's question to `search_code` only on the production toolset; `eval/agent_eval.py --extra-files N`
  (result files tagged `_extraN`). Tests in `tests/test_extra_files.py`. Prompt unchanged (`docs_hint_narrow`).
- **Protocol (fixed now):** `docs_hint_narrow` with `--extra-files 3` on the 10 fresh questions (default
  prompt: 4/10; counts as working at >= 8/10) and the original 15 (5/15; gate 12/15, the user decides Phase 6).
  Then the 62-question code + docs + prsearch run must not lose against the default (code 36/40, docs 10/12,
  prsearch 10/10) before it becomes the default. Single deterministic runs, one repository; the 15 original
  questions are the ones this was designed while looking at (descriptive), the fresh 10 were scored under earlier arms
  but nothing was tuned on them. Expected ceiling from retrieval alone: 24/25.

- **Result (2026-11-11): `--extra-files 3` did not move the check.** `docs_hint_narrow` + 3 extra files: fresh
  **4/10** (default 4/10; gate 8/10), original 15 **5/15** (default 5/15; gate 12/15). Not met; the 62-question run
  was not started; the default stays off (0). Full offline suite: passed, exit 0.
- **Why: retrieval is no longer the bottleneck, answer writing is.** Re-creating what the model was shown for the
  10 fresh questions (its own top 5 + 3 extras), enough gold files were in front of it for **10/10**, yet its
  answers named enough of them for only **4/10** (so the earlier ceiling of 24/25 for retrieval cannot be reached
  by the answer). Reading the answers: (a) it names the symbols it describes but often not the file (for example
  "`requests.exceptions.JSONDecodeError`" or `HTTPError` with no `src/requests/exceptions.py`), and the check needs the
  literal path in the text, so a substantively right answer can miss; (b) it often builds the answer from one or
  two chunks and ignores the rest (`archfresh-001`: 3 gold files shown, 1 named); (c) test files take part of the
  results: **18%** of the top-5 chunks for these questions are in `tests/`, and `archfresh-003` was answered
  mainly from `tests/test_requests.py`. Dropping `tests/` chunks unless the question mentions tests would lift
  the offline count from 16 to 18 of 25 and code gold hits from 39 to 40 of 40, but none of the 40 code questions
  has its gold in `tests/`, so that set cannot show what such a rule would cost a question about the tests.
  Not built.
- **What this says about the check itself:** "answer contains at least half the gold file paths" measures how many
  paths the model writes, not whether the architecture explanation is right. No person has read these answers and
  the 30 labels are still unreviewed (docs/ARCHITECTURE_LABEL_REVIEW.md), so 5/15 and 4/10 should not be read as
  "the answers are bad".

### 2026-11-11: reading the architecture answers, and the next two arms (protocol fixed before any run)

- **Review sheet:** `docs/ARCHITECTURE_ANSWER_REVIEW.md` holds the default prompt's 25 answers beside their gold
  files, with a Y / P / N line for a person to fill in (judge the explanation, not path formatting). Nothing has been
  filled in by a person yet.
- **Mechanical counts on those 25:** strict pass 9 (the check), **loose pass 9** (also accepting the file name or the
  dotted module), 2 abstained. Path formatting is therefore not what is costing passes at the question level.
- **First read by Claude (not a person's verdict; one reader, no second opinion):** about 7 answers are right and
  complete enough, about 10 are right but cover one layer of several, about 6 are wrong: `architecture-004` says
  `HTTPAdapter` owns request preparation, `-003` puts `Session.send` in `models.py`, `-011` cites
  `src/requests/packages/urllib3/fields.py` (not in this version), `-015` says `SOCKSProxyManager` translates
  failures, `archfresh-003` is answered from test files, `-008` never mentions `structures.py`. Three of the good
  answers score as misses (`architecture-005`, `archfresh-004`, `archfresh-005`: right content, fewer than half the gold
  paths written). So the failure is mostly real (thin answers and some invented attributions), and the check is only
  mildly harsh.
- **Built (both off by default):** `agent_code_skip_tests` (`search_code`'s top 5 and its extras leave out test
  files unless the question or the model's query is a test question per `retrieval.intent.detect_intent`;
  `--skip-tests`), and prompt variant `architecture_cover` = the `docs_hint_narrow` prompt unchanged with a
  different last-message reminder (`ARCH_COVER_REMINDER`): one short item per participating file, cited, leave out
  files that do not take part, never name a file or class that is not in the results. Tests in
  `tests/test_extra_files.py` and `tests/test_prompt_variants.py`.
- **Protocol (fixed now), one Colab session, all on the 25 architecture questions (original 15 + fresh 10):**
  (F1) `docs_hint_narrow --extra-files 3 --skip-tests` (retrieval changes only; compare with 4/10 and 5/15 for the
  default and 4/10, 5/15 for extras alone); (F2) F1 with `--variant architecture_cover` (adds only the reminder).
  Count reported: the check's strict pass, plus a second reading of the answers for wrong attributions (named files
  or classes not in the results), since the new reminder is aimed at those. Gates unchanged: fresh >= 8/10 to call
  an arm working; Phase 6 needs 12/15 and the user's decision. A winner must then not lose on the 62-question run
  (code 36/40, docs 10/12, prsearch 10/10) before it becomes a default; `--skip-tests` in particular needs the code
  questions checked, because none of them has its gold in `tests/`. Single deterministic runs, one repository.

- **Result (2026-11-12), 25 architecture questions, one Colab session, single deterministic runs:**

  | Arm | Fresh 10 (gate 8) | Original 15 (gate 12) |
  |---|---|---|
  | default prompt (earlier runs) | 4/10 | 5/15 |
  | `--extra-files 3` alone | 4/10 | 5/15 |
  | F1: `--extra-files 3 --skip-tests` | **6/10** | **9/15** |
  | F2: F1 + `architecture_cover` reminder | **6/10** | **11/15** |

  No arm reaches either gate; F2 is one short on the original 15. Zero abstentions in F1 and F2 (the default had
  2 in the original read). One `search_code` call per question in all of them. The extra files help only with
  tests skipped: the same extras without `--skip-tests` did nothing, F1 is the first arm that moved the count.
  The original 15 are the questions these changes were designed while looking at, so the 9 and 11 are descriptive;
  the fresh 10 (6/10) are the held-out figure.
- **What `architecture_cover` changed:** 19 of its 25 answers were JSON lists of one cited item per file (the
  parser allows a list of strings), against 0 before. Reading the 16 list answers printed in the run: no invented
  file paths (the earlier read found `src/requests/packages/urllib3/fields.py`), `architecture-015` now
  correctly points at `HTTPAdapter.send`; but descriptions still slip, e.g. `architecture-001` calls the auth
  classes part of the network transport and `archfresh-010` labels the `default_headers` chunk
  `resolve_proxies`. The 9 string answers were not read for this.
- **Bug found and fixed:** when the model answers with a list, `agent/graph.py` skipped citation repair, snapping and
  the SHA check (they only ran on strings), and the API/UI got a list. The F2 numbers above were produced
  before the fix (their citations were not validated). Now a list becomes a bulleted string before the checks
  (`tests/test_agent_graph.py::test_a_list_answer_becomes_bullets_and_gets_its_citations_checked`). The check's
  scorer already turned lists into text, so the pass counts themselves are valid.
- **Decision:** nothing is made the default. `agent_code_extra_files`, `agent_code_skip_tests` and
  `architecture_cover` stay off by default. To adopt F2 it must pass the 62-question code + docs + prsearch run
  without losing (code 36/40, docs 10/12, prsearch 10/10), and the user decides Phase 6 (11/15 against a 12/15 gate).
- **Regression result (2026-11-16), F2 (`--variant architecture_cover --extra-files 3 --skip-tests`), 67 questions of
  `agent_eval_v1.jsonl`, one Colab session, single run** (`eval/agent_eval_results/agent_eval_production_architecture_cover_extra3_notests_20261116T195549.json`):
  code 34/40 (3 abstained), docs 5/12 (5 abstained), probes 5/5, prsearch 10/10. **Fails** the bars (code 36, docs 10).
  Docs is the big loss: 5 of 12 abstained and docs-051 called `search_code`. Not yet diagnosed which of the three
  switches causes it; the docs path never uses extras or skip-tests, so the `architecture_cover` reminder is the
  first suspect (it is not applied to `search_docs` results by design? unverified). **Decision:** F2 is not adopted;
  all three settings stay off by default. Phase 6 stays "built, not adopted".
- **Follow-up (2026-11-16), F1 = default prompt and reminder + `--extra-files 3 --skip-tests`, code + docs (52 questions,
  one run)** (`..._docs_hint_narrow_extra3_notests_20261116T201253.json`): code 37/40 (1 abstained), docs 10/12
  (0 abstained). Both meet the bars (36, 10); prsearch and probes not re-run (those tools never use the changed code).
  So the docs loss in F2 (5/12) came from the `architecture_cover` reminder, not from extras or skip-tests. F1 is
  neutral on code/docs but only reached 6/10 fresh and 9/15 architecture, below the Phase 6 gates (8/10, 12/15).
  F2 gains 2 more architecture questions but costs docs, so a reminder that applies only to code searches is the
  untested next idea. Still nothing adopted.
- **F2b (2026-11-16): `architecture_cover` reminder after `search_code` only; docs keep the default reminder**
  (`production_docs_reminder`, `Toolset.docs_reminder`; tests green). code + docs, 52 questions, one run
  (`..._architecture_cover_extra3_notests_20261116T212535.json`): code 34/40 (3 abstained), docs 10/12. Docs is fixed,
  but code is 34 against the bar of 36 (F1 had 37). **Fails.** The cover reminder itself costs about 3 code
  questions (abstentions), identical to F2's 34. Architecture questions were not re-run. Nothing adopted; all
  settings stay off by default.
- **Stronger answer model built, not yet run (2026-11-16).** Offline gate check first: a cue-word gate for the cover
  reminder flags 15 of 25 architecture questions and 0 of 40 code questions, but only one of F2's two architecture
  wins is gated, and the cue words were chosen after reading these sets, so the reminder is dropped. The user then
  chose **Qwen3.8-27B** as the answer model. The pinned transformers 4.57.6 has no `qwen3_5` model type, so it is
  served by vLLM 0.33.0 (own venv, `cyankiwi/Qwen3.8-27B-AWQ-INT4` at revision e3a70090, thinking off, eager,
  GPU utilization 0.88, no reranker; same build and vLLM version as the Phase 7 judge run). New:
  `agent/vllm_llm.py` (`VllmChatModel`, same `Model` shape), `tests/test_vllm_llm.py`,
  `notebooks/serve_model_27b_on_colab.ipynb`. Colab-only dependency; pyproject, uv.lock and the 14B notebook are
  unchanged. Full suite green. Not measured: output format and quality with this model, memory, speed.
- **Qwen3.8-27B (vLLM AWQ-INT4, A100 40 GB, no reranker) on the architecture questions, F1 settings (extras 3, skip
  tests, default prompt and reminder), 2026-11-16, single deterministic runs.** Unlike the 14B (always one search) the
  27B searches again (up to 4 times) when the first result is not enough. About 30 s per question.
  At the evaluated 12,000-character tool-result budget: original 15 **7/15** (4 abstained, 1 failed), fresh 10
  **6/10** (2 failed). The abstentions are the shared budget running dry: later searches came back nearly empty.
  With `--tool-chars 24000` (new eval flag, `_chars24000` in the file name): original 15 **11/15**, fresh 10 **7/10**
  (0 abstained, avg 1.2 searches). Gates: 12/15 and 8/10, so **both still missed by one question**. Five of the seven
  misses are `failed: budget` with the last reply a second `search_code` call, i.e. the failure comes on the third
  generation, after two large results. Probable cause (not confirmed, the trace does not name the cap): the 8,192
  input-token cap (the server's context is that cap plus 512). The other two misses name 1 of 3 gold files.
  14B with the same settings: 9/15, 6/10. Files: `..._v1_extra3_notests_chars24000_20261116T222842.json`,
  `..._fresh_v1_..._chars24000_20261116T223419.json`. Not run: code/docs/prsearch/probe regression for the 27B.
- **27B with a 12,288 input-token cap, 2026-11-17 (A100 80 GB this session, no reranker), single deterministic runs.**
  Built: the 27B notebook sets `REPO_RAG_AGENT_MAX_INPUT_TOKENS` to 12288 (vLLM `max_model_len` = cap + 512) and
  reports it in health as `max_input_tokens`; `eval/agent_eval.py --max-input-tokens N` refuses to run when the
  server reports another value (tag `_in{N}`). Full suite, ruff and mypy green. All runs below use F1 settings
  (`docs_hint_narrow`, extras 3, skip tests) with `--tool-chars 24000 --max-input-tokens 12288`.
  Architecture: original 15 **13/15** (gate 12; was 11 at the 8,192 cap), fresh 10 **8/10** (gate 8; was 7), 0
  abstained. Misses: architecture-002 and archfresh-002 (`failed`, 4 searches), architecture-011 and archfresh-006
  (1 search, file coverage). Supports the cap as the cause of the earlier budget failures.
  Regression: code **39/40** (gate 36; miss code-033), docs **11/12** (gate 10; miss docs-051), prsearch **10/10**,
  probes **5/5** (all 5 abstained, as expected), history **11/11**, refs **10/10**, kind **5/5**, commit **5/10**
  (misses commit-079 no tool call, 080, 083, 085, 088; 1 abstained, 2 failed). 14B F1 for comparison: code 37/40,
  docs 10/12, architecture 9/15 and 6/10. About 20 to 60 s per question.
  Caveats: the fresh set clears its gate with no margin; commit questions remain a weak spot; one repo, labels not
  human-reviewed; the 27B needs its own vLLM server, no reranker, and limits above the evaluated defaults
  (24,000-character result budget, 12,288 input tokens). **Not adopted as the default; that is the user's decision.**
  Files: `eval/agent_eval_results/..._chars24000_in12288_*.json` (architecture 20261117T083842 and T084459;
  code/docs/prsearch/probe T091315; history/refs/commit/kind T091905).
- **Why the five commit questions fail (27B, 2026-11-17).** commit-079: reply was a flat tool call
  (`{"type":"tool_call","name":"search_commits","query":...}`), `failed: parse`, no retrieval. commit-080: right
  trail (PR 1808, #1728, PR 1729) but hit the 4-call limit (`failed: budget`); gold `8bfe0d873f38`. commit-083:
  answered `4d498db3ed23` (also fixes #2279) vs gold `96efb5ec9638`. commit-085: abstained; gold `2814664e91d9`
  has the message "Fixes #20.", nothing to match on. commit-088: answered `4fef3b5586e1`, the merge commit of PR 1888
  (not in `history_commits`, only `pull_requests.merge_commit_sha`) vs gold `8b792be81bfa`, a commit of that PR.
- **Flat tool-call parse fix (2026-11-17).** `parse_reply(tolerant=True)` (production only) now accepts a flat tool
  call when the tool is known and every extra key is an argument the tool declares; the arguments are still
  validated, the strict bake-off parser is unchanged. `tests/test_tolerant_parse.py` +2 tests (flat call parses
  only when tolerant; wrong key, wrong type, unknown tool, missing argument still fail). Full suite, ruff, mypy
  green; notebooks regenerated; zip hash `dae75df3` -> `5ee1023f`. Rerun on a new A100 40 GB server (same limits,
  `max_input_tokens` 12288): commit **5/10** (unchanged; 079 now parses and makes 2 calls but still misses),
  code **38/40** (gate 36), docs **11/12** (gate 10). New code miss code-011 is a model reply with one extra
  closing brace (`...}}}`, invalid JSON, failed the retry too); it passed on the 80 GB session, so it is output
  variance between servers, not the parser change (not tested). No trailing-brace repair added. Open: the
  4-call limit (080), PR merge SHA to commits mapping (088). Files: `..._chars24000_in12288_20261117T094627.json`
  (commit) and `..._20261117T101206.json` (code, docs).
- **Commit questions on the 27B: PR merge mapping, call limit, prompt variant, one label (2026-11-17, A100 40 GB,
  F1 settings, `--tool-chars 24000 --max-input-tokens 12288`, single deterministic runs).**
  (1) `pr_commits()` (`history/git_tools.py`) lists the commits a merge brought in (`first_parent..second_parent`);
  the production `get_pr` result gains `pr_commits` (up to 5 SHA + subject) next to `merge_commit_sha`; unchanged
  for records with no usable merge commit. Checked on the real clone: PR 1888 lists `8b792be81bfa` first. Tests:
  `tests/test_git_tools.py` +3, `tests/test_pr_commits_tool.py`. The agent and tools run on the laptop, so this did
  not need a new Drive zip. With only this change, commit stayed **5/10**: on 088 the model still answered the
  merge SHA `4fef3b5586e1` and ignored `pr_commits`.
  (2) Tool-call limit 6 with 7 turns (`REPO_RAG_AGENT_MAX_TOOL_CALLS=6`, `REPO_RAG_AGENT_MAX_TURNS=7`, not recorded in
  the result file): commit still **5/10**. 080 used all 6 calls (search_commits, search_prs, get_pr 1808,
  get_references 1728, get_pr 1729, search_commits) and failed on the budget. More calls only let it wander
  longer; **not adopted, limits stay 4 and 5.**
  (3) Why the other misses fail (rank of the gold commit in the commit search, hybrid, no reranker, the model's own
  query): 085 fused rank 22 (dense 26, BM25 92); its stored text starts with `.` because
  `strip_issue_references` removes "Fixes #20." entirely, leaving a code diff. 080 fused rank 18 (dense 23, BM25
  11; BM25 weight is 0.3). 079 not in the top 300 (gold is a docstring change with no word overlap with the bug).
  The model sees only the top 5. The earlier reranker test (2026-11-07, 14B + 4B reranker) lowered the agent's
  commit score 3/10 to 1/10; not tried on the 27B (no reranker fits next to it on a 40 GB card).
  (4) **Label:** commit-083 also accepts `4d498db3ed23`. Both commits fix #2279 in the same template file and
  `4d498db3ed23` (2014-10-14) is the one that changes "Gittip" to "Gratipay". Decided after seeing the miss, so it
  is a scoring change, not a model gain. **First done wrongly:** the line in `agent_eval_v1.jsonl` was patched,
  which changed the dataset sha256 (`d8bf41dcb3a6` to `577b5674816d`) and broke the tests that pin it. **Fixed the
  same day:** the dataset file is back to its original bytes (`d8bf41dcb3a6`) and the extra answer lives in
  `ALSO_ACCEPTED` in `eval/agent_eval.py`, applied in `score()` (a pass through it carries
  `detail == "accepted through ALSO_ACCEPTED"`). Two result files from the patched period record the other hash,
  `577b5674816d` (the commit run `..._20261117T172419.json` and the regression `..._20261117T175513.json`); their
  questions are identical to the dataset on disk except that one accepted-answer list. Re-scoring saved commit
  results offline with the new `score()`: the three runs without the prompt change go from 5/10 stored to **6/10**
  (083's reply `4d498db3ed23` is now accepted), the prompt-variant run stays **7/10**. Stored `passed` fields in
  older result files were not rewritten.
  (5) **Prompt variant `commit_pr_commits`** (`agent/prompts.py`, allowed in `config.py`; off by default, the
  default stays `docs_hint_narrow`): `docs_hint_narrow` plus one paragraph: the merge commit holds no change,
  name the matching commit from `pr_commits`. Commit **7/10** (088 now answers `8b792be81bfa`). With the 083
  label applied to every run, the same questions score **6/10 without the prompt change and 7/10 with it**, so the
  prompt itself is +1 (088). Without the 083 label the figures are 5/10 and 6/10. n = 10, one run.
  Misses: 079, 080, 085.
  Regression with the variant: code **40/40**, docs **11/12** (docs-051 as before), prsearch 10/10, probes
  5/5, history 11/11, refs 10/10, kind 5/5. Code was 38 to 39/40 earlier; that difference is within what
  moved between servers without any change, so not credited to the prompt.
  Files: `..._commit_pr_commits_extra3_notests_chars24000_in12288_20261117T172419.json` (commit) and
  `..._20261117T175513.json` (the rest). **Not adopted as the default; that is the user's decision.**
- **Why 20 to 30 s per question, and CUDA graphs (27B on vLLM, A100 40 GB, 2026-11-17/18).** Per-question timing
  (`time_one.py`, scratchpad script, not in the repo; client wall time vs server `seconds`, tokens, tunnel round
  trip) in eager mode: the answer generation (about 7,400 to 8,000 input tokens, 133 to 164 output tokens) takes 19
  to 23 s on the server; a fit over 13 generations gives prefill about 3,150 tokens/s and **decoding about 7
  tokens/s** (136 ms/token). The Cloudflare tunnel adds about 0.3 to 0.6 s per call (0.7 to 1.9 s per question);
  the laptop adds about 20 s once per process (embedding-model warm-up). Eager mode was run for the 27B because
  of CUDA-graph memory worries; the weights alone should allow about 70 tokens/s, so it was launch-bound.
  Built (off by default): `server_command(enforce_eager, speculative_tokens, max_num_seqs)`; notebook switches
  `ENFORCE_EAGER = True`, `MTP_TOKENS = 0`; health reports `decode_mode`; `root_cause()` prints the engine's real
  error when vLLM fails to start. MTP is possible: the checkpoint has 15 `mtp.*` tensors (shard 5) and
  `mtp_num_hidden_layers: 1`; vLLM 0.33.0 accepted `{"method":"mtp","num_speculative_tokens":3}`.
  First try (graphs + MTP together) failed at startup: `ValueError: max_num_seqs (256) exceeds available Mamba
  cache blocks (179)` (graph capture itself and KV planning were fine; eager mode skips the check). Fix:
  `--max-num-seqs 8` when graphs are on (the agent sends one request at a time).
  **CUDA graphs alone** (`decode_mode: cuda-graphs`, `--max-num-seqs 8`, no MTP, same limits as before): about
  **35 output tokens/s**, answer step 3.8 to 4.4 s instead of 19 to 22 s; code-009 24.9 -> 6.4 s, architecture-005
  28.6 -> 7.5 s, history-053 8.0 -> 2.5 s, commit-081 8.8 -> 2.9 s. Startup compile about 4 minutes. Full
  regression (128 questions, about 10 min, mean **5.0 s** per question, was 23 s): architecture original 15
  **12/15** (eager 13; gate 12, no margin), fresh 10 **8/10** (8), code **39/40** (39), docs **11/12** (11),
  prsearch 10/10, probes 5/5, history 11/11, refs 10/10, kind 5/5, commit **5/10** (6/10 with the 083 label;
  commit-087 flipped to a miss: answered `17dd3c0a7689` "fixed typo" vs gold `f48c78c9eebd`). The two modes are
  not bit-identical (architecture-005 answer 164 tokens vs 157), so a few answers move either way. **Not adopted:
  the notebook default stays eager; the user decides.** File: `..._chars24000_in12288_20261118T022710.json`.
  **CUDA graphs + MTP** (`decode_mode: cuda-graphs + mtp3`, `{"method":"mtp","num_speculative_tokens":3}`, same
  `--max-num-seqs 8`, same limits): answer step 2.8 to 3.2 s (about 45 to 52 output tokens/s); code-009 4.7 s,
  architecture-005 5.1 s (same 164 output tokens as the graphs-only run), history-053 1.7 s, commit-081 2.4 s.
  Full regression, mean **4.2 s** per question over the 103 non-architecture questions (graphs only 5.0 s, eager
  23 s; code 6.0, docs 3.9, commit 4.7): architecture original 15 **13/15**, fresh 10 **8/10**, code **39/40**,
  docs **11/12**, prsearch 10/10, probes 5/5, history 11/11, refs 10/10, kind 5/5, commit **5/10** (087 a miss as
  in the graphs run). All gates met; MTP over graphs alone is about 16% faster over the whole set (more on long
  answers, little on short ones). The architecture 13/15 vs 12/15 is within run-to-run movement, not credited to
  MTP. Acceptance rate not read (no metrics endpoint is exposed through the tunnel). **Adopted by the user on 2026-11-18
  as the 27B notebook default (decision log); eager stays available through the two switches.** File: `..._chars24000_in12288_20261118T024701.json`.

### 2026-11-18: architecture gold labels reviewed (`docs/ARCHITECTURE_LABEL_REVIEW.md`)

Claude drafted a verdict for each of the 40 labels and the user accepted them as the review ("the human review is
done"); the reviewer's own checking is not recorded, so provenance is owner-submitted, unverified. A (15) and B (10)
were checked against the psf/requests clone at `611c6162cb`. C (flask, 15) was judged without opening the source
(there is no flask clone). 34 kept. Changes: **A-11** drop `adapters.py` (it only forwards the encoded body);
**B-004** drop `status_codes.py` (`raise_for_status` uses numeric ranges, not `codes`); **B-002** add `models.py`
(`prepare_auth` wraps the netrc tuple in `HTTPBasicAuth`, models.py:685); **B-006** add `sessions.py` (`Session.send`
dispatches the digest `handle_401` hook, sessions.py:791); **B-009** add `adapters.py` (`build_response` sets
`encoding`, adapters.py:385). Need 1 → 2 for A-04 and C-15. C-09/10/11 assume the merged `AppContext` model;
not confirmed at `d086db856b`.
**Applied (same day):** the A and B changes are in `eval/agent_eval.py` (`ARCH_LABEL_REVIEW` for the original set, the
`ARCH_FRESH` edits and `ARCH_FRESH_NEED` for the fresh set; A-04 need 2; A-11 gold `models.py` + `utils.py`, need 2)
and the two datasets were regenerated (`agent_eval_architecture_v1.jsonl` sha256 `7ed2a1c55719`,
`agent_eval_architecture_fresh_v1.jsonl` sha256 `403f135f5445`); `tests/test_agent_eval.py` updated. The frozen
`architecture_v1/` set is untouched. The 27B gate numbers (13/15 and 8/10) did not change with the new labels.
**Not applied:** the flask (C) changes, because the flask labels were never checked against source. A-11 need 2 was my
choice, not the reviewer's. Saved result files were not rewritten; they keep the labels they were scored with.

### 2026-11-18: finished experiments archived under `experiments/`

The user chose to move finished experiment code out of the live tree (and did the file moves under `experiments/`
themselves; nothing committed).
- **Builders moved** (as they were, paths not adjusted, so they are not runnable from there): `notebooks/build_answer_bakeoff.py`,
  `build_answer_judge.py`, `build_commit_depth_resume.py`, `build_summary_prose_eval.py` now live in `experiments/notebooks/`.
- **Tests:** complete copies of the five affected test files are in `experiments/tests/`. In `tests/`, the 13 tests
  that check archived artifacts were removed, and `tests/test_commit_depth_resume.py` and
  `tests/test_answer_bakeoff_notebook.py` were deleted (the latter imported a builder that moved). `tests/` now holds
  only tests of live code.
- **Result files** moved to `experiments/eval/agent_eval_results/`. `eval/agent_eval.py` still has
  `RESULTS_DIR = eval/agent_eval_results`, and another session wrote a result into that old path after the move; which
  path should be the default is not decided.
- **Suite state:** the last full run failed only on notebook and zip freshness tests (serve, retrieval, policy, judge,
  pr-search notebooks), because `src` files (`config.py`, `graph.py`, `prompts.py`, `tools.py`, `git_tools.py`,
  `bakeoff.py`) were edited by another session after the notebooks were rebuilt. The zip hash recorded earlier
  (`554b0f10`) is stale. **Next:** rebuild every `notebooks/build_*.py`, run ruff, mypy and the full suite once
  `src` stops changing, then re-upload `repo_rag_ask_source.zip` and the changed notebooks to Drive.

### 2026-11-18 (later): commit questions, 57/113 to 70/113: wider shortlist (k=15) and a longer commit-only tool-call cap (6)

Goal: raise the weak spot, bug report to fix commit (5/10 on the old 10-question set). Everything below is psf/requests,
Qwen3.8-27B on the Colab A100 40 GB (CUDA graphs + MTP), production toolset, `--extra-files 3 --skip-tests
--tool-chars 24000 --max-input-tokens 12288`, single runs.
- **Where the old failures were.** Replaying the model's own `search_commits` queries against the local index: of the 5
  misses, 4 had the gold commit outside the top 5 (retrieval), one was a selection failure (gold at rank 4).
- **Retrieval check** (`eval/commit_retrieval_check.py`, results in `eval/commit_retrieval_results/`, 141 pairs, bug text as
  the query, a proxy because the agent writes its own query): gold in the top 5 for 79 (56%), top 10 for 94 (67%),
  top 20 for 106 (75%), top 50 for 119 (84%); 22 beyond 50. Sweeps of BM25 weight (0.1 to 1.0), title fusion and
  dense-only moved the count by at most 3 pairs: within noise, nothing adopted; BM25 weight 1.0 hurts (74 at top 5).
  PR-text enrichment is untestable here: only 8 of 141 gold commits have a PR. Showing more candidates is the lever.
- **New 113-question set** `eval/datasets/agent_eval_commit_full_v1.jsonl` (all test pairs, sha256 da7d08dd0509; the old
  10-question commit set is unchanged). Build with `eval/agent_eval.py build-commit-full`; run with `--dataset`.
- **Shortlist** `agent_commit_k` (5 to 30): the first 5 hits in full, the rest as one line (short SHA, subject, files,
  symbols; `commit_brief` in `agent/tools.py`), so 15 candidates cost about what 5 full ones do.
- **Bug found by the run:** with k above 5 the citation check in `agent/graph.py` raised `KeyError: 'text'` on the one-line
  rows; the first k=15 and k=20 runs crashed and are discarded. Fixed (uses `text` or `subject`), regression test added.
- **Results (113 questions):**

  | Setting | Passed | Abstained | Failed (all "budget") |
  |---|---|---|---|
  | k=5, 4 tool calls (before) | 57 (50%) | 8 | 12 |
  | k=15, 4 tool calls | 65 (58%) | 8 | 13 |
  | k=20, 4 tool calls | 65 (58%) | 8 | 15 |
  | k=15, 6 tool calls | 70 (62%) | 14 | 1 |
  | k=15, 8 tool calls | 70 (62%) | 14 | 0 |
  | k=15, 10 tool calls | 70 (62%) | 14 | 0 |
  | k=15, commit cap 6 as default (end to end) | 70 (62%) | 14 | 1 |

  k=5 to 15: 11 questions gained, 3 lost. k=15 to 20: 4 gained, 4 lost (noise), so 15. Cap 4 to 6: 5 gained, none lost;
  6 to 8 and 10: no question changed. Mean time 4.2 s to 4.4 s per question. Most "failed" rows were the model asking for
  a fifth tool call (`Tool-call cap`), not a token or character budget.
- **Applied:** `agent_commit_k` default 15; `agent_commit_max_tool_calls` default 6, in force only once the model has
  called `search_commits` (there is no question-type router), other questions keep `agent_max_tool_calls` = 4. The cap for
  other question types was not re-measured, and no other type was run with the new defaults in this entry.
- **Not measured:** about 40 of 113 still fail, mostly wrong commits (28) and abstentions; the 10-question commit set
  (5/10) was not re-run with the new defaults; `--max-tool-calls` in the eval overrides only the global cap.
- **Side fixes:** mypy error in `metadata_db.top_contributors` (explicit types); `tests/test_retrieval_followup.py` and
  `eval/analyze_retrieval_failures.py` point at the archived `experiments/eval/retrieval_results/colab_full/` zip.
- **Next:** the summary pilot (LLM-written commit summaries added to the indexed text; cheap pilot on the 141 gold commits
  plus about 1,000 others first; needs a GPU), then the exact-SHA reranker measurement at depth 50.

### 2026-11-18 (later): commit-summary pilot built, NOT yet run

Tests whether a short plain-English `Summary:` line (written by the 27B from the commit's own message and diff only, no bug
report, issue or PR) makes the gold fix commit rank higher. Cheap first: 1,143 commits (all 143 gold shas of the 141 pairs, found
in the local index, plus 1,000 random others, seed 20261118) instead of all 4,883.
- **Files:** `eval/commit_summary_pilot.py` (`prepare`, `summarize`, `package`, `score`; resumable; data in
  `eval/commit_summary_pilot/`), `notebooks/build_commit_summary_pilot.py` -> `commit_summary_pilot_on_colab.ipynb`
  (embeds `src/repo_rag` and the script; you upload `pilot_input.zip`), `tests/test_commit_summary_pilot.py` (12 tests).
- **Design:** the summary line goes before `Files:` so the 2,048-token embedder cap never cuts it. Gold and random commits get
  the same prompt. `score` compares dense-only and the agent's hybrid (dense + BM25, weighted RRF) with and without the summary,
  hit@1/5/10/20, MRR, and how many gold commits moved up or down. Ranks are inside a 1,143-commit corpus, so they show
  direction only; a gain must be re-measured through the agent.
- **Checked offline:** `prepare` on the real index (143 of 143 gold found, 141 of 141 pairs usable), `package` and `score` with
  fake summaries and random vectors. The summary prompt was NOT tried on the model (the Colab server had timed out), so summary
  quality is unseen.
- **To run:** a 27B server session for `summarize` (about 1,143 x 4 s, one request at a time), then a GPU session for the
  notebook, then `score`. Go to the full 4,883 commits only if the pilot clearly helps.

### 2026-11-18 (later): Claude reviewed the 27B's 52 code and docs answers (replaces the 27B judge for this run)

The user asked for Claude, not the 27B, to be the judge. I read all 52 answers of the 27B run `..._101342` (guards on,
see `eval/claude_review_27b/README.md`) against the cited lines from `eval/judge_answers.py prepare`, and against the
pinned source (`611c6162cb`) wherever the extracted evidence was cut off. Checked and found right where I first doubted
them: `super_len` (urllib3 2.x encoding), the proxy bypass rules, the `\Z` header regex, `Response.close`.
- **Faithful 50/52 (2 partial), correct 49/52 (3 partial).** Code 38/40 correct, docs 11/12. Mechanical score was 40/40 and 11/12.
- **Not both top:** code-006 and code-032 (no charset: the answer says `text` falls back to `apparent_encoding`, but for
  `text/*` the encoding is ISO-8859-1 from `get_encoding_from_headers`, so detection runs only when the encoding is None;
  code-032 is the garbled-text question and misleads the most); code-025 (says "~/.netrc or ~/.netrc" and cites
  `NETRC_FILES` at line 74, it is line 78); docs-051 (accurate but misses the main reason, connection reuse).
- **Reading:** the mechanical score is close to Claude's on this run (40/40 against 38/40 correct), far closer than on the
  earlier answers (mechanical pass overstated: 33/50 correct then). Not a fair before/after: different model (27B, those were
  the 14B), different reviewer, and these 52 are the questions the system was tuned on, so expect lower on new questions.
  Claude is one AI reader; no human labels exist. Verdicts: `eval/claude_review_27b/verdicts.json`.
- **Checker blind spot:** a citation to a file without an extension (`Makefile:1-27`) is not resolved by the judge tooling's
  citation regex, so docs-044 reached the judge with no evidence.

### 2026-11-18 (later): Claude reviewed the exact-fact and commit answers (files in `eval/claude_review_27b/`)

- **Exact facts** (15 answers of S1 run `..._115543`, checked against git and `metadata.db`): 10 correct, 4 partial (hist-1 titles only and a merge commit listed as a change; people-3 "Kenneth Reitz" by count with no recency; pr-2 a bare "- 5797"; rel-1 skips the security fix), pr-3 a right abstention worded as a retrieval failure. hist-2, people-1/2/4, pr-1, issue-1 (10 open issues with timeout in the title, matches the database), rel-2/3, stats-1/2 verified. **I first thought the database was truncated (4,883 of 6,495 commits) and was wrong:** the 1,612 missing are merge commits, excluded by design (`git rev-list --no-merges --count` = 4,883); stats-1 matches `git shortlog --no-merges` exactly. The `identities` table has 0 rows, so aliases such as "Ian Stapleton Cordasco" (21 commits) are not merged; the order does not change.
- **Commit questions** (113, run `..._ck15_20261118T171718`): 70 strict passes. Reading the 43 others with git: 6 are valid alternatives to the gold (011 and 106 fix the same issue the same day, 019 is the code fix where the gold only edits the test, 041 and 110 are one-line attempts and follow-ups at the same place, 077 is the merge commit of the gold's PR), so about **76/113 (67%)** by this reader; 051 and 060 have release commits as gold (label noise). **039 is misleading:** the answer names `c3444f8c7` as an earlier fix, but it introduced the redirect cache that caused the bug. 14 abstentions all use the generic "No relevant implementation was found".
- **The main finding:** 67 of the 70 passing commit answers are a bare SHA (median 12 characters), so the user gets no reason and no confidence. That is a product gap the score does not show.
- Limits: one AI reader; alternatives judged from messages and diffs, not by running the code.

### 2026-11-18 (later): commit answers carry facts and leads; `commit_explain` prompt variant (built, NOT measured live)

Response to the review above (67 of 70 passing commit answers were a bare SHA, 14 abstentions were the generic sentence).
- **`src/repo_rag/agent/commit_answer.py`**, on by default (`agent_answer_facts`): under the model's text, for each commit it names, date, author, subject, rank among the search results and "a text match, not confirmed against the code"; for an abstention after `search_commits`, the 3 closest candidates as "unverified leads" (`abstain` stays true); for count tools a note that merge commits are not counted. All from tool rows, never from the model.
- **Scoring is unchanged by construction:** the result keeps `model_answer` and `eval/agent_eval.py` scores that; abstentions still fail. Test: leads containing the gold SHA do not pass.
- `search_commits` rows now also carry `date` and `author` (about 500 characters over 15 rows), which could shift which commit the model picks; unmeasured.
- **Prompt:** the first attempt edited the shared commit paragraph and `tests/test_prompt_variants.py` failed (the baseline prompt hash is pinned). It is now the variant `commit_explain` (`commit_pr_commits` plus "SHA plus one sentence why" and "review comments are not in the data"); the default stays `commit_pr_commits`.
- Full suite, ruff and mypy pass. Zip hash `b63296e8`. **Next:** one server run of the 113 commit questions with `--variant commit_explain` (and one with the default) to check the score is still about 70 and the answers now explain.

### 2026-11-18 (later): fresh question set `agent_eval_fresh_v2` (73 questions, NOT yet run)

Every number so far is on questions the system was developed against. This set is new, written after the others were frozen, and
built by `eval/build_fresh_v2.py` (deterministic; sha256 b4f215712346): **code 30, docs 10, history 14, refs 4, kind 4,
prsearch 6, probes 5**. Same record format as `agent_eval_v1`, so `eval/agent_eval.py run --dataset eval/datasets/agent_eval_fresh_v2.jsonl` scores it as is.
- **Checks in the builder and `tests/test_fresh_v2.py` (5 tests):** no question text and no code gold range appears in `agent_eval_v1`,
  `agent_eval_fresh_v1` or `requests_qa` (one planned question was dropped for overlapping a gold range); code gold is a whole definition from the
  symbol table at `611c6162cb` with its source hash; each docs file is checked to contain its topic; facts come from git and `metadata.db`, and
  the busiest-author-per-year answers were checked against `git shortlog --no-merges`. New docs topics avoid the ten of `agent_eval_fresh_v1`.
- **No `commit` questions:** all 141 bug to fix pairs were used for tuning or selection, so there is no unseen gold for that type.
- **Limits:** written by Claude, one author, no human check of the questions; the author had read parts of the repository in this session, so the questions
  are new to the system but not blind. The mechanical scorer is the same as before (cited range overlaps gold; docs path named), so it has the same optimism:
  read a sample of the answers as well. The code questions use whole definitions, a different spread of files from the tuned set.
- **Next:** one server run on this set with the default settings, then read about 20 answers; compare the pass rate per type with `agent_eval_v1`.

### 2026-11-19: answer facts for PRs and people, one-session runner (worktree `quality/answers`, NOT measured live)

Work done in the git worktree `D:\Repo_chatbot_quality` (branch `quality/answers`, from commit `2bd1c8e`), so it does not collide with the main folder.
- **`agent/commit_answer.py`:** short answers now also get, from the tool rows only, the title and state of each PR or issue number they name (a bare "- 5797"
  becomes "- #5797 (pull request, merged): Switch to charset_normalizer") and, for contributors from `top_contributors`, their commits and first and most recent
  commit dates (answers "who knows it best" with recency). Only for answers of at most 300 characters; numbers need three digits or `#`. The eval still scores `model_answer`.
- **`eval/colab_session.py`:** runs every measurement in one server session (fresh set default and with guards, commit set default and `commit_explain`, the 103 tuned
  questions, the two architecture sets, two repeats of the fresh set for noise, then the summary step), resumable, refuses a server that is not the cuda-graphs + MTP,
  12,288-token profile, writes `report.md` (per type counts, flips, noise) and `answers_sample.md`. Runbook: `docs/COLAB_SESSION.md`. Tests: `tests/test_colab_session.py` (7).
- **Checks:** ruff and mypy clean; full suite green apart from one test that needs the local `data/` checkout (the worktree now has a junction to the main `data/`); the new tests
  (`test_commit_answer`, `test_colab_session`, `test_fresh_v2`) pass. Zip hash **`2e86916f`**. `tests/`, `experiments/` and `.env` are ignored files, copied into the worktree by hand.
- **Not done:** nothing here is measured on the live model; the reranker is not in the session (no room next to the 27B).

### 2026-11-19 (later): the session measured live (27B, cuda-graphs + mtp3, 12,288-token cap, shipped prompt `commit_pr_commits`, `--workers 4`)

Results are in `eval/colab_session_results/latest/` (`report.md`, `answers_sample.md`, one log and result JSON per step). Single runs, greedy decoding.
- **Fresh set `agent_eval_fresh_v2` (73 new questions):** 72/73 in each of three runs, with **zero questions flipping between runs**; guards on (`--citation-snap`, abstain retry 1, tool-error retry 1): 72/73, also no flips. The one miss is `fresh2-code-029` (the "no charset" wrong diagnosis). The set is at its ceiling, so it shows no harm from the guards, not a gain; it was written by Claude after the tuned sets and is read by one AI reader.
- **Tuned 103 questions:** code 40/40, docs 11/12, history 11/11, kind 5/5, prsearch 10/10, refs 10/10, commit 7/10, probes 5/5 (correct abstentions). No regression against the earlier measurements.
- **113 bug reports:** default **74/113** (14 abstained, 2 failed; 70 before the extra row fields and appended facts); `commit_explain` **78/113** (10 abstained, 1 failed; 6 gained, 2 lost). Bare-SHA answers: 86 of 99 with the default prompt (median model answer 12 characters), 0 of 103 with `commit_explain` (median 302). Of the two losses, commitfull-061 is partly the scorer: the model found the PR but said no commit SHA is recorded (empty `merge_commit_sha`). Not read against the diffs: `commit_explain` makes more specific claims (e.g. commitfull-006 names the exact changed line). Default NOT changed; it is the user's decision.
- **Architecture (25 answers read):** 12/15 and 9/10 mechanical; the answers are specific and cited. Loose citation seen (architecture-004 cites the same range for two classes); architecture-002 produced no valid reply.
- **Parallel mode:** `eval/agent_eval.py --workers N` and `serve_model(..., concurrency=N)` (27B notebook `CONCURRENCY = 8`; zip hash **`c2a718e2`**). About 1.9x faster than serial (fresh set 172 s vs 327 s), not the 3 to 5x guessed. Same answers as serial on the checks done (three identical fresh runs).
- **Mistakes found and fixed on the way (the runs they spoiled are kept under `eval/colab_session_results/run1_*`, `run2_*`, `run3_*` and must not be quoted):**
  - `agent_eval.py --variant` defaults to `baseline`, not the shipped prompt. The first "default" steps ran the old prompt (docs 1/10 on the fresh set, 10/10 with the `history_grounded` prompt that had been bundled with the guards). `colab_session.py` now passes `--variant commit_pr_commits` to every step and the guards step changes only the guards.
  - `--workers` first built agents in worker threads at once: "Cannot copy out of meta tensor" (embedding model) and "Could not connect to tenant" (Chroma). Then agents built in the main thread failed on SQLite ("objects created in a thread can only be used in that same thread"). Now the embedding model is loaded once and shared behind a lock, and each worker builds its own agent inside its thread, one at a time.
  - Stopping the Colab server mid-run produced a `commit_explain` result with 72 failures; its state was cleared and re-run.
- **Still to do:** the summaries step (resumable, serial, about 80 minutes) feeds the pilot; then `commit_summary_pilot.py package`, the pilot notebook on a second GPU session and `score`. Reranker at depth 50 still not measured.

### 2026-11-19 (later): commit-summary pilot packaged for review

Codex wrote 1,076 summaries in commit order using only the first 6,000 characters of each
indexed commit. The 67 original Qwen3.8-27B rows were preserved byte for byte. New rows
were passed through the pilot cleaner and carry `source: codex`. Codex did not read the
pilot queries or gold links.

`eval/commit_summary_pilot/pilot_input.zip` contains 1,143 commits (0.4 MB). Validation
found 0 duplicate SHAs, 0 summaries over 60 words, and 0 numbered issue references.
The pilot tests passed (12 of 12) and Ruff passed on `eval`.

**Validity:** Report any result as **summaries by Codex imitating the 27B**, with a
mixed corpus of 1,076 Codex and 67 original 27B summaries. This measures the mixed
corpus. The shipped 27B's summary quality has yet to be measured across the pilot,
and this result alone cannot justify having the 27B summarize all 4,883 commits.
Sample review, GPU embedding, and scoring are pending.

### 2026-11-19 (later): commit-summary pilot scored (summaries by Codex imitating the 27B)

Claude's audit of the summaries: 1,143 unique rows covering `commits.jsonl`; three spot checks against the diffs faithful; no sign of leakage. Codex rows are shorter than the 27B's (mean 165 characters and 24 words against 231 and 34) and open differently ("The Requests package..." against "This commit ..."). Scored from `pilot_vectors.npz` (T4, 0.6B embedder, 141 gold commits in a corpus of 1,143; `eval/commit_summary_pilot/result.json`):

| | hit@1 | hit@5 | hit@10 | hit@20 | MRR |
|---|---|---|---|---|---|
| dense, no summary | 69 | 102 | 114 | 121 | 0.596 |
| dense, with summary | 79 | 108 | 117 | 122 | 0.654 |
| hybrid (RRF), no summary | 66 | 100 | 114 | 124 | 0.579 |
| hybrid (RRF), with summary | 77 | 110 | 118 | 124 | 0.642 |

Hybrid, pairwise: 44 gold commits ranked higher, 16 lower, 81 unchanged. Caveats: the corpus is 1,143 commits, not the 4,883 in the real index (full-index hybrid hit@5 was 79/141, so these ranks are much easier); the summaries are Codex's, shorter than the 27B's; one run, one repository; the real test is the 113-question agent run on a re-indexed store. Reading: a real gain in this setting, not yet shown at full scale or with the 27B's own summaries. Summarising all 4,883 commits is the user's decision.

### 2026-11-19 (evening): 27B summaries for all 4,883 commits; retrieval up, agent answers not

Wrote a 27B summary for every commit (`eval/summarize_commits.py run --workers 8`, 4,816 new rows in about 20 minutes at ~300 commits/min; 0 errors; mean 239 chars, none with `#N`). Codex's 1,076 were not used. Tools: `summarize_commits.py prepare|run|status|package|score|build-index`; nothing under `src/` changed.

Full-corpus retrieval (141 bug reports, 4,883 commits; vectors by the 0.6B embedder on Colab, `result_all.json`):

| | hit@1 | hit@5 | hit@10 | hit@20 | MRR |
|---|---|---|---|---|---|
| hybrid, no summary | 46 | 80 | 97 | 105 | 0.442 |
| hybrid, 27B summary | 49 | 90 | 103 | 112 | 0.483 |
| dense, no summary | 55 | 77 | 93 | 105 | 0.474 |
| dense, 27B summary | 58 | 94 | 102 | 111 | 0.518 |

Hybrid pairwise: 59 gold commits ranked higher, 30 lower, 52 unchanged. Hybrid is not better than dense alone here (not investigated).

Agent test: experiment index `data/exp_summaries` (copy of `data/` with the summary text and vectors swapped in; real index untouched), the 113 commit questions, same server settings, `--workers 8`:

| prompt | without summaries | with summaries | gained / lost |
|---|---|---|---|
| `commit_explain` | 78/113 | 72/113 | 4 / 10 |
| `commit_pr_commits` (default) | 74/113 | 71/113 | 9 / 12 |

Both are lower, but neither difference is significant (sign tests p about 0.18 and 0.66). Read: better retrieval did not turn into more correct answers. Lost questions were not mainly ones where the model skipped the PR/issue route (1 of 10 and 2 of 12). One checked case (commitfull-019): the model named the true code fix, while the dataset's single gold sha is the later test-fix commit, so single-sha gold punishes a defensible answer. Not diff-checked beyond that. Decision: do NOT put summaries in the shipped index on this evidence.

Follow-up: the agent's own first `search_commits` query for each of the 113 questions, replayed on both indexes (`agent_query_ranks.json`, `lost_ranks.json`; scripts were scratch). Gold commit in the top N:

| | top 1 | top 3 | top 5 | top 10 | top 30 |
|---|---|---|---|---|---|
| index without summaries | 34 | 54 | 63 | 75 | 88 |
| index with summaries | 33 | 60 | 68 | 83 | 90 |

Rank better with summaries on 31 questions, worse on 21, same on 61. So retrieval on the agent's own (short, model-written) queries also improved, by 5 questions at top 5 and 8 at top 10, yet the pass count fell. Of the 17 distinct lost questions, 5 had the gold outside the top 30 on both indexes, 3 (022, 053, 109) had it pushed from inside the top 5 to outside it, and 9 had it unchanged or moved within the top 5. So the drop is not explained by summaries hiding the answer; the likely remainder is run-to-run variation in what the model picks, plus single-sha gold. Not measured: the run-to-run noise on this set (the fresh set had none across three runs, the commit set has not been repeated). The one experiment that would settle it is repeating the summary-index run once or twice with the same prompt.

Validity check of the 4,883 summaries (Claude): all present, 0 over 60 words (max 59, mean 35), 0 with `#N`, 0 think/label leaks, 52 one-sentence rows (rule says two or three), 31 duplicated texts, 4,097 start with "This commit". Grounding: 4,424 backticked names, 8 not found verbatim in the commit text (all fair readings, e.g. `None`, `False`); 1,007 numbers, 34 not found in 27 commits, almost all correct inferences (`MMXVII` to `MMXVIII` read as 2017 to 2018, `1,500,000` as 1.5 million). Two real errors found among the 27 read against their diffs: a76f4925b5 says the sleep was restored to 0.01 s, the diff changes .01 to 1 (direction reversed); 593734b770 names "status code 400", which is not in the diff. Then read by hand against message, file list and diff (about 140 distinct commits, 3% of the 4,883; not all of them): 30 random (seed 7), 45 whose summary states a direction or a from-to change (seed 11), 24 of the 32 commits over 5,000 characters plus 25 random short ones (seed 23), and the 27 flagged above. Clear errors, all a direction or a figure: a76f4925b5 (sleep .01 to 1 reported as restored to 0.01), 593734b770 (invented "status code 400"), 0a4b7356 (the diff ADDS a 991-line `__collections.py`, the summary says it removes it). Mild overreach, not wrong: 1bc377b1 calls `r.content` "deprecated", bf2e7352 calls `pycon` "invalid", 1311c96e calls a logo link "broken", 9d4fd3c9 quotes diff text, ad311497 and several vendored-library bumps add generic "bug fixes and improvements" filler. Estimate: about 2% of summaries have a wrong direction or figure (3 of about 140; 95% upper bound near 6%); the other 97% were faithful. Conclusion: valid for use as search text; they describe commits, they are not an authority on what a commit changed. Only 32 commits exceed 5,000 characters, so the 6,000-character input cut rarely matters.

### 2026-11-19 (night): Codex read all 4,883 summaries; Claude checked all 902 of its `wrong` rows

Codex read every row (`docs/REVIEW_CODEX_ALL_SUMMARIES_REPORT.md`, `review_all/verdicts_001..056.jsonl`): ok 3,162, filler 662, wrong 902 (18.47%), cannot_tell 157. Its report recommends blanking all 902. That figure is too high, because Codex was strict: it faulted bugs in the commit's own code, quibbled over wording, and put benefit language under `wrong` instead of `filler`.

Claude read the commit text, the summary and Codex's reason for all 902 (about 134 in a first pass, then the remaining 768 in order) and gave each row one verdict. Results, also in `docs/REVIEW_CLAUDE_OF_CODEX_WRONG.md` and `review_all/claude_check_of_codex_wrong.jsonl`:

| Claude verdict | Rows | Share of 902 | Share of 4,883 |
|---|---:|---:|---:|
| real error (contradicts the commit, invents a change, wrong name or number) | 127 | 14% | 2.6% |
| minor imprecision or overclaim | 223 | 25% | 4.6% |
| Codex over-strict, summary fine | 550 | 61% | 11.3% |
| cannot decide from the 6,000 characters | 2 | 0.2% | 0.04% |

Real errors are mostly: a direction or order reversed (2439 auth and body swapped, 3223 a revert described backwards, 3188 `while True`/`while 1` swapped, 3194 sleep .01 to 1 described as the reverse); a "syntax error" or "missing newline" fixed that the diff does not fix (1068, 1700, 1772, 3779, 4845, 4373, 2208); invented details (3204 "screw project", 382 "DeepMind Korea office", 379 "Prettier", 2850 "Korean language extension"); and imports called absolute that became relative (1483, 1574, 4434). Real plus minor is 350 rows (7.2%). The 2.6% matches the earlier hand estimate of about 2% (3 of about 140).

Not checked by Claude: Codex's 3,162 `ok`, 662 `filler` and 157 `cannot_tell` rows. An earlier spot-check of 40 `ok` rows found no error Codex missed (small sample). These verdicts are one reader's judgement, not ground truth; the real/minor line was Claude's call.

Decision unchanged: do not put summaries in the shipped index (retrieval improved, agent answers did not). If they are ever used, blank or rewrite only the 127 real rows, not all 902, and leave out whitespace, AUTHORS and version-bump commits. The run-to-run noise measurement stays unrun and is not needed for this decision.

### 2026-11-19 (night): why commit questions fail, and what was built to fix it (nothing measured on the agent yet)

The 113 commit questions pass about 70% (78 and 74 with the two prompts). Offline analysis, all on the real index:

- **Where the gold commit was (eval/commit_failure_sort.py, `eval/commit_failure_sort/failures_20261119.json`).** Replaying the agent's own first `search_commits` query for each miss: of 35 misses (commit_explain) the gold was in the top 5 for 4, at rank 6-15 (shown as a one-line row) for 8, at 16-30 for 5 and **not in the top 30 for 18**; for the 39 misses of the shipped prompt 7, 11, 5 and 16. So 23 of 35 never had the gold in the 15 commits the model saw. The agent picking wrongly from a good shortlist explains about a third; retrieval is the main loss. The model searched only once in most of the misses.
- **Labels (eval/commit_lenient_score.py).** A commit that names the same issue number as the gold or the reported issue counts as a hit: +0 for commit_explain (78 to 78), +2 for the shipped prompt (74 to 76; commitfull-011 and 106, two commits fixing the same issue). Label noise is small. By gold type: code changes 64/93 pass, docs or release notes only 8/10, tests only 4/4, version bumps 2/6.
- **Retrieval variants (113 questions, hybrid, gold SHA hits at 1/5/15/30, MRR; scratch scripts, not in the repo).**

| search | @1 | @5 | @15 | @30 | MRR |
|---|---:|---:|---:|---:|---:|
| real index, the model's query (today) | 34 | 62 | 80 | 89 | 0.427 |
| real index, the question text | 36 | 62 | 81 | 89 | 0.432 |
| real index, both fused (RRF) | 35 | 68 | 88 | 91 | 0.444 |
| real index, both fused, the question as the agent sends it (with the framing sentence) | 40 | 65 | 84 | 93 | 0.460 |
| summary index, the model's query | 33 | 68 | 86 | 92 | 0.438 |
| summary index, both fused | 42 | 73 | 89 | 94 | 0.500 |
| summary index, both fused, the question as the agent sends it | 42 | 75 | 89 | 93 | 0.501 |

  The rows that use the bug text alone leave out the sentence the harness puts in front of it ("A user reports the bug below ..."); with it the question alone is worse (real index @5 59, @15 78, @30 85), so the honest fused gain on the real index is +3 at @5 and +4 at @15, not +6 and +8. Fused with the question's weight at 0.5 gave @5 69, @15 85, MRR 0.462 on the real index; 1.0 and 2.0 were within noise of it. Richer BM25 text changed nothing (the full patch, code files first, 4,000 characters: within +-2 at every depth), and a larger BM25 weight hurt (0.3 to 0.6: @5 62 to 55). 22 questions have the gold outside the top 30 for every form; many are terse messages ("Fix #1293", "fixed #1723") whose issue numbers are scrubbed from the index text on purpose.
- **`is_fix` filter.** On this set it lifts @5 from 62 to 82, but all 115 gold commits pass the heuristic by construction (the data protocol requires it; only 1,101 of 4,883 commits do), so the number is an upper bound, not evidence. Measured as an agent run it carries that caveat.
- **Not done, and why.** Indexing issue or PR text as commit text (or mapping a matched issue to its fix) would hand the answer to this eval: each question is the text of the issue whose fix is the gold, and the protocol sets `linked_issue_text: false` for that reason. It could only be measured with the question's own issue held out, which leaves almost nothing to find. **Reranker dropped (user decision):** it did not perform well in the agent test and adds answer time; the notebook and check script built for it were deleted.
- **Built, all off by default, tests green:** `agent_commit_query = "fused"` (model query + the question, weighted RRF, `agent_commit_fuse_question_weight`); the `get_commit` tool and prompt variant `commit_verify` (read the top candidates, search again if none fits; `agent_commit_read_chars`); `agent_commit_hide_summary` (the summary helps the search, the model reads the raw commit); `MetadataDB.find_commit`; `eval/agent_eval.py --commit-query fused --fuse-weight --hide-summary` (result files now carry `_idx-<dir>`); `eval/commit_session.py` (one Colab session: control twice for noise, fused, verify, fused_verify, fused_fix_only, summary_hidden, with the sign-test p-value against the control and the lenient score). All notebooks regenerated: the serve notebook's source hash changed, so upload the new `notebooks/repo_rag_ask_source.zip` to Drive before starting the server.
- **Live result (27B, 113 commit questions, `commit_explain` prompt, `eval/commit_session_results/report.md`, 5 runs of the winning settings and 3 identical controls):**

| step | passed | won / lost vs control_a | sign-test p |
|---|---:|---:|---:|
| control_a, control_b, control_c (identical) | 76, 76, 78 | 2/2, 4/2 | 1.0, 0.69 |
| fused (model query + question) | 79 | 6 / 3 | 0.51 |
| fused, question weight 0.5 | 76 | 3 / 3 | 1.0 |
| verify (`get_commit`) | 69 | 2 / 9 | 0.065 |
| fused + verify | 69 | 2 / 9 | 0.065 |
| fused + fix-only filter (circular, upper bound) | 85 | 13 / 4 | 0.049 |
| summary index for search, summary hidden from the model | 78 | 10 / 8 | 0.82 |
| **summary index, fused, summary hidden** (run 1 / run 2) | **82 / 84** | 12 / 6, 13 / 5 | 0.24, 0.096 |

  The controls differ on 4 of 113 questions pairwise (8 across all three), so the run-to-run noise is about +-2 passes. The winning setting passed 82 and 84, above all three controls (best 78); pooled per question (two runs against three) it passes more often on 16 questions and less on 7 (sign test p = 0.093). That is a likely gain of about +6 passes (76.7 to 83 on average), not a proven one; neither single run is significant. **`verify` lost 7:** in 100 of 113 questions the model made exactly one `get_commit` call and almost never searched again; reading the diff made it more confident in a plausible earlier or related commit (nine lost questions all answered, one abstained after four calls). Not adopted. `fused` alone (+3) and the summary index alone (+2) are within noise; the combination is what moved. Lenient scoring changes at most +1 in every step. Nothing was made a default.
- **Decision for the user:** the one candidate worth shipping is the summary index used for search only, with `agent_commit_query=fused` and `agent_commit_hide_summary=true`. It needs `data/exp_summaries` (summary text embedded for all 4,883 commits; 127 of the summaries have a clear error, which now only affects ranking, not what the model reads) to replace the shipped commit index, and a repeat on a different repository before any claim about generalising.

### 2026-11-19 (late): commit shortlist/fusion fixes, opt-in `commit_bm25_weight`, and offline agent-query replay

- **Bug fixes (`src/repo_rag/agent/tools.py`, `graph.py`, `commit_answer.py`):**
  - `commit_brief` now reads `metadata["symbols"]` and falls back to scanning all lines of `page_content` instead of only `lines[:3]` (which previously dropped `Symbols:` on 421 of 2,233 commits with symbols in `all_commits.jsonl`, and on summary-bearing brief rows where `Summary:` pushed `Symbols:` past line 2), and `_brief_subject` emits a `(no message) ...` diff snippet instead of `Files: ...` when `strip_issue_references` clears the whole message (`commit_answer.subject_of` uses `(no commit message)`).
  - `ProductionTools._fused_search` now calls `_search(..., k=depth)` at full `retrieval_k` (50, capped at `rerank_candidate_depth` when `reranker_mode == "on"`) before `weighted_rrf` instead of truncating each list to `agent_commit_k` (15) before fusion; `strip_question_framing` strips the conversational lead-in before the pasted bug report; and `RepoAgent._answer` attaches `_question` only on the first `search_commits` call per question so follow-up model queries are not pulled back to the initial question.
  - Updated the `search_commits` schema description to advertise top 5 in full plus up to 10 brief candidates.
- **Opt-in `commit_bm25_weight` (`src/repo_rag/config.py`, `retrieval/retrievers.py`, `eval/agent_eval.py`):** `Settings.commit_bm25_weight` (default `None`) overrides `bm25_weight` for `kind == "commits"` only; wired `--commit-bm25-weight` into `eval/agent_eval.py`. No production default changed.
- **Offline CPU replay of the 113 `control_a` (`20261119T215016`) first queries on `data/` (`eval/commit_retrieval_check.py --agent-replay`, `eval/commit_retrieval_results/agent_replay_20261119.json`):**

| config | @1 | @5 | @10 | @15 | @30 | not_in_30 | MRR@30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `model_bm25_0.3` (current default) | 33 | 61 | 75 | 81 | 88 | 25 | 0.421 |
| `model_bm25_0.1` | 33 | 60 | 73 | 82 | 89 | 24 | 0.416 |
| `model_bm25_0.0` (dense only) | 31 | 61 | 71 | 79 | 88 | 25 | 0.396 |
| `fused_d15_raw_bm25_0.3` (old depth-15 + framing) | 36 | 68 | 75 | 84 | 89 | 24 | 0.435 |
| `fused_d50_clean_bm25_0.3` (depth-50 + framing strip) | **39** | **69** | **80** | 83 | **90** | **23** | **0.458** |
| `fused_d50_clean_bm25_0.1` | 36 | 66 | 78 | 81 | 90 | 23 | 0.440 |
| `fused_d50_clean_bm25_0.0` | 38 | 63 | 76 | 82 | 90 | 23 | 0.443 |

  Unlike full raw issue bodies, on the agent's short queries and clean fused queries `bm25_weight = 0.3` beats `0.1` and `0.0` (`@1`, `@5`, `@10`, MRR), so `commit_bm25_weight` stays `None`.
- **Timing note for `eval/commit_session_results/report.md`:** `control_c` (`78/113`, `23:03:42`) and `summary_fused_hidden_b` (`84/113`, `23:17:26`) ran after the `tools.py`/`graph.py` edits (`22:53:41`); earlier steps ran before those edits.
- **Checks:** 12 modified files pass `ruff check` and `ruff format --check`; `mypy` clean (`114 source files`); all 17 `notebooks/build_*.py` regenerated (`notebooks/repo_rag_ask_source.zip` short SHA-256 `8a8865f1`); full offline suite `991 passed, 1 deselected in 1151.49s` (`--basetemp=data/pt_full -p no:cacheprovider`). Nothing committed.

### 2026-11-20 (early): `--repo` support, Flask indexing notebook, pre-Colab fixes, and runbook

- **Multi-repo commit-eval plumbing and Flask dataset:**
  - `eval/agent_eval.py`, `eval/commit_session.py`, `eval/summarize_commits.py`, and `eval/commit_lenient_score.py` now accept `--repo` (`psf/requests` preserves all original paths; other repos isolate their datasets, summary working folders, summary indexes, and session results). Note: concurrent `commit_bm25_weight` edits in `config.py` / `agent_eval.py` coexist cleanly.
  - Built the `136`-question Flask test dataset (`eval/datasets/agent_eval_commit_full_pallets__flask_v1.jsonl` from `eval/datasets/phase5_bugfix_v2/repos/pallets__flask/pairs.jsonl`, `136` test + `34` dev pairs). Verified `eval/datasets/agent_eval_commit_full_v1.jsonl` (`psf/requests`, `113` questions) is clean against git.
  - Added `notebooks/build_flask_index.py` -> `notebooks/index_flask_on_colab.ipynb` (pinned at `d086db856be187255b8ec61ef409357393020f32`, full history, `float32`, `INDEX_PRS = False`, T4-sized `COMMIT_EMBED_BATCH = 4`).
- **Four pre-Colab fixes found during audit:**
  1. `notebooks/build_flask_index.py` `OUTRO` (and regenerated `index_flask_on_colab.ipynb`): `D:\Repo_chatbot\data\phase5_repos\pallets__flask` is a `--no-checkout` clone with no working tree, and `import-index` clears `clone_path`; updated `OUTRO` to `git clone --no-hardlinks` into `D:\Repo_chatbot\data_flask\repos\pallets__flask` and check out `d086db856be1` before `import-index`, `index-history --offline`, and `index-symbols`.
  2. `eval/summarize_commits.py`: `score_all()` now swaps/restores `pilot.QUERIES` alongside `COMMITS`/`SUMMARIES`/`RESULT`, `configure()` is idempotent, `main()` accepts `--data-dir` (used by `prepare()` and `build_index()`), and `prepare()` fails fast if the repo has no indexed commits in that `data_dir`.
  3. `eval/commit_session.py`: `configure()` is idempotent (keeps immutable `BASE_COMMON`) and defaults `EXPERIMENT_INDEX` to `data/exp_summaries_<slug>` for non-Requests repos; `report()` closes the read-only `metadata.db` connection cleanly.
  4. Added unit tests in `tests/test_flask_index_notebook.py`, `tests/test_summarize_commits.py`, and `tests/test_commit_session.py`.
- **Runbook:** Wrote `docs/FLASK_COMMIT_RUNBOOK.md` covering Colab T4 indexing, separate local import (`D:\Repo_chatbot\data_flask`), 27B commit summarization, summary embedding, and the 136-question Flask `commit_session.py` evaluation. No Flask GPU step has run yet.
- **Checks:** `ruff check` passes across all edited/evaluated files; `uv run mypy` clean (`114 source files`); targeted offline suite (`test_flask_index_notebook.py`, `test_commit_session.py`, `test_summarize_commits.py`, `test_agent_eval.py`, `test_commit_summary_pilot.py`, `test_commit_shortlist.py`, `test_commit_verify.py`, `test_agent_graph.py`, `test_commit_answer.py`, `test_retrievers.py`, `test_config.py`, `test_colab_notebook.py`) `149 passed in 8.08s` (`--basetemp=data/pt_flask_audit -p no:cacheprovider`). Nothing committed.

### 2026-11-20 (morning): Flask Colab index verified locally, symbols indexed, and summary prep checked

- **Colab index verification:** Uploaded `repo_rag_index.zip` (`11.31 MB`, `pallets/flask` @ `d086db856be187255b8ec61ef409357393020f32`) matches `D:\Repo_chatbot\data_flask` byte-for-byte on IDs/texts (`max_vec_diff = 1.49e-08`): `429` code chunks, `653` doc chunks, `3,829` commits (`history_commits = 3,829`, `history_files = 9,317`, `history_symbols = 15,454`). `eval/commit_summary_pilot/pallets__flask/` already holds `all_commits.jsonl` (`3,829` commits) and `queries.jsonl` (`170` bug-fix pairs).
- **Local checkout + symbol index:** Cloned `D:\Repo_chatbot\data\phase5_repos\pallets__flask` (`--no-hardlinks`) into `D:\Repo_chatbot\data_flask\repos\pallets__flask` and checked out `d086db856be187255b8ec61ef409357393020f32`, then ran `uv run repo-rag index-symbols pallets/flask --checkout D:\Repo_chatbot\data_flask\repos\pallets__flask`: indexed `1,622` definitions (`code_symbols = 1622`), `3,894` call sites (`code_calls = 3894`), and `639` imports (`code_imports = 639`) from `80` files.
- **Verification:** `uv run repo-rag search "how does Flask push the request context" --repo pallets/flask --kind commits -k 3` returned `e71a5ff8de93`, `7b5015010bc8`, and `47288231fe8f`; `build_agent("pallets/flask", ...)` against `D:\Repo_chatbot\data_flask` succeeded (`RepoAgent`); `D:\Repo_chatbot\data\manifest.json` remains `['psf/requests']` only (untouched). Next: Stage 3 of `docs/FLASK_COMMIT_RUNBOOK.md` (start 27B A100 server and run `summarize_commits.py --repo pallets/flask run`).

### 2026-11-20 (evening): Flask replication of the commit experiments, and `commit_explain` becomes the default prompt

- **Second repository: pallets/flask** (pinned `d086db85...`, 3,829 non-merge commits, 136 test questions from `phase5_bugfix_v2`, file `eval/datasets/agent_eval_commit_full_pallets__flask_v1.jsonl`; same protocol as requests: an issue to its own linked fix commit). Indexed on a free T4 (`notebooks/index_flask_on_colab.ipynb`, float32), imported into the separate folder `D:\Repo_chatbot\data_flask`; the shipped psf/requests index was not touched. The `--repo`, `--data-dir` and `--summary-index` options exist in `eval/agent_eval.py`, `eval/commit_session.py`, `eval/commit_lenient_score.py` and `eval/summarize_commits.py`; results go to `eval/commit_session_results/pallets__flask/`.
- **Summaries:** the 27B wrote all 3,829 (0 errors), embedded on a T4 with `commit_summary_pilot_on_colab.ipynb` (new switch `EMBED_BASELINE = False` skips re-embedding the unchanged text; the last cell now saves to Google Drive first, because the browser download failed silently twice). `summarize_commits.py --repo pallets/flask build-index` wrote `data/exp_summaries_pallets__flask` (3,829 commits carry a Summary line). Retrieval only, 170 queries over 3,829 commits, hybrid: hit@1 93 to 99, hit@5 141 to 146, MRR 0.675 to 0.712; 46 gold commits moved up, 20 down (dense: 51 up, 21 down).
- **Live result (27B, 136 Flask commit questions, `commit_explain`, `--workers 8`):**

| step | strict | abstained | won / lost vs control_a | sign-test p |
|---|---:|---:|---|---:|
| control_a, control_b, control_c | 101, 100, 101 | 23, 24, 24 | n/a | n/a |
| fused | 103 | 22 | 7 / 5 | 0.774 |
| summary index, fused, summary hidden | 106 | 18 | 12 / 7 | 0.359 |
| the same, repeated | 107 | 17 | 12 / 6 | 0.238 |

  Against the mean of the three controls, the two summary runs are ahead on 14 questions and behind on 9 (p = 0.405). Identical controls differ on 5 of 136 questions. Same direction as requests (82 and 84 against 76 to 78) but nothing is significant on its own; read it as a likely gain of about 4 to 6 points, not a confirmed one. Cost at answer time: none (summaries are used for search only, the model sees the raw commit); the one-time cost is a 27B summary pass plus a GPU embedding run per repository. A first accidental duplicate session was killed before it finished and its log deleted; the reported runs come from one chain.
- **Decision (user, 2026-11-20): `commit_explain` becomes the default `agent_prompt_variant`** (was `commit_pr_commits`). Evidence: 78/113 against 74/113 on requests, bare-SHA answers 0 of 103 against 86 of 99, stable at 100 to 101 of 136 on Flask. **Not re-measured on code, docs, history and architecture questions under this prompt**: run `eval/colab_session.py` in the next 27B session (its "default" steps still pass `commit_pr_commits` on purpose, so they stay comparable; add `--variant commit_explain` runs for the 103 tuned questions before quoting that nothing regressed). Tests updated (`test_prompt_variants`, `test_references`, `test_commit_verify`), all `notebooks/build_*.py` regenerated (the serve notebook embeds the source hash: upload the new `notebooks/repo_rag_ask_source.zip` to Drive before the next server session).
- **Not decided:** shipping the summary index with `agent_commit_query=fused` and `agent_commit_hide_summary=true`. Recommendation: keep both off by default and document them as opt-in (needs a summary pass per repository).

## 3. Decision log


| Date | Decision | Why | Alternatives considered |
|---|---|---|---|
| 2026-11-19 | Continue the commit-summary pilot with 67 original 27B rows and 1,076 labelled Codex rows | The original server session ended. The mixed-writer pilot can measure this corpus and provides limited evidence about the shipped model | Resume with the 27B if its server becomes available and measure an all-27B corpus |
| 2026-11-01 | Implement the approved v2 bakeoff; freeze common prompt/parser only after complete shared dev matrices; retain raw failures and unchanged gates | User-approved four revisions and dev amendment; model execution belongs to user's A100 notebook | Local model runs, partial dev selection, cached-score analysis, post-test tuning or dependency changes |
| 2026-11-01 | Freeze Phase 7 protocol/splits/probes before model research; propose four exact revisions and wait for approval | User-required order and no new dependency; preserve raw labels, test gates and separate summary approval | Load models or build the harness before approval; silently upgrade Transformers; select on test |
| 2026-11-01 | Adopt no architecture-summary change; keep code search unchanged; prepare E3 on F1 selected by the frozen dev tie order | Router fails all guards; best fusion gain +0.0333 misses the required +0.05 and code guard; labels unreviewed by a person | Promote a test-selected fusion row or tune the routing grid again on v1 |
| 2026-10-31 | Freeze architecture_v1 before retrieval; keep summaries and richer facts opt-in; require explicit approval for E3 | Apply the brief's dev/test gates without changing labels or choosing a prose model for the user | Tune on test questions, silently relabel v1, or enable summaries in default code search |
| 2026-10-26 | Automatically release the commit-depth Colab runtime after verified Drive save and flush | User requested avoiding idle compute after the run; retain results before releasing the allocation | Leave an idle GPU attached or kill the Python kernel without releasing the runtime |
| 2026-10-26 | One top200 quality pool and shared scores for depth50/100/200, 8B primary | Isolate depth; stable ties; historical short-list depth50 stays a separate control | Re-score three depths independently or compare against a different fused pool |
| 2026-10-26 | Commit BM25 weights selected on 66 dev queries with +0.01 hysteresis; test descriptive | Avoid test tuning and churn; conditional fresh 8B comparison if nondefault wins | Choose from the best test MRR or combine untested depth/weight changes |
| 2026-10-26 | Frozen round-robin timing subset across all repos, fixed retrieval_k=200 and per-repo ratios | Isolate scored depth while keeping candidate lists comparable; do not average percentiles | Omit Requests by concatenating first 60, or claim prefix quality reuse measures latency |
| 2026-10-26 | Freeze Phase 5 v2 with context-preserving scrub and message-only evaluation candidates; keep v1 immutable | Review found descriptions deleted and possible label-driven pool membership | Overwrite v1 or spend GPU time on the flawed corpus |
| 2026-10-26 | Restrict per-repo/link-path 8B guards to >=30 connected test groups | One flip among 14 FastAPI questions cannot support a 0.02 precision claim; small strata descriptive with no safety conclusion | Treat every small-sample flip as failure or ignore its score |
| 2026-10-26 | Keep own-fix metrics and add same-file proxy plus BM25 query-budget sensitivity | Product similar-fix relevance and unequal neural budgets are not established by exact own-fix Hit/MRR | Claim the proxy resolves semantic usefulness |
| 2026-10-25 | Reject callee under frozen fresh-question rule; keep expansion none and move to Phase 5 without retuning | Fresh proof 20/30 to 17/30, paired -0.10 [-0.2424,+0.0345]; old/text guards pass but qualifying gain fails | Adopt the repeated two old fixes, promote secondary both, or tune after seeing fresh outcomes |
| 2026-10-25 | Freeze 30 new multi-block questions including 12 Click cases; retain old 120 byte-for-byte | Fairer primary assessment than post-hoc saved-list fixes; disclose old-source overlap and lack of independent review | Adopt expansion from two repaired old questions |
| 2026-10-25 | Wire store-backed expansion after enabled code reranking only, default none | Preserve laptop/hybrid behaviour and evaluate the actual production option with scoped evidence | Enable on first-stage lists despite measured regressions |
| 2026-10-25 | Hash preregistration on disk without a git commit; fixed callee primary and proof/regression/text gates | User forbids commits; freezes precede retrieval; both stays secondary; rejection leads to Phase 5 | Commit without authorization; repeatedly tune on the new questions |
| 2026-10-25 | Freeze 90 stratified diagnostic questions; retain old15 and report added75 separately | User requested a larger comparison; explicit intent guards and cue-free/multi-block questions diagnose different failures | Fifteen targeted paraphrases only; claim random representativeness or guaranteed significance from count |
| 2026-10-25 | Lock and validate the depth protocol; report direct 50-20 with per-type guards and actual policy timings | Prevent weight tuning, repeated-call pseudoreplication, or averaging latency percentiles | Select defaults from pooled MRR alone or reuse one fifty-pair call as twenty-candidate timing |
| 2026-10-25 | Do not prefer depth50 from the old paired comparison; leave settings unchanged pending new measurements | Old MRR20 is marginally higher with intervals touching zero; old timing scored fifty once for both views | Claim measured speedup or change defaults from an old implementation-only sample |
| 2026-10-25 | Separate evaluation completion from pending rerank-default deployment decision | User requested precise status; A100 is experimental and deployment hardware is undecided | Imply a default/deployment approval from phase checkbox |
| 2026-10-25 | Add rerank_candidate_depth, default 50, independently of retrieval_k | Evaluate production 20/50 without changing existing opt-in behaviour or weakening first-stage recall | Set default 20 immediately; mix unscored tail results |
| 2026-10-25 | Keep ambiguous stress alternatives separate from frozen v2 | Test/demonstration may be sufficient for unspecified intent; all 24 old test queries give explicit cues | Relabel old results or call same-assistant/paraphrased evidence independent |
| 2026-10-25 | Complete Phase 4 after audited v2 results and keep code reranking opt-in | Fresh rerank20 gain is positive, but labels lack independent review and measured A100 latency is ~0.9–1.0 s; production depth differs | Automatic promotion from pooled numerical guards alone |
| 2026-10-25 | Keep test penalty 1.0 and omit reserved BM25 quota | Penalty cuts test MRR 0.705 → 0.352; quota gives no fresh gain and one legacy rank regression | Global source bias; quota as an unmeasured default |
| 2026-10-25 | Preserve v2 reports, full compressed traces and matching source ZIP | Make the 129-question result auditable after future code changes | Overwrite v1/v2 labels or discard score/latency provenance |
| 2026-10-25 | Use production headers for the prechosen v2 deployment matrix; preserve v1's common-header comparison separately | Production reranking must be assessed with the text it will actually receive | Promote from a different header policy without parity evidence |
| 2026-10-25 | Freeze the 4 primary v2 policies; lexical quota 5 of 20; old dev weights remain fixed | Evaluate ranking without another tuning loop on 16 old questions | Search weights/quota after inspecting fresh outcomes |
| 2026-10-25 | Keep reranking opt-in and Phase 4 unchecked until the v2 GPU result and decision | Local CUDA is unavailable; code/tests/BM25 are not GPU or latency measurements | Mark complete from prepared notebook or fake scorer tests |
| 2026-10-25 | Prioritize ranking/instrumentation and production parity before changing database or model | Original AST exact top50 covers all gold; missing candidate logs and code/commit rerank instruction mismatch limit deployment conclusions | Replace HNSW or embedding model from current top10 MRR alone |
| 2026-10-25 | Keep v1 gold frozen; independently review minimal evidence/alternatives in a versioned follow-up | Whole-function gold can mark the exact requested branch incomplete; original test cases have now been inspected | Relabel existing gold to match retrieved chunks or continue tuning against old test results |
| 2026-10-25 | Keep Phase 4 open after the valid initial full run; expand test/example gold before a penalty policy | All original gold targets implementation files; dev/test share that bias | Promote the 0.8 test penalty from the current benchmark |
| 2026-10-25 | Report HNSW=exact only as a small-index plumbing observation | 270–565 vectors/index cannot establish scale recall or best-index choice | Claim ANN quality/scale from 100% observed recall |
| 2026-10-25 | Keep code reranking opt-in after the separate real 20/50 runs | Point estimates improve but held-out improvement intervals span zero; no separate depth timing | Treat main rerank=false as no reranker experiment; enable globally |
| 2026-10-25 | Distribute Phase 4 source as a separate ZIP loaded from Drive; keep the notebook small | User explicitly rejected long encoded cells and will upload the ZIP to Drive | Embedded Base64 archive |
| 2026-10-25 | `bm25/` is the canonical current report path; archive original reports under `bm25_legacy/` | Avoid stale-looking main reports and retain before/after evidence | Require readers to discover updated metrics under `bm25_review/` |
| 2026-10-25 | Use uv for Phase 4 notebook dependencies, with pip only as an optional uv bootstrap | User requested uv; target the notebook kernel explicitly while keeping locked direct versions | Install the full dependency set through pip |
| 2026-10-25 | Add union line coverage and ≥90%-complete gold alongside overlap metrics | Any-overlap recall cannot measure answer completeness | Replace old metrics and lose comparability |
| 2026-10-25 | Bootstrap evidence groups, pair by question; show pooled fixed comparisons and held-out tuned results separately | Shared-answer questions are dependent; pooling a selected configuration would mix tuning and assessment | IID question bootstrap; pooled tuned headlines |
| 2026-10-25 | Keep compact result JSON/Markdown versionable; preserve original baseline | Reproducible audit evidence is part of the project | Commit vector/Chroma caches; keep only rounded tables |
| 2026-10-25 | Type-check maintained Phase 4 in the default command; explicitly exclude legacy research scripts | The original 27-file clean check did not cover new eval code | Hide all eval errors or imply whole-tree typing |
| 2026-10-25 | Phase 4 gold = hashed source line ranges; overlapping evidence groups stay within dev/test | Fair across chunkers; paraphrases cannot leak shared answer code between splits | Matching our symbol metadata (biased); random question-level split |
| 2026-10-25 | Common source-derived headers across all strategies; exact dense is the primary controlled comparison | Separates structural chunking from contextual headers and HNSW approximation | Native headers only on AST; HNSW-only comparison |
| 2026-10-25 | Reference astchunk is dev-only; unsupported JavaScript is N/A | Keep the corpus identical rather than silently changing parser or files | Parse JS as TypeScript; omit unsupported files |
| 2026-10-25 | Keep production weights unchanged until full GPU results are reviewed | BM25-only checks do not establish hybrid-search quality | Tune defaults from the initial lexical table |
| 2026-10-21 | LangChain + LangGraph for orchestration | The user asked for it. LangGraph gives routing, tool loops and memory | LlamaIndex |
| 2026-10-21 | Our own cAST-style chunker on tree-sitter | `astchunk` has 4 languages and no symbol metadata | Use `astchunk` directly; LangChain `LanguageParser` |
| 2026-10-21 | ChromaDB behind an interface, plus BM25 + RRF | Simple, local, first-class LangChain package | LanceDB (native hybrid), kept as a later option |
| 2026-10-21 | GitPython on a local blobless clone for history | Full history cheaply, no API rate limits | GitHub REST API for commits |
| 2026-10-21 | Two issue-linking paths (direct + via PR) | fastapi: 8 direct links in 7,903 commits | Regex only |
| 2026-10-21 | SQLite exact-answer layer and typed tools; no LLM-written SQL | Vector search can't answer who/when/how many reliably. Safety | Text-to-SQL |
| 2026-10-21 | Python 3.11+, uv, ruff, mypy, pytest | Modern, fast toolchain | pip/poetry |
| 2026-10-21 | Exclude jina-code-embeddings | CC-BY-NC license | n/a |
| 2026-10-21 | Benchmark CodeRankEmbed vs Qwen3-0.6B vs nomic-embed-code | Covers small / mid / large and code-specialist / general | Qwen3-4B/8B, jina, CodeSage |
| 2026-10-21 | Defer vLLM | Doesn't affect the model choice; Colab risk | Add now with a parity check |
| 2026-10-21 | Add langchain as the AI showcase and scale repo | 491 direct links, 17k commits, relevant domain | llama_index, transformers |
| 2026-10-21 | Qwen3-Embedding-0.6B @ 512d as embedding model (final after run 2) | T2 nDCG@10 0.558 vs ≤0.29 for the nomic models; fastest indexing; Apache-2.0; laptop-viable | nomic-embed-code 7B (best T1 only), CodeRankEmbed |
| 2026-10-21 | Weighted fusion instead of equal-weight RRF | Equal RRF dropped T1 from ~0.84 to ~0.59 | Plain RRF |
| 2026-10-22 | One query instruction for all collections; commit docs = message + files + symbols + 800-char diff | T3 grid: no instruction beat the code one; 800 chars matches 2,000 and indexes 1.4x faster | Commit-specific instruction; 2,000-char diff |
| 2026-10-22 | Qwen3-Reranker-4B over top 50, GPU-only (`reranker_mode=auto`); bge removed | T3 0.494 to 0.687 (beats 7B embedder); bge hurt (-0.12); laptop CPU too slow | Qwen3-4B embedder (+0.004 after rerank), Qwen3-Reranker-0.6B (+0.06), bge |
| 2026-10-22 | Merge only within the same enclosing definition; signature header only for single-def chunks | Keeps class headers with their methods, stops cross-boundary merges, avoids misleading headers | Plain cAST sibling merge |
| 2026-10-22 | Own Markdown/RST heading scanner instead of MarkdownHeaderTextSplitter | LangChain's splitter rewrites whitespace, breaking line numbers and citations | MarkdownHeaderTextSplitter |
| 2026-10-22 | Accept over-budget unsplittable nodes (docstrings, decorator stacks) | 1.5% of chunks, max 2.4x budget, well under the embedder's 2,048-token cap; splitting a docstring mid-paragraph hurts more | Line-split long string literals |
| 2026-10-22 | Comments attach forward only within budget; over-budget multi-line leaves split by line; merge backtracks on shared lines | Real JS/TS/Java/Go repos had 116 over-budget chunks (max 11,481); now 31 (0.6%) with 0 split defs | Always attach; force-join on shared lines |
| 2026-10-22 | JS/TS bindings (`x = function`, `{k: () => }`, `const C = class`) are definitions | Express defines most of its API this way | Declarators only |
| 2026-10-22 | Own BM25 index and weighted-RRF fusion instead of EnsembleRetriever | Need metadata filters on both signals, a persisted index, and tunable weights | EnsembleRetriever + BM25Retriever (langchain-community) |
| 2026-10-22 | Collections named after the embedding space; manifest records it | Vectors from different models must never mix; changing the model forces a re-index | One collection per kind |
| 2026-10-22 | Wrap sentence-transformers as our own LangChain Embeddings, no langchain-huggingface | Exact control of instruction, padding and truncation as benchmarked; one fewer dependency | langchain-huggingface HuggingFaceEmbeddings |
| 2026-10-22 | ~~Retry Chroma's transient hnsw-reader error~~ (superseded 2026-10-24) | It wasn't transient; see the 2026-10-24 entry | |
| 2026-10-24 | Move indexes as plain data (jsonl + npy), not Chroma files; import rebuilds BM25 | Chroma/HNSW files are OS- and version-specific; text + vectors are portable and let import merge per repo | Copy `data/chroma` |
| 2026-10-24 | Colab notebook embeds our source + uv.lock pins and runs the real CLI; a test keeps it fresh | Repo isn't on GitHub; no second implementation to drift | pip install from a repo; reimplement indexing in the notebook |
| 2026-10-24 | Build Colab indexes in float32 (`embedding_dtype`) | Matches CPU laptop queries; bf16 differs by cosine >= 0.9998, so bf16 stays an option | bfloat16 default |
| 2026-10-24 | Chroma `hnsw.sync_threshold=2`, no retry | Root cause of intermittent 'hnsw segment reader' errors (20 -> 0 failures in 5,760 reads); retry hid it | Retry reads; reopen client |
| 2026-10-25 | Open-source answer models; `llm_provider=local`, names unselected until Phase 7 | Explicit user request; Phase 5 retrieval needs no Anthropic key | Paid LLM default |
| 2026-10-25 | Search commit-only symbols_diff800 text; keep linked issue text in SQLite | Same leakage-safe formatter in production and evaluation | Append query issue text to searchable commits |
| 2026-10-28 | Fact-set expected answers come from raw sync nodes and plain `git`, never from the tools | A tool checked against itself proves nothing; ties and undefined orders are skipped | Generate answers from `MetadataDB` |
| 2026-10-28 | PR/issue docs embed title + labels + capped body; state/author stay metadata | Exact fields belong to the typed tools; search is for meaning | Embed reviews and comments |
| 2026-10-31 | PR/issue search: title-query fusion on by default; PR reranking is opt-in per call, to be switched on by the agent when a GPU exists | Fusion: Hit@5 +0.07 httpx, +0.02 flask, never worse. Reranker (Qwen3-8B) beat fused on both repos (Hit@1 +0.18 / +0.05, MRR +0.13 / +0.05), intervals overlap at n=45/58 | Rerank by default (laptop has no GPU); tune weights on 45 pairs (overfit risk); more PR text indexed (needs another GPU embed) |
| 2026-10-25 | Freeze 329 real pairs and a fixed 4B/8B matrix, report actual shortfalls | Both linking paths verified; no tuning on test or fabricated 200/repo counts | Retune weights on legacy questions |
| 2026-11-10 | Phase 7 answer-quality judge: quantized Qwen3.8-27B (FP8 default, INT4 fallback), gated by negative controls | User decision D3 ("Qwen3.7-28B quantized only", resolved to Qwen3.8-27B); the Phase 4 judge is a reranker and cannot grade answers | Prompt-only manual check; MiniCheck-7B (faithfulness only); a second cross-family judge |
| 2026-11-11 | Label the filled 50-answer sheet as AI review by Codex; retain independent human calibration as open | User requested Codex review of every answer; AI marks must not be presented as a person's independent evidence. Two correctness marks are provisional due to missing reference topics | Treat AI labels as human validation or declare the judge trusted without its output |
| 2026-11-18 | `agent_commit_k` = 15 and `agent_commit_max_tool_calls` = 6 (commit questions only: in force once `search_commits` has been called) | 113 bug to fix questions on the 27B: 57/113 before, 65 with k=15, 70 with k=15 and 6 calls; k=20 and 8 or 10 calls add nothing; other question types keep 4 calls and were not re-measured | k=20 (no gain, more failures); a global cap of 6 (not measured on the other types); BM25 or title-fusion tuning (within noise) |
| 2026-11-18 | The 27B serve notebook defaults to CUDA graphs + MTP (3 draft tokens) with `--max-num-seqs 8` (`ENFORCE_EAGER = False`, `MTP_TOKENS = 3`); `server_command` itself still defaults to eager | User decision after measuring: mean 4.2 s per question vs 23 s in eager mode, all Phase 6 and regression gates still met on the 128 questions (architecture 13/15 and 8/10, code 39/40, docs 11/12, prsearch 10/10, probes 5/5; commit 5/10, one question below the eager run); about 4 to 6 minutes of startup compile; not bit-identical to eager, a few answers move either way | Eager (23 s, `ENFORCE_EAGER = True`, `MTP_TOKENS = 0`, still available as the two switches); CUDA graphs without MTP (5.0 s) |
| 2026-11-18 | Defaults become the Qwen3.8-27B profile: `llm_model` = `cyankiwi/Qwen3.8-27B-AWQ-INT4` (revision `e3a70090`), `agent_code_extra_files` 3, `agent_code_skip_tests` True, `agent_tool_result_chars` 24000, `agent_max_input_tokens` 12288; Phase 6 ticked; the 14B stays as `LOCAL_14B_MODEL/REVISION` and the two notebooks that load it pin it | User decision after the 27B met every gate (architecture 13/15 and 8/10, code 39/40, docs 11/12, prsearch 10/10, probes 5/5); the laptop agent never loads the model, the 27B cannot be loaded locally | Keep the 14B default and the old limits (14B fails the architecture gate: 9/15 and 6/10) |
| 2026-11-18 | Default prompt variant becomes `commit_pr_commits` (`docs_hint_narrow` plus one paragraph: when `get_pr` returns `pr_commits`, name the matching PR commit, never the merge commit) | User decision (option A). Measured on the 27B in eager mode: commit 7/10 vs 6/10 (088 fixed; both counts include the commit-083 label), code 40/40, docs 11/12, prsearch 10/10, probes 5/5, history 11/11, refs 10/10, kind 5/5. n = 10 for commit, one run; **not re-measured on graphs + MTP**, where the last full run used `docs_hint_narrow` (commit 5/10, 088 a miss) | Keep `docs_hint_narrow` (088 stays a miss) |
| 2026-11-18 | No fallback model: the 14B (`Qwen2.5-Coder-14B-Instruct`) is unsupported under the current 27B defaults (user decision). Its settings stay in `config.py` as `LOCAL_14B_*` and its notebooks pin it explicitly, but it was not re-measured under the new limits. | The 27B profile met every gate; keeping a second model would need a full re-measure for no stated need. |
| 2026-11-20 | Default prompt variant becomes `commit_explain` (`commit_pr_commits` plus "SHA plus one sentence why" and "review comments are not in the indexed data"). Summary-index search (`agent_commit_query=fused`, `agent_commit_hide_summary`) stays off by default | User decision after the commit runs: 78/113 vs 74/113 (requests), 0 of 103 bare-SHA answers vs 86 of 99, 100 to 101 of 136 on Flask; summary index 106 and 107 of 136 vs controls 100 to 101 (not significant) | Keep `commit_pr_commits`; make the summary index the default |

---

## 4. Tools and libraries in use

Phase 7 implementation: `agent/bakeoff.py`, `agent/bakeoff_tools.py`, the common prompt;
`eval/answer_bakeoff_{io,backend,scoring}.py`, `run_answer_bakeoff.py`,
`analyze_answer_bakeoff.py`, `prepare_answer_bakeoff_capsule.py` and
`notebooks/build_answer_bakeoff.py`. Existing dependencies only; tests use `ScriptedAnswerModel`.

Phase 7 preparation adds `eval/prepare_answer_bakeoff.py` (standard library only) and the
frozen inputs/proposal in `eval/datasets/phase7_bakeoff_v1/`. It loads no models. The
implementation/review checkpoint is `docs/PHASE7_BAKEOFF_PROTOCOL_REVIEW.md`.

Architecture summaries add `eval/build_architecture_dataset.py`, `eval/run_architecture_eval.py`,
`eval/analyze_architecture.py`, `eval/prepare_summary_prose.py`, `eval/run_summary_prose.py` and
`notebooks/build_summary_prose_eval.py`. New typed lookup: `MetadataDB.file_summary_titles`.
All use existing dependencies. `docs/ARCHITECTURE_LABEL_REVIEW.md` is the human review sheet;
`docs/SUMMARIES_BUILD_REVIEW.md` records commands, gates, limitations and the E3 proposal.

Phase 5b tooling: `history/pr_index.py`, `eval/build_facts.py`, `eval/run_facts.py`, `eval/run_pr_search_eval.py`; existing dependencies only (numpy for the bootstrap).

Commit-depth tooling: `eval/run_commit_depth_eval.py`, `eval/analyze_commit_depth.py`,
`eval/retrieval/commit_depth.py` and `notebooks/build_commit_depth.py`; existing dependencies
only. These add shared-score depth comparisons, dev-only weights, production-call timing
and hash/prefix reconstruction.

| Tool | Version (locked) | Used for |
|---|---|---|
| uv | 0.11.33 | Env and dependency management |
| Python | 3.13 (local), ≥3.11 supported | Runtime |
| typer / rich | 0.27.3 / 15.0.0 | CLI and terminal tables |
| pydantic / pydantic-settings | 2.14.1 / 2.16.1 | Config |
| GitPython | 3.2.1 | Clone, `ls-files`, commit access |
| sqlite3 / urllib.request | Python stdlib | Typed read-only history queries, atomic links and cached GitHub GraphQL; no new dependency |
| pytest / ruff / mypy | 9.1.1 / 0.17.1 / 2.4.0 | Tests, lint/format, types |
| git | 2.47.1 | Blobless clones, history mining |
| tree-sitter / tree-sitter-language-pack | 0.26.0 / 1.22.1 | AST parsing (grammars cached per user) |
| langchain-core / langchain-text-splitters | 1.6.13 / 1.1.3 | `Document`, fallback splitters |
| chromadb / langchain-chroma | 1.5.9 / 1.1.x | Vector store (persistent, cosine) |
| sentence-transformers / transformers / torch | 5.7.0 / 4.57.6 / 2.14.1 (CPU) | Local Qwen3 embeddings and reranker |
| rank-bm25 | 0.2.2 | Keyword index |
| astchunk / pyrsistent | 0.1.0 / 0.20.0 | Phase 4 reference cAST, dev-only; its four grammar wheels are transitive dev dependencies |
| sentence-transformers, rank-bm25, pandas | ≥5.1 (Colab) | Embedding benchmark only (not project deps) |
| Colab A100 runtime | Python 3.13.16, torch 2.11.0+cu130, NumPy 2.1.3 | First real Phase 4 full run; local Windows environment unchanged |
| Offline failure analyzer | eval/analyze_retrieval_failures.py | Existing NumPy/scorer/BM25; optional cached tokenizer/parser; no new dependencies or model inference |
| V2 runner and summarizer | eval/run_followup_eval.py; eval/summarize_followup.py | Fixed policies, full traces, true warm timing, per-repo/pooled decision checklists |
| V2 Drive workflow | notebooks/build_phase4_followup.py | Compact generated notebook + separate ZIP, pinned uv runtime; no Base64 |
| Ambiguous production-path runner | eval/run_ambiguous_eval.py; notebooks/build_ambiguous_eval.py | 15 alternative-proof cases; hybrid/rerank20/rerank50, real warm timing, separate Drive ZIP |
| Expanded comparison workflow | eval/build_comparison_dataset.py; eval/summarize_comparison.py; notebooks/build_comparison_eval.py | Frozen 90 questions, protocol validation, direct paired depths, cohort/type reports, compact uv/Drive notebook; no new dependencies |
| Fair expansion workflow | eval/build_expansion_dataset.py; eval/run_expansion_eval.py; eval/analyze_expansion_run.py; notebooks/build_expansion_eval.py | 30 fresh + 120 regression, store-backed option, paired gates, volume/displacement and full audit; no new dependencies |

Planned but not installed yet: langchain, langgraph, langchain-anthropic, chromadb,
PyGithub, fastapi, streamlit (see plan.md §6). httpx arrived as a langchain-core dependency.

---

### 2026-10-25: Audit the 90-question comparison results and review the labels

Imported `phase4_comparison_results.zip` (A100; hybrid, rerank20, rerank50 on 90 questions).
Pooled MRR@10 **0.748 / 0.896 / 0.893**, complete proof@10 **0.867 / 0.933 / 0.944**; rerank20 minus
hybrid **+0.149 [+0.081, +0.219]**, rerank50 minus rerank20 **-0.003 [-0.009, 0.000]**. Warm p50
rerank20 **890-970 ms**, rerank50 **2,110-2,350 ms**, hybrid about 66-70 ms. Rerank20 ranks first
on **73/90** and has the evidence in the top 10 on all 90; it beats hybrid on 32 and loses on 7.
Six multi-block questions have no complete proof. Pooled numbers recomputed from `per_question.json`.

Reviewed the 17 non-rank-1 cases by reading the accepted proofs and the top three chunks
(`eval/PHASE4_COMPARISON_LABEL_REVIEW.md`): **14 real ranking errors** (5 test over implementation,
2 wrong test, 7 related wrong source chunk) and **3 labels too strict** (express-001, flask-008,
flask-015). Frozen labels untouched; `datasets/phase4_comparison_v3/review_overlay.json` adds three
alternative proofs. My rescorer reproduced stored MRR/proof for all 270 rows first. With the overlay
rerank20 is MRR **0.915**, rank-1 **76/90**. This is a model review, not human; overlay chosen after
seeing results, so both columns are reported. No code, dependency or dataset changes; not committed.

### 2026-10-25: Rerank ordering and intent-aware instruction

Acting on the label review (`eval/PHASE4_COMPARISON_LABEL_REVIEW.md`). Measured offline on saved
pair scores, no GPU (`eval/analyze_rerank_tiebreak.py`). Log-odds alone orders exactly like P(yes)
(monotonic), so the change that matters is combining log-odds with the first-stage rank:
`logit(p) - lam * ln(rank)`. Candidate pool re-ordered, rerank over 20, lam 0 / 0.1 / 0.25 / 0.5 / 1 / 2 / 4:
v2 (129 q) MRR **0.8893 / 0.8854 / 0.8961 / 0.8919 / 0.8690 / 0.8258 / 0.7561**; v3 (90 q)
**0.8963 / 0.8963 / 0.9093 / 0.9130 / 0.9204 / 0.8846 / 0.8555**. Depth 50 on v3:
**0.8930 / 0.8930 / 0.9078 / 0.9115 / 0.9204 / 0.8846**. At lam 0.25: v2 5 better / 1 worse,
v3 3 / 1 (depth 20), 4 / 1 (depth 50). The best lam differs by set (v2 0.25, v3 1.0; 1.0 hurts v2),
so 0.25 is the conservative choice. The gain is small and not significant; v3 and v2 overlap in
repos, so they are not independent confirmations.

Shipped in `retrieval/rerank.py` (`combine_with_first_stage`) and `HybridRetriever._rerank`,
**code collections only** (commit and docs reranking were measured without it and are unchanged):
`rerank_first_stage_weight = 0.25`; 0 restores plain ordering. Intent-aware instructions
(`retrieval/intent.py`, `code_rerank_instruction_implementation/_test`) are behind
`rerank_intent_instructions = False`: they cannot be measured offline (the model must re-read the
pairs), and the detector's regexes were written from the failing phrasings, so on v3 it matches
12/12 implementation and 18/18 test questions by construction, and after widening the test patterns it also matches v2's 24/24
test questions (widened using those same questions, so not a held-out check); v2 implementation 6/99 (the rest carry no wording). **Needs a GPU run before turning on**, with a check that test-intent
queries do not regress. The frozen comparison protocol now pins weight 0 and intent off so a
re-run of the comparison notebook stays comparable. 370 tests pass, ruff and mypy clean, notebooks
regenerated. Not committed.

### 2026-10-25: Prepare the GPU policy run (tie-break and task instructions)

The user offered a GPU. `rerank_intent_instructions` became `rerank_instruction_mode`
(`default` | `intent` | `source_aware`; default unchanged) because the keyword detector turned out
weak on natural wording. New dataset `eval/datasets/phase4_intent_v1/`: the frozen 90 comparison
questions untouched plus **30 natural-wording paraphrases** of the 12 implementation and 18 test
questions (same gold, same groups). Checked offline before any GPU time: the detector fires on
**0 of 12** implementation paraphrases and **11 of 18** test paraphrases (template originals: 12/12
and 18/18), so `intent` is expected to be weak; `source_aware` (one instruction for every code
query that prefers source unless a test or example is asked for) needs no detector.

Pre-registered policies (`eval/retrieval/policies.py`), all reranking the fused top 20 through the
production retriever with one resident Qwen3-Reranker-4B: `plain` (baseline), `tiebreak` (0.25),
`intent`, `source_aware`, `source_aware_tiebreak`. Scores are shared when instructions match, so
tie-break differences are pure ordering. New: `eval/run_policy_eval.py`,
`notebooks/build_policy_eval.py` -> `phase4_policy_on_colab.ipynb` + `phase4_policy_source.zip`
(Drive ZIP, uv install, quality only, no timing), `tests/test_policy_eval.py`. 381 passed, 1 network
deselected; ruff clean; mypy clean. GPU results pending; nothing committed.

### 2026-10-25: Audit the policy run and turn the tie-break off

Imported `phase4_policy_results.zip` (A100; 5 policies, 120 cases, 41 source hashes/run match local
code) into `eval/retrieval_results/phase4_policy_run/`; analysis `eval/analyze_policy_run.py`.
MRR@10 plain **0.883**, tiebreak **0.890**, intent **0.908**, source_aware **0.890**,
source_aware_tiebreak **0.886**. Paired minus plain: tiebreak +0.006 [-0.013, +0.026]; intent
+0.025 [+0.008, +0.045]; source_aware +0.007 [-0.015, +0.033]. `intent` is carried by the 12 template
implementation questions (+0.167 [+0.042, +0.292], 4/0) and adds +0.000 on the 30 paraphrases; the
detector fires on 0/12 natural implementation and 11/18 natural test paraphrases. The tie-break was
-0.028 on test questions (0/2 worse). Conclusion: none is reliable. `rerank_first_stage_weight`
default **0.25 -> 0.0** (reverting my earlier offline-based default); `rerank_instruction_mode` stays
`default`. Tests updated; notebooks regenerated; results added to `eval/RESULTS.md`. Not committed.

### 2026-10-25: Chunk header fix prepared (`# defines:` line), GPU run pending

Phase 4's misses included chunks that merge several small definitions but are headed only by
their path (requests-026/027 sit inside a `requote_uri` chunk; Express getters inside `req.is`).
`metadata["symbols"]` already listed them; the header did not. Added `Settings.header_defines`
(**off by default**) which writes `# defines: a, b, c` (capped at `header_max_defines_chars=240`, then
`...`) for chunks with more than one definition; `split_header` and the duplicated header regexes in
`tests/test_ast_chunker.py` and `eval/chunker_check.py` know the new line; 3 new chunker tests. With it
on, Requests has 148 of 274 chunks with the line, Express 29 of 384, Flask 226 of 429. Express's
`defineGetter(req, 'secure', ...)` call-style getters are not tree-sitter definitions, so the case that
motivated this (express-026) is NOT covered. Real-repo invariants unchanged (chunk counts 274/ 384/429,
Split 0, text mismatch 0).

BM25-only on the 120 policy cases: MRR **0.478 -> 0.472**, complete proof 0.650 -> 0.658, 8 better /
11 worse; ambiguous 0.489 -> 0.451, test 0.307 -> 0.323. Neutral to slightly negative, so the flag stays
off until dense/fusion/rerank are measured. New runner options `--policies` (adds first-stage-only
`hybrid`) and `--header-defines`, and `notebooks/build_header_eval.py` ->
`phase4_header_on_colab.ipynb` + `phase4_header_source.zip` (header off vs on, hybrid and plain, per
repo). 387 passed, 1 network deselected; ruff and mypy clean. Not committed; GPU run pending.

### 2026-10-25: Audit the header run; keep `header_defines` off

Imported `phase4_header_results.zip` into `eval/retrieval_results/phase4_header_run/`
(`eval/analyze_header_run.py`; 41 source hashes/run match; header on/off flags recorded in run.json).
Header on minus off over 120 cases: hybrid MRR **-0.001 [-0.038, +0.032]**, complete proof -0.008
[-0.027, 0.000] (11 better / 9 worse); plain rerank MRR **+0.011 [-0.017, +0.036]**, proof -0.017
[-0.043, 0.000] (7 / 2). Requests proof 1.000 -> 0.952 (requests-comparison-004, -017). Express
identical (no definitions to list). requests-027 improved, requests-026 did not (its paraphrase did).
Conclusion: neutral, stays off. Results added to `eval/RESULTS.md`. Not committed.

### 2026-10-25: Evidence expansion tried offline; not adopted

New `src/repo_rag/retrieval/expand.py` (`ChunkIndex`, `expand`; modes none/adjacent/callee/both;
list length unchanged; top hits never reordered), 5 unit tests, and `eval/analyze_expansion.py`
which rescored the saved top-10 lists of the header-off run (hybrid and plain rerank) for all 120
cases without a GPU. Settings fixed beforehand (keep 5, seeds 3, per_seed 2). Plain rerank complete
proof 0.950 -> adjacent 0.950, **callee 0.967** (+0.017 [0.000, +0.043], 2 better / 0 worse), both 0.942.
Hybrid 0.883 -> adjacent 0.833 (-0.050 [-0.099, -0.008]), callee 0.833. Multi-block under rerank:
callee +0.111 [0.000, +0.278] (express-014, flask-014 fixed). Remaining four no-proof questions need
caller/callee edges the name heuristic cannot give. Not wired into the retriever; recorded in
`eval/RESULTS.md`. Notebooks regenerated (source ZIPs include the new module). Not committed.

### 2026-10-25: Audit the expansion GPU run; expansion rejected

Imported `phase4_expansion_results.zip` into `eval/retrieval_results/phase4_expansion_run/`.
Fresh 30 (incl. 12 Click): plain complete proof **0.667**, callee **0.567**, both 0.567; MRR 0.766 /
0.761 / 0.765. Callee minus plain proof **-0.100 [-0.242, +0.034]** (1 better, 4 worse). Regression 120
callee 0.967 vs 0.950 (multi-block 0.667 -> 0.778, reproduced). Pre-registered rule failed (gain and
lower bound), so `evidence_expansion` stays `none`. Numbers recomputed from `per_question.json`;
410 passed, ruff and mypy clean at audit time. The offline replay gain did not generalise.
Phase 4 is closed; Phase 5 next. Not committed.

### 2026-10-25: Prepare the stronger-judge run (Qwen3-Reranker 8B vs 4B)

The user asked for a stronger judge and noticed that no Colab cell names a model. True: models are
named only in `Settings` (`local_embedding_model`, `reranker_model`) and loaded implicitly inside
`LocalEmbeddings` (`store/embeddings.py`) and `QwenReranker._load` (`retrieval/rerank.py`), so the
Hugging Face download happened silently the first time the eval ran. The new notebook has an explicit
download cell (embedding 0.6B about 1.2 GB, reranker 4B about 8 GB, 8B about 16 GB) and says so in
its intro.

Frozen protocol `eval/datasets/phase4_judge_v1/protocol.json` (written before any 8B retrieval):
same hybrid first stage, rerank the fused top 20, top 10, no expansion/tie-break, one code instruction;
only the judge differs; 150 frozen expansion questions (30 fresh incl. 12 Click + 120 regression).
Adopt the 8B only if pooled MRR gain >= +0.03 with a positive 95% lower bound, complete proof not
lower (lower bound >= -0.02), and no stratum or cohort loses more than 0.02. New: `eval/run_judge_eval.py`
(both judges resident, one process per repo, rough warm timing and peak GPU memory),
`eval/analyze_judge_run.py` (applies the rule, writes `aggregate/decision.json`),
`notebooks/build_judge_eval.py` -> `phase4_judge_on_colab.ipynb` + `phase4_judge_source.zip`,
`tests/test_judge_eval.py` (5 tests including adopt/reject on synthetic results). Runner validated
on the Requests, Express and Flask checkouts (`--mode validate`); Click has no local checkout. GPU
result pending; an LLM-as-judge pilot is the next candidate if the 8B does not pass. Not committed.

### 2026-10-25: Audit the judge run; 8B passes the frozen rule

Imported `phase4_judge_results.zip` into `eval/retrieval_results/phase4_judge_run/` (43 source hashes
per repo match; protocol hash verified by `eval/analyze_judge_run.py`). Pooled 150 questions MRR@10
hybrid **0.726**, 4B **0.860**, 8B **0.926**; 8B minus 4B **+0.066 [+0.026, +0.109]**; complete proof
0.893 -> 0.913 (+0.020 [0.000, +0.044]); fresh multi-block 0.766 -> 0.850; implementation +0.142;
ambiguous -0.005. Rank 1: 113 -> 131 of 150 (28 better / 9 worse). Paraphrases 0.844 -> 0.950.
Warm p50 rerank about 1.3-1.4 s (8B) vs about 0.9-0.97 s (4B); peak GPU 29 GB with both resident.
All five pre-registered checks passed, so `adopt_8b` is true, but no default was changed: `reranker_model`
also drives commit reranking, measured with the 4B only. Recorded in `eval/RESULTS.md`. Not committed.

### 2026-10-25: Default reranker switched to Qwen3-Reranker-8B; 8B failure analysis

At the user's request `Settings.reranker_model` now defaults to `Qwen/Qwen3-Reranker-8B` (was 4B),
for code, docs and commit reranking alike. Caveat recorded in the setting's docstring: bug -> fix
reranking was measured only with the 4B (nDCG@10 0.494 -> 0.687), so Phase 5 must re-measure it
with the 8B. Frozen evaluations still pin the 4B through their protocol files, so they stay
reproducible. `tests/test_config.py` updated; notebooks regenerated.

Failure analysis of the 8B judge run (150 questions): **19 not ranked first, 13 with no complete
proof**. Not-first kinds: 10 a test outranks the implementation (7 on Express/JS; 5 of the 8 fresh
ones), 6 another chunk of the right file, 3 a wrong chunk of the same kind. By stratum: 9 multi-block,
5 ambiguous, 3 implementation, 1 test, 1 example. All 13 no-proof questions are multi-block (8 of
30 fresh, 5 of 18 regression); single-block strata have complete proof 1.000. Compared with the
4B, the remaining misses concentrate in multi-block and in test-over-implementation on Express.
No absolute retrieval target had ever been written in plan.md; the verdict and proposed targets are
in the chat summary. Not committed.

### 2026-10-25: Claude review of the Phase 5 build (BM25 preflight)

Reviewed Codex's Phase 5 (commit indexer, issue linker, metadata.db, history index, bug_match, CLI, frozen
`phase5_bugfix_v1`, GPU harness). Verified: 444 tests pass, ruff and mypy clean; the BM25 test table
reproduces exactly from the stored rows and from an independent BM25 rebuild (0 ranking differences over 263
test questions); no gold SHA in any query. Findings in `docs/PHASE5_REVIEW_FINDINGS.md`. Top issues: the
scrub drops the whole closing line, deleting the subject of 41% (Requests) and 59% (Flask) of gold commits
and the entire message of 29% / 56%; keeping the rest of the line raised BM25 MRR 0.350 -> 0.369 (Requests)
and 0.666 -> 0.699 (Flask); the benchmark measures issue -> its own fix, not similar past fixes; the 8B rule
can fail on one FastAPI question; metadata.db stores raw messages with emails (186/178/710 messages)
despite its docstring. GPU matrix not run, so nothing neural was reviewed. Not committed.

### 2026-10-26: Claude review of the Phase 5 v2 fixes

Verified Codex's v2: 452 tests pass, ruff and mypy clean; v1 frozen hashes intact; v2 corpora rebuild; same 329
questions as v1; 0 empty gold messages; 0 email-like strings in searchable docs; v2 BM25 test numbers reproduce
exactly (Requests MRR 0.3683, Flask 0.6995, FastAPI 0.5595). New finding: File Hit@5 is inflated by shared
doc/changelog files (Flask 0.941 -> 0.625, Requests 0.814 -> 0.726 when they are excluded), so the frozen metric
should use source paths. Details in `docs/PHASE5_REVIEW_FINDINGS.md` (Round 2). GPU matrix still not run.
Not committed.

### 2026-10-26: Claude audit of the Phase 5 v2 GPU run (and a correction)

Audited `phase5_bugfix_v2_results.zip` from the raw rows: held-out test (n=263) MRR@10 BM25 0.550, dense 0.685, hybrid
0.677, rerank4b 0.823, rerank8b 0.847; Hit@5 0.673 / 0.795 / 0.795 / 0.905 / 0.905. 8B minus 4B +0.0245 MRR
[+0.0030, +0.0465], 27 better / 14 worse: the frozen rule passes narrowly, and the gain is rank-1 only (Hit@5 equal,
Hit@10 0.913 vs 0.920) at about +45% latency. BM25 adds nothing over dense at the first stage for commits. Candidate-pool
ceiling (gold in the fused top 50): Requests 0.912, Flask 0.956, FastAPI 0.929, so most remaining misses are first-stage
recall. Code-fix subset (208 q) hybrid 0.648 / 4B 0.817 / 8B 0.845 vs docs-only (55 q) 0.786 / 0.845 / 0.858. Provenance
matches; metadata.db has no real emails. **Correction:** my earlier claim that File Hit@5 is inflated by changelog files
(Flask 0.941 -> 0.625) was a denominator error; the correct source-path value is 0.914, so no re-freeze was needed. Details in
`docs/PHASE5_REVIEW_FINDINGS.md`. Not committed.

### 2026-10-26: Brief for Codex: commit reranking depth and BM25 weight

User approved testing a deeper candidate pool and re-checking the BM25 weight for commits. Wrote
`docs/CODEX_HANDOFF_COMMIT_DEPTH.md` (Codex builds, Claude reviews): one 200-deep fused pool per question with
pair scores shared across depth 50/100/200 prefixes (8B primary, 4B secondary); BM25 weight grid {0, 0.1, 0.3, 0.5, 1.0}
chosen on dev only (keep 0.3 unless +0.01 MRR); timing on a 60-question subset; pre-registered adoption rule
(Hit@5 +0.02 with positive lower bound, MRR not lower than -0.005, source-path subset non-negative, no large-repo loss
over 0.02). Motivation: the v2 pool ceiling is 0.912 (Requests), 0.956 (Flask), 0.929 (FastAPI), close to the 8B Hit@10.
No code was written by me; nothing is committed.

### 2026-10-27: Claude audit of the commit depth / BM25 weight checkpoint

Partial checkpoint (no aggregate; timing Requests only). Verified hashes, revisions and source. Rebuilt depth rankings
from the 200 stored probabilities. 8B held-out test (263): depth 50 Hit@5 0.920 MRR 0.8587; depth 100 0.928 / 0.8587
(Hit@5 +0.0076 [0.0000, +0.0192]); depth 200 0.932 / 0.8624 (+0.0114 [0.0000, +0.0265]). The pre-registered rule needs
+0.02 with a positive lower bound, so it FAILS: keep depth 50. Cost on Requests p50 3.9 / 7.6 / 15.1 s. Unplanned: a pool
built from retrieval_k=200 at the same rerank depth 50 beat the v2 control by Hit@5 +0.0152 [+0.0037, +0.0309] (hypothesis,
needs a confirmatory run; GPU rerun noise is about one question). BM25 weight: 0.1 selected on dev (66 q, noisy) but with
the reranker 0.1 vs 0.3 is Hit@5 +0.0000, MRR +0.0022 [-0.0005, +0.0068], so keep 0.3. Details: `docs/PHASE5_REVIEW_FINDINGS.md`
Round 4. No defaults changed. Not committed.

### 2026-10-27: Brief for Codex: confirm retrieval_k=200 for commits; finish depth/weight output

User approved. Wrote `docs/CODEX_HANDOFF_COMMIT_CONFIRM.md` (Codex builds, Claude reviews). Part A: run the missing Flask and FastAPI timing under the
frozen protocol, write the mechanical aggregate/decision output, the labelled post-hoc nested-50 row, RESULTS/log/handoff. Part B: pre-registered
confirmation of `retrieval_k=200` with rerank depth 50 (arm A 50 control, arm B 200, arm C 100 descriptive; 8B; shared score cache; arm A run twice
for noise) on a fresh holdout from repos not used so far (suggested pallets/click and encode/httpx, direct links, v2 hygiene and leakage checks, frozen
before retrieval), with the v2 dev split as small supporting evidence. Adoption needs Hit@5 >= +0.01, MRR >= 0, both lower bounds >= -0.01, no large-repo
loss over 0.02, and a noise check; interval wider than 0.06 means inconclusive. If adopted, use a per-kind commit setting rather than the global
`retrieval_k`. Needs GITHUB_TOKEN only to build the dataset. No code written by me; nothing committed.

### 2026-10-27 - Built the retrieval_k=200 confirmation (Claude built it; Codex had no usage left)

- **What:** fresh holdout `eval/datasets/phase5_commit_confirm_v1/` (pallets/click, encode/httpx, pallets/werkzeug, pallets/jinja; pins and protocol frozen before sampling, hash `9c73301e...`). 342 questions after 2 leakage exclusions (click 128, httpx 15, werkzeug 114, jinja 85); not tentative (>=100). Supporting: v2 dev (66).
- **Code (outside `eval/retrieval/` so the depth notebook's source ZIP and Drive checkpoint hash `869432f8c494` stay valid):** `eval/confirm_lib.py`, `prepare_commit_confirm.py`, `run_commit_confirm_eval.py`, `analyze_commit_confirm.py`, `notebooks/build_commit_confirm.py`, `tests/test_commit_confirm.py`. No `src/repo_rag` change.
- **Design:** arms A (retrieval_k 50), B (200), C (100) through the production search path with one score cache per question; arm A scored again without cache (GPU noise); first-stage-only rows; production timing on 40 questions. Rule and outcomes are in the protocol; nothing changes automatically.
- **Deviations, all before any retrieval:** (1) email check tightened (it flagged `actions/cache@v3.0.9`); (2) werkzeug aborted prepare because a linked commit failed the message fix heuristic, so such links are ignored (those commits are not in the pool; click/httpx unaffected); (3) two questions whose gold diff still contains the issue's own number (click #1047, jinja #748, a Sphinx anchor the v2 scrub misses) are excluded in the experiment freeze and listed in `leakage_report.json`. The scrub gap in `src` is NOT fixed (it would break v2 rebuilds); it is an open item.
- **Caveats:** httpx yields only 15 questions (few "Fixes #N" commits); Werkzeug/Jinja share maintainers and style with Flask; labels link-derived, no independent review.
- **Checks:** full suite passed, ruff clean, no new mypy errors in new files (11 older errors in other eval scripts remain).
- **Part A (finish the depth run) needs no new build:** rerun the existing `phase5_commit_depth_on_colab.ipynb` with `phase5_commit_depth_source.zip` and `phase5_commit_depth_checkpoint_869432f8c494.zip` in Drive; it resumes timing and writes the aggregate.
- **Next:** run `phase5_commit_confirm_on_colab.ipynb` on an A100, bring back `phase5_commit_confirm_results.zip` for audit. Nothing committed.

### 2026-10-27 - Audited the retrieval_k=200 confirmation (quality stage complete; timing partial)

- **Input:** `phase5_commit_confirm_checkpoint_c4ee43d91184.zip` (the user stopped the Colab run during the timing stage to save compute). All 4 fresh repos and the 3 v2 dev repos have complete `confirm` runs; only Click has a timing run. No vLLM or other engine was used: it would change the measured production path and needs approval as a new dependency.
- **Analyzer bugs found and fixed by me (mine, from the build):** (1) `analyze` overwrote each row's evidence `group` with "fresh"/"v2dev", so the bootstrap saw ONE group and every interval had zero width (a false pass). Now stored as `run_group`; real groups are 332 for 342 questions. (2) It required a timing run for every fresh repo; it now skips a missing one and records `timing_not_run_for`. New test asserts groups > 1 and the partial-timing path. The offline fake-model test had not checked interval width, which is why (1) slipped through. Notebook and source ZIP regenerated (the user's already-run notebook is unaffected: runs hash only the run scripts).
- **Result (fresh, 342 questions, 8B, rerank depth 50), B (`retrieval_k=200`) minus A (50):** Hit@5 +0.0117 [+0.0000, +0.0264] (about 4 questions), MRR@10 +0.0086 [-0.0008, +0.0204]. Hit@5 A 0.8772, B 0.8889, C 0.8889; pool recall 0.9006 / 0.9094 / 0.9123. A rerun matches A exactly on Hit@5 (noise check passes) but 4 of 128 Click questions change top-10 order and the largest probability difference is 0.057.
- **Per repo Hit@5 B-A:** Click +0.0234 [0, +0.0538] (n=128), Werkzeug +0.0175 (n=114), Jinja -0.0118 [-0.0353, 0] (n=85), httpx 0 (n=15). v2 dev (66): exactly 0 for B and C.
- **Rule outcome (frozen, not changed):** `eligible_pending_review`; all checks and both flags pass (CI width 0.026 <= 0.06). Nothing changes: `default_changed=false`.
- **Honest reading:** the gain is real in sign but tiny (about 4 of 342 questions, lower bound exactly 0, MRR interval touches zero), concentrated in Click/Werkzeug, absent on the dev set, and Jinja lost one. It clears the bar the rule set; it is not strong evidence. Cost: Click timing only (30 calls per arm, A100) p50 A 4.28 s, B 4.66 s (x1.09), C 4.46 s (x1.04): the "no extra reranker cost" claim holds for rerank scoring, but B is about 9% slower end to end on this one repo. Other repos untimed. In timing, B's top-10 matched the quality run in only 50% of calls (A 100%), consistent with score differences from batch composition near ties, which also bounds how much weight 4 questions can carry.
- **Not done / open:** if adopted it is a separate per-kind commits setting (keep code and docs at 50) with its own review; timing for 3 repos; independent label review; code/docs at 200 never measured. Results: `eval/retrieval_results/phase5_commit_confirm_run/` (summary.md, decision, comparisons, latency, diagnostics, experiment freeze; `per_question.json` 7 MB left out, rebuild from the checkpoint ZIP). Nothing committed.

### 2026-10-28 - Phase 5b slice 1: git tools and exact-reference detector

- **Added:** `src/repo_rag/history/git_tools.py` (`blame`, `file_history` incl. `git log -L` function history with PR numbers parsed from a trailing `(#N)`, `diff_between`, `releases_containing` earliest-first, `GitToolError`) and `src/repo_rag/retrieval/exact_refs.py` (`find_exact_refs`: `#N`/PR/issue numbers, SHAs, version tags, `@handles`, paths; hex-only words and plain numbers are not SHAs). Config: `git_tool_max_results=100`.
- **Safety:** fixed git commands through GitPython (no shell); refs and paths validated (no leading `-`, no `..`, no absolute paths); only author names returned, never emails; results capped and `*_truncated` flags reported; "who pushed" still not claimed.
- **Bugs hit while building:** `str.splitlines()` also splits on `\x1e`/`\x1f`, which silently emptied every log result (fixed with `split("\n")`, git-side `%x1e/%x1f` separators); GitPython needs `YYYY-MM-DD HH:MM:SS +0000` dates in test commits; `blame -L` clamps an end line past EOF rather than erroring.
- **Tests:** `tests/test_git_tools.py`, `tests/test_exact_refs.py` (36 tests, real temp git repos with tags and two authors). Full suite after regenerating all 12 notebooks: 495 passed before regeneration plus 103 notebook/freshness tests re-passed (13 stale-notebook failures were expected: every source ZIP bundles `src/repo_rag`). Ruff clean; mypy clean (71 files).
- **Side effect:** all local notebooks and source ZIPs were regenerated, so local `phase5_commit_depth_source.zip` now hashes `97fef224d1ac`, not the one behind Drive checkpoint `869432f8c494`. The Drive copies of the old notebook/ZIP and that checkpoint still work together; use those (not the regenerated ones) if the depth run is ever resumed.
- **Not done in 5b:** GitHub PR/review/issue sync, the `metadata.db` PR/review/tag/identity tables, `query_commits` filters by author/path/symbol/date, `top_contributors`, `change_frequency`, `get_pr`/`search_prs`, the auto-generated fact set. Phase 5b stays unticked. Nothing committed.

### 2026-10-28 - Phase 5b slice 2: GitHub PR / review / issue sync and metadata.db tables

- **Authorization:** the user approved read-only GraphQL calls with `GITHUB_TOKEN` from `.env` (never printed). Live calls made: `sync-github encode/httpx --max-prs 30 --max-issues 30` twice, into a scratch data dir (not the real `data/`).
- **Added:** `history/github_sync.py` (`GitHubSync`, `parse_pr`, `parse_issue`; newest-first by `updatedAt`, first sync bounded by `max_prs`/`max_issues` = 500, later syncs fetch only items updated since the saved watermark and merge by number; raw nodes in `data/github_cache/sync/`; a pass is saved only if it finished; `complete` says whether the oldest item was reached), `history/github_index.py` (sync then load), `GitHubClient.fetch` (uncached, error-checked), new `metadata.db` tables `pull_requests`, `pr_reviews`, `pr_closes`, `issues`, `github_sync` plus typed methods `get_pr`, `find_prs`, `pr_for_commit` (refuses an ambiguous prefix), `prs_closing_issue`, `get_issue_meta`, `find_issues`, `github_coverage`; CLI `sync-github` and `show-pr`. Stdlib `urllib` is kept (CLAUDE.md mentions `httpx` for 5b, but adding a dependency needs the user's approval).
- **Rules honoured:** no SQL from the LLM (fixed parameterized queries, label match is whole-label, tested with an injection string); emails redacted from titles and bodies; only logins shown; `show-pr` prints "git does not record who pushed"; DB opened read-only for queries; a database written before these tables reports no coverage instead of crashing.
- **Live check:** 30 newest PRs (19 open, 10 closed, 1 merged) and 30 issues for encode/httpx synced; reviews 25, closing links 5. PR #1's merge commit `7626fe7c` exists in the local clone, its commit time (13:21:03 +0100) matches `mergedAt` 12:21:04Z, and `git tag --contains` gives 0.0.3 then 0.1.0 (the git_tools slice). The second sync fetched 1 item per kind (the boundary item), so incremental works against the real API.
- **Semantics to remember:** `issues.closed_by_prs` lists PRs that close the issue, or would close it if it is still open (open PRs included). `complete=false` means the index holds only the newest N PRs/issues; tools must say so when a number is not found. If an item changes while a pass is paging, ordering can shift by one; the next sync re-reads from the watermark. Reviews are the first 50 per PR; `reviews_total` flags a cut list. Closing issue references in other repositories are not keyed.
- **Tests (full suite 524 passed, 1 deselected; ruff and mypy clean; notebooks regenerated):** `tests/test_github_sync.py` (16 tests: fake paginated GraphQL, cap and completeness, incremental merge, failed pass leaves state untouched, GraphQL errors, deleted users and null merge data, DB round trip, exact filters and injection string, per-repo replace, read-only refusal, pre-existing DB without the tables, CLI).
- **Not done in 5b:** identity merging (`.mailmap`/login), `tags` table in `metadata.db`, `query_commits` author/path/date filters, `top_contributors`, `change_frequency`, the `pull_requests`/`issues` vector collections and `search_prs`/`search_issues`, the auto-generated fact set. Phase 5b stays unticked. Nothing committed.

### 2026-10-28 - Phase 5b slice 3: identities, tags and the commit-level exact tools (offline)

- **Added:** `history/git_facts.py` (`read_tags`, `read_identity_map`, `read_login_links`), `history/facts_index.py` (`index_git_facts`), `metadata.db` tables `tags`, `identities`, `author_logins`, and typed methods `find_commits`, `count_commits`, `top_contributors`, `change_frequency`, `history_window`, `list_tags`, `get_tag`, `tags_for_commit`; CLI `index-facts`, `contributors`, `hot-files`. No new dependency, no network.
- **Identity merging is evidence-only, names-only:** (1) `.mailmap` through git's own `%aN` against raw `%an` over the same bounded history (no emails are read into results); a raw name that the mailmap sends to more than one canonical name is left unmerged and counted as ambiguous; (2) GitHub login links from squash/rebase merge commits (exactly one parent): the commit author is the PR author, so (canonical name, PR author login) is counted with evidence and conflict counts; real merge commits are skipped because their author is the merger. Names are never merged by similarity. `author` filters accept a canonical name or a linked login, case-insensitively; a raw alias is not a name.
- **Window rule:** every ranking/count returns the `HistoryWindow` it covers (indexed non-merge commits and their date range) because the index is the newest `max_commits` and merges are excluded; CLI prints "older history is not counted". Dates are compared in UTC (`datetime()`), `since` inclusive, `until` exclusive, strict `YYYY-MM-DD[THH:MM[:SS]]` validation. Path filters match a file or a directory (`path == p` or prefix `p/`), never a bare string prefix. Symbol filter matches a qualified name or its last component. File counts use the path as named at each commit, so a rename splits the count (documented; use `git_tools.file_history` to follow renames).
- **Compatibility:** older databases without the new tables still answer (names pass through unmerged, logins blank). `find_commits` is the plan's `query_commits` but returns light summaries; the old `query_commits` (full records) is unchanged for Phase 5.
- **Live check (encode/httpx, 300 newest non-merge commits, scratch data dir):** top authors 84/73/20/9/7 and top files 96/53/24 commits match `git log --no-merges -300` exactly; 88 tags loaded; 0 mailmap merges and 0 login links (this repo has no mailmap and the only synced merged PR is a true merge commit, which is correctly skipped).
- **Pitfalls hit:** GitPython misreads `YYYY-MM-DD HH:MM:SS -0500` author dates (it stored 18:30-05:00 for 23:30-05:00): tests use `<unix> -0500`. `git for-each-ref` accepts `%1f`, `git log` needs `%x1f`.
- **Tests (full suite 542 passed, 1 deselected; ruff and mypy clean; notebooks regenerated):** `tests/test_git_facts.py` (18 tests on a real repo with `.mailmap`, lightweight and annotated tags, an ambiguous name, a timezone edge case and a real merge commit; hostile strings; legacy database).
- **Not done in 5b:** PR/issue vector collections with `search_prs`/`search_issues`, the auto-generated exact-answer fact set (>=95% target), agent wrappers (Phase 7). Phase 5b stays unticked until the fact set passes. Nothing committed.

### 2026-10-28 - Phase 5b slice 4: PR/issue search and the exact-answer fact set (Phase 5b ticked)

- **Added:** `history/pr_index.py` (`pr_document`, `issue_document`, `index_pr_issues`): `pull_requests` and `issues` Chroma collections (named after the embedding space like the others) plus per-kind BM25, built from the synced `metadata.db` rows. Document text = `PR #N: title`, labels, body capped at `pr_body_chars` (3000); emails redacted; state, author, labels, dates and `number` ride along as metadata. `HybridRetriever.search(kind="pull_requests"|"issues", filters={"state": ...})` and `repo-rag search --kind pull_requests --state MERGED`; `repo-rag index-prs <owner/name>`. `export-index`/`import-index` pick the new kinds up through `SEARCH_COLLECTIONS`. No new dependency.
- **Fact set (`eval/build_facts.py`, `eval/run_facts.py`, `eval/datasets/httpx_facts.jsonl`):** 60 auto-generated facts (seed 42) over 12 types: PR merged-by/author/title/state, issue author, issue to closing PR, first release containing a PR, blame of a line, commits touching a path in a month, top committer for a year, most-changed file for a year, tag to commit. **Expected answers come from sources the tools do not use:** the raw GraphQL nodes in the sync cache and plain `git` subprocesses (`merge-base --is-ancestor` over every tag for release, `git log -L` for blame, `git log --full-history --no-merges` filtered in Python for counts and rankings); the generator never imports `MetadataDB` or `git_tools`. Ties (equal top counts, tags created in the same second) are skipped, not guessed. At most 2 zero-count month facts.
- **Result (live encode/httpx, 1,482 non-merge commits and 88 tags in `metadata.db`, 500 PRs and 500 issues synced, scratch data dir):** **60/60 facts pass through the tools alone** (target 95%): every type 100%. Results: `eval/facts_results/httpx/`.
- **Limits of that number:** the GitHub facts (merged-by, author, title, state, closing PRs) read the same synced GitHub data twice (raw node vs the `metadata.db` row), so they check parsing, storage and typed lookups, not GitHub itself. The git facts are a real second reading (different git commands). One repository, 60 facts, tools called with the intended arguments: choosing the tool and arguments from a question is Phase 7 (the plan reruns this set through the agent, target 90%).
- **Issue -> fixing-PR search (`eval/run_pr_search_eval.py`):** pairs from GitHub closing links, query = issue title plus the first 1,000 body characters (never the issue number), corpus = all 500 PRs. **BM25 only (no model): 45 pairs, Hit@1 0.356 [0.222, 0.489], Hit@5 0.600 [0.467, 0.733], Hit@10 0.733 [0.600, 0.867], MRR@10 0.464 [0.339, 0.590]** (bootstrap 10,000, seed 42; random Hit@5 is about 0.010). **Dense and hybrid are not measured:** embedding PR docs on the laptop CPU runs at 0.10 docs/s (about 3 hours for 1,000 docs), so `index-prs` with the real model needs a GPU. Run `eval/run_pr_search_eval.py --strategies bm25,hybrid` after that index exists.
- **Tests (full suite 552 passed, 1 deselected; ruff and mypy clean; all 12 notebooks regenerated):** `tests/test_pr_index.py` (6: metadata and redaction, hybrid search with a state filter, idempotent re-index and manifest, error without a sync, CLI output, eval helpers) and `tests/test_facts.py` (4: every generated fact answered correctly on a real two-author repo with annotated tags, known answers, wrong/missing/unknown-tool handling, determinism).
- **Pitfalls hit:** a capped first sync never grows on later syncs (the cap applies to the first pass); deleting the scratch cache and re-syncing was needed to get 500. One `merge-base --is-ancestor` subprocess per (PR, tag) took over 10 minutes for all PRs, so release facts look at a random 30 merged PRs.
- **Not done:** dense/hybrid PR search measurement (GPU), agent tool wrappers and the end-to-end fact run (Phase 7). Phase 5b ticked in plan.md. Nothing committed.

### 2026-10-28 - Colab notebook for the GPU PR-search measurement (prepared, not run)

- **Added:** `notebooks/build_pr_search_eval.py` -> `notebooks/pr_search_on_colab.ipynb` (self-contained, like `index_on_colab`): embeds the `repo_rag` source with uv.lock pins, a **redacted** snapshot of the encode/httpx sync cache (`eval/datasets/httpx_pr_snapshot/`, 500 PRs + 500 issues, email addresses replaced by `<email omitted>`; the raw cache had three real ones) and `eval/run_pr_search_eval.py`. It loads the snapshot into `metadata.db` without any network or token, runs `repo-rag index-prs`, a smoke search, then `run_pr_search_eval.py --strategies bm25,hybrid`, and downloads `pr_search_results.zip` plus `pr_index.zip` (importable locally).
- **Offline checks (`tests/test_pr_search_notebook.py`, 5 tests):** notebook is fresh, payload equals snapshot + script, snapshot has no emails, the notebook's load + index + BM25-eval steps run on the payload with fake embeddings (500 PRs, 500 issues, 45 pairs), and PR/issue vectors survive export/import.
- **Not run on Colab.** Expected: one run on a T4/A100; record Hit@1/5/10 and MRR for bm25 vs hybrid next to the BM25 baseline (Hit@5 0.600, 45 pairs).

### 2026-10-29 - GPU PR-search measurement (Colab run)

Ran `notebooks/pr_search_on_colab.ipynb` on a Colab GPU (float32). Results in
`eval/facts_results/httpx_pr_search_gpu/`. 45 issue -> closing-PR pairs over 500 PRs:

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.356 [0.222, 0.489] | 0.600 [0.467, 0.733] | 0.733 [0.600, 0.867] | 0.464 [0.339, 0.590] |
| hybrid | 0.467 [0.311, 0.622] | 0.800 [0.667, 0.911] | 0.822 [0.711, 0.933] | 0.582 [0.460, 0.701] |

Hybrid is higher on every metric; BM25 reproduces the laptop baseline exactly. The 95% intervals
overlap (n=45), so this is a likely gain, not a proven one. `pr_index.zip` (500 PR + 500 issue
vectors) can be merged with `repo-rag import-index`.

### 2026-10-29 - Improving PR search: misses and fused title query

Imported `pr_index.zip` into a scratch data dir and ran the 45 queries on CPU. Of the 9 hybrid
Hit@5 misses, six had the closing PR at rank 9 to 22; several top hits were other PRs on the same
topic that GitHub did not list as the closer (the metric undercounts: counting any PR that
mentions the issue number gives hybrid Hit@5 0.844). Long issue bodies dilute the query. New
strategy `hybrid_fused` in `eval/run_pr_search_eval.py` = full query + title-only query, fused by
RRF (k=60); no re-embedding. Same 45 pairs, `eval/facts_results/httpx_pr_search_fused/`:

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.356 | 0.600 | 0.733 | 0.464 |
| hybrid | 0.467 | 0.800 | 0.822 | 0.582 |
| hybrid_fused | 0.467 | 0.867 [0.756, 0.956] | 0.956 [0.889, 1.000] | 0.638 |

**Caveat:** the variant was chosen after seeing these 45 pairs (title-only alone scored Hit@10
0.978; title + 300 body chars was worse), so it needs a second repo before it is claimed.
Next likely gain: reranker on the top 50 (GPU); the misses are mostly near-misses.
The Colab notebook now runs `bm25,hybrid,hybrid_fused`.

### 2026-10-30 - Second-repo check of PR search (pallets/flask, Colab GPU)

`notebooks/flask_pr_search_on_colab.ipynb` (snapshot `eval/datasets/flask_pr_snapshot/`, newest
500 PRs + 500 issues). Results in `eval/facts_results/flask_pr_search_gpu/`. 58 pairs:

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.621 | 0.793 [0.690, 0.897] | 0.845 | 0.689 |
| hybrid | 0.741 | 0.914 [0.845, 0.983] | 0.931 | 0.807 |
| hybrid_fused | 0.828 | 0.931 [0.862, 0.983] | 0.931 | 0.868 [0.785, 0.943] |

Same ordering as httpx (bm25 < hybrid <= hybrid_fused), so hybrid over BM25 replicates on a second
repo. Fused vs hybrid: Hit@1 and MRR rise (0.741 -> 0.828, 0.807 -> 0.868) but Hit@5/10 are
about equal, so the fused title query's extra Hit@10 gain seen on httpx (0.822 -> 0.956) did
NOT repeat; flask hybrid already had Hit@10 0.931. Fused is never worse on either repo. Intervals
still overlap with n=58.

### 2026-10-30 - Title-query fusion wired into `search` for PRs and issues

`HybridRetriever.search` now also searches the first line (the title) of a multi-line PR/issue
query and fuses the two rankings (RRF, k=60); one-line queries, code/docs/commits and
`rerank=True` are untouched. Setting `title_query_fusion` (default True) turns it off. The
eval's `hybrid_fused` now runs through this code (`hybrid` sets it off). Re-measured on httpx
through the real path: hybrid Hit@5 0.800, hybrid_fused Hit@5 0.867 / Hit@10 0.956 / MRR@10
0.638, identical to the earlier standalone numbers. New test in `tests/test_pr_index.py`; full
suite, ruff and mypy clean; all notebooks regenerated.

### 2026-10-30 - PR search reranker experiment prepared (not run)

`search(kind="pull_requests", rerank=True)` now reranks the fused top `rerank_candidate_depth`
(50) with its own instruction `pr_rerank_instruction` ("Given a GitHub issue, judge whether this
pull request resolves the reported problem"); before, PRs would have used the commit-fix
instruction. Opt-in, GPU only (`reranker_mode`), unmeasured, so nothing enables it by default.
`eval/run_pr_search_eval.py` has a new strategy `hybrid_fused_rerank`; both PR-search notebooks
(httpx, flask) now run `bm25,hybrid,hybrid_fused,hybrid_fused_rerank` and need an A100/L4
(Qwen3-Reranker-8B does not fit a 16 GB T4). New test with a fake reranker; full suite, ruff and
mypy clean; all notebooks regenerated. **Pending:** run both notebooks and record whether the
reranker lifts Hit@5/Hit@1 over `hybrid_fused` (0.867/0.467 httpx, 0.931/0.828 flask).

### 2026-10-31 - PR search reranker on httpx (Colab GPU)

`eval/facts_results/httpx_pr_search_rerank/`, 45 pairs, Qwen3-Reranker-8B over the fused top 50:

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| hybrid_fused | 0.467 | 0.867 | 0.956 | 0.638 |
| hybrid_fused_rerank | 0.644 [0.511, 0.778] | 0.933 [0.844, 1.000] | 0.956 | 0.767 [0.669, 0.859] |

The first three rows reproduce earlier numbers. The rerank gain is mostly at the top of the list
(Hit@1 +0.18, MRR +0.13); Hit@10 is unchanged (the reranker only reorders the 50 candidates).
Intervals overlap with n=45, and flask is not yet measured, so reranking stays opt-in.

### 2026-10-31 - PR search reranker on pallets/flask (Colab GPU)

`eval/facts_results/flask_pr_search_rerank/`, 58 pairs, same setup as httpx:

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| hybrid_fused | 0.828 | 0.931 | 0.931 | 0.868 |
| hybrid_fused_rerank | 0.879 [0.793, 0.948] | 0.966 [0.914, 1.000] | 0.966 | 0.915 [0.849, 0.971] |

Reranking is higher on every metric on both repos (httpx: Hit@1 0.467 -> 0.644, MRR 0.638 ->
0.767; flask: Hit@1 0.828 -> 0.879, MRR 0.868 -> 0.915). Gains are larger where the baseline is
weaker. Intervals still overlap at n=45 and n=58, but the direction is the same on both repos.

### 2026-10-31 - Phase 6 slice 1: symbol table and `find_callers` (built, hand-checked)

New: `chunking/symbols.py` (definitions via `AstChunker.symbol_records`, call sites and Python/JS
imports via tree-sitter, Python/JS/TS/Go/Java), `store/symbol_table.py` (tables `code_symbols`,
`code_calls`, `code_imports` in `metadata.db`; typed read-only `get_symbol`, `find_callers`),
`ingest/symbol_indexer.py` and CLI `index-symbols --checkout`, `symbol`, `callers`. CPU only, no
model: psf/requests gives 807 definitions and 2,660 call sites in 3 s.

Calls are resolved by name and every row carries how it matched (`same class`, `imported`,
`qualified`, `same file`, `receiver name`, `name only`) plus how many definitions share the name;
non-test code ranks before tests. A bare `f()` never matches a Python/JS/Go method.
Hand-check against `git grep` on requests (`eval/symbol_check_results.md`, `eval/check_callers.py`):
6 of 7 symbols match the textual call sites exactly; the 7th differs only by a docstring and a
doctest. **Limit:** for method names shared by several classes (`send`: 43 rows, 1 better than
"name only") most rows are name-only and may call another class's method; no type inference.
Tests: `tests/test_symbols.py` (11). All notebooks regenerated.
**Not done in Phase 6:** repo/dir/file summaries (needs the answer model), `find_callees`,
Java/Go import resolution, agent wiring.

### 2026-10-31 - Phase 6 slice 2: repo map (file / directory / repository summaries)

New `summaries/repo_map.py` + `summaries/index.py`, collection `summaries`, CLI
`repo-rag index-summaries --checkout` and `search --kind summaries`. **No model:** a file summary
is its module docstring (title lines and underlines skipped), classes with their methods and
one-line docstrings, functions, and project/library imports; a directory summary lists its files
with one line each; the repository summary has the cleaned README (badges, links, code fences
removed) and the layout. Capped by `summary_max_chars` (3,000). A `Summarizer` protocol lets a
local model add a "Summary:" paragraph later; its output is cached by sha256(model, kind, path,
facts), so re-indexing never repeats a call. No model is chosen or wired yet (decision for
Phase 7). psf/requests gives 43 summaries (36 file, 6 directory, 1 repository). Plain
`import pkg.sub` now also records an import binding (also used by `find_callers`).

**Measured (BM25 only, 40 requests questions, file-level gold,
`eval/facts_results/requests_summary_bm25/`):**

| Ranker | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| code chunks, collapsed to files | 0.550 | 0.950 | 1.000 | 0.707 |
| file summaries | 0.275 | 0.675 | 0.875 | 0.452 |
| both, fused | 0.425 | 0.850 | 0.975 | 0.625 |

**Summaries do not help find the file for a code question; mixing them into code search makes it
worse.** That is expected (they describe files, the questions ask about behaviour) and is why
`summaries` is its own collection, not a default part of code search. Their purpose is
architecture questions, for which there is no labelled set yet, so their value is **unmeasured**.
Dense search over summaries is also unmeasured (needs a GPU to embed). Tests:
`tests/test_summaries.py` (6). Notebooks regenerated.

### 2026-10-31 - Repo-map summaries with hybrid search (same 40 requests questions)

`eval/run_summary_eval.py --data-dir` adds hybrid (dense + BM25) rankers. The requests code index
was exported and imported into a scratch dir (nothing re-embedded) and the 43 summaries were
embedded on the laptop CPU. `eval/facts_results/requests_summary_hybrid/`, file-level gold:

| Ranker | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| code chunks, BM25 | 0.550 | 0.950 | 1.000 | 0.707 |
| summaries, BM25 | 0.275 | 0.675 | 0.875 | 0.452 |
| both, BM25 fused | 0.425 | 0.850 | 0.975 | 0.625 |
| code chunks, hybrid | 0.700 | 1.000 | 1.000 | 0.838 |
| summaries, hybrid | 0.350 | 0.775 | 0.925 | 0.530 |
| both, hybrid fused | 0.575 | 0.950 | 1.000 | 0.727 |

Hybrid lifts summaries (MRR 0.452 -> 0.530) but code chunks lift more (0.707 -> 0.838), so the
conclusion holds: for file lookup on code questions the summaries are worse than chunks, and
fusing them in lowers MRR (0.838 -> 0.727). Caveats: the summaries are the model-free version (no
prose paragraph yet), and these are code-behaviour questions, not architecture questions;
the architecture use stays unmeasured until such a set exists. Keep `summaries` out of default
code search.

### 2026-10-31 - Summaries improvement handed to Codex

`docs/CODEX_HANDOFF_SUMMARIES.md`: a pre-registered brief. Q0 build and freeze an architecture question
set (15 per repo, requests and flask, labels unreviewed by a person); E1 summaries as a router (narrow the
path, then chunk search) vs fusion; E2 richer model-free facts (all docstrings, raises/constants, commit and
PR titles, README section); E3 local-model prose on Colab (model choice needs the user's approval, check
faithfulness mechanically). Success rule: pooled test Recall@5 +0.05 with the paired lower bound above 0,
no repo moves the wrong way, code-question MRR@10 drops by at most 0.01. No code was changed for this.

### 2026-11-01 - Review of the summaries build; prose grounding check fixed (before any model run)

Claude reviewed Codex's architecture-summaries build. Verified: the four dataset hashes match; 1,280
saved result rows (both modes, all fact variants) recompute from the raw rankings with 0 mismatches;
the success rule in code matches the brief. Result: no variant is eligible. Hybrid test Recall@5 (20
questions): chunks 0.829, summaries 0.688, fused 0.779, router 0.704; F1 (all docstrings) lifts
summaries to 0.783 and fused to 0.863 (+0.033 over chunks, CI [-0.035, +0.091], code MRR -0.050),
below the +0.05 / positive-lower-bound / -0.01 rule. F3 (history titles) and F4 (docs sections) do not
help. Defaults unchanged; summaries stay out of default code search.

**Fixed:** `summaries/prose.py` rejected every word absent from the facts. Tested on real facts: a
faithful natural paragraph failed ("defines", "manages", "redirects") and a near-copy failed
("class" vs "Classes"), while invented numbers ("12 methods", "version 2") were never compared, so
the Colab run would have discarded nearly all model output. New rule (`CHECK_VERSION`
`identifiers-numbers-v2`): reject a paragraph containing a code-like name not in the facts (dotted,
slashed, snake_case, camelCase/Pascal with a later capital, backticked, followed by `(`, or a
capitalised word inside a sentence; plurals and 'Python' for 'python' accepted) or any number not
in the facts. Plain lowercase words and acronyms are free. Still lexical grounding, not entailment;
spelled-out numbers pass. `SUMMARY_PROMPT` reworded to match (no stop-word list).
**Proposal amended before any model run:** `model_proposal_v1.json` (+ freeze) kept; v2 frozen with
`supersedes_sha256` and the amendment reason; the architecture dataset freeze is untouched and still
verifies. Capsule `summary_prose_inputs.zip` rebuilt (sha256 c9f0a25e...); all notebooks regenerated.
Tests: `tests/test_summary_facts.py` prose tests rewritten (11 in file).

**Still open (not changed):** E2's fact choice is a tie on dev (the directory router scores 0.825 for
every variant), so F1 was picked by list order, and `fact_deltas.md` (router arm) is all zeros; the
informative summaries/fused deltas are in `fact_deltas.json` and the fact table in
`SUMMARIES_BUILD_REVIEW.md`. F3 has PR titles for 0 / 9 files only (no Requests PR snapshot). The
10-question dev split is too small to select on. The analyzer reuses the runner's scoring functions.
Labels are unreviewed by a person. E3 still needs the user's explicit model approval.

### 2026-11-01 - Model decisions and Phase 7 bake-off brief

The user replied "continue" to Claude's recommendation to use Qwen3-4B-Instruct-2507 for the summary
prose and to run a measured bake-off for the answer model. Recorded as agreement with that plan;
`summary_prose_on_colab.ipynb` still stops at its approval cell, which the user sets when running it.
Current models in use: embeddings Qwen3-Embedding-0.6B @512, opt-in GPU reranker Qwen3-Reranker-8B; no
generating model is chosen yet. `docs/CODEX_HANDOFF_PHASE7_BAKEOFF.md` is the pre-registered brief:
candidates (<= 4, open weights, one A100) proposed and user-approved before any download; facts (60,
model sees only the question) plus code/architecture questions plus 10 unanswerable probes; mechanical
scoring (exact answer, tool choice, citation range overlap, abstention); eligibility rule: fact accuracy
>= 0.90, invalid citations <= 0.05, probe abstention >= 0.80. No code was changed for this brief.

### 2026-11-01 - Review of the Phase 7 bake-off checkpoint; four candidates approved

Claude reviewed Codex's protocol/input freeze (`eval/datasets/phase7_bakeoff_v1/`,
`docs/PHASE7_BAKEOFF_PROTOCOL_REVIEW.md`). `prepare_answer_bakeoff.py verify` passes (130 questions;
facts 15 dev / 45 test, code 16 / 24, architecture 0 / 20, probes 2 / 8); questions.jsonl carries no
labels; gates convert to 41/45 facts, 7/8 probes, 4/44 answerable abstentions; the three Hugging Face
revisions, licences and parameter counts match the live API. Codex stopped at the approval gate (no
harness, no model). **Defect fixed:** `pyproject.toml` gained `eval.prepare_answer_bakeoff` after the
notebooks were regenerated, so the three Phase 5 notebooks embedding it were stale (3 failing tests);
all notebooks regenerated, full suite, ruff and mypy pass. Open findings: the dev fact types are the
easy ones (release chain, date-range counts, blame, tag lookup are all in test), the preparer has no
tests, ids encode the fact type so the harness must never pass them.

**The user approved all four candidates** ("Approve all four"): Qwen3-4B-Instruct-2507,
Olmo-3-7B-Instruct, Qwen3-14B, Qwen2.5-Coder-14B-Instruct, at the pinned revisions only, for this
bake-off only. Recorded in `eval/datasets/phase7_bakeoff_v1/approval.json`. The dev-split amendment
was not decided and is not part of the approval.

### 2026-11-01 - Phase 7 bake-off v2: dev split amended (before any model run)

At the user's request ("Amend dev split") Claude amended the bake-off dev split. **v1 is untouched and still
verifies** (the v1 preparer must stay byte-identical, so v2 has its own preparer). v2 adds 11 dev-only httpx
facts of the hard tool patterns, about entities not in the 60: release_first x2 (2732, 2799), commits_in_month x2,
blame_line x2, tag_commit x2, issue_closed_by x2, hot_file x1. Registered rule: `eval/build_facts.py` seed
20261101 on the frozen snapshot and checkout; drop facts whose (type, args) or id is among the 60; keep the
first two per type in builder order (`eval/build_dev_extra_facts.py`, output
`eval/datasets/httpx_facts_dev_extra.jsonl`). Expected answers come from raw GitHub nodes and plain git, not the
tools. hot_file had only one fresh fact. Result: 141 questions, facts 26 dev / 45 test, total dev 44 / test 97.

Unchanged and checked by `eval/prepare_answer_bakeoff_v2.py verify`: all 97 test questions and labels
byte-for-byte, the gates (41/45 facts, 7/8 probes, 4/44 answerable abstentions), scoring, generation, tools, timing,
and the four approved candidate revisions (`phase7_bakeoff_v2/approval.json` carries the approval over; it names
the later amendment request). Disclosed overlap: two extra commits_in_month facts share the file
httpx/_models.py with a test fact (different months). Test types are no longer held out as whole types.
New tests `tests/test_answer_bakeoff_freeze.py` (6) also cover the v1 preparer. pyproject mypy module list
extended, notebooks regenerated.

## 5. Open items

- [ ] (2026-11-20) Re-measure code, docs, history and architecture questions under the new default prompt `commit_explain` (next 27B session, `eval/colab_session.py`).
- [ ] (2026-11-20) Decide whether the summary index (fused search, summaries hidden) ships; it needs a 27B summary pass and a GPU embedding run per repository.
- [x] (2026-11-20, done: see the 2026-11-20 (evening) entry; fused 103, summary index 106 and 107 of 136 against controls 100 to 101) Run `docs/FLASK_COMMIT_RUNBOOK.md` on Colab: index `pallets/flask` (`d086db856be1`, `float32`, T4), generate 27B commit summaries for all 3,829 commits, embed them (`commit_summary_pilot_on_colab.ipynb`), and run `eval/commit_session.py --repo pallets/flask` on the 136 test questions to test whether `summary_fused_hidden` replicates on a second repository.
- [ ] (2026-11-19) Review a sample of the Codex summaries, run the pilot notebook on a GPU, and score the vectors. Report the result as summaries by Codex imitating the 27B. The mixed-writer result alone cannot justify a 27B run over all 4,883 commits.

- [x] (2026-11-11) Review all 50 non-abstained saved code/docs answers with Codex; fill CSV verdicts/notes and run mechanical comparison. This is explicitly AI review, not human calibration.
- [ ] (2026-11-11) Have a person verify the AI review; repair the Makefile citation capture and missing proxy/SSL/timeout reference sections in a versioned follow-up, then compare with actual judge output. Do not claim human-vs-judge agreement from the AI-marked sheet.

- [ ] (2026-11-05) Commit questions 3 to 4 of 10: gold fix outside the top 5 for 5 of 10. Test reranking the top 50 next to the answer model on the A100; also bare-SHA answers carry no explanation.
- [ ] (2026-11-05) Docs 8 of 12 and code 36 of 40 after the parse fix; remaining misses are tool choice (usage questions to search_code, symptom-style code questions to search_commits) and retrieval range, see the entry above.
- [ ] (2026-11-05) All agent_eval numbers are single exploratory runs; run-to-run variation not measured.
- [ ] (2026-11-05) PR/issue text search tool and `index-prs` embeddings still not built; FastAPI backend and URL indexing from the UI (Phase 8) not started.

- [x] Freeze Phase 7 protocol, splits and ten probes before candidate research/model loads; propose at most four exact revisions.
- [x] Approve four Phase 7 revisions and v2 dev amendment; build harness/fake tests/capsule/A100 notebook/mechanical analyzer.
- [ ] User runs the v2 A100 notebook, reviews dev, records the dev-selected freeze, then returns raw test results for recomputation; no model measurements yet.
- [x] Q0: freeze 30 architecture questions and the unchanged 40-question Requests code regression set before retrieval.
- [ ] Human review of architecture_v1 labels (currently **unreviewed by a person**); corrections require a new version.
- [x] Complete/audit final hybrid E1/E2 runs and apply the registered success rule; retain negative results (no adoption).
- [x] Approve summary model separately; four answer-model experiment revisions also approved (2026-11-01).
- [ ] Run the generated E3 Colab notebook with its approval cell set and review raw prose/rejection audit; this Phase 7 task leaves that experiment untouched.

- [x] Phase 5b: PR/issue collections and search, tools, auto-generated fact set (60/60 through the tools on httpx).
- [ ] Run `repo-rag index-prs` with the real model on a GPU, then `eval/run_pr_search_eval.py --strategies bm25,hybrid` to measure dense/hybrid issue -> fixing-PR search (BM25 baseline: Hit@5 0.600 on 45 pairs).
- [ ] Phase 7: rerun `eval/datasets/httpx_facts.jsonl` end to end through the agent (target 90%); generate fact sets for a second repo.
- [ ] Decide whether the marginal `retrieval_k=200` result (`eligible_pending_review`) merits a per-kind commits setting; if yes it needs its own change, tests and review. Default stays 50 until then.

- [x] Audit the original Phase 5 v2 GPU artifacts and build the frozen commit depth/BM25 weight runner, paired analysis, timing subset, exact CLI report capture and Drive notebook.
- [ ] Run/review `phase5_commit_depth_on_colab.ipynb` on A100, audit raw pairs/prefixes and per-repo timing, then decide whether any commit-only config change merits review/tests. No automatic default change.
- [x] Correct Phase 5 review blockers, preserve v1, freeze v2, add localization/query-budget checks and small-stratum rule; sanitize SQLite payloads and regenerate notebooks.
- [ ] Independently judge alternative historical fixes with time-aware new-bug queries before claiming the product similar-fix goal is validated. Same-file overlap is only a proxy.
- [x] Implement task-specific rerank instructions, injected regression test, production-header v2 profile and full candidate/pair/token/warm-cold timing instrumentation. Real GPU execution/latency was subsequently audited in the returned v2 run.
- [x] Version and author-review required/supporting evidence for the 8+2 cases; add 49 new questions and freeze original gold/source. Independent review remains open.
- [x] Audit the returned three-repo A100 matrix, assess fresh/test/example quality and p50/p95, record keep-opt-in decision, publish reports/update write-ups and tick Phase 4.
- [x] Initial Phase 4 full GPU run and separate 4B code-reranker experiments at 20/50: validated for both repos. Code reranking remains opt-in; no independent top-20 timing or universal enablement conclusion.
- [ ] Independently review v2 labels and alternative evidence before default promotion. Mixed-target GPU assessment is complete; retain frozen labels and failures. Requests has no example questions; pooled examples number six. Add fresh source-intent/multi-block cases for any new policy.
- [x] Expose production scoring depth separately from retrieval depth; default 50 preserved; fixed-input production replay and tests pass.
- [x] Expand the 15-case set to frozen90 with source-reviewed ambiguous/multi-block/test/implementation/example strata; validate hashes/full-corpus answerability, measure BM25 and preserve original15 scores.
- [x] Audit the 90-question depth run and subsequent 120-question policy/header runs; retain measured limitations and plain reranking/original headers. Independent label review remains open separately.
- [x] Implement the Codex expansion brief: freeze fresh 30 including Click, preserve old 120, wire explicit default-none expansion, prepare actual production GPU quality matrix/audit and run local regression checks.
- [x] Run the new expansion A100 notebook, audit returned results and apply the fixed callee rule: REJECTED (fresh proof 0.667 -> 0.567). Proceed to Phase 5 without further expansion tuning on these labels. Independent review remains pending.
- [x] Implement Phase 5 commit ingestion, both verified/cached linking paths, SQLite links, fixture integration and leak-resistant corpus/metrics; configure GitHub token and freeze real inputs before retrieval.
- [ ] Perform the Phase 5 manual scale tier. The original fixed A100 4B/8B matrix has been audited; the depth/weight extension remains pending. See docs/COMMIT_DEPTH_BUILD_REVIEW.md; do not reopen expansion tuning.
- [ ] Choose deployment hardware and latency requirements before measuring that deployment path or promoting reranking by default; A100 remains experiment-only.
- [ ] ANN scale check: larger corpus, explicit build/search settings, recall and latency. The observed 1.0 recall on 270–565 vectors is a plumbing check.
- [ ] Have a second human review the 80 source-grounded questions; expand beyond the Requests seed (used for earlier model selection) and new Express questions before broad generalization.
- [ ] Add a matched-budget sensitivity comparison and independently authored/paraphrased questions before attributing measured differences to AST structure alone. Bootstrap intervals are pointwise/exploratory, not multiplicity-adjusted confirmatory evidence.
- [ ] Time a Colab run (GPU type + the index cell's duration) on a bigger repo, to settle the "indexes in under 10 minutes on a GPU" criterion. The first run worked but wasn't timed.
- [ ] Delete the stray `D:\Users` folder (my mistake: one script file inside empty folders).
- [ ] First git commit (nothing has ever been committed; the user has said "not yet" so far).
- [ ] `AGENTS.md` is a stale copy of an old CLAUDE.md: the user to decide delete vs point at CLAUDE.md.

### 2026-11-10 (later): Codex review of the judge check and review sheet, fixed

Codex found three issues in `eval/review_sheet.py` (lint and 11 tests passed for it; it also saw a stale-bytecode failure in the fake-GPU preflight test). All checked against the code and real:
- **Evidence was cut at 45 lines** in the review sheet, so a person could not check claims against everything the judge sees. Removed the truncation; the sheet shows the full cited and reference lines.
- **Re-running `sheet` overwrote the CSV**, wiping a person's verdicts. It now carries over verdicts and notes by id and keeps any already-marked answer even when a smaller `--sample` would not pick it.
- **`compare` was incomplete without a judge.** It now also reports person verdict counts, mechanical vs person, and mechanical vs judge, and says so when there is nothing to compare. Key renamed `agreement` -> `person_vs_judge`.
- **Preflight test flake:** the fake `torch.py` was rewritten per call under one directory, so a stale `.pyc` was reused. The test now disables bytecode writing and drops the cached module.
Tests: `tests/test_review_sheet.py` (5) and `tests/test_answer_judge.py` (8) pass twice in a row. The full suite was not re-run (the other session is still editing `src`).

### 2026-11-11: AI evidence audit of the 50 saved answers (draft record, not Phase 7 acceptance)

Codex read all 50 non-abstained answers (38 code, 12 docs; 2 abstentions excluded) in `eval/review/answer_review_v1/` against the complete cited and reference lines, and filled `review_sheet.csv`. **These are AI verdicts, not independent human labels** (`ai_review.json`: `independent_human_review: false`; `review_summary.md`). `eval/review_sheet.py compare` output:
- Faithful 29/50 (58%), partial 17, unfaithful 4. Correct 30/50 (60%), partial 18, incorrect 2. Both top verdicts: 22/50 (44%). Partial is not counted as a pass.
- Mechanical score vs this review: 64% agreement. Mechanical pass 42 answers (27 correct, 15 partial); mechanical miss 8 (3 correct, 3 partial, 2 incorrect). So the mechanical pass rate overstates answer quality, by one AI reader's judgement.
- `docs-047` and `docs-050` partial labels are provisional (reference evidence insufficient); not proven answer errors.
- The `compare` key `reviewed_by_person` counts these AI rows; read it as "reviewed", not "a person".
**Not established:** that the Qwen judge agrees with a person. Phase 7 stays open until the judge run (blocked: A100 busy) and a comparison against marks from someone who is not an AI (10-15 answers is a start).

### 2026-11-11: Phase 7 judge run (Colab A100, results in `eval/review/answer_review_v1/`)

Judge `cyankiwi/Qwen3.8-27B-AWQ-INT4` (compressed-tensors, snapshot e3a70090), vLLM 0.33.0, 194 requests in 332 s, 0 unparsable (0/194), no `RUN_ERROR`. The earlier offline fixes held on a real GPU.
- **Gates (frozen protocol): judge trusted = True.** Negative controls rejected: faithfulness 47/48 (98%), correctness 48/48 (100%); 0 unresolved citations; 2 abstentions skipped.
- **Answers:** faithful 26/48 (54%; partial 19, unfaithful 3); correct 31/50 (62%; partial 14, incorrect 5). Code 23/38 correct, docs 8/12.
- **Mechanical score vs judge:** 66% agreement (mechanical pass 42: judge says 28 correct, 12 partial, 2 incorrect; mechanical miss 8: 3 correct, 2 partial, 3 incorrect). Same direction as the AI audit: mechanical pass overstates quality.
- **Judge vs the Codex (AI) review, not a person:** top-verdict agreement faithful 41/48 (85%), correct 37/50 (74%); exact-match 83% and 68%. 16 correctness disagreements listed in `agreement.json`. Judge and Qwen answer model share a family; self-preference untested.
**Still open for Phase 7:** no non-AI marks exist, so "the judge agrees with a person" is unmeasured; the negative controls passing shows it rejects mismatched evidence, not that its 54%/62% are calibrated. Quote the rates as judge estimates, with the AI-audit rates (58%/60%) beside them.

### 2026-11-11 (later): Claude's re-read of the 20 contested answers; Codex review corrected; judge scored against it

I read the 20 answers (unique ids) where Codex and the judge differed, from `items.json` only, before looking at either set of verdicts (`eval/review/answer_review_v1/claude_review.json`, with a reason per id). **This is a third AI reader, not a person**; the 30 answers where Codex and the judge agreed were not re-read, so a shared mistake there would go unseen.
- **Codex corrected in `review_sheet.csv` on 6 labels** (original kept in `review_sheet.codex_original.csv`; each changed row's notes say what it was): code-006 faithful partial->unfaithful; code-015 both partial->faithful/correct (the omitted "header absent" guard is a nuance, not an unsupported claim); code-032 correct partial->incorrect (says `Response.text` uses a deprecated function it never calls; borderline); docs-042 correct partial->correct; docs-050 correct partial->correct (Codex flagged it provisional, reference excerpt is only the file head).
- **Agreement with my reading on the 20:** Codex faithful 18/20 exact, correct 16/20. Judge faithful 14/20 exact, correct 7/20 exact (8/20 on the top verdict). The 20 are chosen for disagreement, so this is not the judge's overall accuracy.
- **Corrected review totals (50):** faithful 30 (partial 15, unfaithful 5); correct 33 (partial 14, incorrect 3). Mechanical vs review agreement now 70%. Judge vs corrected review: faithful exact 87.5% (n=48), correct 74% (n=50).
- **Judge failure patterns seen:** (1) it lowers *correctness* for a cited line range that is wider or off (code-013, 018, 019, 030) though the right code is inside it; that is a faithfulness matter and not in the correctness rubric. (2) For docs, the reference excerpt is only the file head, so it called docs-050 and docs-046 "incorrect" because the reference lacked the section (docs-050 is a right answer). (3) It lets real defects through as "correct": code-004, 011, 023, 036 (Codex caught these). (4) It docks faithfulness for omissions (code-039, 009) and for not seeing a callee's body (code-012). (5) It got code-006, 021 (unfaithful) and 032 (incorrect) right, though for 032 it wrongly said the function was absent from the evidence.
- **Implications:** quote judge correctness rates as an upper-leaning estimate, not a measurement; for docs, build the reference evidence from the section, not the file head, before trusting its correctness verdicts. A non-AI check is still missing.

### 2026-11-11 (later): judge protocol v2 prepared (not run)

Built from the three-reader findings; v1 files and results are untouched (`JUDGE_PROTOCOL` env var selects v1 for reproduction; `tests/test_answer_judge.py` checks v1 prompts still load).
- **Docs reference by section, not file head** (`doc_sections`, `doc_reference` in `eval/judge_answers.py`): the up-to-3 sections of each named file most about the question (question stems found in a heading weigh 3, in a body 1, each by rarity; sections under half the best score dropped; falls back to the file start). Checked by hand on the 12 docs questions: the section holding the answer is now included for docs-045 (Timeouts), 046 (Basic Authentication), 047 (Proxies), 050 (SSL Cert Verification), 042, 051. Some rows still carry a noise section (e.g. "Custom Headers" for docs-050).
- **Prompts v2:** faithfulness judges only what the answer asserts (omissions and a wider cited range are fine); correctness ignores cited paths and line numbers and calls an answer partial when it omits something the question asks for. Deliberately NOT added: "don't mark incorrect when the reference doesn't cover the question", because the negative controls rely on a mismatched reference producing `incorrect`.
- **Protocol:** `eval/datasets/phase7_answer_judge_v2/protocol.json` (same model, seed, controls, limits, gates). **v2 numbers are in-sample:** the changes were designed on the same 50 answers the reviewers read, so an improvement in agreement with the corrected review is expected and is not evidence about new answers.
- Notebook and bundle regenerated for v2; `eval/review/answer_review_v2/items.json` is the CPU-side prepare output for inspection. 15 tests pass (answer judge + review sheet). Full suite not run (the other session is still editing `src`).
- Next: run `notebooks/answer_judge_on_colab.ipynb` on the A100, then `eval/review_sheet.py compare` against the corrected review (copy `review_sheet.csv` into the v2 folder first).

### 2026-11-11 (later): judge v2 run (Colab A100, results in `eval/review/answer_review_v2/`)

Same model (`cyankiwi/Qwen3.8-27B-AWQ-INT4`), vLLM 0.33.0, 194 requests in 327 s, 0 unparsable. Gates pass again: controls rejected faithfulness 47/48, correctness 48/48. Judge says: faithful 25/48 (52%; partial 19, unfaithful 4), correct 41/50 (82%; partial 4, incorrect 5; code 31/38, docs 10/12).
Against the corrected review (faithful 30/50, correct 33/50), **in-sample** (v2 was designed on these answers):
- **Correctness:** exact agreement 80% (v1: 74%). All 17 disagreements are the judge being kinder or harsher in one direction mostly: 8 cases where the review says partial and the judge says correct (code-004, 023, 027, 031, 036, 039, docs-045, 047), 2 where the judge says incorrect against a review partial (code-022, docs-046), and none where the judge is stricter than a review "correct" at the top level. The judge now gives only 4 partials against the review's 14, so **it overstates correctness (82% vs 66%)**; v2's "ignore citation numbers" fix removed the false downgrades but exposed that it does not catch incompleteness.
- **Faithfulness:** exact agreement 75% (v1: 87.5%), worse. It marks 8 faithful answers partial (code-003, 009, 012, 013, 024, 036, 038, docs-047), so the omission rule did not hold; it reports 52% against the review's 60%.
- On the 20 contested answers v2 matches the Claude re-read on 14/20 (faithfulness) and 13/20 (correctness), against 14 and 7 for v1.
- Mechanical vs judge agreement 78% (v1: 66%).
**Reading:** the judge is a reasonable detector of clearly wrong or unrelated answers (controls 98-100%; 3 of the review's 3 incorrect answers flagged) and not a measure of completeness or fine faithfulness. Quote its rates as ranges bounded by the corrected AI review (faithful ~52-60%, correct ~66-82%), not as accuracy. No non-AI labels exist.

### 2026-11-11 (later): blind human check sheet (15 answers)

`eval/review/human_check_v1/`: `review_sheet.md` + `review_sheet.csv`, blank, no judge/Codex/Claude verdicts and no mechanical score (`review_sheet.py sheet --blind`). The 15 are the answers where the readers split: judge-v2 "correct" vs review "partial" (code-004, 023, 036, 039, docs-045, 047), judge "incorrect" vs review "partial" (code-022, docs-046), judge "partial" vs review "faithful" (code-003, 009, 012), judge "faithful" vs review "partial" (code-011, 016), and Claude vs Codex (code-015, 032). A person fills `faithful` and `correct`; then copy `eval/review/answer_review_v2/judgments.jsonl` into that folder and run `compare` for human vs judge, and compare by hand against `answer_review_v1/review_sheet.csv` and `claude_review.json`. The sample is chosen for disagreement, so it measures who is closer to a person on hard cases, not overall accuracy.

### 2026-11-11 (later): no human labels will be made now

The project owner does not have the repository knowledge to mark the 15-answer sheet (`eval/review/human_check_v1/`), and an AI filling it would only add a fourth AI reading, not a human one, so it is left blank and optional. **Phase 7's judge check is therefore validated against AI readers only** (Codex, Claude, and the judge itself); state that wherever its numbers are quoted. If a teammate who knows the code can mark it later, the sheet and `review_sheet.py compare` are ready.

### 2026-11-11 (later): owner-submitted marks for the 15-answer blind sheet (provenance unverified)

The project owner pasted marks for the 15 answers (`eval/review/human_check_v1/review_sheet.csv`; 5 faithfulness cells left blank: code-009, 011, 016, 032, docs-045). **Independence is unverified:** the owner had said they lack the repository knowledge, and several reasons match Codex's notes almost word for word (code-011, 032, 036, docs-045, 046), so treat these as possibly AI-derived, not as clean human labels. Agreement with them (exact; cells marked only):
- Judge v2: faithfulness 5/10, correctness 10/15. Judge v1: 6/10, 8/15.
- Codex original: 8/10 and 10/15; Codex as corrected: 9/10 and 10/15. Claude re-read (12 of the 15 ids): 8/8 and 7/12.
- The marks are more lenient than both AI readers on completeness: correct where Codex and Claude say partial for code-004, 023, 039, docs-047 (the judge agrees with the marks on 023, 039, 047 and 004). So "the judge overstates correctness" is not settled: it depends on whether omissions make an answer partial, a rubric choice.
- The judge is furthest from the marks on code-022 (judge unfaithful/incorrect, marks faithful/partial), docs-046 (incorrect vs partial), code-032 (incorrect vs partial), and it marks faithful answers partial on code-003, 012, 036.
- 15 answers chosen for disagreement: this says who is closer on hard cases, not overall accuracy.

### 2026-11-11 (later): full test suite green; README answer-quality table

`uv run pytest -q -p no:cacheprovider --basetemp=data/pt_full1 -x`: all 850 tests passed, exit 0, run once with nothing else running (the notebook-freshness tests also pass now that the other session's edits have settled). README: added "Answer quality (code and docs answers, AI-checked)" with the judge v2 and corrected AI-review rates and the no-human-labels limit; the "Labels are author-checked" limit now says the same.

### 2026-11-11 (later): citation snapping and an offline citation check (built, not wired into the agent)

New `src/repo_rag/agent/citation_snap.py` (+ `eval/citation_check.py`, `tests/test_citation_snap.py`, 9 tests). Motivation: the reviews found cited ranges that are the whole chunk, miss the function the sentence names (code-006), or run past the file (code-021).
- `citation_support`: for each `path:a-b`, the backticked names said since the previous citation that no citation to that file shows (a name inside a definition that contains the citation counts as shown).
- `snap_citations`: re-points a range that shows none of the named definitions to their definition ranges from the symbol table, or narrows a wide range that contains all of them; keeps a range that already shows part of a named definition or sits inside a definition the answer names elsewhere; only when the new range overlaps a retrieved chunk; never edits prose or adds a citation; returns every change.
- **Offline result on the 50 saved code/docs answers** (`eval/citation_check.py --review ...`): the check flags 9 answers: 4 the review rates partial, 2 unfaithful (code-006, 021), 3 faithful (code-005, 038, docs-047) so precision about 6/9; it flags 6 of the 20 non-faithful answers, so recall is about 30%: a cheap partial signal, not a faithfulness measure. 6 answers get snapped (code-006, 007, 014, 019, 021, 038). Rescoring mechanically: no passed answer loses overlap with the gold range, and code-006 gains it.
- **Not measured:** whether snapped answers score better with the judge or a reader. The snapped answers are in `eval/agent_eval_results/agent_eval_production_20261109T013636_snapped.json`; running judge v2 on them (one A100 run, notebook needs a different answers file) is the measurement. Until then it is not wired into `agent/graph.py` (the retrieved chunks, not the whole file, would also limit it there, so it can only act less than in this offline run).

### 2026-11-11 (later): judge run on the snapped answers prepared (not run)

`notebooks/build_answer_judge.py` now builds a second notebook and bundle for the snapped answers: `notebooks/answer_judge_snapped_on_colab.ipynb` + `notebooks/answer_judge_snapped_bundle.zip` (Drive names `answer_judge_snapped_bundle.zip` and `answer_judge_snapped_results.zip`, so they do not overwrite the earlier run). Same judge, protocol v2, seed and 50 answers; only the `path:start-end` references differ (6 answers). A local `prepare` on it gave 52 items, 48 controls, 194 prompts, 0 unresolved citations; code-006's cited evidence now holds `apparent_encoding` and `text` and code-021's holds `dispatch_hook`. The unsnapped notebook text was also corrected (docs reference is by section in v2). `tests/test_answer_judge.py` (11) checks both notebooks and bundles are current and that the question file's hash is unchanged. Read the result against the v2 run: faithfulness on the 6 changed answers, and no change elsewhere (the controls should still pass).

### 2026-11-11 (later): judge v2 on the snapped answers (results in `eval/review/answer_review_snapped_v2/`)

Same judge and protocol; 194 requests in 329 s, 0 unparsable; controls rejected 47/48 and 48/48 (gates pass). Faithful **29/48 (60%)** vs 25/48 (52%) for the unsnapped answers; unfaithful 2 vs 4; correct 41/50 (82%) unchanged.
- Of the 6 answers whose citations changed: code-006 and code-021 went unfaithful -> faithful (their new ranges hold `text`/`apparent_encoding` and `dispatch_hook`, which I checked in the prepared evidence), code-038 partial -> faithful; code-007, 014, 019 unchanged; correctness unchanged on all six.
- Of the 44 answers with the same text, 1 verdict moved (code-009 faithfulness partial -> faithful): judge run-to-run/batching noise of about 2%, so a +4 gain is well above it but a single flip is not a finding.
- The corrected review's labels are for the original citations, so "exact agreement with the review" is unchanged (36/48, 40/50) and is not a fair test of the snapped versions; the judge's change on 006/021 agrees with what the snapped ranges contain.
- **Reading:** snapping fixes citation-range faithfulness failures without changing correctness, no gold overlap was lost (checked earlier), and nothing got worse. It is measured on 50 answers from one repo with one judge: three verdict improvements, no regressions.
- **Decision pending:** wire `snap_citations` into `agent/graph.py` (needs the symbol table at query time and the retrieved chunks, so it can only act less than offline); not done.

### 2026-11-11 (later): citation snapping wired into the agent behind `agent_citation_snap` (default off)

`Settings.agent_citation_snap` (default False). When on, `RepoAgent._check` runs `snap_citations` after `repair_citations` and lists each change under `result["citations"]["snapped"]` (the key is absent when off, so the saved result shape is unchanged). `RepoAgent` takes an optional `snap_lookup`; `build_agent` supplies one backed by the symbol table, and a database without symbols reads as "nothing found" (a `sqlite3.Error` returns `[]`; any error inside snapping leaves the answer as it was). Tests: 3 in `tests/test_agent_graph.py` (off by default, narrows and reports, no lookup or a failing lookup never costs the answer) and one assertion in `tests/test_agent_runtime.py`, which also surfaced the no-table case. `index_on_colab.ipynb` regenerated (src changed). Default stays off until a live run on the A100 measures it end to end (the offline gain, 25/48 to 29/48 faithful, used the whole cited file as the retrieved set and so acts more than the live agent can); enable with `REPO_RAG_AGENT_CITATION_SNAP=true`.

Full suite after the snapping change: green on the fifth run (all tests pass, exit 0). Getting there took regenerating every notebook that embeds `src/repo_rag` (about 18 builders: indexer, ask, serve, policy, header, pr_search, flask_pr_search, retrieval, phase4 followup, summary_prose, judge, expansion, comparison, ambiguous, commit_depth, commit_depth_resume, commit_confirm, bugfix, answer_bakeoff), so zips already on Drive for those no longer match their notebooks; re-upload before using one. Two runs failed on six freshness tests that passed when run alone: my `ruff format src` had reformatted files I had not meant to touch (`agent/prompts.py`, `agent/tools.py`, `store/symbol_table.py`, which the other session may be editing), changing the bytes the notebooks embed. Lesson: format only the files you changed.

### 2026-11-16: A100 session prepared (live snap check, `grounded_complete` prompt, section 1 questions, fact set); not run

Plan and pre-set decision rule in `docs/A100_RUN_PLAN.md`. New: `agent_prompt_variant="grounded_complete"` (`docs_hint_narrow` plus a block saying to answer only from the returned chunks and to give what was asked; generic wording, no example is an eval question's subject); `eval/agent_eval.py run --citation-snap` (tag `_snap`, rows now record the `citations` report); `eval/section1_check.py` (32 plan.md section 1 questions adapted to psf/requests, automatic citation/abstention/SHA flags, writes `eval/section1_results/*.md`; `tests/test_section1_check.py`, 6 tests). All ~19 notebooks that embed `src/repo_rag` regenerated again; full suite green (exit 0, no failures) with ruff clean.

### 2026-11-16: A100 session results (answer model Qwen2.5-Coder-14B served from Colab; agent and tools on the laptop)

Files in `eval/agent_eval_results/`: A `agent_eval_production_docs_hint_narrow_20261116T202742.json`, B `..._docs_hint_narrow_snap_20261116T203735.json`, C `..._grounded_complete_snap_20261116T204724.json`, D `agent_facts_20261116T205314.json`; E `eval/section1_results/section1_docs_hint_narrow_snap_20261116T205919.{md,json}`.
- **A control, B snap, C stricter prompt (code, docs, prsearch; 62 questions each):** identical pass counts: code 36/40, docs 10/12, prsearch 10/10. A matches the earlier measurement of this prompt. B vs A: no pass flips; 6 answers changed, all citation ranges (code-008, 014, 019, 020, 021, 038; mostly narrowed test ranges plus the code-019/021 fixes). C vs B: 39 of 62 answer texts changed, two flips (code-006 miss to pass, code-023 pass to miss), median length 392 vs 375 characters, abstentions unchanged. By the rule in `docs/A100_RUN_PLAN.md`, C does not qualify on pass counts (needs +2); its target (unsupported claims, incomplete answers) is only visible to the judge, not yet run. Snapping: pass counts are not lowered; the judge check on B vs A is still pending.
- **D facts:** 57/60 (95.0%), target 90% met (was 59/60 on 2026-11-08). Misses: blame_line httpx/_config.py:114, tag_commit 0.28.1 and 0.6.4. Cause of the two-fact drop not investigated (git tools are untouched by the prompt and snapping changes).
- **E section 1 (32 questions adapted to psf/requests, `docs_hint_narrow` + snap): the "cited, sensible answers" criterion is NOT met.** Automatic checks flagged 13. My reading (an AI reader): about 12 answers are sensible and supported (arch-3, explain-1, explain-2, people-1/2/4, pr-1, pr-2, rel-2, stats-1, stats-2, bug-1); 9 abstained or failed on questions that have answers (arch-2, arch-4, explain-3, nav-1, people-3, issue-1, rel-1 abstained after one search or a get_symbol only; hist-1 blame tool called with an invented path `src/utils/url.py`; hist-2 budget failure); the rest are weak: arch-1 (names two files, no architecture), explain-4 (docstring claims `__len__`/`fileno`/`tell` the cited lines do not show), bug-2 (cites the pinned HEAD sha as a commit touching proxies and timeouts), bug-3 and pr-2 and rel-2 (bare SHA or number with no explanation), impact-1 (invents a `RedirectSession` class), impact-2 (searched code, not pyproject), pr-3 (vague), and oos-1 answered the out-of-scope cache question instead of abstaining.
- **Not run:** judge scores on A, B, C answers; section 1 with `grounded_complete`.

### 2026-11-16 (later): fixes for the section 1 failures, code written, NOT finished or measured

Written (ruff clean; `tests/test_prompt_variants.py` and `tests/test_agent_graph.py` pass; nothing else run):
- `file_history` tool (`agent/tools.py`): newest-first commits for a file or one function; a symbol name alone is resolved through the symbol table, and an ambiguous or unknown name returns an error that lists the candidate paths. Offered only in the new prompt variant `history_grounded` (= `docs_hint_narrow` + a history paragraph + the `grounded_complete` block).
- `Settings.agent_tool_error_retries` (default 0): a failing tool call goes back to the model as `TOOL_ERROR_NEXT` instead of ending the run (hist-1 died on an invented blame path).
- `Settings.agent_abstain_retries` (default 0): an abstention before two tool calls is answered once with `ABSTAIN_RETRY` (7 of 32 section 1 questions abstained after one search). Both guards are off by default until measured, and both are wired through `Toolset` and `run_question`.
**Still to do:** unit tests for the two guards and `file_history` (none written yet); regenerate every notebook that embeds `src/repo_rag` (they are stale now) and rerun the full suite; `eval/agent_eval.py` has no flags for the two guards yet (set `REPO_RAG_AGENT_ABSTAIN_RETRIES=1`, `REPO_RAG_AGENT_TOOL_ERROR_RETRIES=1`); an A100 run to measure them (code/docs/prsearch pass counts, the 5 out-of-scope probes must still abstain, then section 1). The weak answers (arch-1, bug-2 citing the pinned HEAD sha, impact-2, oos-1 answering) are not addressed beyond the grounded paragraph.

### 2026-11-16 (later): tests for the section 1 fixes; suite state

`tests/test_agent_guards.py` (8 tests: abstain guard on/off/once, tool-error recovery, `file_history` offered only in `history_grounded`, bare-symbol resolution and error messages) pass; `file_history` is listed in `HISTORY_TOOLS`, not `ADDED_TOOLS` (a test pins the default production tool set). `agent_eval.py` and `section1_check.py` take `--abstain-retries` / `--tool-error-retries`; `docs/A100_RUN_PLAN.md` has Round 2 (runs F, G, H) with its pass rule. Notebooks regenerated. The last full run had two failures that are not from these changes: another session is adding 27B/32B serve notebooks (`tests/test_serve_notebook.py` expects `notebooks/serve_model_27b_on_colab.ipynb`, not generated yet) and was editing `config.py` and `local_llm.py` while the suite ran (one freshness race, passed on rerun). Rerun the full suite once that session settles.

### 2026-11-18: A100 run plan rewritten for the 27B

`docs/A100_RUN_PLAN.md` now covers the 27B only (section 1, the fact set, then the guards, `file_history` and snapping behind a rule fixed in advance). The 14B plan (first session and Round 2) was moved to `experiments/docs/A100_RUN_PLAN.md` in the repository reorganisation and is historical. Nothing was run; the times in the plan are estimates from earlier 27B runs. The 14B's section 1 and fact results are under `experiments/eval/`.

### 2026-11-18: section 1 and fact set on the 27B; root causes and fixes (Colab A100 40 GB, vLLM, CUDA graphs + MTP)

Server: `cyankiwi/Qwen3.8-27B-AWQ-INT4`, `max_input_tokens` 12288. Agent on the laptop with the production defaults (3 extra files, tests skipped, 24,000 tool characters). All single runs; the automatic section 1 flags are not a correctness check, and I (an AI reader) did not verify every unflagged answer.

- **S1 baseline** (`section1_docs_hint_narrow_20261118T093045`): 6 of 32 flagged (14B: 13). Failures: arch-2 (parse), hist-1 (budget), hist-2 (budget), pr-3 and rel-1 (abstained); nav-1 cited a range that does not show `DEFAULT_REDIRECT_LIMIT`.
- **Root causes** (re-asking the failing questions with full traces):
  - arch-2: the first two `search_code` results used 23,625 of the 24,000 tool characters, the next two came back as `{"value": [], "omitted": N}` with no reason, and the model searched blind until the turn cap. Separately the 27B often ends a tool call with one `}` too many, which cost a parse retry and, when it repeated, the question (three `parse` failures at 1.6 s).
  - hist-1: `blame` on a 81-line range returned 10,550 characters; the third reply (a correct per-commit summary) hit the 512-token output cap, so the JSON was invalid. That cap is applied on the server and was left unchanged.
  - hist-2: no tool lists a file's recent commits; the model tried `count_commits` with invented dates and hit the 4-call cap.
  - rel-1: the `tags` table in `data/metadata.db` was empty for psf/requests (162 tags exist in git), and there was no tool for "what changed between two tags".
  - pr-3: `pr_reviews` holds reviewer, state and time only, no review text; not fixable by tuning (the abstention is correct, its wording is generic).
- **Changes:** `file_history` now takes a line range from the symbol table (`git log -L :name:` cannot find an indented method: "no match" on `prepare_url`); new `compare_refs` tool (commit count and newest 15 between two refs; reads no blobs, so no fetches on the blobless clone), offered only in `history_grounded`; setting `agent_budget_note` (off by default, `REPO_RAG_AGENT_BUDGET_NOTE=true`) adds a note to a result emptied by the budget; tolerant parsing accepts extra closing braces after a complete JSON object (production only, strict bake-off parsing unchanged); `eval/agent_facts.py` falls back to `experiments/data_scratch/answer_bakeoff_capsule` (the capsule moved). Data: `repo-rag index-facts psf/requests` loaded 162 tags and 139 author-login links into `data/metadata.db` (the run was slow: the batch lookup falls back to one `git log` per missing commit, each fetched from GitHub by the partial clone, about 1.7 s each).
- **S3** (guards + snapping + `history_grounded`): 5 flagged; hist-2 fixed by `file_history`; hist-1 failed on the `-L` bug above.
- **Reruns after each fix:** 4 flagged (hist-1, hist-2 answered), then 3 (arch-2 answered after the brace fix), then **1 of 32 flagged, 2 abstained**: pr-3 (data limit) and oos-1 (correct). rel-1 now answers via `compare_refs`: 5 commits, cited by SHA and PR number (not checked against git by me). nav-1 was not flagged in the last run; nothing was changed for it, so it may flip.
- **S2 facts on the 27B: 59/60 (98.3%)**, target 90% (14B: 57/60). The 14B's misses (blame_line, tags 0.28.1 and 0.6.4) now pass; one miss: `commits_in_month:tests@2019-12-01` (used `change_frequency` after two `count_commits`).
- **Regression with all guards on** (`history_grounded`, snapping, abstain/tool-error retries 1, budget note; code, docs, prsearch, probe, history, refs, kind): code 39/40 then 40/40 and 40/40, docs 11/12 (docs-051 missed in every run), prsearch 10/10, history 11/11, refs 10/10, kind 5/5, probes 5/5 all abstained; baseline code 39/40. By the rule in `docs/A100_RUN_PLAN.md` no pass was lost. The code 39 to 40 change is one question and is not credited to a fix.
- **Not changed:** defaults stay off (guards, snapping, `history_grounded`, budget note); adopting them is the user's decision. Phase 7 stays unticked: the judge has not run on 27B answers, no person has labelled answers, and sensible-answer review of section 1 is by flags plus partial AI reading. Files: `eval/section1_results/section1_*_20261118T0930..1007*`, `eval/agent_eval_results/agent_facts_20261118T094552.json` and the three `..._20261118T094233 / 095807 / 101342` regression files. Nothing committed.

### 2026-11-18 (later): the 19 exact-fact section 1 answers checked against git and `metadata.db` (S1 run `..._20261118T100718`)

Offline check by an AI reader (script in the session scratchpad, not in the repo): each answer compared with `git` at the pinned commit `611c6162cb` and the database. The automatic section 1 flags (1 of 32) do not test correctness, and this check shows they overstate quality for exact questions. Code and docs answers were not re-verified beyond the automatic citation checks.
- **Verified correct (9):** rel-1 (the 5 commits and their SHAs match `git log v2.30.0..v2.31.0`), rel-2 (v2.27.0, first tag containing PR #5856's merge commit), rel-3 (2.4.2, first tag containing the commit that added `json=`), people-2 (Joren Hammudoglu, newest commit on lines 40-80), people-4 (opener, reviewers, merger match the database), stats-1 (top 5 match `git shortlog`: 2209, 329, 234, 203, 65 commits), hist-2 (three newest commits match), pr-1 (description matches the PR body), pr-2 (PR 5797 is the charset_normalizer switch, but the answer is a bare "- 5797").
- **Accurate but incomplete or loosely relevant (3):** hist-1 lists the newest 8 of about 50 commits touching the function, in git's order, and does not say it is truncated; bug-1 cites 497695864b48, which is a 2012 HISTORY.rst line recording the fix, not the code change; bug-2 returns 8 bare SHAs, of which several do not touch proxy and timeout handling (docs wording, urllib3 vendoring), and gives no explanation.
- **Wrong or misleading (5):** people-1 and people-3 ("Nate Prewitt"): the path-based tools count commits only under the current path, which does not follow the 2023 move to `src/`, so on `src/requests/auth.py` Nate Prewitt has 4 commits, while with the old path included Kenneth Reitz has 33 (auth.py) and 118 (sessions.py); stats-2 lists the pre-move paths (`requests/models.py`, `test_requests.py`) for the same reason; issue-1 says no open bugs about timeouts, but the database holds 10 open issues that mention timeout (for example #7639, #7631); bug-3 names 470af31f4a5e (2013-11-24, "Handle 301s 'properly'"), which is already in v2.1.0, so it is not a v2.3 change (the v2.3.0 notes concern Authorization headers on redirects).
- **Unanswerable:** pr-3 (no review text in the data), abstained.
- **Causes, not yet fixed:** `top_contributors`, `change_frequency` and `count_commits` filter by the stored path and do not follow renames; `search_prs` returns the newest few issues and the model generalised from them; bug-3 was answered from a search hit without checking which release contains the commit. Phase 7 stays unticked.

### 2026-11-18 (later): fixes for the five wrong exact answers; tests pass, NOT yet measured on the live model

All inside the `history_grounded` variant, so the default fact set path is unchanged.
- **Renames (people-1, people-3, stats-2):** `git_tools.exact_renames` (100%-identical renames over the pinned history, 34 found on psf/requests in 0.3 s, `GIT_NO_LAZY_FETCH` so a blobless clone fetches nothing); `MetadataDB.count_commits`/`top_contributors` take `alt_paths` (older names of a file, counted together), `change_frequency` takes `renames` (counts under the latest name, a commit touching two names counts once); `ProductionTools._following` applies them. Checked on `data/metadata.db`: `src/requests/auth.py` top authors went from Nate Prewitt 4 to Kenneth Reitz 30, Nate Prewitt 9, Ian Cordasco 8 (raw `git shortlog` says Kenneth Reitz 33: the index holds fewer commits than git, so the ranking agrees and the counts do not); `sessions.py` Kenneth Reitz 97, Ian Cordasco 48, Cory Benfield 24; most-changed files `src/requests/models.py` 735, `tests/integration/test_requests.py` 368, `src/requests/sessions.py` 346. Limit: a file moved and edited in the same commit is not followed.
- **issue-1:** new `find_issues` tool (total matches plus the newest 10, no bodies). On the data, open issues with "timeout" in the title: 10 (7639, 7631, 7350, 5544, 5430, ...).
- **bug-3 and truncation:** prompt guidance only (`_PROD_HISTORY`): confirm with `releases_containing` which release first contains a commit before naming it as the change in a release; say when a `file_history` or `compare_refs` list is partial; use `find_issues` for open/closed issue questions. Prompt guidance is unmeasured until a live run.
- **Tests:** `tests/test_git_facts.py` (aliases, renames, scoped counts), `tests/test_git_tools.py` (exact vs edited renames), `tests/test_agent_guards.py` (find_issues, dispatch, tool listing); the 6 touched test files pass. Full suite after regenerating the 16 notebooks: one failure, `tests/test_retrieval_followup.py::test_v1_frozen_and_v2_question_mix_and_gold_review`, which opens `eval/retrieval_results/colab_full/phase4_v1_source.zip`, now under `experiments/eval/retrieval_results/` after the repository reorganisation (not from this work).
- **Next:** a live S1 and regression run on the 27B (needs a new Colab session), then re-check people-1, people-3, stats-2, issue-1 and bug-3 against git again. Phase 7 stays unticked. Nothing committed.

### 2026-11-18 (later): live run of the rename, issue and release fixes (27B, new Colab session)

Same setup as the earlier runs (`history_grounded`, snapping, abstain and tool-error retries 1, budget note on). Files: `eval/section1_results/section1_history_grounded_snap_abst1_toolerr1_20261118T114337.{md,json}` and `eval/agent_eval_results/agent_eval_production_history_grounded_extra3_notests_chars24000_in12288_snap_abst1_toolerr1_20261118T114949.json`.
- **S1: 4 of 32 flagged, 3 abstained** (pr-3 data limit, oos-1 correct, bug-3). Flags: explain-4 (docstring cites `models.py:654-670`, which shows none of the functions it names; it was unflagged in the previous run, so run-to-run variation), bug-3, nav-1, pr-3.
- **Re-checked against git and `metadata.db`:** people-1 and people-3 now answer Kenneth Reitz (git also ranks him first: 33 commits on auth.py, 118 on the sessions code); stats-2 now lists `src/requests/models.py` (735), `tests/integration/test_requests.py` (368), `src/requests/sessions.py` (346), identical to the database query; issue-1 now says 10 open issues about timeouts (#7639, #7631, #7350, #5544, #5430, #5263, #5185, #4914, #4590, #2392), correct for a title match (20 open issues mention timeout in title or body; only #2392 carries the Bug label). bug-2 now explains each commit (its claims were not verified). bug-3 now abstains instead of naming a wrong commit; the model never called `releases_containing`, so the prompt guidance did not change its behaviour. hist-1 still lists the newest 8 of about 50 commits without saying so; that prompt line had no effect.
- **Regression (93 questions):** code 40/40, docs 11/12 (docs-051), prsearch 10/10, history 11/11, refs 10/10, kind 5/5, probes 5/5 all abstained. No pass lost against the 27B baseline (code 39/40).
- **Left open:** bug-3 (abstains), hist-1 (partial list not disclosed), nav-1 and explain-4 (citation ranges), pr-3 (data). The 19 exact answers were not all re-checked after this run: the five fixed ones were. Phase 7 stays unticked; nothing committed.

### 2026-11-18 (later): `first_release` on commit hits and a partial-list note; live S1 and regression (27B)

Changes (only in `history_grounded`): `search_commits` hits carry `first_release` (earliest tag containing the commit, `unreleased`, or empty when git cannot say; about 0.3 s per hit); `file_history` asks git for one entry more than it shows and appends `{"note": "only the newest 8 changes are shown; older ones exist: say it is partial"}` when more exist; the prompt tells the model not to name a commit whose `first_release` predates the release asked about. 6 touched test files pass.
- **S1** (`section1_..._20261118T115543`): **1 of 32 flagged** (pr-3, data limit), 2 abstained (pr-3, oos-1). hist-1 now says the list is partial ("newest 8 ... older ones exist") and the commits still match git.
- **bug-3 is not fixed.** It now names 969195ad55f9, which git puts first in v2.3.0, but the commit is a 2014 `HISTORY.rst` line about a change from 1.2.1 ("301 and 302 redirects now change the verb to GET"), not a v2.3 change; the v2.3.0 notes list different redirect changes (Authorization and Proxy-Authorization no longer exposed on redirect, CVE-2014-1829 and -1830). The question is vague ("the redirect behaviour change in v2.3"). Recorded as a known weak answer.
- **Regression** (`..._20261118T120311`, 103 questions, includes the commit questions): code 40/40, docs 11/12 (docs-051), prsearch 10/10, history 11/11, refs 10/10, kind 5/5, probes 5/5 all abstained, **commit 5/10** (misses 079, 080, 085, 087, 088; 2 abstained, 1 failed), the same count as the 27B baseline (5/10, misses 079, 080, 083, 085, 088; 087 had flipped in earlier runs). No pass lost.
- Phase 7 stays unticked; nothing committed.


### 2026-11-18: Phase 7 closed with known gaps (user decision)

Ticked in `plan.md` with a status note listing the gaps (bug-3 weak, pr-3 unanswerable from the data, commit questions 5/10, judge not run on 27B answers, no human labels, code and docs answers checked only by the automatic citation checks, guards and `history_grounded` not default). Open items carried forward: decide whether to make `history_grounded`, snapping, the retries and the budget note the defaults (the regression rule was met with them on); run the judge on 27B answers; a person to label a sample; Phase 9.

### 2026-11-18: Phase 8 (API and UI) checked live, one stale caption fixed

Phase 8 was ticked by the user's decision (criterion reworded: build the index in Colab, import the zip, chat in the UI). No model server was running (the Colab session had been released), so the live check covers everything except a real answer.
- **Tests:** `tests/test_api.py`, `test_api_rerank_default.py`, `test_ui_api_client.py`, `test_ui_app.py`: 35 passed before and after the change below.
- **API** (`uv run uvicorn repo_rag.api.server:app`, run locally): `GET /health` 200; `GET /repos` lists `psf/requests` at `611c6162cb` with embedding `Qwen3-Embedding-0.6B_512`, chunks code 274, commits 4883, docs 282, issues 4186, pull_requests 3086, `searchable: true`; `POST /ask` for an unindexed repo gives 404, with no model configured gives 503 with the setup message, with an empty question gives 422; `POST /repos/import` with an empty body gives 400, with a non-zip body gives 400 ("can't open the upload"). The streaming `/ask` and a real import were not run against a live model or a Colab zip today (covered by the offline tests with scripted agents).
- **UI** (Streamlit on localhost, in the in-app browser, pointed at the API): the sidebar shows the API URL, a repository picker (`psf/requests`) and "Connected to the API: 1 repo(s) searchable"; with the API down it says "Could not reach the API" and how to start it; a question with no model configured ends in a collapsed "Failed" box whose expanded text is the 503 message (the reason is hidden until the box is opened).
- **Fixed:** the sidebar caption was hard-coded to "Qwen2.5-Coder-14B (provisional)"; it now names `settings.llm_model` (shows `Qwen3.8-27B-AWQ-INT4`). Verified in the browser.
- **Not verified:** a real answer with the sources panel and GitHub permalinks against the live 27B; the UI after a real `POST /repos/import`.
- The notebook-freshness tests fail while another session edits `src/repo_rag/history/metadata_db.py`; regenerate the 16 notebooks after that session settles. Nothing committed.

### 2026-11-18 (later): Phase 8 real answer through the API
- Ran the API against the live Colab 27B (A100, cuda-graphs + mtp3). POST /ask on psf/requests, 'How does Session.send decide which adapter to use?': stream start, prestep, answer, check, remember, result. The answer cites src/requests/sessions.py ranges for send, get_adapter, mount and __init__, plus 8 sources. The get_adapter range 831-886 is one chunk that also holds merge_environment_settings (get_adapter is at line 870), so the source label names the other function.
- Not run: the Streamlit UI rendering of this answer (covered only by the scripted-agent tests). Server stopped.
