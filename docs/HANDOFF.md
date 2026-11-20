# HANDOFF: read this first

For a new session (human or AI) picking up this project cold. It is meant to be enough to
understand what exists, why it is the way it is, what is true right now, and what to do next.
It points to the deeper documents rather than repeating them; the pointers are in §12.

- **Last verified:** 2026-11-01 for the Phase 7 v2 implementation and regenerated notebooks; historical
  checkpoints below retain their original dates.
- **How to keep it true:** when a phase finishes, a decision changes, or a gotcha is found,
  update §3 (status), §9 (pitfalls) and §11 (what's next). §14 lists commands that check this
  file's claims.
- Pronouns: refer to the user as "they". The user's git author name is `abhishek-paul-d`.

---

## 0. TL;DR

- **Phase 7 status (2026-11-18, latest):** 27B answer model; S1 section 1 is 1 of 32 flagged by the automatic checks, facts 59/60, regression held with the guards on (see PROJECT_LOG 2026-11-18 entries). A manual check of the 19 exact answers against git and `metadata.db` found 5 wrong or misleading (renames, issue filter, release check); fixes are written and unit-tested but **not yet run on the live model** (needs a new Colab session; plan in `docs/A100_RUN_PLAN.md`). **Phase 7 was closed with known gaps on 2026-11-18 by the user's decision** (see plan.md Phase 7 for the gaps); the next open work is Phase 9 and the carried-forward items in PROJECT_LOG 2026-11-18. Legacy bullets below are from earlier dates.
- **Phase 7 v4 dev revision 1 scored (2026-11-02):** `dev_1`/`smoke_1` are in
  `eval/facts_results/answer_bakeoff_v4_dev/`. Qwen3-14B is the only model with no failed dev gate
  (facts 67/71, citation-complete 14/16, 0 parse failures, probes 2/2). Coder fails answerable
  abstention (2/16), 4B fails fact exactness and probes, Olmo fails four gates. Test untouched; the
  open decision is `CONFIRM_FREEZE` on revision 1 vs a revision 2. Numbers in PROJECT_LOG 2026-11-02.
- **Phase 7 bake-off v4 (2026-11-01; dev revision 0 and 1 now run, see above; supersedes v3 for what to run):** v3 ran
  (`eval/facts_results/answer_bakeoff_v3/`): no model qualified, mostly format failures; see
  PROJECT_LOG. v4 (`eval/datasets/phase7_bakeoff_v4/`, `eval/prepare_answer_bakeoff_v4.py`, design in
  `docs/PHASE7_BAKEOFF_V4_DESIGN.md`) has 45 fresh test facts, 89 dev questions, a worked-example
  prompt and corrected-plus-strict fact scores. First run
  `uv run python eval/prepare_answer_bakeoff_v4.py verify`; never edit the v1-v4 preparers.
  Upload `notebooks/answer_bakeoff_v4_bundle.zip` to the top level of My Drive (no folder);
  Run all stops after dev. Review `dev_summary.json`, decide on prompt revision 1 or 2 (all four
  models again, new smoke), then set `CONFIRM_FREEZE = True`.
- **Phase 7 bake-off v3 (2026-11-01, superseded by v4 above; was the v2 line below for what to run):** review found
  35/500 httpx PR and 10/500 issue records exceed the 12,000-character tool-result budget (almost
  all `body`), so the model lost author/merged_by/merge_commit_sha for those (v2 dev audit 25/26).
  Protocol **v3** (`eval/datasets/phase7_bakeoff_v3/`, `eval/prepare_answer_bakeoff_v3.py`) differs
  from v2 only by cutting a PR/issue `body` over 2,000 characters; questions, labels, splits, gates,
  scoring, budget and models are identical. Now 0/500 each omitted. First run
  `uv run python eval/prepare_answer_bakeoff_v3.py verify` (verifies v1 and v2 too); never edit
  the v1/v2/v3 preparers; a further change needs a v4 directory. The notebook writes to Drive
  folder `repo_rag_answer_bakeoff_v3`. The notebook no longer embeds base64: upload
  `notebooks/answer_bakeoff_bundle.zip` to the top level of My Drive first (the
  notebook checks its sha256). Still no model run or measurement.
- **Phase 7 v2 bakeoff build (2026-11-01):** all four pinned candidates are approved.
  Harness, fake-model tests, redacted capsule, A100 notebook and raw-transcript analyzer are
  ready; read `docs/PHASE7_BAKEOFF_BUILD_REVIEW.md`. Use
  `eval/datasets/phase7_bakeoff_v2/`: **141 questions, 44 dev / 97 test, facts 26 / 45**.
  V1, both preparers, test questions and gates stay frozen; first run
  `uv run python eval/prepare_answer_bakeoff_v2.py verify`.
  The user runs `notebooks/answer_bakeoff_on_colab.ipynb` on **one A100 40 GB**. It logs
  a 14B question's peak memory before the full matrix, selects/freezes the common
  prompt/parser on dev, then unlocks test. Return `answer_bakeoff_results.zip`.
  **No models were loaded/downloaded here and no model measurements exist yet.**
  No dependency/default change; the separate summary experiment is untouched.
  All 17 notebook builders were regenerated; full validation is in PROJECT_LOG.
  Final checks: **638 tests passed** (JUnit **827.627 s**), Ruff clean, mypy clean for
  **95 source files**; both preparers verify and the notebook/source identity is fresh.
  Known limitation: the bounded-tool audit passes **25/26** dev facts; the full record for
  `issue_author:1428` exceeds v2's context budget and is omitted. V2 stays unchanged.

- **Current summaries build (2026-11-01):** E1/E2 from `docs/CODEX_HANDOFF_SUMMARIES.md` are complete.
  Q0 is frozen in `eval/datasets/architecture_v1/` (30 questions, labels **unreviewed by a person**).
  E1/E2 runner: `eval/run_architecture_eval.py`; raw results under
  `eval/facts_results/architecture_v1_{bm25,hybrid}/`. Multi-path routing and F1-F4 facts are opt-in.
  No variant meets the frozen rule. The selected router lowers architecture test Recall@5
  0.8292 -> 0.7042 and code MRR 0.8375 -> 0.7667. F1 wins the frozen dev tie order for E3.
  `summary_prose_on_colab.ipynb` and `summary_prose_inputs.zip` are ready; summary-model
  Qwen3-4B-Instruct-2507 was approved in the separate 2026-11-01 decision recorded in
  PROJECT_LOG; its notebook still has an explicit approval cell. No summary model has run.
  Review `docs/SUMMARIES_BUILD_REVIEW.md`.
  Summaries remain excluded from default code search; no staging/commit/push authorized.

- **Current commit-depth build (2026-10-26):** `docs/COMMIT_DEPTH_BUILD_REVIEW.md` is the
  implementation/review entry point for `docs/CODEX_HANDOFF_COMMIT_DEPTH.md`. Frozen one-pool
  depth50/100/200, dev-only BM25 weight grid, separate warm timing and original depth50
  controls are implemented. Use `phase5_commit_depth_on_colab.ipynb` plus its matching
  `phase5_commit_depth_source.zip`; new GPU quality/timing is pending. No defaults changed.
  Its final cell verifies the saved ZIP, flushes Drive and automatically releases the Colab
  runtime. Use the updated notebook; the source ZIP is unchanged. Results remain in My Drive.

- **Phase 5 review corrections (2026-10-26):** current inputs are `phase5_bugfix_v2`,
  with description-preserving scrub, same-file localization proxy, a 2,000-character BM25
  sensitivity row, small-stratum rule correction and email-redacted SQLite writes/reads.
  Use `notebooks/phase5_bugfix_v2_on_colab.ipynb` and `phase5_bugfix_v2_source.zip`.
  V1 inputs/results are unchanged historical evidence; the original v2 GPU run is now audited.
  Same 329 labels/splits, not a fresh holdout; own-fix lookup is not similar-fix validation.

- **Phase 5 build checkpoint (2026-10-25):** read `docs/PHASE5_BUILD_REVIEW.md` for the current
  implementation, commands, artifacts and limits. Commit ingestion/linking/SQLite/search and
  `ask --bug` evidence are implemented. 329 real linked-issue questions are frozen (Requests
  141, Flask 170, FastAPI 18); real BM25 preflight and old120-code regression are recorded.
  The original 4B/8B GPU comparison is audited; deeper comparison and manual scale remain open.
  Use the v2 notebook/Drive ZIP listed above. Bundled inputs need
  no GitHub token in Colab. Current validation is in the latest PROJECT_LOG entry; older test
  counts and 'not started' preparation notes below are historical.
- **Open-source models only (user decision):** `llm_provider=local`, answer/summary models
  unselected until Phase 7. Phase 5 needs no Anthropic key. Local GITHUB_TOKEN is configured
  and read-only GitHub lookups were authorized and verified; no credentials enter artifacts.

- **The project:** `repo-rag`, a RAG chatbot that indexes a GitHub repository and answers
  questions about it (architecture, explain code, match a bug report to past fix commits,
  history/people/PR/release questions, onboarding). The headline feature, and the user's
  resume angle, is **AST-based chunking with tree-sitter** instead of character chunking.
- **Stack:** Python 3.11+ (3.13 locally), `uv`, LangChain + LangGraph, ChromaDB, BM25,
  tree-sitter, local Qwen3 embeddings. Windows 11 laptop **without a GPU**; heavy jobs run on the
  user's Colab (A100 40 GB).
- **Done:** Phases 0 to 4, including the audited three-repo A100 follow-up and production
  decision. Last full offline suite: 596 passed, 1 network deselected (703.96 s);
  Ruff clean (283 formatted files), mypy 84 maintained files and 5 new evaluation scripts clean.
- **Current:** Phase 4 is closed. Keep plain opt-in reranking (GPU only), original headers, weights
  unchanged, no test penalty, no quota, no expansion (`evidence_expansion=none`). The fresh 30-question
  expansion test failed its frozen rule (see below). Labels everywhere are author-checked; independent
  review is pending. A100 is experiment-only; deployment undecided. **Commit depth/weight comparison is built; its new GPU assessment is pending.**
- **Rerank policy run (2026-10-25):** GPU run of tie-break and task-instruction policies found no
  reliable gain, so code reranking keeps plain P(yes) ordering and one instruction
  (`rerank_first_stage_weight=0`, `rerank_instruction_mode=default`). See `eval/RESULTS.md` and
  `eval/retrieval_results/phase4_policy_run/`. Remaining misses need structural fixes (chunk
  headers, evidence expansion).
- **Chunk header experiment (2026-10-25):** `Settings.header_defines` (off) adds a `# defines:` header
  line to chunks that merge definitions. GPU run was neutral (MRR -0.001 hybrid, +0.011 rerank; proofs
  slightly down), so it stays off. Results: `eval/retrieval_results/phase4_header_run/`.
- **Evidence expansion (2026-10-25): REJECTED.** On 30 fresh multi-block questions (incl. 12 Click)
  callee expansion lowered complete proof 0.667 -> 0.567 (-0.100 [-0.242, +0.034]); the earlier offline
  gain (+2 questions) did not generalise. Results: `eval/retrieval_results/phase4_expansion_run/`,
  rule and brief: `docs/CODEX_HANDOFF_EXPANSION.md`. Multi-block needs real call edges both ways
  (Phase 6 symbol table). Code stays, default `none`. No git commit or default change is authorized.
- **Reproduction audit:** eval/retrieval_results/phase4_expansion_full/ preserves compact
  chunks/traces, frozen source/labels, independently regenerated statistics, validation and
  case diagnoses. All 450 rows and all 45 source hashes/run match; all 1,787 chunks reproduce.
  Primary CI spans zero, so report failure of the adoption rule rather than universal harm.
  Concrete next implementation brief: docs/PHASE5_HANDOFF.md. No new expansion GPU run.
- **Reranker is now Qwen3-Reranker-8B (2026-10-25).** It beat the 4B on 150 questions (MRR 0.860 ->
  0.926, +0.066 [+0.026, +0.109]; rerank about +45% latency; GPU-only) and is the default for ALL
  reranking. The original Phase 5 commit run now measures both: the 8B gain is rank-1 only
  with unchanged Hit@5; the depth/weight extension is pending. Remaining misses: 19/150 not ranked first (10 test over implementation, mostly
  Express), 13 multi-block without a complete proof. Results: `eval/retrieval_results/phase4_judge_run/`.
- **Working split from 2026-10-25: Codex builds, Claude reviews and improves.** The user said all
  building happens in Codex. The next build is Phase 5; brief: `docs/CODEX_HANDOFF_PHASE5.md`
  (includes the review protocol). The earlier expansion brief is `docs/CODEX_HANDOFF_EXPANSION.md`.
- **Phase 5 review (2026-10-25):** Codex's build is in review, with ranked findings in
  `docs/PHASE5_REVIEW_FINDINGS.md`. The v2 checkpoint above addresses the scrub, 8B rule,
  secondary metric, refreeze and notebook before the GPU run.
- **Phase 5 GPU result reviewed (2026-10-26):** rerank 4B/8B hugely beat hybrid on commits (test MRR 0.677 ->
  0.823 / 0.847); 8B over 4B is small (+0.0245, rank-1 only). Remaining misses are first-stage recall
  (pool ceiling 0.91-0.96). Phase 5 stays unticked until the manual scale tier and the data/CLI items in
  `docs/PHASE5_REVIEW_FINDINGS.md` (Round 3) are done.
- **Agent toolset (built 2026-11-04, NOT bake-off tested):** `Settings.agent_toolset` = `production` (default) adds `search_docs`, `search_commits`,
  `get_references` (written #N mentions, both directions) and `find_callees` to the evaluated 13 tools and uses `PRODUCTION_PROMPT`; `evaluated` is the
  measured setup byte for byte. psf/requests PRs (3,086) and issues (4,186) are synced into `data/metadata.db`; **#27 is an issue, not a PR**. Commit search
  needs commits embedded: re-run `notebooks/index_on_colab.ipynb` with `SHALLOW_CLONE = False`, `INDEX_HISTORY = True` and `PIN_COMMIT = "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60"` (upstream has moved on;
  pinning keeps the index, the local clone and the golden line ranges consistent), then `repo-rag import-index` and `repo-rag index-history psf/requests --checkout data/repos/psf__requests --offline` (not done yet).
  Commit indexing is now batch-prefetched (about 13x faster per commit in a 60-commit local measurement, see PROJECT_LOG 2026-11-05); the Colab indexer notebook has `EMBED_BATCH`/`WRITE_BATCH`.
  Measure it: `uv run python eval/agent_eval.py run --url ... --token ...` (93 questions, exploratory, no gates). Not run yet.
- **Streamlit UI with a remote model (built 2026-11-04, UI not yet run against a real GPU server):** `ui/app.py` (laptop: retrieval, tools, citation
  check) calls `agent/remote_llm.py` `RemoteChatModel`, served by `notebooks/serve_model_on_colab.ipynb` (Drive: `repo_rag_ask_source.zip` only; stdlib HTTP
  server behind a pinned, hash-checked cloudflared quick tunnel, random bearer token). Run: `uv run streamlit run src/repo_rag/ui/app.py`, paste URL + token.
  Public repos only (prompts cross a public tunnel). `streamlit` is a project dependency but excluded from Colab pins (`UI_ONLY` in `build_colab_indexer.py`).
- **Phase 7 Colab ask path (2026-11-03) ran once on a GPU 2026-11-04:** the model load, tool loop and citation check work (`agent/local_llm.py`,
  `agent/runtime.py`, `notebooks/ask_on_colab.ipynb`; Drive: `repo_rag_ask_source.zip`, `repo_rag_index.zip`). Observed: PR questions failed because no PR data is
  indexed (the reason is now reported), and an architecture question abstained after one search (the known v5 failure; use `peek()` to see whether retrieval or
  the model is at fault). Still to do: end-to-end eval (golden set, 90% fact set).
- **Phase 7 agent core built (2026-11-03):** `agent/graph.py` `RepoAgent` (LangGraph) + `agent/citations.py`; see PROJECT_LOG. Needs: a real model adapter
  (deployment decision: GPU/quantized/hosted), `repo-rag ask`, router only with evidence, end-to-end eval incl. the 90% fact-set target.
- **Answer model (provisional, 2026-11-03):** Coder-14B, prompt arm 2, config defaults set; use `agent/citations.py` to validate/repair citations
  before returning an answer. Next: build the LangGraph agent (Phase 7) and evaluate it end to end on fresh questions.
- **Phase 7 v5 RUN (2026-11-03):** tuned Coder-14B (revision 2 prompt) passes facts 45/45, probes 14/16, abstentions 4/45, citations 37/45 but fails
  invalid citations 5/90 (limit 4); Qwen3-14B fails probes (5/16). Results: `eval/facts_results/answer_bakeoff_v5/`; PROJECT_LOG has the table. Decision open.
- (built, now run) **Phase 7 v5 BUILT (2026-11-03):** tuned-prompt test for Coder-14B on fresh questions; upload `notebooks/answer_bakeoff_v5_bundle.zip`,
  run `notebooks/answer_bakeoff_on_colab.ipynb` (design: `docs/PHASE7_BAKEOFF_V5_DESIGN.md` section 10). Source is fixed until after the test run.
- **Phase 7 v4 test DONE (2026-11-03):** no candidate qualifies; Qwen2.5-Coder-14B fails only answerable abstentions (6/44 vs <=4),
  Qwen3-14B fails probes (5/8) and abstentions (5/44). Results in `eval/facts_results/answer_bakeoff_v4_test/`; PROJECT_LOG has the table.
  Next step is the user's decision (provisional model for the agent vs a v5 test with fresh questions). No default changed.
- **Phase 5 depth run FINISHED (2026-11-02):** rule says keep depth 50 (100/200 fail Hit@5 gain), weight stays 0.3,
  timing and CLI sample recorded; results in `eval/commit_depth_results/`, see PROJECT_LOG. Remaining for Phase 5: independent
  labels and similar-fix usefulness review.
- (done) **Phase 5 depth run: resume notebook (2026-11-02):** `notebooks/phase5_commit_depth_resume_on_colab.ipynb` +
  `phase5_commit_depth_resume_source.zip` continue from `phase5_commit_depth_checkpoint_869432f8c494.zip`
  (weights, depth and selected stages done; timing, analysis and CLI sample left). Both go at the top level of My Drive.
- **Open task for Codex (2026-10-26):** pre-registered commit reranking depth (100/200 vs 50) and BM25 weight
  test is built: `docs/COMMIT_DEPTH_BUILD_REVIEW.md`. Defaults stay unchanged until it is run and reviewed.
- **Commit depth/weight checkpoint reviewed (2026-10-27):** depth 100/200 fail the pre-registered rule (keep 50);
  BM25 weight stays 0.3 for reranked search. Hypothesis to confirm: `retrieval_k=200` with rerank depth 50 (+0.015
  Hit@5, post hoc). Run is incomplete (timing for Flask/FastAPI, aggregate and decision file). See Round 4 in
  `docs/PHASE5_REVIEW_FINDINGS.md`.
- **Open task for Codex (2026-10-27):** finish the depth/weight output and run a pre-registered confirmation of
  `retrieval_k=200` (rerank depth 50) on a fresh holdout. Brief: `docs/CODEX_HANDOFF_COMMIT_CONFIRM.md`. Defaults unchanged.
- **Confirmation built (2026-10-27):** `notebooks/phase5_commit_confirm_on_colab.ipynb` + `phase5_commit_confirm_source.zip` test `retrieval_k=200` vs 50/100 for commits on 342 fresh questions (4 repos) plus v2 dev. Rule frozen in `eval/datasets/phase5_commit_confirm_v1/protocol.json`. Run and audited (see next bullet). Open: Sphinx-anchor scrub gap (`.. _#N:`) in `src` (2 questions excluded).
- **Confirmation result (2026-10-27, quality complete, timing Click-only):** `retrieval_k=200` vs 50 on 342 fresh questions: Hit@5 +0.0117 [0.0000, +0.0264] (about 4 questions), MRR +0.0086 [-0.0008, +0.0204]; dev set 0 change. Rule outcome `eligible_pending_review` but marginal; default unchanged. Fixed my analyzer's one-group bootstrap bug on the way. Details: PROJECT_LOG 2026-10-27 audit entry, `eval/retrieval_results/phase5_commit_confirm_run/`.
- **Phase 5b started (2026-10-28):** `history/git_tools.py` (blame, file/function history, diff_between, releases_containing) and `retrieval/exact_refs.py` are built and tested (36 tests; full suite green after regenerating all notebooks). The GitHub PR/review/issue sync (`history/github_sync.py`, `repo-rag sync-github`, `show-pr`, `metadata.db` PR/review/issue tables and typed getters) is built, tested and live-checked on encode/httpx (read-only, authorized). Also built offline: tags + `.mailmap`/login identity merging (`git_facts.py`, `repo-rag index-facts`) and the exact commit tools `find_commits`, `count_commits`, `top_contributors`, `change_frequency` (CLI `contributors`, `hot-files`), all returning the history window they cover. **Phase 5b is done (2026-10-28):** `pull_requests`/`issues` collections (`repo-rag index-prs`, `search --kind pull_requests --state MERGED`) and the auto-generated fact set (`eval/datasets/httpx_facts.jsonl`, 60 facts, expected answers from raw sync nodes and plain git): **60/60 pass through the tools alone** (`eval/facts_results/httpx/`). Issue->fixing-PR on httpx (45 pairs, GPU run 2026-10-29, `eval/facts_results/httpx_pr_search_gpu/`): Hit@5 BM25 0.600 [0.467, 0.733], hybrid 0.800 [0.667, 0.911]; intervals overlap. `hybrid_fused` (full + title query) Hit@5 0.867, Hit@10 0.956. Second repo pallets/flask (58 pairs, `eval/facts_results/flask_pr_search_gpu/`): Hit@5 bm25 0.793, hybrid 0.914, fused 0.931; fused mainly lifts Hit@1/MRR there. Phase 7 reruns the fact set through the agent (target 90%). Regenerating notebooks changed the local depth ZIP hash; the Drive copies/checkpoint `869432f8c494` are the ones to resume.
- **Not done:** general LLM answering, GPU embedding of PRs/issues, agent wrappers for the exact tools,
  the API and UI. `repo-rag ask --bug` returns commit evidence; general `ask` remains Phase 7.
- **Three things that will bite you immediately** (details in §1): nothing has ever been
  committed to git; `AGENTS.md` is stale and wrong; the user has said "don't commit yet" every
  time it came up.

---

## 1. Read these warnings before touching anything

1. **`main` has no commits.** `git log` fails. 24 files are staged (an old snapshot), 10 are
   modified after staging, and about 76 are untracked: everything from Phase 2 on exists only in
   the working folder. Losing the folder loses the project. **Do not commit unless the user asks.**
   They have said "don't commit yet" repeatedly and have not asked for a commit. Offer it (§11
   has a suggested grouping) and wait.
2. **`AGENTS.md` records older stack choices.** It names `voyage-code-3`, the non-commercial
   jina fallback, bge reranking and `EnsembleRetriever`; current measured implementation choices
   are described in §5 and config.py. Follow the active user/developer instructions and
   applicable repository rules; this handoff does not override them. Leave AGENTS.md unchanged
   unless the user requests an update.
3. **A stray folder `D:\Users\` exists** (outside the repo), created by a path typo (`..\..` in a
   file path). It contains one leftover script inside empty folders. The sandbox refused to let
   the previous session delete it, so the user must. Never write there.
4. **Do not quote performance numbers we haven't measured.** `CLAUDE.md` requires every claim
   about AST chunking to be backed by `eval/`. Several earlier statements in this project were
   corrected after measurement (§9). When unsure, say "not measured".
5. **GITHUB_TOKEN is configured in the local `.env`.** Read-only GitHub lookups were authorized
   for this work. Never display or bundle it, or put it in a URL, notebook cell or commit.
   No Anthropic key is required; Phase 7 will use an open-source answer model.

---

## 2. What the project is, and why

**The user's original brief:** "Code-Aware GitHub Repository RAG Assistant": a developer
companion that answers codebase architecture questions, explains legacy functions, and matches
incoming bug reports to past Git commit fixes. Original stack idea: tree-sitter, ChromaDB or
LanceDB, LlamaIndex, GitHub API. **Resume highlight: AST parsing instead of arbitrary character
chunking to keep functions and classes intact, improving retrieval precision.** The user asked for
**LangChain** (not LlamaIndex) plus "other necessary packages", asked for prior art to reference,
and asked for `CLAUDE.md` and `plan.md` first.

**Question types it must serve** (`plan.md` §1 has about 40 example questions):

| Kind | Example | Answered by |
|---|---|---|
| Meaning | "How does auth flow from the API to the DB?", "What does `X` do?" | hybrid dense + BM25 search over code/docs, then an LLM |
| Bug to past fix | "`KeyError: 'charset'` on upload, fixed before?" | search over an index of *commits* (+ linked issue text), reranked |
| Exact facts | "Who merged #1234?", "What changed between v2.30 and v2.31?", "Top contributors to `auth/`" | **not** similarity search: `metadata.db` (SQLite) and git tools, called through typed functions |
| Onboarding | "How do I run the tests?" | docs/config chunks |

Two rules follow from that split: **the LLM never writes SQL**, and **names, dates and counts in
answers must come from tool output, not from the model**. "Who pushed" is not recorded by git:
report author, committer and PR `merged_by`, and say so.

**Prior art we built on** (full table in `plan.md` §2): the cAST paper (Zhang et al., EMNLP
Findings 2025, arXiv 2506.15655, code `yilinjz/astchunk`, which reports +4.3 Recall@5 over
line-based chunking on RepoEval), `codebase-qa-agent` (closest end-to-end project),
`RepoRAGX` (LangChain + Chroma baseline we aim to beat), LangChain's `LanguageParser`.
We reimplemented cAST ourselves because `astchunk` covers only 4 languages and gives no symbol
metadata.

---

## 3. Status

| Phase | State | Notes |
|---|---|---|
| 0 Scaffolding | done | uv project, pydantic-settings config, Typer CLI, ruff/mypy/pytest |
| 1 Ingestion | done | blobless clone (full history, files on demand), `git ls-files` walker, code/doc/config tagging |
| Embedding + rerank benchmark | done | 4 Colab runs; decisions in §5 and `eval/RESULTS.md` |
| 2 AST chunker | done | 6 languages + docs chunker; verified on 7 real repos |
| 3 Store + baseline retrieval | done | Chroma + BM25 + weighted-RRF fusion; acceptance query passes |
| Colab GPU indexing + index transfer | done | notebook + `export-index`/`import-index`; one real run verified |
| **4 Evaluation harness** | **evaluation complete; default pending** |90-case and120-case policy/header assessments recorded; fresh30+old 120 expansion GPU test prepared, independent review pending |
| 5 Commit history + bug-to-fix | implementation/preflight ready; GPU pending | 329 frozen questions; docs/PHASE5_BUILD_REVIEW.md |
| 5b PRs, issues, people, releases, `metadata.db` | done | 60/60 facts through tools; `plan.md` §3.7 to 3.9 |
| 6 Symbol table + repo summaries | **done 2026-11-18** (ticked by the user's decision) | Architecture-answer gate met with the 27B (vLLM, CUDA graphs + MTP): 13/15 and fresh 8/10 (proxy metric, labels not human-reviewed, fresh set no margin); defaults are now the 27B profile (extra files 3, skip tests, 24,000 chars, 12,288 input tokens); E1/E2 audited, no summary change qualifies; human label review still open |
| 7 LangGraph agent (5 routes) | v2 bakeoff built; user Colab run pending | Approved four-model harness/notebook/analyzer; no model measurements or full agent yet; docs/PHASE7_BAKEOFF_BUILD_REVIEW.md |
| 8 FastAPI + Streamlit | not started | |
| 9 Polish/stretch | not started | incremental re-index, MCP server, LanceDB, live GitHub tools |

**What works today** (all from the CLI, `uv run repo-rag ...`):

| Command | Does |
|---|---|
| `index <url> [--no-history] [--force]` | clone, chunk, embed new/changed chunks, build BM25, write the manifest |
| `search "<q>" --repo owner/name [-k N --kind code\|docs --language --symbol-kind --path --rerank]` | hybrid search, no LLM, prints ranked chunks with `path:start-end` |
| `export-index <zip> [--repo ...]` / `import-index <zip>` | move an index between machines (§8) |
| `config` | show effective settings (secrets masked) |
| `index-history owner/name --checkout PATH [--github]` | bounded commit index and typed issue links |
| `ask --bug --repo owner/name "report"` | retrieved past-fix candidates with commit citations; no LLM |

**Does not exist yet** (the layout in `CLAUDE.md` is a *target*; check the real tree):
`retrieval/query_rewrite.py`, `agent/graph.py`, `agent/tools.py`, `api/server.py`, `ui/app.py`.
The symbols, summaries and exact-history modules listed in older versions of this paragraph
are now implemented. `agent/prompts.py` currently holds the E3 facts-only summary prompt;
the general answer agent is still Phase 7.
Phase 4 now adds `eval/run_retrieval_eval.py`, `eval/retrieval/`, `eval/datasets/`,
`eval/build_qa_datasets.py`, and `notebooks/build_retrieval_eval.py`.

**Tests:** latest full offline run **596 passed, 1 deselected in 703.96 s**. Ruff lint/format
checks are clean; mypy checks **84 maintained files plus 5 new evaluation entrypoints**.
The sandbox blocks Git's local `file://` helper; the complete offline suite passes in the
normal Windows environment without skipping those tests. Phase 5 adds real-git/history/link and
read-only database integration, leakage and Unicode checks, history-only transfer preservation,
the complete offline fake-model comparison/audit, and exact Drive-ZIP input preservation.
All 120 old code BM25 rankings/metrics reproduce. Prior Phase 4 GPU/replay evidence is
preserved; no new neural score is implied by these offline checks.

---

## 4. Architecture and code map

```
INDEX    repo-rag index <url>
  clone (GitPython, blobless)  ->  file_walker (git ls-files; drops vendor/lock/min/generated/
  binary/>1MB; tags each file code | doc | config)
     code   -> AstChunker (tree-sitter cAST split-then-merge; text-splitter fallback)  -> "code" chunks
     doc/config -> docs_chunker (verbatim heading sections; small configs stay whole)   -> "docs" chunks
  -> VectorStore.sync(): embed ONLY new/changed chunks (Qwen3-0.6B, 512 dims), delete vanished
     ones, update metadata-only changes without re-embedding   -> Chroma (HNSW, cosine)
  -> Bm25Index.build() per (repo, kind) -> pickle ;  Manifest.put() -> data/manifest.json

QUERY    repo-rag search "<q>"
  embed query (task instruction)  ->  dense: Chroma HNSW nearest chunks, filtered by repo
                                  ->  keyword: BM25 over identifiers (camelCase/snake_case split)
  -> weighted RRF  (dense 1.0, bm25 0.3 [PROVISIONAL], k 60)  -> optional rerank (GPU only, opt-in)
  -> top-k chunks with path:start-end

MOVE     export-index -> zip of plain data (jsonl + float32 npy), NOT Chroma files
         import-index -> replace the named repos' chunks (re-embeds nothing), rebuild BM25
         notebooks/index_on_colab.ipynb runs "index" + "export-index" on a Colab GPU
```

HNSW (approximate nearest neighbours) is only the dense half. BM25, the reranker and the future
exact-answer tools do not use it. We did not choose HNSW; it is Chroma's default. The effect of
its approximation on recall has **not been measured** (the benchmark used exact search).

**Source map** (`src/repo_rag/`):

| File | Purpose |
|---|---|
| `config.py` | every setting and default (pydantic-settings, env prefix `REPO_RAG_`). No magic numbers elsewhere |
| `cli.py` | Typer: index, index-history, search, export/import-index, ask --bug, config; general ask pending |
| `ingest/github_client.py` | parse GitHub URLs, blobless clone into `data/repos/<owner>__<repo>`, fast-forward existing clones |
| `ingest/file_walker.py` | which files to index and as what; skip lists live here |
| `ingest/indexer.py` | `chunk_repo`, `index_repo`: the whole index pipeline |
| `chunking/languages.py` | extension to language map; per-language tree-sitter node-type config for python, javascript, typescript, tsx, java, go |
| `chunking/ast_chunker.py` | the main feature: cAST chunker (see below) |
| `chunking/fallback.py` | LangChain `RecursiveCharacterTextSplitter.from_language` wrapper |
| `chunking/docs_chunker.py` | own Markdown/RST heading scanner (LangChain's rewrites whitespace) |
| `chunking/common.py` | the **only** place chunk `Document`s, metadata, ids and the contextual header are built |
| `store/embeddings.py` | `LocalEmbeddings` (LangChain `Embeddings` over sentence-transformers): instruction, left padding, 2048-token cap, Matryoshka truncation |
| `store/vectorstore.py` | Chroma collections named `<kind>__<embedding_id>`; `sync`, `search`, `export_records`, `replace_records` |
| `store/bm25.py` | code-aware tokenizer + persisted BM25 |
| `store/manifest.py` | `data/manifest.json`: which repos, which commit, which embedding |
| `store/transfer.py` | export/import format |
| `retrieval/retrievers.py` | `HybridRetriever`, `weighted_rrf`, filters |
| `retrieval/rerank.py` | Qwen3-Reranker per its model card; `reranking_enabled()` |

**How the AST chunker works** (`ast_chunker.py`): parse with tree-sitter; fall back to the text
splitter if more than 10% of bytes are parse errors, the language is unknown, or anything raises
(it never raises). *Split:* an oversized node is replaced by its children, recursively; size is
non-whitespace characters, budget 1500. *Merge:* greedily join neighbours **only when they share
the same enclosing definition** (a class header stays with its first methods; a method never
merges with the next top-level function). Comments/decorators attach to the definition after
them, only as far as the budget allows. Chunks are whole source lines, verbatim, never overlapping.
Each chunk carries a qualified `symbol` (`Class.method`, Go `Store.Get`), `symbol_kind`,
`parent_symbol`, `signature`, and `symbols` (every definition inside it). The text embedded starts
with a header: `# path:`, `# in class:`, `# signature:`.

**Chunk metadata (flat scalars, Chroma-compatible):** `repo, commit_sha, path, language,
start_line, end_line, symbol, symbol_kind, parent_symbol, signature, symbols, chunk_id, chunker`
(`ast|fallback|docs`), plus `doc_kind` for docs. `chunk_id = sha1(repo:path:symbol:start_line)`
(fallback/docs pieces use an index instead of the symbol). Re-indexing is idempotent.

---

## 5. Decisions and the evidence behind them

All numbers are measured unless marked otherwise. nDCG@10 / Hit@5 explained in §13.

| Decision | Evidence | Where |
|---|---|---|
| **Embedder: `Qwen/Qwen3-Embedding-0.6B` @ 512 dims**, one query instruction for all collections | Real tasks (mean of T2r + T3 nDCG@10): Qwen3 **0.612**, nomic-embed-code 7B + BM25 0.550, 7B alone 0.513, CodeRankEmbed 0.426. Qwen3 wins questions (T2r 0.736 vs 0.431 / 0.416) but trails the 7B on bug-to-fix (T3 0.488 vs 0.596). 512 dims = 1024 dims (T2r 0.738 vs 0.736). The 7B needs a GPU at query time, a 3.5x bigger index and ~4x slower indexing | `eval/RESULTS.md` |
| `jina-code-embeddings` excluded | licence is CC-BY-NC | `eval/RESULTS.md` |
| **Reranker: `Qwen/Qwen3-Reranker-4B` over the top 50, GPU only** (`reranker_mode=auto`) | bug-to-fix T3 nDCG@10 **0.494 to 0.687** (+0.192, 95% CI [+0.133, +0.251]), beating the 7B embedder (0.596). 1.83 s per query on an A100. A bigger embedder adds only +0.004 once reranked. **`bge-reranker-v2-m3` made it worse (0.373, -0.121)** | `eval/RESULTS.md` |
| Reranking is **off by default for code search** | initial 20/50 code runs now exist; held-out improvement intervals span zero, so keep opt-in | `retrieval/retrievers.py`, `eval/RESULTS.md` |
| Commit docs = message + files + touched symbols + 800 chars of diff | T3 tuning grid (5 instructions x 4 formats, dev/test split): best combo +0.017 on test, CI [-0.017, +0.050] = not significant; 800 chars matches 2000 and indexes 1.4x faster | `eval/RESULTS.md` |
| **Always filter retrieval by repo** | +0.13 to +0.18 nDCG@10 for every model | `eval/RESULTS.md` |
| Fusion = **weighted** RRF, BM25 weight **0.3 (provisional, untuned)** | equal-weight RRF *hurt* Qwen3 (mean 0.612 to 0.585). On 36 questions BM25 trades top-1 for top-5: Hit@1 0.667 to 0.556, Hit@5 0.944 to 0.972. Too few questions to tune: Phase 4 | `docs/PROJECT_LOG.md` |
| Own BM25 + fusion, not `EnsembleRetriever` | need metadata filters on both signals, a persisted index, tunable weights | `CLAUDE.md` |
| Own `LocalEmbeddings`, not `langchain-huggingface` | exact control of instruction/padding/truncation as benchmarked | `CLAUDE.md` |
| Chroma `hnsw.sync_threshold=2` | **a Chroma 1.5.9 defect**; see §9 | `config.py` |
| Indexes move as plain data, not Chroma files | HNSW binaries differ across OS/versions | `store/transfer.py` |
| Build GPU indexes in float32 | matches CPU laptop queries; bf16 differs by cosine >= 0.9998, same top-1 36/36 (CPU stand-in), so bf16 is a safe speed-up | `docs/PROJECT_LOG.md` |
| vLLM deferred | only affects indexing speed; matters only if the 7B model had won | `docs/PROJECT_LOG.md` |

**AST chunker verification** (`eval/chunker_check.py`, results in `eval/chunker_check_results.md`):
**0 of 12,309 definitions that fit the budget were split**, across 7 real repos
(psf/requests, pallets/flask, encode/httpx, expressjs/express, trpc/trpc, google/gson,
spf13/cobra) and the 6 AST languages. 0 fallbacks, 0 lost lines, 0 overlaps, 0 duplicate ids.
31 of 5,425 chunks (0.6%) exceed the budget (unsplittable: a giant line, Go table-test rows).
All 57 gold symbols of the 36 benchmark questions land in a chunk that names them.

**Pipeline sanity check** (36 benchmark questions, top 5, psf/requests): gold found 34/36 dense
only, 35/36 hybrid. **Not a like-for-like comparison with the benchmark** (our chunks merge small
definitions, so one chunk can hold several gold symbols) and **not evidence that AST beats
character chunking.**

---

## 6. Performance facts (measured on the user's laptop unless stated)

| Fact | Value |
|---|---|
| CPU embedding speed (Qwen3-0.6B, fp32) | **0.16 code chunks/s (about 6 s per chunk)**, 0.33 doc chunks/s |
| Indexing psf/requests on CPU | 556 chunks (274 code, 282 docs): **36.5 min**, including a one-off 1.2 GB model download |
| One search query on CPU | about **0.6 s** (609 ms measured; 269 ms on a Colab CPU) |
| GPU indexing | A100, bf16: **239 to 311 docs/s** (embedding benchmark). fp32 on a GPU and the full Colab notebook run are **not timed** |
| Chunking a whole repo | 0.3 to 0.5 s (requests, flask, httpx) |
| Rerank latency | Qwen3-Reranker-4B 1.83 s/query top-50, 0.6B 0.58 s, bge 0.16 s (A100); 0.6B reranker on this laptop's CPU: about 23 s for 5 candidates, i.e. unusable |
| Export size | 556 chunks (text + metadata + 512-d vectors) = 1.24 MB |

**Consequence:** large repos must be indexed on a GPU. Searching is fine on a CPU.

**Colab-built vs CPU-built index, same repo and commit:** all 556 chunk ids/texts/metadata
identical; vector cosine >= 0.9999998; 36/36 benchmark questions returned identical top-8 results.

---

## 7. Environment and how to work here

- **Machine:** Windows 11 Home, **no GPU** (torch is the CPU build `2.14.1+cpu`), Python 3.13.14,
  `uv` 0.11.33, git 2.47.1. Repo at `D:\Repo_chatbot`. Two shells are available: **Git Bash**
  (paths like `/d/Repo_chatbot`) and PowerShell. `uv run ...` for everything.
- **Commands** (also in `CLAUDE.md`):
  ```bash
  uv sync
  uv run pytest                      # ~80 s, offline. DO NOT add -q: pyproject already sets it and a second -q hides the summary
  uv run pytest -m network           # the one test that clones from GitHub
  uv run ruff check --fix . && uv run ruff format . && uv run mypy
  uv run python notebooks/build_colab_indexer.py     # regenerate the Colab indexing notebook (LAST, after formatting)
  uv run python eval/chunker_check.py psf/requests=data/repos/psf__requests
  ```
- **pytest in a sandbox:** if the system temp folder is not writable (it wasn't in earlier
  sessions: `PermissionError` creating `tmp_path`), add
  `--basetemp=<a writable folder>` and `-p no:cacheprovider`.
- **Unicode output:** set `PYTHONIOENCODING=utf-8` when printing rich panels or the benchmark text
  to a Windows console, or you get `cp1252` encode errors.
- **Hugging Face cache** (`~/.cache/huggingface`): `Qwen3-Embedding-0.6B` (1.2 GB) is present, so
  searching works offline. `Qwen3-Reranker-4B` is **not** downloaded (it needs a GPU anyway);
  `Qwen3-Reranker-0.6B`, CodeRankEmbed and MiniLM are cached from experiments. HF warns about
  symlinks and a missing `hf_xet` on every load: harmless.
- **tree-sitter grammars** download on first use into a per-user cache (already populated here);
  a fresh machine needs network once.
- **The user's Colab:** A100 40 GB. The pattern is: we generate a notebook, the user runs it, and
  uploads a `results.zip` (benchmarks) or `repo_rag_index.zip` (indexes) back. They add secrets in
  Colab's key panel with "Notebook access" on; the benchmark notebooks need `GITHUB_TOKEN` for the
  bug-to-fix task (T3).
- **Config defaults worth knowing** (`uv run repo-rag config`): `max_chunk_chars=1500`,
  `embedding_dims=512`, `retrieval_k=50`, `rerank_top_n=8`, `dense_weight=1.0`,
  `bm25_weight=0.3`, `rrf_k=60`, `chroma_sync_threshold=2`, `reranker_mode=auto`,
  `embedding_dtype=auto` (bf16 on GPU, fp32 on CPU), `compute_device=auto`.

---

## 8. Data and artifacts on disk

| Path | What | In git? |
|---|---|---|
| `data/` | local index: `chroma/`, `bm25/`, `manifest.json`, `repos/psf__requests` (clone) | ignored (`data/` is in `.gitignore`) |
| `data/manifest.json` | one repo: `psf/requests` @ `611c6162cbc4`, 274 code + 282 docs chunks, embedding `Qwen3-Embedding-0.6B_512`. **This index was built on Colab and imported**, replacing the CPU-built one | ignored |
| `eval/retrieval_results/phase4_v2_full/` | audited v2 scores, traces, provenance, frozen source and production decision; raw chunks/vectors in ignored data/phase4_followup_colab | untracked; preserved in working folder |
| `eval/datasets/phase4_comparison_v3/` | frozen 90 questions, readable QUESTIONS.md, evidence overlap manifest and pre-chosen depth protocol | untracked; preserved in working folder |
| `notebooks/phase4_comparison_on_colab.ipynb` + `phase4_comparison_source.zip` | earlier90-case depth workflow; later audit is recorded in PROJECT_LOG | untracked; preserve evaluated snapshots separately |
| `notebooks/phase4_expansion_on_colab.ipynb` + `phase4_expansion_source.zip` | current next-run pair:30 fresh multi-block+120 old, none/callee/both after rerank20 | untracked; GPU decision pending |
| `eval/datasets/phase4_expansion_v1/` | frozen fresh questions, unchanged old 120 copies, protocol/freeze digests, readable QUESTIONS.md | untracked; independent review pending |
| `eval/retrieval_results/phase4_expansion_preflight/` | actual BM25150 and default-off production replay120, not new neural scores | untracked; preserved locally |
| `eval/retrieval_results/comparison_v3_preflight/` | actual 90-case BM25 baseline and unchanged original-15 regression evidence | untracked; not GPU results |
| `eval/embedding_bench/` | benchmark code, generated notebooks, and result folders `results/` (run 2), `results_run1/`, `t3_results/`, `rerank_results/` | untracked, small |
| `notebooks/index_on_colab.ipynb` | **generated** (do not hand-edit); embeds `src/repo_rag` and the `uv.lock` versions | untracked |
| scratchpad (session-temporary) | a backup of the CPU-built index (`local_cpu_backup.zip`) and many stress scripts. **Ephemeral; do not rely on it** | n/a |

**Index transfer in practice:** run the notebook on Colab (set `REPO_URLS`; public repos only),
download `repo_rag_index.zip`, then `uv run repo-rag import-index <zip>`. Import replaces only the
named repos, re-embeds nothing, rebuilds BM25, and **refuses an index built with a different
embedding model/dims** (the embedding id is part of the collection name and the manifest).

**Benchmark repos and pinned SHAs** (embedding benchmark corpus): psf/requests `611c6162`,
pallets/flask `d086db85`, encode/httpx `b5addb64`. The real-repo chunker check also used
expressjs/express `0dee2c52`, trpc/trpc `1c9bf51e`, google/gson `e2292f9c`, spf13/cobra `adbc8813`.

**Test-repo plan for later phases** (`plan.md` Phase 5): requests (main), flask (second),
fastapi (tests the commit-to-PR-to-issue linking path: only 8 of 7,903 commits say "fixes #N"
because it squash-merges), langchain (AI showcase and scale test; 491 direct links, 17k commits),
express (second language). Issue-link counts (commits that explicitly close an issue): requests
182, flask 215, fastapi 8, langchain 491, llama_index 56, transformers 316.

---

## 9. Pitfalls and lessons (each one cost real time)

**27B serving speed (2026-11-18):** the answer model is Qwen3.8-27B on vLLM
(`notebooks/serve_model_27b_on_colab.ipynb`). The notebook now defaults to CUDA graphs + MTP (3 draft tokens)
with `--max-num-seqs 8`: about 4 s per question vs 23 s in eager mode (decoding about 45 to 52 vs 7 tokens/s);
every gate was still met (PROJECT_LOG 2026-11-17/18 and the decision-log row). Eager is `ENFORCE_EAGER = True`,
`MTP_TOKENS = 0`. Lessons: CUDA graphs without `--max-num-seqs` fail at startup (`max_num_seqs (256) exceeds
available Mamba cache blocks (179)`; eager skips the check); a failed start now prints a `root cause:` line;
startup compiles for 4 to 6 minutes; the health line shows `decode_mode`, so check it before running anything.
Colab: the GPU session releases after 10 idle minutes (the tunnel then returns Cloudflare error 1033 and the
agent swallows it into empty rows), a reused runtime can break imports (`yaml` circular import: delete the
runtime and run from the top), and the Drive zip and the `.ipynb` must both be replaced (compare the hash and
the cell line numbers). Result JSONs record the laptop's `llm_model` default, not the served model. Do not use
shell heredocs for text with backslashes (they turn `\n` into real newlines); use Edit.

**Answer bakeoff implementation (2026-11-01):** preserve both hash-checked preparers;
never run format/fix tools on them. Model text must be redacted by decoded JSON string
tokens: scanning escaped JSON mistakes a newline before a decorator for an email and can
create invalid escapes. Capsule SQLite redaction rebuilds/vacuums the DB and checks logical
text; binary bytes are not email text. Raw Git objects contain emails, so use the pinned
public clone/tag recipe and redact typed results. Freeze dev before test and never overwrite
raw runs. Regenerate all notebooks/source ZIPs after `src`, `pyproject.toml` or `uv.lock`
changes; the new notebook has a freshness/portable-payload test.

**Answer bakeoff preparation (2026-11-01):** fact labels are not tool schemas. In
particular, `release_first` requires `get_pr` then `releases_containing`, and `field`
is a hidden scoring projection, never a hint passed to the model. Freeze hashes are
immutable: version amendments instead of overwriting v1. A model config's saved
`transformers_version` is not proof that the installed runtime supports its architecture;
Qwen3.5 requires a runtime change despite its old serialization-version field. Check the
actual registry and official card. No weights before exact-revision approval.

**Summary evaluation (2026-10-31):** the Flask evaluation checkout is shallow (one commit).
For F3, use the existing full clone under `data/phase5_repos` to walk ancestors of the
evaluation pin, through `summary_history_repos_dir`; never use that clone's HEAD as the pin.
The sandbox also needs a writable copy of the existing parser cache, otherwise AST parsing
tries a blocked download and silently falls back; the experiment refuses mismatched cached
code vectors. Final source changes require regenerating every notebook. The first E3 cell
keeps model approval unset. Architecture labels remain **unreviewed by a person**.

1. **Chroma 1.5.9 intermittently fails reads right after writes** with
   `Error creating hnsw segment reader: Nothing found on disk`. It looked like a rare flake for
   a day (I even added a retry, which also failed). It is a Chroma defect: Chroma's default
   `hnsw.sync_threshold=1000` keeps new vectors only in memory, and a read that needs the vector
   index (a vector query, or `get` with embeddings) sometimes can't find it. Reproduced with raw
   `chromadb` (two collections, interleaved writes/reads): **20 failures in 5,760 reads at the
   default, 0 at `sync_threshold=2`**, no measurable cost. Fix: `chroma_sync_threshold=2`, applied
   to new collections and `modify()`-ed onto existing ones when opened. The retry was removed.
   If the error ever returns, **re-measure with a raw-chromadb loop before adding a retry.**
2. **The Colab notebook embeds our source, so it goes stale.** `tests/test_colab_notebook.py`
   fails if it does. Order matters: `ruff format` first, then regenerate the notebook, then test.
   Generated `.ipynb` files are excluded from ruff (ruff rewrites them).
3. **Do not use shell heredocs for text containing backslashes or `\n`.** They were mangled
   several times (a regex, a `\n` in a string, `D:\Users` triggering a Python `\U` escape). Use
   the editor/Write tools for such content.
4. **A Windows sleep gap looks like a hung test.** One run reported "1 day, 12:22:44" because the
   laptop slept mid-run. Check wall-clock before suspecting a hang.
5. **Chroma exact ties are ordered arbitrarily**, and the hash-based fake embedder creates many
   exact ties. Tests that compare rankings across two stores must compare per-chunk similarities,
   not tie-sensitive order.
6. **BM25 on tiny corpora:** a term in 1 of 2 documents has IDF exactly 0. Test fixtures need
   filler chunks.
7. **`get_settings()` is cached per process.** Tests that change env vars must call
   `get_settings.cache_clear()`.
8. **Test fixtures under `tests/fixtures/` must stay byte-for-byte**; they are excluded from ruff.
9. **`MarkdownHeaderTextSplitter` rewrites whitespace**, which breaks line numbers and
   citations; that is why `docs_chunker.py` has its own scanner.
10. **sentence-transformers 5.7 renamed `tokenizer_kwargs` to `processor_kwargs`.** Left padding
    matters for Qwen3 (last-token pooling); verified applied.
11. **Python 3.11 compatibility:** the project supports `>=3.11`, so no `def f[T]()` syntax.
12. **Don't call a failure "rare" or "transient" without evidence.** That cost a day (pitfall 1).
13. **Don't claim hardware works before it has run.** "A free T4 works" was written before any GPU
    run and had to be softened.
14. **Tests can outrank implementation, but do not globally penalize them.** V2 test-target
    MRR falls 0.705 → 0.352 with the 0.8 penalty. Use an explicit path filter when the user
    requests it; a future intent-aware policy needs fresh assessment.
15. Defaults that surprised people: HF repos need `trust_remote_code` for CodeRankEmbed and its
    remote code needs `transformers<5`; the project pins `transformers>=4.51,<5`.
16. The sandbox blocks risky `rm -rf` and some temp paths. Don't try to work around a blocked
    command; tell the user what you wanted to do and why.
17. **Phase 4 parser coordinates:** temporary tree-sitter `Point.row/column` access crashed
    natively on this Windows/CPython 3.13 build. Tuple access, already used by the production
    chunker, passed. The reference adapter substitutes only coordinate access; its cAST
    boundaries and text reconstruction remain upstream behavior. Reference JavaScript support
    is absent: mark Express N/A instead of silently using the TypeScript parser.
18. **Reference line envelopes may contain missing text:** upstream astchunk can omit oversized
    multiline leaf text and leave blank padding. The adapter exports represented source spans;
    missing nonblank source earns no hit/coverage credit. Five of 270 Requests reference chunks
    differed from the complete source-line slice; three had gaps, two were partial lines.
19. **Read all run flags in an uploaded result archive:** the main full runs have rerank=false,
    but separate `rerank_experiment/` runs have rerank=true. All original gold is implementation
    code, so a test penalty is rewarded by the dataset. HNSW=exact on 270–565 vectors is a
    plumbing result, not scale validation. Do not infer a universal ranking/penalty policy.
20. **Original v1 Phase 4 is not full production-policy evaluation:** its common headers differ from
    production (Express signatures 7/ 384 versus 53/ 384; bodies/spans match). The eval explicitly
    supplies a code rerank instruction. Production's former commit-prompt bug is now fixed
    with task-specific instructions and an injected-scorer test; code reranking remains opt-in.
    Original candidate lists/scores beyond ten were not exported; exact reconstructed depth
    must not be presented as logged ANN@50. Gold may be broader than the required answer.

21. **V2 measures an eval policy; the new production path needs real GPU timing.** The CLI
    now has --rerank-depth (default 50), still returns 8 by default, and stays opt-in. Report
    original 20-candidate/top10 results as such; the new production A100 experiment is pending.
    A100 is experiments only, not chosen deployment hardware.

---

## 10. Working with this user

Observed, not guessed:

- They want to **understand**, not just receive code: they asked "explain the flow", "give me full
  explain on what u did and why", "what is built so far", "on what code did u test this". Lead with
  the answer, then the reasoning; diagrams and tables are welcome. Keep it plain; avoid jargon
  without a one-line explanation (they asked what `GITHUB_TOKEN` and `VOYAGE_API_KEY` are for).
- They want **decisions made by measurement**: they asked to compare models on speed, quality "and
  other factors", and to test on repos with real history. Report honest numbers, including bad ones
  and gaps ("only Python was tested on real repos").
- They write tersely, sometimes with typos or fragments ("next stagr", "status"). Interpret
  generously; a bare "status" means "report progress now".
- They expect a **maintained project log** (`docs/PROJECT_LOG.md`, required by `CLAUDE.md`).
- They have **not approved any commit**. Ask before committing, pushing, or deleting.
- They run long jobs themselves on Colab and bring results back; build the notebook, say exactly
  what to click, and what a healthy run looks like.
- They interrupt or background long commands; prefer background runs with a watcher for anything
  over a minute or two.

---

## 11. What to do next

### Latest (2026-11-20, night): Flask re-run with the `search_prs` fix is done

`search_prs` used to raise on Flask (no PR data indexed) and ended turns in an abstention; fixed in `agent/tools.py` (returns a note). Flask re-run (136 questions): controls 105 and 105, `fused` 105, summary index (fused, summaries hidden) 110 and 110 (14 won, 9 lost, p 0.405: not significant; it also abstains less, 14 against 19). The summary index stays opt-in. Of the 31 baseline misses, 25 are retrieval misses (gold outside the top 5 of the model's first query), 19 are abstentions, and some are label ambiguity (four questions share one gold commit; the model often names a sibling fix). Details: PROJECT_LOG 2026-11-20 (night, later). `eval/commit_failure_sort.py` now takes `--repo`, `--data-dir`, `--dataset`, `--run`. The `commit_explain` regression check is done too: tuned 98 against 99, fresh 72 against 72, architecture 13/15 and 9/10 (12/15 and 9/10 before); one bug-style code question went to a commit answer (`code-033`). No regression beyond noise. Results: `eval/commit_explain_check/`.

### Latest (2026-11-20, evening): Flask replication, `commit_explain` is now the default

Details and tables: PROJECT_LOG 2026-11-20 (evening). Flask (136 questions, `data_flask`): controls 101, 100, 101; fused 103; summary index, fused, summaries hidden 106 and 107. Same direction as requests, nothing significant alone; the summary index is NOT shipped (recommendation: opt-in, it costs a 27B summary pass and a GPU embedding run per repo). The default prompt is now `commit_explain` (user decision); code, docs, history and architecture have not been re-run under it, so do that first in the next 27B session (`eval/colab_session.py`). Notebooks were regenerated: upload the new `notebooks/repo_rag_ask_source.zip` to Drive before starting the server notebook. Pitfalls: a Colab `files.download` can fail silently (the pilot notebook now saves to Google Drive first); `pkill` is not available in this bash, kill by PID with PowerShell, and never start a second `commit_session.py` while one runs; `commit_session.py --repo pallets/flask` needs `--data-dir D:/Repo_chatbot/data_flask` and `--summary-index`.

### Latest (2026-11-19, night): commit questions, what failed and what is built (not yet run on the agent)

Details and numbers: PROJECT_LOG 2026-11-19 (night). Summaries were not the cause of the 113-question ceiling: of 35 misses, 23 never had the gold commit in the agent's 15 shown, 18 not even in the top 30
(`eval/commit_failure_sort/failures_20261119.json`). Built, all default off: `agent_commit_query=fused`, `get_commit` + prompt `commit_verify`, `agent_commit_hide_summary`, and
`eval/commit_session.py` (control twice, fused, verify, fused_verify, fused_fix_only, summary_hidden; sign test and lenient score). Offline (113 questions, question as the agent sends it), gold in the top 5: 62 today, 65 fused, 75 fused on the summary index.
**Ran live (27B):** controls 76, 76, 78; `verify` 69 (worse, do not use); `fused` 79; summary index + `fused` + `hide_summary` **82 and 84** (likely about +6, pooled p = 0.093, not proven); fix-only 85 is circular. Table in PROJECT_LOG.
**Next:** the user decides whether to adopt the summary index for search (`data/exp_summaries` replaces the commit index; set `agent_commit_query=fused`, `agent_commit_hide_summary=true`). Before claiming it generalises, repeat on a second repository. Do not use the reranker (user decision), do not index issue or PR text for commits (leaks this eval), treat `is_fix` filtering as an upper bound here.

### 2026-11-19: the quality session ran live; decisions waiting for the user

Work is in the worktree `D:\Repo_chatbot_quality` (branch `quality/answers`, uncommitted; copy new tests to `D:\Repo_chatbot\tests` by hand). Runbook `docs/COLAB_SESSION.md`; results in
`eval/colab_session_results/latest/`; details in PROJECT_LOG 2026-11-19 (later). Headline (shipped prompt, 27B): fresh set 72/73 three times with no flips, guards change nothing there; 113
bug reports 74/113 default and **78/113 with `commit_explain`** (answers explain instead of bare SHAs: 0 bare against 86 of 99); tuned set no regression.
**Waiting on the user:** make `commit_explain` the default (recommended; its specific claims are not diff-checked), keep guards off (no measured gain), whether the summary pilot justifies summarising
all 4,883 commits (summaries step running; then `package`, pilot notebook, `score`).
Pitfalls: `agent_eval.py --variant` defaults to `baseline`, NOT the shipped prompt (always pass `--variant commit_pr_commits`); `--workers` builds one agent per thread, one at a time, with a shared locked
embedding model (never build agents concurrently); the server notebook and zip must be the worktree's (`c2a718e2`; the main folder's are older and have no `concurrency`); runs under
`eval/colab_session_results/run1_*`, `run2_*`, `run3_*` are spoiled and must not be quoted. Parallel mode is ~1.9x, not 3 to 5x.

### 2026-11-18, evening: commit questions 57/113 to 70/113; next is the commit-summary pilot

Defaults now: `agent_commit_k` = 15 (first 5 commits in full, the rest one line) and
`agent_commit_max_tool_calls` = 6 (applies once the model has called `search_commits`; other
questions keep 4). Measured on the new 113-question set `eval/datasets/agent_eval_commit_full_v1.jsonl`
(27B, single runs): 57 before, 65 with k=15, 70 with k=15 and 6 calls; k=20 and 8 or 10 calls add
nothing. The cap for non-commit questions was not re-measured. Full table and the retrieval sweeps
(BM25 weight and title fusion are within noise) are in PROJECT_LOG 2026-11-18 (later). Pitfall found
on the way: any new field in a commit row must also be handled in `graph._check` (it read
`row["text"]` and crashed on the shortlist rows). Next: pilot of LLM-written commit summaries
(141 gold commits plus about 1,000 others, GPU embedding, compare ranks with and without), then the
exact-SHA reranker measurement at depth 50. The 10-question commit set (5/10) has not been re-run
with the new defaults.

Claude's review of the 27B answers (code/docs 49/52 correct, 15 exact answers 10 fully right, commit
113: 70 strict, about 76 with valid alternatives, 67 of 70 passes are bare SHAs) is in
`eval/claude_review_27b/`; the findings and the product gaps are in PROJECT_LOG 2026-11-18 (later).
Pitfall: the metadata.db has 4,883 commits because merge commits are excluded by design, not truncated.

### Latest (2026-11-08, night): parse retry built, fresh question set built; next is one short GPU session

Done offline since the entry below: parse failures are answered once with feedback
(`agent_parse_retries`, production only, tested; not yet measured live) and
`eval/datasets/agent_eval_fresh_v1.jsonl` (20 new questions; `eval/build_fresh_questions.py`). The
Drive zip to upload is `f0fde72b` (hash in PROJECT_LOG). Plan for that session (30 to 35 min): fresh set
with baseline and `docs_hint_narrow`; `prs_narrow` with the retry; baseline to confirm nothing moved.
Commands: `uv run python eval/agent_eval.py run --url U --token T --no-rerank --variant V --dataset
eval/datasets/agent_eval_fresh_v1.jsonl`. After that: decide whether `docs_hint_narrow` becomes the
default, then E3 and the label review (Phase 6), the judge check (Phase 7), see `docs/FINISH_PLAN.md`.

**A100 session 2026-11-16:** A/B/C all 36/40 code, 10/12 docs, 10/10 prsearch (snap and `grounded_complete` change no pass counts); facts 57/60; section 1 NOT met (9 of 32 abstained or failed, several weak; see PROJECT_LOG 2026-11-16). Next: judge A/B/C answers, then fix the eager abstentions and the history-tool path guessing.

**Snapped answers judged (2026-11-11):** faithful 29/48 vs 25/48, correct unchanged; code-006, 021, 038 improved, no regressions, noise about 1 verdict in 44. Next: wire `snap_citations` into `agent/graph.py` (needs symbol table + retrieved chunks), then re-run live.

**Citation snapping built (2026-11-11, offline, not wired in):** `agent/citation_snap.py`, `eval/citation_check.py`. 6 of 50 saved answers snapped, none lose gold overlap; effect on judge scores unmeasured (snapped answers file is next to the saved answers). Next: judge v2 on it, then wire into `graph.py` if it helps.

**Judge v2 RAN (2026-11-11):** gates pass; correctness agreement with the corrected AI review 80% (v1 74%) but it overstates correctness (82% vs 66%; few partials), faithfulness agreement fell to 75% (v1 87.5%). Use as a wrong-answer detector, not a completeness measure. Details in PROJECT_LOG. Still no human labels.

**Judge v2 prepared (2026-11-11, not run):** docs reference by section, prompts that ignore citation line numbers; protocol `eval/datasets/phase7_answer_judge_v2/`; notebook and bundle regenerated. v2 results are in-sample on the 50 answers. Run it on the A100, then compare with the corrected `eval/review/answer_review_v1/review_sheet.csv`.

**Phase 7 third-reader check (2026-11-11):** Claude re-read the 20 contested answers, corrected 6 Codex labels, and scored the judge: faithfulness fairly reliable (87.5% exact vs corrected review), correctness unreliable on docs (reference evidence is only the file head) and it docks correctness for citation ranges. See PROJECT_LOG 2026-11-11 (later). All still AI readers; no human labels.

**Phase 7 judge RAN (2026-11-11):** gates passed (controls rejected 98%/100%, 0 parse failures); judge says 54% faithful, 62% correct vs the AI audit's 58%/60%; agreement with the AI review 85%/74% on the top verdict. Still no non-AI marks, so calibration against a person is unmeasured. Details: PROJECT_LOG 2026-11-11.

**Phase 7 AI audit (2026-11-11):** Codex filled `eval/review/answer_review_v1/review_sheet.csv` for all 50 answers: 29/50 faithful, 30/50 correct, 22/50 both; mechanical score agrees 64% (it overstates quality). These are AI verdicts, not human labels (`review_summary.md`), so they do not close Phase 7. Still needed: the judge run on the A100, then `eval/review_sheet.py compare`, plus a few marks from a non-AI reader.

**Phase 7 judge check (built 2026-11-10, not run):** D3 = quantized Qwen3.8-27B (user wrote "Qwen3.7-28B"). Upload
`notebooks/answer_judge_bundle.zip` to the top of My Drive, run `notebooks/answer_judge_on_colab.ipynb` on an A100,
return `answer_judge_results.zip`. Judge is trusted only if it rejects >= 90% of negative controls
(`eval/datasets/phase7_answer_judge_v1/protocol.json`). Default model is the 4-bit `cyankiwi/Qwen3.8-27B-AWQ-INT4`
(the official block-wise FP8 is untested on an A100); vLLM is pinned `==0.33.0` in its own venv, `cu130`, sampler
JIT off, eager mode; a preflight cell checks the machine first. Three Colab tries failed before the model ran
(hash mismatch, torch/torchaudio clash, FlashInfer `ninja`); all fixed offline, none run on a GPU yet. No A100?
`eval/review_sheet.py` makes a human review sheet (`eval/review/answer_review_v1/`). PROJECT_LOG 2026-11-10.

### Earlier (2026-11-08, evening): Stage 1 A/B done; the narrow docs hint is promising, not adopted

PROJECT_LOG §2 has the table. Baseline 35/40 code, 7/12 docs, 10/10 prsearch; `docs_hint_narrow` 36/40,
10/12, 10/10 (gained docs-045/046/047 and code-035, lost nothing); `prs_narrow` 6/10 prsearch (the model
writes `"type": "search_prs"`; strict parser rejects it). **Decoding is deterministic: repeats were
identical, so they measure no noise.** Next: (1) write ~20 fresh questions (10 "how do I" usage, 10
"how does" code, none matching any current question) and run baseline vs `docs_hint_narrow` once before
making the hint the default; (2) the user's decision on the parse slip (retry with feedback / accept the
slip / stay strict); (3) E3 notebook and the label review (Phase 6); the plan is in `docs/FINISH_PLAN.md`.
The Drive zip hash at this point is `6afc2787`. The user commits and handles GitHub themselves.

### Earlier (2026-11-08): everything staged for one combined Colab session

Read PROJECT_LOG §2 entries "First live `/ask`..." and "Offline preparation for one combined Colab
session". Done offline, not yet run on a GPU: follow-ups now see the last 2 answered turns
(`graph.with_history`; one of two vague follow-ups worked live, the other failed `parse` and a failed
run now returns `raw_reply`); the model server drains refused request bodies (this was the flaky
`test_a_wrong_or_missing_token_is_refused`, reproduced and fixed); the UI is a client of the API;
`agent_prompt_variant` A/B switch (`baseline` pinned by SHA-256); `eval/agent_facts.py` for the Phase 7
fact gate on **httpx** (the gate's fact set is httpx, not requests; tools alone 60/60).
**The combined session ran (2026-11-08); results are in PROJECT_LOG §2.** Fact gate 59/60 on httpx
(target 90%, with stated deviations); baseline unchanged at 35/40 code, 7/12 docs; the 3-turn follow-up
thread worked once; `docs_hint` fixed 3 docs questions but lost 2 code ones; `no_search_prs` gained back
3 questions that missed since `search_prs` was added, at a cost of 6/10 PR-search questions. No variant is
adopted. **Next:** a narrower `docs_hint` and a reworded `search_prs` paragraph, measured against
`baseline` (repeat runs if possible); the E3 notebook (`summary_prose_on_colab.ipynb`, model approved
2026-11-08) is a separate Colab run; the architecture labels still need a person. To use the Drive zip
again: `notebooks/ask_source.zip` must match the hash in PROJECT_LOG. Nothing is committed.

### Earlier (2026-11-07, later): Phase 8 API built; URL-to-index still not possible

Read `docs/PROJECT_LOG.md` §2, entry "Phase 8, API backend". `uv run uvicorn repo_rag.api.server:app`
(localhost only, no auth) gives `GET /repos`, streaming `POST /ask` (SSE), `POST /repos/import`
(raw zip body). The Streamlit sources panel now links to GitHub at the indexed commit. Phase 8 is
not ticked: a fresh user cannot paste a URL and wait for indexing, because indexing needs a GPU.
Drive zip hash now starts `78651e79`. Untested: `/ask` against a live model. Nothing is committed.

### Earlier (2026-11-07): reranker out-of-memory fixed; it still lowers the agent's commit score (1/10 vs 3/10)

Read `docs/PROJECT_LOG.md` §2, entry "Reranker memory fix confirmed". Fixed and confirmed on a GPU:
no out-of-memory, `rerank_failures` 0. But with correct reranking the agent scores 1/10 on commit
questions against 3/10 without. The agent sends short queries; the reranker does not put more gold
commits in the top 5 for those, and the model picks differently from the reordered list. Open
decision, now made (user, 2026-11-07): the UI reranker is off by default (`agent_commit_rerank`,
env `REPO_RAG_AGENT_COMMIT_RERANK`); the Drive zip hash now starts `2c0a2cce`. One unexplained
single failure of `test_remote_llm.py::test_a_wrong_or_missing_token_is_refused` (WinError 10053);
5 of 5 repeats passed.
The text below (2026-11-06) is the earlier state and its "not trustworthy" warning no longer applies.

### Earlier (2026-11-06): reranker and `search_prs` run; reranker out-of-memory fix not yet tested

Read `docs/PROJECT_LOG.md` §2, entry "First run of the commit reranker, `search_prs` and the
103-question set". Single runs on an A100 40 GB (14B model plus `Qwen3-Reranker-4B`): code 34/40,
docs 7/12, history 11/11, refs 10/10, kind 5/5, probe 5/5, **prsearch 10/10 (new)**, commit 1/10 with
the reranker vs 3/10 without. The commit number is **not trustworthy**: the server's reranker ran out
of GPU memory (`/rerank` 500) and the agent silently fell back to the fused order; the result rows did
not count it. Fix made but untested on a GPU: `rerank.py` uses `logits_to_keep=1`; `agent_eval.py` now
records `rerank_failures`. Next: replace `repo_rag_ask_source.zip` on Drive (hash starts `3128bc40`),
start a fresh A100 session, rerun `--types commit` with the reranker, require `rerank_failures` = 0,
then compare with 3/10. Also unexplained: code-032 stalled 300 s, new misses code-035, docs-046,
docs-047. Nothing is committed.

### Earlier (2026-11-05): agent evaluated with the real model; commit search is the weak spot

Read `docs/PROJECT_LOG.md` §2, entry "First agent evaluation with the real model". Streamlit UI + Colab model server work (`notebooks/serve_model_on_colab.ipynb`, Drive zip must be named
`repo_rag_ask_source.zip` and match the notebook's SHA-256: regenerate all notebooks after any `src/repo_rag` change). `eval/agent_eval.py run --url <u> --token <t> [--types code|docs|history|refs|kind|commit|probe]`
(`--toolset evaluated` for the old tools). Latest per type (single exploratory runs, no gates): code 36/40, docs 8/12, history 11/11, refs 10/10, kind 5/5, commit 3 to 4/10, probe 5/5.
Next: (1) commit questions: gold is outside the top 5 for 5 of 10, so try reranking the top 50 next to the answer model; (2) look at the docs and code misses; (3) PR/issue text search; (4) Phase 8 FastAPI.
Pitfall: a quick tunnel dies when the Colab session idles (HTTP 530); a run during that is invalid, not a result. Nothing is committed.

### Latest (2026-11-01): run the v2 answer bakeoff on Colab

Read `docs/PHASE7_BAKEOFF_BUILD_REVIEW.md`. Run `notebooks/answer_bakeoff_on_colab.ipynb`
on an A100 40 GB; the source/data capsule is embedded. The four exact approved revisions
are checked before loading. A 14B memory preflight precedes all four dev runs; then the
dev-only freezer records prompt/parser/source/index hashes before test. At most two shared
revisions are permitted. Keep raw failures and return `answer_bakeoff_results.zip`.
Recompute with `uv run python eval/analyze_answer_bakeoff.py report --output <results>`.
There is no model-quality result or winner yet; defaults stay unchanged. Full Phase 7 remains
unchecked. The notes below retain earlier preparation/approval history.

### Latest (2026-11-01, evening): bake-off candidates approved; dev split amended to v2

The user approved all four candidates at the pinned revisions and asked for the dev-split amendment.
**Use protocol `eval/datasets/phase7_bakeoff_v2/`** (141 questions, facts 26 dev / 45 test, test and gates
identical to v1; v1 stays frozen and must keep verifying: never edit `eval/prepare_answer_bakeoff.py`).
Verify with `uv run python eval/prepare_answer_bakeoff_v2.py verify`. Next step for Codex: build the
harness, fake-model tests, Colab notebook and dev-frozen prompt per `docs/CODEX_HANDOFF_PHASE7_BAKEOFF.md`,
reading the v2 splits. Any further change needs a v3 directory. No answer model has been loaded.

### Earlier (2026-11-01): frozen answer bakeoff; model approval pending

The first two stages of `docs/CODEX_HANDOFF_PHASE7_BAKEOFF.md` are prepared for review in
`docs/PHASE7_BAKEOFF_PROTOCOL_REVIEW.md`. Inputs and proposal have separate SHA-256 freezes
in `eval/datasets/phase7_bakeoff_v1/`. Facts split by type (15 dev / 45 test), original code
split preserved (16 / 24), architecture test only (20), probes 2 / 8. Labels remain
unreviewed by a person. No weights downloaded or models loaded. Wait for exact-revision
approval before implementing the harness, fake-model tests and generated A100 notebook.
After any later `src/repo_rag` change regenerate every notebook. Phase 7 remains unchecked;
this checkpoint does not implement the full LangGraph agent or choose a winning model.

### Earlier (2026-11-01): choose the answer model

Next is `docs/CODEX_HANDOFF_PHASE7_BAKEOFF.md`: a pre-registered, measured bake-off for the Phase 7
answer model (Codex builds, Claude reviews; the user has a Colab A100 40 GB). The prose grounding check
was fixed and the summary-model proposal re-frozen as v2 before any model run (see PROJECT_LOG,
2026-11-01); Qwen3-4B-Instruct-2507 is the summary model, run still behind the notebook's approval cell.

### Earlier (2026-11-01): architecture summaries assessed; E3 approval pending

Phase 5b and Phase 6 slices 1 and 2 are built. The architecture experiment from
`docs/CODEX_HANDOFF_SUMMARIES.md` is now implemented; read `docs/SUMMARIES_BUILD_REVIEW.md` for
commands, artifacts and limitations and `docs/ARCHITECTURE_LABEL_REVIEW.md` for the 30 frozen
questions (**unreviewed by a person**). The final E1/E2 assessment is recorded in PROJECT_LOG;
no variant qualifies for adoption.
E3 requires explicit approval of the proposed summary model before its Colab run.
The older notes below describe the earlier commit-depth work.

### Current priority: run and review the commit depth/weight notebook

Read `docs/COMMIT_DEPTH_BUILD_REVIEW.md` and the original specification
`docs/CODEX_HANDOFF_COMMIT_DEPTH.md`. Upload `phase5_commit_depth_source.zip` to Drive and
run `phase5_commit_depth_on_colab.ipynb` on A100. Return `phase5_commit_depth_results.zip`.
The original v2 GPU run is independently audited; the new experiment has no GPU results yet.
CPU replay of saved dev lists chooses weight 0.1, but the notebook runs its own first stage
and selects on dev only. Production defaults, independent label review and manual scale
remain unchanged/open. No paid key or new GitHub call is needed. Do not reopen expansion.

The returned A100 expansion run is valid and fails the frozen primary: fresh proof 20/30
-> 17/30, delta -0.10 [-0.2424,+0.0345], one gain/four losses. Old guard proof 114/120 -> 116/120
repeats the original two fixes; all old-stratum/text gates pass, but fresh gain fails. Both
also scores 17/30 fresh proofs. Leave evidence_expansion=none and do not iterate further.

All 450 outputs and aggregate JSON reproduce. All 45 code hashes per run match the frozen/local
source at import; rebuilding clean pinned repos reproduces all 1,787 chunk texts/metadata.
The source ZIP is unchanged. All four runs used A100-SXM4-40GB and identical model revisions.
See phase4_expansion_full/README.md for audit limits and exact records; phase4_expansion_run/
preserves the other imported report. This is quality only, without independent latency or
generated-answer validation; independent label review remains open.

Follow [the Phase 5 handoff](PHASE5_HANDOFF.md) for local commit ingestion and direct/via-PR
issue linking. GITHUB_TOKEN is unset: local git/fixture work can start, GitHub API work needs
the token configured in the environment or .env. Never paste credentials in chat. Phase 5 is
not implemented or checked off yet. Preserve the reviewed data and no-commit instruction.

### Historical preparation of the completed expansion run

The expansion brief is implemented: **30 newly authored multi-block cases**, six each for Requests,
Express and Flask plus twelve Click, and **120 unchanged regression guards**. Click pin:
2247b35ea1c47c727d7a06e51fa280e12a863ff6. All fresh proofs validate and need multiple production
chunks;26 connected fresh groups. Labels/protocol were frozen and hashed before query search.
No commit was made because the user forbids commits. Source overlap with old cases is disclosed;
same-assistant source review is not independent review.

Run **phase4_expansion_on_colab.ipynb** with **phase4_expansion_source.zip** uploaded to My Drive.
The actual production search executes plain rerank20, callee and both (450 outputs total).
One resident scorer shares exact identical pair scores; **quality only, no latency measurement**.
Return phase4_expansion_results.zip. The analyzer audits frozen source/data, full outputs,
twenty-pair/fusion parity, expansion reconstruction, scores, returned characters and displaced IDs.
Primary callee must gain >= 0.05 in proof on the new 30 with lower 95% bound > 0, lose <= 0.02 MRR/proof
on every existing pooled stratum (including multi-block), and grow text <= 10% separately on new and old.
Both is secondary. If callee fails, keep expansion off and move to Phase 5 without further tuning.
An invalid/incomplete run needs repair, not an adoption/rejection claim. Deployment and
independent label review remain open; defaults never change automatically.

Production flag: REPO_RAG_EVIDENCE_EXPANSION=none|adjacent|callee|both, default none. Expansion
only applies after requested/enabled code reranking and preserves query filters, kept hits and
result count. Insertions are marked unscored; the CLI displays related evidence. The lazy chunk
index is cached: call forget_cached_indexes() after reindexing a reused retriever, as for BM25.
No read/expansion is added to the ordinary default hybrid/laptop path.

Local measured BM25 on new 30: MRR 0.5981 / proof 0.4000. Old 120 remains 0.4776/ 0.6500. Holding old
recorded pools/probabilities fixed, current default-off production reproduces all 120 rankings
and metrics. Full450-output pipeline/audit also passes with test models; those synthetic scores
are not benchmark evidence. No real expansion GPU result has been provided for this new set.

### Earlier evaluation context

For the complete context, fixed protocol, run instructions and returned-result audit for the
earlier90-question depth comparison, read [its historical handoff](phase4_comparison/HANDOFF.md).
The 90 run was subsequently audited (PROJECT_LOG): MRR hybrid/20/50=0.748/0.896/0.893,
proof=0.867/0.933/0.944;50-minus20 MRR CI touches zero. The preparation notes below predate
that audit. Current work is the fresh expansion test above, not repeating the old90 experiment.

### Evaluation complete, rerank default pending; preserve evaluated snapshots

The returned `phase4_followup_results.zip` is audited and published under
`eval/retrieval_results/phase4_v2_full/`. Read its README and the leading section of
`eval/RESULTS.md`. The upload SHA-256 is
`6f15971e3675ea887b397739bb4939f974ba59453b64b0eb3ad32b2f77ea75f1`.
All 38 source hashes per run match the frozen v2 source ZIP; 129 questions / 1,032 score rows /
1,548 timed calls reproduce from pinned source, exported vectors, lists and pair scores.
Neural inference and GPU elapsed times were not independently rerun locally.

Fresh-question MRR hybrid → rerank20: Requests 0.699 → 1.000, Express 0.635 → 0.949,
Flask 0.739 → 0.957; pooled 0.701 → 0.966, paired +0.265 [0.175, 0.355]. Pooled numerical
coverage/test/example checks pass. Independent label review remains pending (only six example
questions; Requests has no example stratum). These are chunk retrieval metrics, not answer accuracy.

**Production decision:** keep code reranking opt-in, weights unchanged, no test penalty or
reserved BM25 quota. Warm rerank20 p50/p95 is 925/1028 ms (Requests), 976/1028 (Express),
889/944 (Flask), versus hybrid roughly 61/64 ms. Production now exposes --rerank-depth / REPO_RAG_RERANK_CANDIDATE_DEPTH, default 50;
result count defaults to 8. The v2 policy scores 20 and evaluates top10. New production-path
A100 measurement is prepared, not yet run; deployment hardware remains undecided.

All 24/24 fresh test questions explicitly say test/tested; their scores do not assess
ambiguous intent. A same-assistant source review is in eval/PHASE4_LABEL_REVIEW.md; independent
review remains outstanding. The next comparison now uses **90 questions**, 30 each on Requests,
Express and Flask: 36 ambiguous without source-type/symbol cues, 18 multi-block, 18 test,
12 implementation and 6 examples. The original 15 questions remain a separately reported
retained subset alongside 75 additions. Shared accepted evidence is connected into 83 groups;
58 repo/file pairs are covered. All hashes and full-corpus answerability validate. Source-authored
labels are not an independent holdout; old-v2 evidence overlap is recorded in the manifest.
Use **notebooks/phase4_comparison_on_colab.ipynb** with **phase4_comparison_source.zip**.
The fixed hybrid/rerank20/rerank50 matrix has 810 independent measured calls, three per policy
and question, but quality uses repetition zero. Report direct paired 50-20, all/per-repo/type,
added75/retained15 and separate per-repo p50/p95. Labels/protocol/source travel in the results ZIP.
Actual expanded BM25 MRR is Requests 0.5009, Express 0.6346, Flask 0.3479; complete-proof scores
0.5667/0.7333/0.5333. All old15 BM25 rankings/metrics reproduce unchanged. No expanded GPU
results exist yet; do not substitute the old15 notebook or mistake baseline scores for reranking.
A100 is experiment-only. Leave the laptop on hybrid; GPU reranking remains explicit opt-in.

The blanket penalty demonstrably hurts test intent (MRR 0.705 → 0.352). The quota matches
fresh scores but regresses legacy Requests-034 through a close-score tie. Pair traces explain
Requests-036/Express-028: tests outrank the required implementation even though gold is in the
pool. Requests-029 loses required helper context below top10. Retain these failures for future
assessment rather than editing frozen labels to remove them.

The three 274/ 384/429-vector indexes give ANN recall 1.0 at 1/5/10/50; this is plumbing, not
scale evidence. V2 uses one AST corpus and cannot establish a new AST-versus-character advantage.
V1's narrow completeness evidence and reference/budget limitations remain in the historical
sections of RESULTS and `eval/PHASE4_FAILURE_ANALYSIS.md`. Both v1/v2 matching source ZIPs and
original labels are preserved. Do not treat legacy questions, repetitions or header variants
as independent fresh experiments.

**Direct old depth comparison:** `eval/retrieval_results/rerank_depth_comparison/` now pairs
rerank20/50 on all 80 original questions. Requests MRR 0.8800/0.8779, completeness 0.925/0.900;
Express MRR 0.7692/0.7679, completeness 1.000/1.000. All paired MRR intervals include zero.
The original scorer processed fifty pairs once and reused scores for both depths, so no
independent twenty-candidate timing or speedup was measured. All 160 depth rows and original
intervals reproduce; no new inference. New cue-free production-path A100 comparison is pending.

**Next actions:** first complete the expanded 90-question A100 experiment and obtain independent
label review before default promotion. Deployment timing waits for chosen hardware.
Phase 5 commit ingestion, direct/via-PR issue linking and bug→fix evaluation
are next in the roadmap; no implementation has started. Before a later reranking-default
proposal, obtain independent label/alternative-evidence review, assess source intent and
multi-block context on a fresh set, and choose a deployment latency target. Matched chunk
budgets, more languages, ANN scale and end-to-end answer quality remain open follow-ups.

### Then (see `plan.md` for each)

- **Phase 5** (commit history, bug-to-fix): needs `GITHUB_TOKEN`. Two issue-linking paths are
  required (direct "fixes #N" and commit to PR to issue via GraphQL `closingIssuesReferences`).
  Eval rule: index commits **without** their "Fixes #N" text and query with the issue text only.
  Carry over the T3 findings (§5).
- **Phase 5b**: PR/issue index, the SQLite `metadata.db`, typed git tools (`blame`, `log -L`,
  `tag --contains`), exact-reference detection (`#N`, SHAs, tags, `@handles`), and an
  auto-generated fact-check set.
- **Phase 6**: symbol table and call edges (tree-sitter queries), file/dir/repo summaries.
- **Phase 7**: LangGraph agent with an exact-ref pre-step and 5 routes
  (architecture, explain, bug_match, navigate, history). Needs `ANTHROPIC_API_KEY`.
- **Phases 8 and 9**: API/UI, then stretch goals.

### Pending decisions for the user (ask, don't assume)

1. **Commit?** Nothing is committed. A reasonable grouping if they agree (each commit should pass
   the tests on its own; if that proves fiddly, one commit is fine): (a) scaffold, config, CLI,
   ingestion; (b) chunking + chunker check; (c) embedding benchmark + results; (d) store,
   retrieval, indexer; (e) index transfer + Colab notebook; (f) docs (CLAUDE.md, plan.md,
   project log, this file). Re-run `uv run pytest` before each.
2. **`AGENTS.md`:** delete it, or replace it with a pointer to `CLAUDE.md`?
3. **Delete `D:\Users\`** (§1.3).
4. **Time a Colab run** (GPU type and the index cell's duration) on a bigger repo, which would
   settle the plan's "under 10 minutes on a GPU" criterion. The one real run worked but wasn't timed.
5. Phase 4 now uses Express as the second repo/language alongside Requests; review the new
   Express questions and run its GPU comparison before making a cross-language quality claim.

---

## 12. Where everything is documented

| File | Authoritative for | Size |
|---|---|---|
| `docs/HANDOFF.md` (this file) | orientation, current truth, next steps | |
| `CLAUDE.md` | tech stack, repo layout (target), **rules and conventions**, commands | ~200 lines |
| `plan.md` | goal, success criteria, architecture (§3), design of every future phase, risks, question taxonomy | ~510 lines |
| `docs/PROJECT_LOG.md` | **chronological history of what was done and why**, decision table (§3), tools/versions (§4), open items (§5) | ~700 lines |
| `eval/RESULTS.md` | measured results behind every model decision | ~240 lines |
| `docs/PHASE7_BAKEOFF_PROTOCOL_REVIEW.md` | frozen protocol/splits/probes, exact-revision model proposal, pending approval and limitations | |
| `docs/PHASE7_BAKEOFF_BUILD_REVIEW.md` | current v2 implementation, approved Colab workflow, dev/test unlock, capsule provenance and recomputation | |
| `eval/chunker_check_results.md` | chunker invariants on 7 real repos | |
| `eval/embedding_bench/README.md` | how the embedding benchmark and its tasks (T1, T2, T2r, T3) work | |
| `AGENTS.md` | **stale, ignore** (§1.2) | |

**Suggested reading order for a new session:** this file, then `CLAUDE.md`, then `plan.md`
§0 and the section for the phase you are about to do, then the last few entries of
`docs/PROJECT_LOG.md`. Read `eval/RESULTS.md` before changing any model or retrieval setting.

**Maintenance rules:** `CLAUDE.md` requires a dated `docs/PROJECT_LOG.md` entry after any change
to code, data, decisions or measured numbers, and a tick in `plan.md` when a phase's acceptance
criteria pass. Keep this file's §3, §9 and §11 current.

---

## 13. Glossary

- **RAG:** retrieval-augmented generation. Find relevant chunks, give them to an LLM to answer.
- **Chunk:** a piece of a file that is indexed and retrieved. Ours are whole source lines.
- **AST / tree-sitter:** the syntax tree of code; tree-sitter parses many languages. AST chunking
  splits along function/class boundaries instead of every N characters.
- **cAST:** the split-then-merge algorithm from the paper (split oversized nodes, merge small
  neighbours up to a size budget).
- **Embedding / dims:** a vector for a text; 512 numbers here (Matryoshka truncation of 1024).
- **Dense search / HNSW:** nearest-neighbour search over embeddings; HNSW is the approximate index.
- **BM25:** classic keyword scoring. Good at exact identifiers.
- **RRF:** reciprocal-rank fusion. Combine rankings by `weight / (k + rank)`.
- **Reranker:** a cross-encoder that rescores (query, document) pairs; slower, more accurate.
- **nDCG@10 / Hit@5 / MRR / Recall@k:** ranking-quality metrics. Hit@5 = was a correct item in the
  top 5. nDCG@10 rewards putting correct items higher. Higher is better; 1.0 is perfect.
- **T1 / T2 / T2r / T3:** the benchmark tasks. T1 docstring to code (auto-generated); T2 36 hand
  written developer questions about requests; T2r = T2 searching only requests' chunks (how
  production filters by repo); T3 = a GitHub issue to the commit that fixed it (144 real pairs).
- **Blobless clone:** `git clone --filter=blob:none`: all commits, file contents fetched on demand.
- **Embedding id:** `<model>_<dims>` (e.g. `Qwen3-Embedding-0.6B_512`); names the vector space.
- **Via-PR linking:** squash-merge repos record `(#PR)` in commits and put "Fixes #N" in the PR body.

---

## 14. Check this file is still true

```bash
git log --oneline | head -3          # "does not have any commits yet" means §1.1 still applies
git status --short | wc -l           # size of the uncommitted pile
uv run pytest                        # expect: 323 passed, 1 deselected (or more, if tests were added)
uv run repo-rag config               # compare against §7 defaults
uv run repo-rag search "where are retries configured" --repo psf/requests -k 3
                                     # expect adapters.py HTTPAdapter.__init__ / HTTPAdapter in the top 3
python -c "import json;print(json.load(open('data/manifest.json')))"   # expect psf/requests, 274 + 282
uv run python notebooks/build_colab_indexer.py && git status --short notebooks   # no semantic change expected
ls D:/Users 2>&1                     # the stray folder (§1.3)
head -5 AGENTS.md                    # still the stale copy? (§1.2)
```
