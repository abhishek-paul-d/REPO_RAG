"""Application settings. All env vars and tunable defaults live here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The 14B answer model the local loader (`LocalChatModel`) was evaluated with. Notebooks that load
# it on a GPU pass these explicitly; the default `llm_model` is the 27B served by vLLM.
LOCAL_14B_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
LOCAL_14B_REVISION = "aedcc2d42b622764e023cf882b6652e646b95671"


class Settings(BaseSettings):
    """Loaded from environment variables and `.env`.

    Tunables use the `REPO_RAG_` prefix (e.g. `REPO_RAG_MAX_CHUNK_CHARS`).
    API keys use their conventional unprefixed names.
    """

    model_config = SettingsConfigDict(
        env_prefix="REPO_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Secrets ---
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    github_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GITHUB_TOKEN", "github_token")
    )
    voyage_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("VOYAGE_API_KEY", "voyage_api_key")
    )

    # --- Storage ---
    data_dir: Path = Path("data")

    # --- Models ---
    llm_provider: Literal["local"] = "local"
    # Answer model: Qwen3.8-27B, 4-bit AWQ, served by vLLM from Colab (user decision 2026-11-18; it
    # met the Phase 6 gates with CUDA graphs + MTP, see docs/PROJECT_LOG.md). The laptop agent never
    # loads it. It cannot be loaded locally: `LocalChatModel` needs transformers, whose pinned
    # version has no `qwen3_5`, and its chat-template hash below is the 14B's. The previous,
    # provisional answer model (user decision 2026-11-03; Qwen2.5-Coder-14B with prompt arm 2
    # passed 4 of 5 v5 gates, eval/RESULTS.md) is kept as the LOCAL_14B_* constants below.
    llm_model: str | None = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
    llm_model_revision: str | None = "e3a700906e6a73ba4290b1873dc2ba8198527ab5"
    llm_prompt_arm: int = 2
    # "evaluated" is exactly the tools and prompt measured in the Phase 7 bake-offs; "production"
    # adds search_docs, search_commits and get_references (agent/tools.py), which are not measured.
    agent_toolset: Literal["evaluated", "production"] = "production"
    # Limits the answer model was evaluated under (phase7_bakeoff_v5 protocol "generation").
    agent_max_tool_calls: int = 4
    agent_max_turns: int = 5
    # Tool calls allowed once the model has searched commits (2026-11-18, 113 bug->commit
    # questions at k=15): 4 calls 65/113 with 13 budget failures; 6 calls 70/113 with 1; 8 and 10
    # calls 70/113, no change. Other questions keep `agent_max_tool_calls`; not re-measured there.
    agent_commit_max_tool_calls: int = Field(default=6, ge=1, le=12)
    # 24,000 characters and 12,288 input tokens (the Qwen3.8-27B profile, user decision 2026-11-18);
    # the 14B was evaluated at 12,000 and 8,192: set those with REPO_RAG_ env vars to reproduce it.
    agent_tool_result_chars: int = 24000
    agent_max_input_tokens: int = 12288
    agent_max_new_tokens: int = 512
    # The evaluated chat-template call and the template's hash, so a different template is refused.
    llm_template_arguments: dict[str, Any] = {"add_generation_prompt": True, "tools": None}
    llm_chat_template_sha256: str | None = (
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
    )
    # Where the answer model runs when it is not on this machine (agent/remote_llm.py serves it,
    # e.g. from notebooks/serve_model_on_colab.ipynb). The token is the shared secret.
    llm_server_url: str | None = None
    llm_server_token: SecretStr | None = None
    api_max_import_bytes: int = Field(default=2 * 1024**3, gt=0)
    api_url: str = "http://127.0.0.1:8000"
    """Where the Streamlit UI finds the API (`uvicorn repo_rag.api.server:app`)."""
    """Largest index zip `POST /repos/import` accepts (api/server.py); the request is refused
    with 413 once the upload passes it."""
    summary_model: str | None = None
    """Open-source local models only; no paid key required."""
    # Chosen by eval/embedding_bench (see eval/RESULTS.md). Qwen3-Embedding-0.6B led the
    # developer-question task by a wide margin and is Apache-2.0 and laptop-viable.
    embedding_provider: Literal["voyage", "huggingface"] = "huggingface"
    embedding_model: str = "voyage-code-3"
    """Used when embedding_provider == "voyage" (untested paid option)."""
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dims: int | None = Field(default=512, gt=0)
    """Matryoshka truncation. 512 matched full 1024 dims on our eval at half the index size.
    None keeps the model's full dimension."""
    embedding_query_instruction: str = (
        "Instruct: Given a developer question or a description of behaviour, "
        "retrieve the code that implements it\nQuery:"
    )
    """Qwen3 query instruction, used for every collection (code, docs, commits). T3 tuning found
    no commit-specific instruction that beat it (eval/RESULTS.md). Documents get no prompt."""
    embedding_max_tokens: int = Field(default=2048, gt=0)
    """Per-chunk token cap, the same one the embedding benchmark used."""
    embedding_batch_size: int = Field(default=16, gt=0)
    index_write_batch: int = Field(default=64, gt=0)
    """Chunks embedded and written to the vector store per step. A GPU does better with a larger
    batch (the model sorts each batch by length, so there is less padding): the Colab notebooks
    raise this together with `embedding_batch_size`."""
    history_prefetch_chunk: int = Field(default=5000, gt=0)
    """Missing blobs requested per fetch while preparing to read a blobless clone's history."""
    compute_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_dtype: Literal["auto", "float32", "bfloat16"] = "auto"
    """auto = bfloat16 on a GPU, float32 on a CPU. Index vectors built on a GPU and queries
    embedded on a laptop CPU should use the same dtype: set float32 when building elsewhere."""
    commit_diff_chars: int = Field(default=800, gt=0)
    """Commit docs = message + files + touched symbols + this much diff. 800 matched 2,000 on
    bug -> fix retrieval and indexes about 1.4x faster."""
    pr_body_chars: int = Field(default=3000, gt=0)
    """PR and issue docs = title + labels + this much of the body."""
    summary_max_chars: int = Field(default=3000, gt=0)
    """Cap on one repo-map summary (file, directory or repository), facts plus any model prose."""
    summary_fact_features: tuple[str, ...] = ()
    """Experimental facts: docstrings, runtime, history, readme. Empty preserves the baseline."""
    summary_fact_max_chars: int = Field(default=16000, gt=0)
    """Explicit larger cap when summary_fact_features is nonempty; baseline stays at 3,000."""
    summary_history_limit: int = Field(default=12, gt=0)
    summary_history_repos_dir: Path | None = None
    """Optional full-history clones for ancestry when the source checkout is shallow."""
    summary_section_chars: int = Field(default=1200, gt=0)
    summary_sections_per_file: int = Field(default=3, gt=0)
    summary_prose_max_tokens: int = Field(default=160, gt=0)
    title_query_fusion: bool = True
    """PR/issue search also searches a multi-line query's first line and fuses the rankings."""
    reranker_model: str = "Qwen/Qwen3-Reranker-8B"
    """The 8B judge beat the 4B on code questions: MRR@10 0.860 -> 0.926 over 150 questions
    (eval/RESULTS.md, Phase 4 judge run), at about +45% rerank latency. Bug -> fix reranking was
    measured only with the 4B (nDCG@10 0.494 -> 0.687 over the top 50); re-measure it with the 8B in
    Phase 5. bge-reranker-v2-m3 made bug -> fix worse (-0.12)."""
    reranker_mode: Literal["auto", "on", "off"] = "auto"
    """auto = rerank only when a CUDA GPU is available. On a laptop CPU even the 0.6B reranker
    took ~23 s for 5 candidates; the 4B took ~1.8 s per query (top 50) on an A100, the 8B is
    about 1.4x slower."""
    agent_prompt_variant: Literal[
        "baseline",
        "no_search_prs",
        "docs_hint",
        "docs_hint_narrow",
        "prs_narrow",
        "architecture_hint",
        "architecture_hint_reminder",
        "architecture_cover",
        "grounded_complete",
        "history_grounded",
        "commit_pr_commits",
        "commit_explain",
    ] = "commit_pr_commits"
    """Production-toolset prompt. Default `commit_pr_commits` since 2026-11-18 (user decision): it
    is `docs_hint_narrow` plus one paragraph telling the model to name a PR's own commit from
    `pr_commits`, not the merge commit; on the 27B it took commit-088 from a miss to a pass (commit
    7/10 vs 6/10, n = 10) with code, docs and the other types unchanged. Before that the default
    was `docs_hint_narrow` (user decision 2026-11-09): on the
    fresh set it took usage questions from 1/10 to 9/10 with code unchanged at 8/10, and on the
    62-question set docs went 7/12 to 10/12 and code 35/40 to 36/40 (single deterministic passes).
    `baseline` is the text every earlier evaluation used (pinned by SHA-256 in
    tests/test_prompt_variants.py). The others are for A/B runs of `eval/agent_eval.py --variant`;
    `prs_narrow` lost 4 prsearch questions and `docs_hint` was contaminated: not candidates."""
    agent_commit_query: Literal["model", "question"] = "model"
    """What `search_commits` searches for. `model` (default) is the query the model writes.
    `question` replaces it with the user's current question as typed (no thread history), in case
    a short model-written query loses the details of a pasted bug report. Unmeasured on the agent
    (see docs/PROJECT_LOG.md, 2026-11-09): A/B it with
    `eval/agent_eval.py --commit-query question` before changing the default."""
    agent_code_extra_files: int = Field(default=3, ge=0, le=6)
    """Production toolset only: after `search_code` returns its top 5 chunks (unchanged), add up to
    this many more, each from a file not already shown, taken from a search with the user's whole
    question. Offline on psf/requests (2026-11-11), 5 chunks showed 13 of 25 architecture questions
    at least half their gold files (3 files for the model's own query); +3 files from the question
    gave 24 of 25, and code questions cannot lose a result because the top 5 are untouched. 3 is the
    default since 2026-11-18 (the 27B profile); 0 is what every earlier evaluation ran; see
    docs/PROJECT_LOG.md."""
    agent_code_skip_tests: bool = True
    """Production toolset only: `search_code` leaves chunks from test files out of its top 5 (and
    out of the extras) unless the question or the model's query explicitly asks about tests
    (`retrieval.intent.detect_intent`). Offline on psf/requests, 18% of the top-5 chunks for the
    architecture questions were in `tests/`; dropping them gave enough gold files for 18 instead of
    16 of 25 questions and 40 instead of 39 of 40 code questions (none of whose gold is in
    `tests/`, so a test-seeking question is not covered by that number). On by default since
    2026-11-18 (the 27B profile)."""
    agent_extra_chunk_chars: int = Field(default=1000, gt=0)
    """Each extra chunk is cut to this many characters so the extras stay within
    `agent_tool_result_chars`; the metadata still gives the chunk's full line range."""
    agent_commit_k: int = Field(default=15, ge=5, le=30)
    """How many commits `search_commits` returns. The first 5 are shown in full (message, files, a
    diff excerpt); any beyond that are one line each (subject, files, symbols), so 15 to 20 cost
    about what 5 full ones do. On the 141 psf/requests bug -> fix pairs the gold commit is in the
    top 5 for 79, the top 10 for 94 and the top 20 for 106 (`eval/commit_retrieval_check.py`,
    2026-11-18, bug text as the query). Agent runs on the 113 test pairs (27B): 57 passed at 5,
    65 at 15, 65 at 20 (more failures), so the default is 15; `--commit-k` A/Bs it."""
    agent_answer_facts: bool = True
    """Append facts from the tool results to an answer (`agent/commit_answer.py`): for a commit
    answer the commit's date, author, subject and rank, or the closest candidates as unverified
    leads when nothing matched; for count answers a note that merge commits are not counted.
    `result["model_answer"]` keeps the model's own text, and the eval scores that."""
    agent_commit_fix_only: bool = False
    """`search_commits` only returns commits classified as fixes (`is_fix=true`), as `ask --bug`
    does. Unmeasured on the agent; see `agent_commit_query`."""
    agent_citation_snap: bool = False
    """After the citation check, re-point or narrow a cited `path:start-end` using definition ranges
    from the symbol table (`agent/citation_snap.py`): a range that shows none of the functions the
    sentence names, or one much wider than them. Offline on 50 saved psf/requests answers it changed
    6, lost no gold overlap, and judge v2 faithfulness went 25/48 to 29/48 with correctness
    unchanged (docs/PROJECT_LOG.md, 2026-11-11). Not yet run through the live agent, so off by
    default; the result lists each change under `citations.snapped`."""
    agent_abstain_retries: int = Field(default=0, ge=0, le=2)
    """Production toolset only: when the model abstains before it has made two tool calls, answer it
    once with a message asking it to search again (`prompts.ABSTAIN_RETRY`) instead of accepting the
    abstention. The 2026-11-16 section 1 run abstained after one search on 7 of 32 questions the
    repository answers. Unmeasured: off until an A/B on the agent_eval set shows the correct
    abstentions on out-of-scope probes survive (they were 5/5)."""
    agent_budget_note: bool = False
    """Production toolset only: when a tool result is empty because the per-question result budget
    (`agent_tool_result_chars`) is used up, add a note saying so (`prompts.BUDGET_NOTE`). Section 1
    arch-2 (2026-11-18): two searches used 23,625 of 24,000 characters, the next two came back empty
    with no reason, and the model searched blind until it ran out of turns. Unmeasured; off keeps
    the old behaviour. Enable with `REPO_RAG_AGENT_BUDGET_NOTE=true`."""
    agent_tool_error_retries: int = Field(default=0, ge=0, le=3)
    """Production toolset only: when a tool call fails, give the model the error and let it call
    again (or answer) instead of ending the run with `failure="tool"` (`prompts.TOOL_ERROR_NEXT`).
    Each recovery uses a turn and a tool call from the existing caps. Section 1, 2026-11-16: hist-1
    died on a blame call with an invented path. Unmeasured; 0 keeps the old behaviour."""
    agent_parse_retries: int = Field(default=1, ge=0, le=3)
    """Production toolset only: how many times a reply that fails the format check is answered with
    a message naming the problem, instead of ending the run. Added after `prs_narrow` showed the
    model writing "type": "search_prs" (the right tool, a wrong field) and the strict parser
    discarding it. The evaluated (bake-off) toolset stays strict whatever this says. 0 turns it
    off."""
    agent_history_turns: int = Field(default=2, ge=0, le=10)
    """Earlier answered turns of a thread shown to the model so follow-ups ("what does it
    return?") resolve. 0 turns the feature off. A thread's first question, and any question in a
    thread with no answered turn, reaches the model exactly as before."""
    agent_history_answer_chars: int = Field(default=600, gt=0)
    """Each earlier answer is cut to this many characters before it is shown to the model."""
    agent_commit_rerank: bool = False
    """Let the Streamlit agent re-score `search_commits` results on the model server's reranker.
    Off by default: with the agent's short queries it scored 1/10 on the commit questions against
    3/10 without it, 0 rerank failures (docs/PROJECT_LOG.md, 2026-11-07)."""
    rerank_instruction: str = (
        "Given a bug report, judge whether this commit's changes fix the reported bug"
    )
    code_rerank_instruction: str = (
        "Given a developer question, retrieve the code that implements the requested behaviour"
    )
    pr_rerank_instruction: str = (
        "Given a GitHub issue, judge whether this pull request resolves the reported problem"
    )
    docs_rerank_instruction: str = (
        "Given a developer question, retrieve documentation that answers the question"
    )
    rerank_instruction_mode: Literal["default", "intent", "source_aware"] = "default"
    """Which task instruction code reranking uses. `intent` switches on a keyword detector for
    'implementation'/'test' wording; `source_aware` sends every code query one instruction that
    prefers source over tests unless the question asks for a test or example. GPU policy run:
    `intent` gained only on the template wording it was written from (+0.167 on 12 such
    questions, +0.000 on the 30 paraphrases); `source_aware` +0.007 MRR, CI [-0.015, +0.033].
    Neither is reliable, so both stay off."""
    code_rerank_instruction_implementation: str = (
        "Given a developer question asking where behaviour is implemented, retrieve the source "
        "code that implements it, not tests or examples that only demonstrate it"
    )
    code_rerank_instruction_test: str = (
        "Given a developer question asking which test checks a behaviour, retrieve the test "
        "that verifies it, not the implementation"
    )
    code_rerank_instruction_source_aware: str = (
        "Given a developer question, retrieve the code that implements the requested behaviour; "
        "prefer source code over tests and examples unless the question asks for a test or example"
    )
    rerank_first_stage_weight: float = Field(default=0.0, ge=0)
    """Code reranking only. 0 orders by the reranker's P(yes). A positive weight orders by
    log-odds of P(yes) plus this weight times -ln(first-stage rank). Offline on saved scores,
    0.25 looked slightly better; the live GPU policy run did not confirm it (MRR +0.006 over
    120 questions, CI [-0.013, +0.026]; test questions -0.028), so it stays off."""
    rerank_max_tokens: int = Field(default=2048, gt=0)

    # --- Ingestion ---
    max_file_bytes: int = Field(default=1_000_000, gt=0)
    update_embed_limit: int = Field(default=200, gt=0)
    """`repo-rag update` refuses to embed more than this many chunks on this machine unless told
    to (`--yes`), because embedding on a CPU is slow (see CLAUDE.md). A convenience guard, not a
    measured threshold: tune it to your hardware."""
    max_commits: int = Field(default=5000, gt=0)
    commit_patch_chars: int = Field(default=100_000, gt=0)
    """Per-file patch retained in history records; document diff uses commit_diff_chars."""
    git_tool_max_results: int = Field(default=100, gt=0)
    """Cap on commits, files or tags one git tool call returns (blame is capped by line range)."""
    max_prs: int = Field(default=500, gt=0)
    max_issues: int = Field(default=500, gt=0)
    """First-sync caps: the newest N by updatedAt; later syncs fetch everything newer."""
    github_sync_page_size: int = Field(default=50, gt=0, le=100)
    github_review_page: int = Field(default=50, gt=0, le=100)
    """Reviews fetched per PR; a PR with more is flagged rather than silently cut."""
    github_timeout_seconds: float = Field(default=30, gt=0)

    # --- Chunking ---
    max_chunk_chars: int = Field(default=1500, gt=0)
    """Chunk size budget in non-whitespace characters (cAST convention)."""
    header_defines: bool = False
    """Add a `# defines: a, b, c` header line to chunks that merge several definitions. Phase 4
    found such chunks described only by an unrelated first name. OFF until a GPU run shows it
    helps: BM25-only on the 120 policy cases it was neutral to slightly negative (MRR 0.478 ->
    0.472; ambiguous 0.489 -> 0.451). Changing it changes chunk text, so re-index."""
    header_max_defines_chars: int = Field(default=240, gt=0)
    """Longest `# defines:` list, in characters; longer lists end in `...`."""
    ast_max_error_ratio: float = Field(default=0.1, ge=0, le=1)
    """Fall back to the text splitter when more than this share of a file's bytes sits in
    tree-sitter ERROR/MISSING nodes. Small errors are tolerated; tree-sitter recovers well."""
    fallback_chunk_overlap: int = Field(default=100, ge=0)
    """Character overlap for fallback (non-AST) chunks. AST chunks never overlap."""

    # --- Retrieval ---
    retrieval_k: int = Field(default=50, gt=0)
    """Candidates per first-stage signal. Recall@50 on bug -> fix: 0.86."""
    rerank_candidate_depth: int = Field(default=50, gt=0)
    """Maximum fused candidates scored when reranking is requested; independent of retrieval_k.
    Keep the previous 50-candidate behaviour by default; 20 is an explicit experiment."""
    rerank_top_n: int = Field(default=8, gt=0)
    evidence_expansion: Literal["none", "adjacent", "callee", "both"] = "none"
    """Experimental additions after reranked code search only; never on the CPU/hybrid path.
    Name matching is not a call graph. Default remains off pending the frozen expansion eval."""
    expansion_keep: int = Field(default=5, ge=1)
    expansion_seeds: int = Field(default=3, ge=1)
    expansion_per_seed: int = Field(default=2, ge=1)
    expansion_max_definers: int = Field(default=3, ge=1)
    dense_weight: float = Field(default=1.0, ge=0)
    bm25_weight: float = Field(default=0.3, ge=0)
    """PROVISIONAL. Fusion is weighted RRF. Equal weights measurably hurt Qwen3 in the embedding
    benchmark (mean nDCG@10 0.612 -> 0.585), so BM25 gets a smaller vote. 0.3 is a guess that
    has not been established as a universal optimum by Phase 4."""
    rrf_k: int = Field(default=60, gt=0)

    # --- Evaluation (never changes production retrieval defaults) ---
    eval_char_overlap: int = Field(default=200, ge=0)
    eval_summary_embedding_batch_size: int = Field(default=4, gt=0)
    eval_cutoffs: tuple[int, ...] = (1, 5, 10)
    eval_bm25_weights: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0)
    eval_test_penalties: tuple[float, ...] = (1.0, 0.8)
    eval_rerank_depths: tuple[int, ...] = (20, 50)
    eval_code_instruction: str = (
        "Given a developer question, retrieve the code that implements the requested behaviour"
    )
    eval_split_seed: int = 42
    eval_dev_fraction: float = Field(default=0.4, gt=0, lt=1)
    eval_coverage_threshold: float = Field(default=0.9, gt=0, le=1)
    eval_bootstrap_samples: int = Field(default=10000, ge=100)
    eval_bootstrap_confidence: float = Field(default=0.95, gt=0, lt=1)
    eval_bootstrap_seed: int = 42
    eval_followup_candidate_depth: int = Field(default=20, gt=0)
    eval_followup_bm25_quota: int = Field(default=5, ge=0)
    eval_followup_latency_repeats: int = Field(default=3, gt=0)
    eval_followup_trace_ids: tuple[str, ...] = (
        "requests-036",
        "requests-028",
        "express-036",
        "express-028",
    )

    # --- Chroma ---
    chroma_sync_threshold: int = Field(default=2, ge=2)
    """Chroma writes its vector index to disk after this many unflushed writes. Its default
    (1000) keeps new vectors only in memory, and a read in that window intermittently fails with
    "Error creating hnsw segment reader: Nothing found on disk". Reproduced with raw chromadb:
    20 failures in 5,760 reads at the default, 0 at 2. Cost: none measurable (2,000 x 512-d
    vectors written in 1.2 s instead of 1.0 s)."""

    @property
    def clones_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.data_dir / "bm25"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def metadata_db_path(self) -> Path:
        return self.data_dir / "metadata.db"

    @property
    def github_cache_dir(self) -> Path:
        return self.data_dir / "github_cache"

    @property
    def embedding_id(self) -> str:
        """Identifies the embedding space; vectors from different ids must never be mixed."""
        model = self.local_embedding_model.split("/")[-1]
        return f"{model}_{self.embedding_dims or 'full'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
