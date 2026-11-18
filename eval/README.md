# Retrieval evaluation (Phase 4)

**Current priority:** [fair evidence-expansion test](datasets/phase4_expansion_v1/README.md).
Thirty fresh multi-block questions (including twelve Click cases) and 120 unchanged regression
guards; actual production plain rerank20 / callee / both, with fixed proof/regression/text-budget
criteria. Use **notebooks/phase4_expansion_on_colab.ipynb** and its matching separate
**phase4_expansion_source.zip** on A100. Expanded GPU results are pending; expansion defaults
to none, reranking remains opt-in, and deployment is undecided. On rejection proceed to
Phase 5 without tuning this set again. Local BM25 and default-none replay are in
retrieval_results/phase4_expansion_preflight/. The older preparation workflow below is historical.

**Current work:** v2 GPU evaluation is complete; rerank default pending. The next run is the
[expanded 90-question comparison](datasets/phase4_comparison_v3/README.md): 30 each on Requests,
Express and Flask, 75 additions plus 15 retained questions, with ambiguous, multi-block,
test, implementation and example strata. Use `notebooks/phase4_comparison_on_colab.ipynb`
and its separate `phase4_comparison_source.zip`. Hybrid, rerank20 and rerank50 execute
independently with fixed settings and real timing. Expanded BM25 preflight is saved;
expanded GPU results and independent review are pending. A100 is experiment-only and deployment
hardware remains undecided. Original v1/v2 datasets and results remain frozen below.

This benchmark measures **retrieval of source code**, before answer generation. It uses the
production file walker, chunker, embedder, Chroma store, code tokenizer and weighted-RRF function.
It writes to `data/retrieval_eval/`, separately from the working index. No LLM, GitHub token or
production index rebuild is needed.

For the first run's question-level diagnosis, see [failure analysis and next steps](PHASE4_FAILURE_ANALYSIS.md).
`analyze_retrieval_failures.py` audits exported chunks/vectors offline, without neural inference.
It writes compact diagnostics under `retrieval_results/failure_analysis/`; optional local
tokenizer/source audits check truncation risk and production-header parity.

## Dataset

`datasets/requests_qa.jsonl` and `datasets/express_qa.jsonl` each contain 40 manually authored
questions, pinned to full commit SHAs. Gold is a path plus inclusive **1-based source lines**,
with a SHA-256 of the exact evidence text. Symbol names are an audit aid, never a scoring input.
`build_qa_datasets.py` resolves manually selected source anchors using stdlib Python AST and
independent JavaScript top-level syntax nodes, not our chunk boundaries.

Each repo has 16 development and 24 test questions. Overlapping gold ranges are grouped before
splitting: paraphrases and shared answer implementations cannot cross the split. Weight and test
file penalty selection uses development MRR@10, then Recall@10 to break ties. Production defaults
are **not changed automatically**. Do not repeatedly tune against the test report.

**Limitation:** 36 Requests questions were used in the earlier embedding-model benchmark; the
Requests test split is held out for the Phase 4 tuning, not for prior model selection. Express
questions are new. Gold has been checked against source anchors and hashes but has not had a
second human review. Broader tasks, repositories and independent labels are needed before a
general accuracy claim.

Question wording and author independence remain limitations: some queries name their target
symbol, which can help lexical retrieval, and these labels were authored by the implementation
agent. Questions have not been rewritten in response to measured outcomes. Independent review
and additional paraphrases remain open work.

## Local validation and cheap checks

Use clean checkouts at the SHAs recorded in the dataset. Validation refuses the wrong commit,
tracked modifications, missing gold files and evidence-hash mismatches.

```powershell
uv run python eval/run_retrieval_eval.py --repo psf/requests --checkout data/repos/psf__requests --mode validate
uv run python eval/run_retrieval_eval.py --repo expressjs/express --checkout data/eval_repos/expressjs__express --mode validate
uv run python eval/run_retrieval_eval.py --repo psf/requests --checkout data/repos/psf__requests --mode bm25
uv run python eval/run_retrieval_eval.py --repo expressjs/express --checkout data/eval_repos/expressjs__express --mode bm25
```

`bm25` is a **lexical-only check**. It cannot establish the embedding pipeline's AST advantage.
The CLI has no fake-embedding option. Tests inject small offline vectors only to verify plumbing.

## Full GPU run

Upload `notebooks/phase4_source.zip` to the top level of **My Drive** in Google Drive.
Open the matching `notebooks/retrieval_eval_on_colab.ipynb` in Colab, select a GPU (A100 is the
development target), and run all cells. Approve the Drive mount when Colab asks. If the ZIP
is in another folder, edit `SOURCE_ZIP` in the first Parameters cell. The notebook verifies the
ZIP's SHA-256 and extracts it into local Colab storage. It contains no Base64 source payload.

The notebook uses **uv** to install runtime dependencies plus the reference library at versions
from `uv.lock`, targeting the notebook's Python interpreter. If uv is absent, pip installs
only uv once. It checks out both public
repos at the pinned commits, runs validation, compares chunkers, and downloads
`phase4_results.zip`. `RUN_RERANK=True` additionally tests the 4B reranker at top-20 and top-50
on AST/header-on for both repos. The first A100 full run and separate reranker runs are now
validated in `retrieval_results/colab_full/`; full notebook wall-clock time and independent
top-20 reranker latency were not recorded.

Equivalent command on a GPU machine:

```powershell
uv run python eval/run_retrieval_eval.py --repo psf/requests --checkout path/to/pinned/requests --mode full
# Additional code-specific reranker experiment (requires CUDA):
uv run python eval/run_retrieval_eval.py --repo psf/requests --checkout path/to/pinned/requests --mode full --strategies ast --headers on --rerank --output data/retrieval_eval_rerank
```

Regenerate after changing source, gold or dependency pins, **after formatting**. The Phase 4
builder writes both the compact notebook and `phase4_source.zip`; upload/download them as a
matching pair. The ZIP includes project Python source and question datasets, not local indexes
or environment files.

```powershell
uv run python notebooks/build_retrieval_eval.py
uv run python notebooks/build_colab_indexer.py
```

## What gets compared

| Strategy | Boundaries | Default budget | Overlap |
|---|---|---|---|
| `char` | Strict fixed character windows | 1500 total characters | 200 characters |
| `lc_lang` | LangChain `from_language` recursive splitter | 1500 total characters | Up to 200 characters |
| `astchunk` | Reference cAST implementation, dev-only dependency | 1500 non-whitespace characters | None |
| `ast` | Our production cAST implementation | 1500 non-whitespace characters | None |

All use the same tracked code corpus, including allowed test/example code. Unsupported reference
languages make that **entire repo/strategy N/A**; the corpus is never silently reduced.
`astchunk==0.1.0` supports Python, Java, C# and TypeScript, but not JavaScript. Its evaluation
adapter uses tuple coordinate access instead of temporary `Point.row/column` access to avoid a
native crash reproduced with this tree-sitter/CPython 3.13 Windows combination. The reference's
split/merge algorithm and reconstructed chunk text are unchanged. This shim is eval-only.
The span audit also found omitted oversized leaf text in upstream reconstruction. The adapter
preserves that text behavior but records `eval_source_spans` as a flat JSON string: lines where
nonblank source was replaced by blank padding receive no hit/coverage credit. The inclusive
start/end envelope is still retained for citations. `source_spans` is exported with each result,
and `chunks_with_source_gaps` counts affected chunks. Three multiline/Unicode/split fixtures and
a dropped-leaf regression check the mapping; a bounded envelope alone is not assumed complete.

Each strategy runs header off and header on. Header off embeds the chunk body; header on
prepends the **same source-derived path/class/signature policy to every strategy**, without
access to gold questions. It uses the smallest definition containing the entire chunk. Spans
across multiple definitions get a path-only header. This is a controlled common-header
comparison, not a byte-for-byte replay of every native production header.

In full mode each variant reports:

- BM25 alone, exact cosine search, and Chroma HNSW using the **same real vectors**.
- Weighted RRF with BM25 weights 0, 0.1, 0.3, 0.5, 1.0 and test-file score multipliers 1.0, 0.8.
- The dev-selected combination and its unchanged held-out test scores.
- HNSW top-k neighbour recall against exact search, with similarity-boundary ties accepted.
- Optional reranker depth comparison with a code-specific instruction, not the bug-fix prompt.

Controls live in `config.py` (`REPO_RAG_EVAL_*`), separate from production defaults. The
primary chunking comparison is **exact dense search with header off**; common-header and hybrid
results show the incremental effects. Candidate depth follows the production setting (50).

## Metrics and audit files

- **Recall@k:** fraction of a question's distinct gold ranges overlapped by the first k chunks,
  macro-averaged over questions. Overlapping retrieved chunks do not count the same range twice.
- **Hit@k:** fraction of questions where at least one gold range is found.
- **MRR@10:** reciprocal rank of the first overlapping chunk; zero if none appears in top 10.
- **Line coverage@k:** fraction of unique gold source lines covered by the union of the first k
  chunks, per question, then macro-averaged. Nested gold ranges and overlapping chunks cannot
  double-count a line; blank lines count. Character chunks touching part of a line count it.
- **Complete gold@k (Gold ≥90%):** fraction of that question's gold ranges individually reaching
  the configured coverage threshold, then macro-averaged over questions. This is not the fraction
  of questions whose entire answer is complete. The default threshold is 0.9.
- **Chunk count/average size:** raw body and embedded text sizes, to expose budget differences.

Summary JSON includes dev, test, and all-question means with **95% pointwise percentile bootstrap
intervals**, using 10,000 resamples and seed 42 by default. The sampling unit is the evidence
group: related questions stay together. Group scores are weighted by their question counts, so
the point estimate remains a question-macro average. `comparisons.json` stores **paired AST minus
baseline differences** on identical question ids, including MRR, recall, coverage and completeness.
Intervals are exploratory and not adjusted for multiple comparisons. They do not establish a
general AST advantage; small samples can yield wide or degenerate intervals.
Header-on/off and all/dev/test rows reuse the same questions: they are views of one experiment
per repository, not independent replications. Never sum their sample sizes as additional evidence.

The Markdown report leads with all 40 questions for fixed configurations and also includes dev
and held-out test tables. Pooled results describe this sample; Requests also influenced previous
model selection. The weight/penalty grid stays in JSON. A configuration selected on dev is
assessed using the test split, never promoted to a pooled headline as an unbiased estimate.
Selection still uses the original development MRR/recall rule, not the newly inspected metric.

A one-line overlap counts as a hit. It does not demonstrate that a complete answer is present.
Whole-class gold is therefore easier to overlap than a short function. The AST budget counts
non-whitespace characters while character baselines count every character: this compares the
planned defaults, not equal token budgets. Do not attribute all differences solely to syntax.

Each run writes `summary.md`, `summary.json`, `comparisons.json`, `selected_on_dev.json`, `run.json` (revision, dataset
and source hashes, settings without secrets, package versions and hardware), plus per-variant
`per_question.json`, `chunks.json` and `ann_and_rerank.json`. Vector caches are keyed by input
texts, model, query instruction, truncation, dtype, device and embedding implementation hash.
Partial completed summaries survive later-variant failures. Cached vectors avoid re-embedding
unchanged inputs. Chroma indexes stay isolated and are excluded from the Colab download.

The first reviewed **full-mode** results are now in `RESULTS.md`. Phase 4 remains unchecked
pending the mixed-target follow-up: about 10 reviewed questions/repo targeting tests/examples,
which the original gold omits. Keep the original dataset frozen. The test-penalty result is
biased by gold location, and HNSW matches on small indexes do not measure ANN quality at scale.

## Results retention and checks

Keep the compact Markdown/JSON summaries, per-question rankings (optionally lossless gzip), paired differences, settings
and source hashes under `eval/retrieval_results/` as versionable evidence. Current BM25 reports
are under `bm25/`, including coverage and CIs. The original overlap-only baseline is archived
under `bm25_legacy/`; `bm25_review/` is an earlier review snapshot of the same experiment. Large vector
caches, Chroma databases, downloaded repos and scratch outputs remain ignored under `data/`.
No git staging or commit is implied by this policy. First real full results and their separate
reranker experiments are under `colab_full/`; uncompressed caches/chunks remain ignored.

`uv run mypy` now checks application code, `eval.retrieval`, the dataset builder and the runner.
Legacy standalone `eval/embedding_bench/` research/notebook scripts and `eval/chunker_check.py`
are explicitly excluded. `uv run mypy eval` checks the maintained eval scope as well. This is
the declared scope; the earlier clean 27-file application-only check did not cover Phase 4.
