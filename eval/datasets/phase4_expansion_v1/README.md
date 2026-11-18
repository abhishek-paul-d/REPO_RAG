# Frozen evidence-expansion experiment

This tests the proposed name-based `callee` expansion after real rerank20 search. Earlier
post-hoc replay of 120 saved lists improved reranked complete proof from 0.950 to 0.967
(two fixes, interval touching zero), while expansion hurt unreranked lists. That motivated
this experiment; it did not justify adopting a default.

## Sample and freeze

**30 newly authored multi-block questions:** six each Requests, Express and Flask, twelve
Click. Click is new to this project's retrieval evaluations, pinned at
`2247b35ea1c47c727d7a06e51fa280e12a863ff6`. Every required source range was author-reviewed
and hashed against clean pinned source. Each question has one sufficient proof whose ranges
are jointly required. All 30 are answerable somewhere in the production corpus; none has a
complete proof within a single indexed chunk. Corpus sizes: 274/384/429/700 respectively.
Fresh cases share **26 connected evidence groups**.

The **existing 120 cases** are copied byte-for-byte from phase4_intent_v1 as regression guards:
36 ambiguous, 18 multi-block, 24 implementation, 36 test and 6 example. Their existing groups
and alternative-proof semantics are preserved. `manifest.json` records evidence overlap
between fresh questions and old cases. New wording over previously inspected code is not
independent evidence. Primary fresh30 and regression120 are assessed separately.

Authored and source-reviewed by Codex; independent label review remains pending. This is a
deliberate source-authored diagnostic sample, not random traffic, guaranteed unseen model
training data, or an independently reviewed holdout. Only two languages are represented.

`freeze.json` records the protocol/manifest hashes before any fresh query retrieval:

- Protocol: `aed33f2b62b61d0eeec45680141ea3cb7eb953733553a09ca5338d3d2a64c34f`.
- Manifest: `8ed5db211f49854fa38eb968ae08c637a34083caabf541cb78a1e39cbdb9bd01`.

The brief requested pre-registration in a committed file but also forbids commits. The user's
no-commit instruction takes precedence: files were frozen and hashed on disk; no git commit
or staging was performed. The authoring script refuses to overwrite a frozen manifest.

## Fixed matrix and success rule

**Plain rerank20**, **rerank20 + callee**, **rerank20 + both**. First stage remains 50 per
signal, dense/BM25 weights 1.0/0.3, RRF k=60, ordinary production headers (`header_defines`
off), one code instruction, plain P(yes) ordering. Expansion keeps the first 5 of top10,
uses 3 seeds, adds at most 2 chunks per seed, skips names defined in more than 3 chunks,
deduplicates and displaces the tail to preserve result count. Added chunks are unscored
related evidence; their score field is a zero placeholder, not a model probability.

The nested `retrieval` object retains the earlier depth protocol as a model/fusion settings
reference. The top-level expansion matrix and runner overrides fix this experiment at
rerank20, ordinary headers and none/callee/both; earlier descriptive all90/depth50 fields
inside that reference do not select this run's questions or policies.

Actual production search executes for every policy. A resident reranker shares exact scores
for identical instruction/query/candidate text inputs, as in the earlier quality-only policy
runner. **150 questions, 450 policy outputs; no independent latency measurement.**

The pre-chosen primary is **callee minus plain on fresh30**. `both` is secondary and cannot
replace the primary if callee fails. Callee is eligible only if all hold:

1. Fresh complete-proof@10 gain is at least +0.05 and its paired 95% lower bound is above zero.
2. No pooled stratum of old120 loses more than 0.02 MRR@10 or complete-proof@10. All five
   strata are checked, including multi-block; per-repo views are also reported diagnostically.
3. Returned text grows by at most 10%, separately on fresh30 and regression120. Growth is
   the ratio of summed returned characters, including contextual headers, minus one.

Complete proof means union coverage of at least 90% of every required range in one accepted
proof. Intervals resample entire connected evidence groups (10,000 samples, seed 42, 95%).
Secondary/subgroup intervals are pointwise exploratory, without multiplicity adjustment.
A CI spanning zero is inconclusive. Repeated views/paraphrases are not independent replications.

The analyzer reports paired differences, per-cohort/type/repo scores, chunks and characters
returned, inserted/displaced IDs, and eligibility. It independently rescores from exported
chunks and reconstructs fusion and expansion from traces. Incomplete or inconsistent runs
are invalid and cannot yield an adoption decision. On rejection, keep expansion off and
proceed to Phase 5 without tuning these labels again. Eligibility never changes a default
automatically; independent review remains open.

## Running

Upload **notebooks/phase4_expansion_source.zip** to My Drive root. Open the matching
**notebooks/phase4_expansion_on_colab.ipynb**, select A100 and Run all. The compact notebook
uses uv, verifies the separate ZIP, clones/checks out all four pinned repos, runs the fresh
and regression matrix, audits it, and downloads **phase4_expansion_results.zip**.

The returned ZIP contains frozen labels/protocol, source ZIP, per-repo chunk text/metadata,
full dense/BM25 lists, fused candidate probabilities, pre-expansion and final rankings,
metrics, volume/displacement and environment/model revisions. A100 is experiment-only;
deployment remains undecided. No expanded GPU results are available at preparation time.

Local source validation can use `uv run python eval/run_expansion_eval.py --labels
eval/datasets/phase4_expansion_v1 --repo <owner/name> --checkout <pinned-checkout> --output
<new-output> --mode validate --device cpu`. `--mode bm25` is only a lexical preflight;
the analyzer refuses it as evidence for the expansion success rule.

## Production limits

`Settings.evidence_expansion` accepts none/adjacent/callee/both and defaults to **none**.
It runs only for requested, enabled code reranking; the ordinary unreranked CPU/hybrid and
docs paths do not expand. Query filters also scope the expansion corpus. A lazily built
store-backed chunk index is cached like BM25; call `forget_cached_indexes()` after reindexing
when reusing a retriever instance. No corpus read is added to the default path.

The heuristic matches names with a regex, not resolved calls. It may follow irrelevant names,
cannot supply callers, skips common names and misses Express call-style getters. Real call
edges belong to Phase 6. This evaluates evidence retrieval, not answer quality, AST-vs-character
causality, ANN scale or deployment latency. Reranking remains opt-in and expansion stays off.
