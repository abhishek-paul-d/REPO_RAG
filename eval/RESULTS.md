# Eval results

Decisions backed by measured numbers. Raw outputs live next to each benchmark.

## Phase 5 final: commit depth, BM25 weight, timing, CLI sample (2026-11-02)

A100 40 GB run of frozen `phase5_commit_depth_v1` (raw files and aggregates in `eval/commit_depth_results/`).
Decision by the pre-registered rule: **keep reranking depth 50 and BM25 weight 0.3; no default changed.**
Depth 100 and 200 fail the Hit@5 gain and positive-lower-bound checks (pooled 8B Hit@5 +0.0076 at
depth 100, 95% CI 0.0 to 0.019, 263 questions / 254 groups).

**Phase 5 "done when" evidence, 8B rerank at depth 50, held-out test, per repo (all test / source-path):**

| Repo (link path) | n | Hit@5 | MRR@10 | n (source) | Hit@5 (source) | MRR@10 (source) |
|---|---:|---:|---:|---:|---:|---:|
| psf/requests (direct) | 113 | 0.8584 | 0.7699 | 103 | 0.8447 | 0.7589 |
| pallets/flask (direct) | 136 | 0.9779 | 0.9326 | 93 | 0.9892 | 0.9409 |
| fastapi/fastapi (via PR) | 14 | 0.8571 | 0.8571 | 12 | 1.0000 | 1.0000 |
| all | 263 | 0.9202 | 0.8587 | 208 | 0.9183 | 0.8542 |

FastAPI's via-PR path finds links and retrieves them; with 14 questions it is descriptive only.
Without reranking (dense), source-path Hit@5 is 0.7788 (requests 0.6893, flask 0.8495).

Depth sweep, 8B, source subset: Hit@5 0.9183 / 0.9279 / 0.9279 and MRR@10 0.8542 / 0.8566 / 0.8573 at
depth 50 / 100 / 200 (4B 0.9087 / 0.9135 / 0.9087). Weight: dev-selected 0.1 (dev MRR 0.7567 vs 0.7337,
66 dev queries), default stays 0.3.

A100 warm timing, p50 s (ratio to depth 50 at depth 100 / 200): requests 3.9 (1.95x / 3.89x),
flask 3.5 (1.91x / 3.60x), fastapi 9.2 (1.95x / 3.76x). Experiment cost, not laptop latency.

CLI sample (`cli_sample.txt`, query in `cli_sample_query.json`): bug report `psf/requests#1000`;
candidate 1 `f69934fb24a9` is the real fix, candidate 2 is unrelated, so relevance is not confirmation.

**Not claimed:** the gold is link-derived (an issue's own fix), not independent labels of similar fixes
for new bugs; usefulness of similar past fixes, answer quality, laptop latency, other repos or
languages and ANN accuracy at scale are unmeasured. Manual scale (pandas, CPU stages) is in
`eval/scale_results/README.md`; embedding was not run there.

## Phase 7 answer model: v4 and v5 bake-offs (2026-11-03)

Mechanical scoring only; citation validity does not prove correctness; architecture and probe labels are unreviewed by a person.
Raw files: `eval/facts_results/answer_bakeoff_v4_test/` and `answer_bakeoff_v5/`. Gates: facts, probes, abstentions, invalid citations.

**v4 test (97 questions, frozen revision 1, 4 models): no candidate qualified.** Qwen2.5-Coder-14B failed only answerable abstentions
(6/44, limit 4) and cited 14 of 44; Qwen3-14B cited 38 of 44 but answered 3 of 8 probes (5/8, needs 7).

**v5 test (fresh seeded questions: 45 answerable, 16 probes; plus the 45 seen v4 facts as a guard).** Tuned prompt = revision 2
(stricter citation sentence, re-search before abstaining, fixed reminder after search results), chosen on dev by the registered rule.

| Arm | Model | Facts | Probes | Abstentions | Citation complete | Invalid refs | Failed gates |
|---|---|---|---|---|---|---|---|
| tuned (rev 2) | Qwen2.5-Coder-14B | 45/45 | 14/16 | 4/45 | 37/45 | 5/90 (0.056) | invalid_citations |
| v4 prompt (rev 0) | Qwen2.5-Coder-14B | 44/45 | 14/16 | 2/45 | 16/45 | 0/21 | citation_complete |
| tuned (rev 2) | Qwen3-14B | 44/45 | 5/16 | 4/45 | 40/45 | 1/84 | probe_abstention |
| v4 prompt (rev 0) | Qwen3-14B | 45/45 | 3/16 | 3/45 | 40/45 | 1/57 | probe_abstention |

The tuned prompt lifted Coder's citation presence from 16 to 37 of 45 with probes unchanged; it did not reduce abstentions (2 to 4), and
v4's 6/44 abstentions did not reproduce on these questions. The one failed gate is one reference over (5 of 90, limit 4.5): a made-up
range, a malformed range, two test names written as paths, one file mixed up. **No model qualified; no gate was relaxed.**
Limits: Flask-heavy architecture split, related questions share functions, unreviewed labels, two repos and one language.

**Provisional choice (user, 2026-11-03):** Qwen2.5-Coder-14B-Instruct (revision `aedcc2d42b62...`, prompt arm 2) with the agent-side
citation checker `agent/citations.py`. Exploratory, not a registered result: on the same 41 answered tuned-Coder outputs the checker
removes all 5 invalid references, attaches 2 citations from chunks the answer names, and leaves every answered question with a valid
citation (41/41, was 37/45 before counting abstentions); it was written after seeing these failures, so it needs fresh questions.

## Commit depth and BM25 weight: frozen, GPU run pending (2026-10-26) [superseded by the section above]

The original Phase 5 v2 A100 run is now independently reconstructed locally from its
raw rankings/traces, including BM25 scores, active settings/model revisions, all 54 declared
historical source hashes and the original fixed 8B rule. This is an audit of existing GPU
outputs, not fresh inference. The source-path cohort is the headline; it includes tests and
configs outside the documented doc/changelog exclusion, not independently labelled code bugs.

**Historical controls, depth50 / retrieval_k=50, held-out test:**

| Cohort | Method | n | Hit@5 | MRR@10 |
|---|---|---:|---:|---:|
| source | dense | 208 | 0.7788 | 0.6676 |
| source | hybrid | 208 | 0.7644 | 0.6484 |
| source | rerank4b | 208 | 0.8990 | 0.8169 |
| source | rerank8b | 208 | 0.9038 | 0.8446 |
| all | dense | 263 | 0.7947 | 0.6845 |
| all | hybrid | 263 | 0.7947 | 0.6773 |
| all | rerank4b | 263 | 0.9049 | 0.8228 |
| all | rerank8b | 263 | 0.9049 | 0.8473 |

The original 8B-vs-4B MRR change is +0.02449, paired 95% CI [+0.00299, +0.04617]
(263 questions / 254 groups). The old frozen rule passes narrowly. Hit@5 is identical;
this supports a small ranking improvement, not more top-five hits. Link-derived labels,
current issue bodies, old T3 overlap and own-fix rather than similar-past-fix construct limits
remain. There are 55 test queries with docs/changelog-only gold, outside the 208-source subset.

The new protocol is `datasets/phase5_commit_depth_v1/protocol.json`, SHA-256
`87181ddde8e33032ffa0d83f35f68d9c0f1925410116a271abbb0e163f693d17`. Its freeze covers the protocol, source/timing cohorts,
original dataset freezes, archived depth50 rows/traces and a compact snapshot preserving all
54 historical source file bytes. Original v1/v2 inputs and result folders remain unchanged.
The original downloaded result ZIP is identified by SHA-256 in the protocol.

New **depth quality** uses one retrieval_k=200 fused pool, weights dense 1.0 / BM25 0.3,
and one top-200 scoring call per query per judge. Depth50/100/200 sort the corresponding
prefixes of the same scores, with probability-descending / fused-rank-ascending ties.
8B is primary; 4B secondary. Historical depth50 is a separate control: shorter first-stage
lists can change the fused order, so it is not the new prefix50 baseline.

**Weight** is a separate first-stage-only grid at retrieval_k=50: 0, 0.1, 0.3, 0.5, 1.0.
Selection uses only 66 dev queries; keep 0.3 unless another weight gains >=0.01 MRR.
CPU replay of the original saved lists gives dev MRR 0.733748 at 0.3 and 0.756728 at 0.1
(+0.022980), so that replay selects 0.1. This is **not a new GPU result or production choice**.
The notebook repeats first-stage retrieval and selection; if it chooses a different weight,
it runs fresh 8B depth50 calls for that weight and 0.3, comparing on test. Test rows for every
weight are descriptive, never used to select. This cannot establish the joint effect of a
new weight and depth200; no joint setting is adopted from these separate experiments.

The prechosen depth rule requires pooled test Hit@5 gain >=0.02 with positive paired lower
bound, MRR delta >=-0.005, nonnegative source-subset Hit@5 change, and no repo Hit@5 loss >0.02
at >=30 connected test groups. Smaller strata remain descriptive. Choose 100 if both pass,
unless 200 beats 100 by >=0.01 Hit@5 with positive paired lower bound. All intervals are
exploratory/pointwise and unadjusted. Passing changes no defaults before review and tests.

Timing is separate: 60 test queries in frozen round-robin order (Requests 23, Flask 23,
FastAPI 14), 3 rotated repetitions, 8B production calls at 50/100/200 with CUDA synchronization.
All timed depths keep retrieval_k=200, matching the quality prefixes; p50/p95 and ratios
are per repo and never averaged. The ratio baseline is **prefix50 with retrieval_k=200**,
not the current production retrieval_k=50, and A100 timing does not resolve laptop deployment.
Shared quality scores do not measure depth latency. Long-query budget views are retained.

Use `notebooks/phase5_commit_depth_on_colab.ipynb` plus `phase5_commit_depth_source.zip`.
It uses uv, explicit downloads/revision checks, separate model processes, frozen bundled
inputs without tokens, and matching-source Drive checkpoints. Return `phase5_commit_depth_results.zip`.
The CLI sample now includes its full bug report and a query JSON; it uses unchanged default
hybrid search, not an experimental depth. No answer-quality claim follows from that sample.

Raw historical audit, source/all cohort tables, dev replay and unchanged 120-question code
regression are in `retrieval_results/phase5_commit_depth_preflight/`. No new depth50/100/200
GPU pair scores, conditional-weight reranking or timings have been measured. Defaults and
Phase 5 checkbox are unchanged; manual scale and independent labels remain open.

Final validation: **460 passed, 1 network deselected in 351.26 s (5:51)**; Ruff lint/format clean (**217 files**), mypy **69 source files** clean. All **57 current source hashes** match the new ZIP. All **113 original v1/v2 dataset/result files** are unchanged. Production config/retriever/CLI hashes are unchanged. Package hashes and checks: `eval/retrieval_results/phase5_commit_depth_preflight/validation.json`.

## Historical Phase 5 review correction: v2 pre-GPU BM25 checkpoint (2026-10-26)

This section records the earlier checkpoint; current GPU/control status is above.

The scrub previously removed whole closing-reference lines, including adjacent fix descriptions.
Production and evaluation now remove reference spans and retain the description. Other issue
reference formats and email-like strings in searchable diff text are also scrubbed. Therefore
this before/after measures the complete v2 formatter correction, not only one regex replacement.
V1 inputs, raw results and source snapshot remain unchanged historical evidence.

V2 protocol: `datasets/phase5_bugfix_v2/protocol.json`, SHA-256
`11910a7b0db2458c0446691b856d2c6a4b8a9bcea976392eeb45f4d615ba1ef8`. The same 329 queries, gold labels, connected groups and splits
are reused byte-for-byte; **this is not a fresh holdout**. No weights were tuned on test.
Evaluation candidates depend only on the broad message heuristic, never the gold link; every
gold passes independently. Production may additionally use verified links. The heuristic remains
broad and the pool sizes differ, so do not compare repositories as equivalent difficulty.

**Full-query BM25, test split, v1 -> v2:**

| Repo | Test n | Pool | v1 Hit@5 | v2 Hit@5 | v1 MRR@10 | v2 MRR@10 | v2 File Hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Requests | 113 | 1,101 | 0.4867 | 0.4867 | 0.3498 | 0.3683 | 0.8142 |
| Flask | 136 | 1,181 | 0.7868 | 0.8235 | 0.6656 | 0.6995 | 0.9412 |
| FastAPI | 14 | 2,442 | 0.7143 | 0.7143 | 0.5952 | 0.5595 | 0.7857 |

Paired test MRR changes for the full formatter correction:

- all: MRR delta +0.0236, paired 95% CI [+0.0116, +0.0372] (263 questions, 254 groups).
- fastapi/fastapi: MRR delta -0.0357, paired 95% CI [-0.1071, +0.0000] (14 questions, 14 groups).
- pallets/flask: MRR delta +0.0339, paired 95% CI [+0.0159, +0.0555] (136 questions, 130 groups).
- psf/requests: MRR delta +0.0185, paired 95% CI [+0.0047, +0.0355] (113 questions, 110 groups).

These intervals describe this reused benchmark, not independent new-bug validation.

The primary task is **issue -> its own linked fix**. It does not validate finding useful
similar historical fixes for a new bug. **File Hit@5** is a secondary localization proxy:
any top-five commit shares an old/new changed path with any gold, including the exact gold
itself. Same-file changes need not address the same bug. A time-aware, independently judged
alternative-fix benchmark is still required to assess the product goal directly.

`retrieval_results/phase5_preflight_v2/` stores corrected raw rankings/scores/source,
`bugfix_summary/summary.json`, `formatter_comparison.json` (paired v2-minus-v1 group CIs),
and `query_length_summary.json`. The prechosen **BM25_2000** row uses the first 2,000 query
characters; primary BM25 still sees the full query. This sensitivity check does not equalize
neural token budgets (embedding: 2,048 tokens; rerank: first 2,000 characters, then 2,048-token
pair). Length strata are descriptive, not independent experiments. Group-bootstrap intervals
are pointwise, exploratory and unadjusted. No universal improvement follows from these rows.

All 120 old code questions were rerun: **0 changed rankings and 0 changed metrics** across
Requests, Express and Flask. This is BM25 regression evidence, not a neural measurement.

The v2 8B rule was frozen before retrieval and any Phase 5 GPU result. Pooled test MRR gain
must be >=0.02 with a positive paired CI lower bound and >=20 connected groups. Repo/link-path
Hit@5 loss guards apply only at >=30 connected test groups; the 14 FastAPI test questions are
descriptive, with no safety conclusion. No automatic default change follows from passing.

Next: `notebooks/phase5_bugfix_v2_on_colab.ipynb` plus `phase5_bugfix_v2_source.zip`, on A100.
Six fixed methods: BM25, BM25_2000, dense, hybrid, 4B and 8B at the same depth 50. Bundled inputs
require no GitHub or paid model key. Current v2 source rejects v1 corpus reconstruction;
reproduce v1 with its archived source ZIP. GPU inference, generated answer quality, deployment
latency, manual scale and independent label review remain pending. Phase 5 stays unchecked.

## Historical Phase 5 v1 BM25 baseline (superseded by v2; 2026-10-25)

The following records the original run. Its corpus scrub is flawed; use v2 above for the next GPU comparison.


Commit indexing, both issue-link paths, typed SQLite link storage and production commit
search are implemented. Phase 5 is **not complete**: the fixed GPU matrix and its real CLI
sample have not run. No commit-reranking improvement for the 8B is claimed.

Frozen inputs: `datasets/phase5_bugfix_v1/`, protocol SHA-256
`314ab8b93072dac058583f8f4c3054543a62c316828cf2d9afdd129a4762089c`.
329 questions / 319 connected groups: Requests 141, Flask 170, FastAPI 18. Dev has 66
questions; test has 263 (254 groups). Counts fell short of the approximate 200/repo target;
the frozen selection/skips report why. FastAPI supplies only 14 test questions.

Queries use issue title/body. The `symbols_diff800` corpus uses commit message, files, real
AST symbols and scrubbed diff, without appended issue text or closing-reference lines/IDs.
From 13,712 frozen records, all 4,724 fix-candidate documents and every connected-group split
rebuild. Historical AST mapping was fixture-tested, not independently re-reviewed on every
commit. `is_fix` is a broad heuristic or a verified closing link, not a
manual judgement that every candidate is a true bug fix.

**BM25 only, held-out test rows:**

| Repository / link path | Test n | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Requests / direct | 113 | 0.2655 | 0.4867 | 0.5575 | 0.3498 |
| Flask / direct | 136 | 0.5735 | 0.7868 | 0.8456 | 0.6656 |
| FastAPI / via PR | 14 | 0.5000 | 0.7143 | 0.7143 | 0.5952 |
| Pooled test | 263 | 0.4373 | 0.6540 | 0.7148 | 0.5262 |

Raw rows, full BM25 candidate scores, run/source hashes, measured source ZIP, interval
tables and validation are in `retrieval_results/phase5_preflight/`. Reproduce with
the source and scripts archived with v1; current scripts target v2.
Intervals in `bugfix_summary/summary.json` resample connected evidence groups and are
exploratory/pointwise. All-question rows are descriptive; no method was tuned on test.

**Code retrieval regression (BM25, same 120 questions):**

| Repo | n | Before / after MRR | Before / after complete proof | Changed rankings / metrics |
|---|---:|---:|---:|---:|
| Requests | 42 | 0.4564 / 0.4564 | 0.5952 / 0.5952 | 0 / 0 |
| Express | 39 | 0.5965 / 0.5965 | 0.7692 / 0.7692 | 0 / 0 |
| Flask | 39 | 0.3813 / 0.3813 | 0.5897 / 0.5897 | 0 / 0 |

`eval/check_phase5_code_regression.py` regenerates the comparison against the immutable
Phase 4 preflight. This is lexical regression evidence, not a new neural or ANN measurement.

Run `notebooks/phase5_bugfix_on_colab.ipynb` with its matching separate Drive ZIP for the
pre-chosen BM25/dense/hybrid/rerank4b/rerank8b comparison at depth 50. Both judges use the
commit instruction; weights are unchanged. Paired intervals, the fixed 8B evidence rule,
raw traces, model revisions and a CLI commit citation are required. No production default
is changed automatically by this notebook. Bundled frozen inputs need no token in Colab.

Limits: link-derived gold is not independent bug-fix review; valid alternative fixes may
be unlabelled; current issue bodies can include post-resolution edits. Requests may overlap
older T3 questions, whose scores are not directly comparable. The small FastAPI subset
does not establish broad via-PR accuracy. Generated answer quality, deployment latency,
the manual scale tier and the 4B/8B commit comparison remain unmeasured. The user chose
open-source models; general answer-model selection remains Phase 7, with no Anthropic key.
Review entry point: `docs/PHASE5_BUILD_REVIEW.md`. Nothing committed.

## Phase 4 judge run: Qwen3-Reranker-8B beats the 4B and passes the frozen rule (2026-10-25)

**Decision: the 8B is eligible as the judge for code reranking; nothing is switched yet.** The
rule in `datasets/phase4_judge_v1/protocol.json` was fixed before the run and every check passed.
Production defaults are unchanged because `reranker_model` is shared with commit reranking, which
was measured with the 4B only. Deployment cost is undecided. Raw results, protocol copy and source
ZIP: `retrieval_results/phase4_judge_run/` (`aggregate/summary.md`, `decision.json`).

Run: A100, 150 frozen questions (30 fresh multi-block incl. 12 on Pallets Click, 120 regression),
hybrid first stage, rerank the fused top 20, top 10, no expansion/tie-break, one code instruction.
Only the judge differs. All 43 source hashes per repo match the local code; model revisions
recorded (4B `22e68366`, 8B `77d193c7`). Labels are author-checked; independent review pending.

| View | n | hybrid MRR | 4B MRR | 8B MRR | 8B minus 4B MRR [95% CI] | 4B proof | 8B proof |
|---|---:|---:|---:|---:|---|---:|---:|
| pooled | 150 | 0.726 | 0.860 | 0.926 | +0.066 [+0.026, +0.109] | 0.893 | 0.913 |
| fresh multi-block | 30 | 0.688 | 0.766 | 0.850 | +0.084 [-0.012, +0.193] | 0.667 | 0.733 |
| regression multi-block | 18 | 0.826 | 0.861 | 0.944 | +0.083 [-0.042, +0.222] | 0.667 | 0.722 |
| regression implementation | 24 | 0.819 | 0.785 | 0.927 | +0.142 [+0.052, +0.240] | 1.000 | 1.000 |
| regression test | 36 | 0.640 | 0.917 | 0.986 | +0.069 [0.000, +0.139] | 1.000 | 1.000 |
| regression ambiguous | 36 | 0.764 | 0.926 | 0.921 | -0.005 [-0.066, +0.057] | 1.000 | 1.000 |

Pooled complete proof +0.020 [0.000, +0.044]. I recomputed these from the per-question files: the
evidence is ranked first on 131 of 150 questions with the 8B (113 with the 4B); 28 questions improve
and 9 get worse (requests-expansion-003, requests-comparison-005 and -016, express-expansion-001 and
-004, express-comparison-001 and -008, express-comparison-020-para, flask-comparison-025), mostly
rank 1 to 2. The gain also appears on the 30 natural-wording paraphrases (MRR 0.844 -> 0.950) and on
the 90 original regression questions (0.896 -> 0.944), so it is not only the template wording.

Cost (indicative, one warm call per question): rerank p50 about 1.29-1.40 s with the 8B vs 0.90-0.97
s with the 4B (about +45%); hybrid about 70 ms. Peak GPU memory 29 GB with both judges and the
embedder resident (the 8B alone is about 16 GB in bf16). Still GPU-only.

Not fixed: 8 of the 30 fresh multi-block questions still have no complete proof (10 with the 4B);
that gap needs call edges (Phase 6). Not measured: answer quality, laptop speed, other languages,
independent label review. The earlier idea of an LLM judge stays the next candidate only if the
8B is judged insufficient.

## Phase 4 expansion GPU test: rejected (2026-10-25)

**Decision: `evidence_expansion` stays `none`; no further expansion tuning on these labels.** The
frozen rule was applied unchanged and callee expansion failed it. Raw results, labels and source
ZIP: `retrieval_results/phase4_expansion_run/` (`aggregate/summary.md`, `aggregate/decision.json`).
The earlier offline replay (section below) overstated the benefit.

Run: A100, expansion wired into the production retriever, plain rerank over 20 vs callee vs both,
30 fresh multi-block questions (Requests, Express, Flask and 12 on Pallets Click) plus the 120
regression cases. Labels and protocol were hashed before retrieval (not committed, per the user).
Labels are Codex-authored and author-checked; independent review is pending. Quality only.

| Cohort | Method | MRR@10 | Complete proof@10 |
|---|---|---:|---:|
| fresh 30 | plain | 0.766 | 0.667 |
| fresh 30 | callee | 0.761 | 0.567 |
| fresh 30 | both | 0.765 | 0.567 |
| regression 120 | plain | 0.883 | 0.950 |
| regression 120 | callee | 0.883 | 0.967 |

Fresh, callee minus plain: complete proof **-0.100 [-0.242, +0.034]** (one question better, four
worse: express-expansion-002, click-expansion-003 and -005, requests-expansion-004; better:
click-expansion-001), MRR -0.005 [-0.016, 0.000]. By repo, proof plain/callee: Express 0.33/0.17,
Click 0.67/0.58, Requests 0.83/0.67, Flask 0.83/0.83. The regression multi-block gain reproduced
(+0.111, 0.667 -> 0.778) but the other regression strata did not change, and returned text grew
<1%. Pre-registered checks: fresh gain **failed**, lower bound positive **failed**, regression
strata passed, text growth passed, so callee is not eligible. I recomputed the fresh numbers from
`aggregate/per_question.json`; they match.

Reading: the offline gain came from the six questions it was designed on and did not generalise.
Name-matched callees add noise as often as evidence. The remaining multi-block gap needs real call
edges in both directions (Phase 6 symbol table). Not measured: answer quality, latency.

**Full reproduction audit:** [compact audit and provenance](retrieval_results/phase4_expansion_full/README.md),
[validation](retrieval_results/phase4_expansion_full/validation.json) and
[range-level case diagnoses](retrieval_results/phase4_expansion_full/case_studies.json).
The uploaded ZIP is 5,297,614 bytes / 38 members, SHA-256
`f3e745d316d2e5a509366eb43f8380ea4be017769184430cf99ded5d685585a2`.
The returned source ZIP is byte-identical to the prepared 158,096-byte package; all 45 source
hashes per run match local code at import. All 450 quality rows/traces, first-stage fusion,
pair parity, expansion reconstruction, metrics, text/displacement counts and aggregate JSON
reproduce. Clean pinned source reproduces all 1,787 chunk texts/metadata; gold/manifest/protocol
hashes validate. Cached vectors are finite/unit float32 at 512 dimensions. All four runs used
NVIDIA A100-SXM4-40GB with identical recorded embedding/reranker revisions. No GPU inference
was repeated locally, and this shared-score quality run provides no independent timing.

A separate direct bootstrap reproduces fresh proof delta **-0.1000 [-0.2424,+0.0345]** on
30 questions in 26 connected evidence groups. The interval spans zero: failure of the
pre-chosen gain rule does **not** establish a universal statistically significant harm.
It is sufficient to reject adoption under this protocol. Callee text growth is +0.332% fresh
and +0.712% old, separately; all five old-stratum guards pass. Secondary both also scores 17/30
fresh proofs, and cannot replace the primary.

The mechanism is required tail evidence being displaced by unscored additions: a mean 3.73
insertions on fresh questions. Four complete proofs lose application-setting branches,
context object search/creation, parser construction or digest challenge fields; Click-001
gains its missing parameter-memo helper. Express-005 loses its only overlapping chunk at
rank 7 as well. These are trace diagnoses, not grounds for another tuning cycle. Real call
edges remain Phase 6 work; their benefit and generated-answer quality are unmeasured here.

Phase 5 is the next implementation task; [handoff](../docs/PHASE5_HANDOFF.md). GITHUB_TOKEN is
unset locally; API-backed PR/issue linking needs it, local git and fixture development do not.
No runtime/default/label changes or commits accompany this audit. The prior runtime suite
remains 410 passing tests; it was not rerun for these data/documentation additions.

## Historical preparation: fair expansion test before the GPU result (2026-10-25)

**Current state:** expansion is wired into actual production code search as an explicit option,
`evidence_expansion=none|adjacent|callee|both`, default **none**. It only runs after requested,
enabled code reranking, preserves the kept first five hits and result count, scopes additions
to the query's repo/filters, and marks added evidence as unscored. The ordinary hybrid/laptop
path is unchanged. No claim that expansion should be enabled follows from the earlier replay.

Following [the expansion brief](../docs/CODEX_HANDOFF_EXPANSION.md), froze
[30 new multi-block questions](datasets/phase4_expansion_v1/README.md) before query retrieval:
six each for Requests, Express and Flask, twelve for Click. Click is a new evaluation repo, pinned at
`2247b35ea1c47c727d7a06e51fa280e12a863ff6`. Required source ranges are hashed/author-checked;
all 30 proofs are complete somewhere in the production corpus and none is complete in a single
chunk. There are 26 connected fresh evidence groups; corpus counts are 274/ 384/429/700. The
prior 120 cases are copied byte-for-byte as regression guards. Prior source overlap is recorded;
new wording and same-assistant review do not establish independence.

The fixed quality-only matrix is **plain rerank20 / rerank20+callee / rerank20+both**, 150
questions and 450 outputs. The GPU notebook calls production search and builds its expansion
index from stored chunks; the model score cache shares identical query/instruction/pair inputs.
This run does not measure latency. The analyzer audits source/protocol/labels/chunks, first-stage
fusion, pair probabilities, reconstructed expansion, all metrics, text volume and displacement.

**Pre-registered primary rule:** callee must improve new 30 complete proof@10 by at least +0.05
with paired group-bootstrap lower 95% bound above zero; no pooled old stratum may lose more
than 0.02 MRR or proof, including the old multi-block stratum; returned text including headers
may grow at most 10%, separately on the fresh 30 and old 120. `both` is secondary, not a replacement
primary. Invalid/incomplete runs cannot support a decision. If rejected, leave expansion off
and proceed to Phase 5 without tuning against these labels. No default changes automatically.
Independent review and eventual deployment assessment remain open; A100 is experiments only.

Protocol and manifest were frozen/hashed on disk rather than committed because the user
forbids git commits. Digests are in datasets/phase4_expansion_v1/freeze.json; old labels and
evaluated source ZIPs are preserved. No new dependencies or production-index writes.

**Measured local BM25 preflight**, after freezing (not the rerank/expansion primary):

| New multi-block questions | Count | MRR@10 | Complete proof@10 |
|---|---:|---:|---:|
| Requests | 6 | 0.7917 | 0.3333 |
| Express | 6 | 0.5208 | 0.5000 |
| Flask | 6 | 0.5889 | 0.6667 |
| Click | 12 | 0.5444 | 0.2500 |
| Pooled fresh | 30 | 0.5981 | 0.4000 |

Old 120 BM25 remains 0.4776 MRR / 0.6500 proof, reproducing the previous rounded 0.478/0.650
header-off baseline. [Local results and before/after table](retrieval_results/phase4_expansion_preflight/README.md)
also verify default-off production ordering by holding the recorded old 20-pair pools and
probabilities fixed: **all 120 rankings and every metric reproduce unchanged**. Per-repo MRR
before/after: Requests 0.8889/0.8889, Express 0.9188/0.9188, Flask 0.8419/0.8419. This is wiring
validation, not new neural/ANN evidence or timing. An offline-model test also exercises all 450
runner outputs and full source/corpus/metric/gate audit; its synthetic quality is not a benchmark.

The next-run pair is **notebooks/phase4_expansion_on_colab.ipynb** and
**notebooks/phase4_expansion_source.zip**. Upload the matching ZIP to My Drive, run the notebook
on A100, and return **phase4_expansion_results.zip**. GPU expansion results/adoption are pending.

Validation after preparation: **410 tests passed, 1 network deselected in 111.76 seconds**;
Ruff clean with **169 files formatted**; mypy **54 maintained source files clean**. All live
notebooks were regenerated after source changes. The new notebook is **7,317 bytes** and source
ZIP **158,096 bytes**, SHA-256 `4e772036c1e5e6c5df9ffc40054d786ccd37d6bc75e244518b39e8d49eef3402`.
No new dependencies, staging or commits. Earlier measured source snapshots remain preserved.

## Phase 4 evidence expansion (offline): helps reranked lists slightly, hurts first-stage lists (2026-10-25)

**Historical checkpoint: not adopted; the module was not yet wired into the retriever.**
It is now connected behind an off-by-default flag for the new fair test described above.
Offline on the saved top-10 lists of the header-off run (no GPU), 120
cases, `eval/analyze_expansion.py`, output `retrieval_results/phase4_header_run/expansion_analysis.txt`.
Settings were fixed before running (keep top 5, expand from the top 3 hits, at most 2 added per hit,
list length unchanged), then not tuned.

| Base list | Expansion | MRR@10 | Complete proof@10 | Proof change [95% CI], better/worse |
|---|---|---:|---:|---|
| plain rerank | none | 0.883 | 0.950 | baseline |
| plain rerank | adjacent chunks | 0.883 | 0.950 | +0.000 [-0.025, +0.025], 1/1 |
| plain rerank | callee names | 0.883 | 0.967 | +0.017 [0.000, +0.043], 2/0 |
| plain rerank | both | 0.883 | 0.942 | -0.008 [-0.036, +0.018], 1/2 |
| hybrid (no rerank) | adjacent | 0.732 | 0.833 | -0.050 [-0.099, -0.008], 1/7 |
| hybrid (no rerank) | callee | 0.729 | 0.833 | -0.050 [-0.108, +0.008], 3/9 |

* On reranked lists only callee expansion helps: complete proof on multi-block questions +0.111
  [0.000, +0.278] (2 of 18 questions: express-014 and flask-014). Returned text size is unchanged.
* On first-stage lists it hurts: each added chunk displaces a ranked 6-10 hit that often holds
  the needed evidence.
* Of the six multi-block questions with no proof, two are fixed. express-018 is reachable but
  falls outside the 2-per-hit allowance. express-013 (`render` is defined in many chunks, so
  name matching skips it), express-016 (`send`) and express-017 (the needed chunk is the CALLER,
  `sendStatus`) need a real call graph in both directions, which is Phase 6 symbol-table work.
* The design was chosen after seeing these failures on the same questions, and two questions is
  too few to adopt a default. A fair test needs new multi-block questions.

## Phase 4 header run: the `# defines:` line is neutral, so it stays off (2026-10-25)

**Decision: `Settings.header_defines` stays False.** Raw results, source ZIP and the paired
analysis: `retrieval_results/phase4_header_run/` (`analysis.txt` from `eval/analyze_header_run.py`).

Run: A100, the 120 policy cases, each repo indexed twice (header off, header on), first stage
only (`hybrid`) and plain reranking over the fused top 20. All 41 source hashes per run match the
local code. Quality only; labels are author-checked.

| Policy | Header on minus off, MRR@10 [95% CI] | Complete proof@10 [95% CI] | Better / worse |
|---|---|---|---|
| hybrid | -0.001 [-0.038, +0.032] | -0.008 [-0.027, +0.000] | 11 / 9 |
| plain rerank | +0.011 [-0.017, +0.036] | -0.017 [-0.043, +0.000] | 7 / 2 |

By repo (plain rerank): Requests +0.004 [-0.069, +0.065], Flask +0.030 [0.000, +0.070],
Express +0.000 (identical: the `# defines:` line needs tree-sitter definitions, and Express's
`defineGetter(...)` and `res.x = function` style gives almost none). The mild MRR gain on
implementation (+0.042 [0.000, +0.104], 2/0) is offset by lost complete proofs: Requests 1.000 ->
0.952 (requests-comparison-004 and -017 under plain rerank) and one Flask question under hybrid.
The motivating cases moved only partly: requests-027 improved (0.5 -> 1.0 for both policies),
requests-026 did not under plain rerank (0.5 -> 0.5; its paraphrase went 0.5 -> 1.0).

No policy shows a reliable gain, and complete proof drifts down, so the extra header line is
not adopted. The remaining misses need a different lever, e.g. expanding a hit to its enclosing
definition or sibling chunks, or a stronger reranker. Earlier BM25-only result: MRR 0.478 -> 0.472.

## Phase 4 rerank policy run: no instruction or ordering change is reliably better (2026-10-25)

**Decision: keep plain P(yes) ordering and the single code instruction.** The first-stage
tie-break default is now 0 (it had been 0.25 on offline evidence); `rerank_instruction_mode`
stays `default`. Reranking remains opt-in. Raw results, source ZIP and paired analysis:
`retrieval_results/phase4_policy_run/` (`analysis.txt` from `eval/analyze_policy_run.py`).

Run: A100, Qwen3-Reranker-4B over the fused top 20 through the production retriever, five
pre-registered policies, 120 cases (the frozen 90 plus 30 natural-wording paraphrases of the
implementation and test questions). All 41 source hashes per run match the local code. Quality
only. Labels are author-checked, and the paraphrases and the detector have the same author.

| Policy | MRR@10 (120) | Minus plain, MRR [95% CI] |
|---|---:|---|
| plain | 0.883 | baseline |
| tiebreak (0.25) | 0.890 | +0.006 [-0.013, +0.026] |
| intent | 0.908 | +0.025 [+0.008, +0.045] |
| source_aware | 0.890 | +0.007 [-0.015, +0.033] |
| source_aware_tiebreak | 0.886 | +0.003 [-0.014, +0.020] |

**`intent` gains only where it was fitted.** On the 12 template implementation questions it adds
+0.167 [+0.042, +0.292] (4 better, 0 worse). On the 30 paraphrases it adds +0.000: the detector
never fires on the 12 natural implementation paraphrases and fires on 11 of 18 test paraphrases
(one better, one worse). It changes nothing on ambiguous, multi-block or example questions, so it
does no harm, but the gain does not carry to real wording.

**`source_aware` is flat.** Implementation +0.062 [-0.014, +0.181] (3 better / 1 worse),
paraphrased implementation +0.097 [0.000, +0.236], test -0.014 [-0.042, 0.000], ambiguous -0.014
[-0.045, +0.010]. Complete proof rises by one question (+0.008, multi-block +0.056 [0.000, +0.150]).

**The tie-break did not reproduce.** Offline on saved scores it looked like +0.007 to +0.015; live it
is +0.013 on the original 90 (3/1) but -0.017 on the paraphrases and -0.028 [-0.069, 0.000] on test
questions (0 better, 2 worse). Weight 0.25 had been picked on the same original questions, so that
gain was partly selection. Treat it as zero.

**What this means for the 14 real misses.** Neither a different instruction nor a first-stage
tie-break fixes them reliably. The remaining levers are structural: chunking that stops small
functions being merged under an unrelated header, evidence expansion for multi-block questions, or
a stronger reranker. Complete proof is unchanged by every policy except for one question.

## Phase 4: evaluation complete, rerank default pending (2026-10-25)

**Decision:** retain current hybrid defaults and explicit opt-in code reranking. Adopt neither
an unconditional test-file penalty nor the reserved BM25 quota. The new run gives positive
ranking evidence for rerank20; default promotion still needs independent label review and
production-path/deployment latency evidence. The evaluation is complete; the decision about
enabling reranking by default remains pending. Deployment hardware is undecided; the user
confirmed that A100 is for experiments only.

Audited reports: [run, provenance and validation](retrieval_results/phase4_v2_full/README.md),
[pooled fresh scores](retrieval_results/phase4_v2_full/aggregate/summary.md),
[paired differences](retrieval_results/phase4_v2_full/aggregate/comparisons.json),
[recorded production decision](retrieval_results/phase4_v2_full/production_decision.json).
The submitted ZIP is 4,609,914 bytes; SHA-256
`6f15971e3675ea887b397739bb4939f974ba59453b64b0eb3ad32b2f77ea75f1`.

### Direct rerank20 versus rerank50 check (original v1 A100 run)

[Paired depth comparison](retrieval_results/rerank_depth_comparison/README.md) rescored all
**160 method/question rows** from the original two-repo experiment and reproduced its stored
bootstrap summaries. All 34 source hashes per run match the preserved v1 source ZIP. These
are the old 40 questions per repo, frozen v1 gold and common headers, **not the new 15-question
ambiguous-intent set or the 49 fresh v2 questions**.

| Repo / original all questions | MRR20 | MRR50 | Gold completeness20@10 | Gold completeness50@10 |
|---|---:|---:|---:|---:|
| Requests / 40 | 0.8800 | 0.8779 | 0.925 | 0.900 |
| Express / 40 | 0.7692 | 0.7679 | 1.000 | 1.000 |

Paired MRR difference **50 minus 20**: Requests −0.00208 [−0.00610, 0.00000], Express
−0.00125 [−0.00375, 0.00000]. Test-only MRR is Requests 0.8486 → 0.8486 and Express
0.7403 → 0.7382. Every interval includes zero; do not call twenty significantly better.
Depth50 lowers first-hit rank on Requests-035 (3 → 4) and Express-007 (4 → 5), while the
Requests completeness tradeoff includes losses on 001/007 and a gain on 014. Broad old labels
have known review caveats. No observed ranking gain here justifies preferring fifty.

**No measured speedup can be reported:** both quality views reuse one fifty-pair model call.
Twenty scores 60% fewer pairs if run separately, but neither latency nor speedup was isolated
by this original experiment. The new production-path notebook independently runs/times hybrid,
rerank20 and rerank50 on the cue-free set; its GPU results remain pending. No default changed.

### Provenance and audit

The three runs used an **A100-SXM4-40GB**, Qwen3-Embedding-0.6B at 512 dimensions/float32,
and Qwen3-Reranker-4B (CUDA bfloat16, batch size 4), both with 2,048-token caps.
The task instruction is the corrected code instruction. Model revisions are recorded in
`preparation.json`: embedding `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, reranker
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`. At import, all **38 source hashes/run** matched local code
and the frozen [v2 source ZIP](retrieval_results/phase4_v2_full/phase4_v2_source.zip).
Dataset hashes and the pre-chosen matrix match; no weights were tuned on this run.

Pinned source, gold and production chunking validate locally: Requests **274 chunks**, Express
**384**, Flask **429**. All **1,032 scored rows**, required/supporting and frozen-v1 metrics,
bootstrap intervals, paired/pooled comparisons, decisions and warm latency quantiles reproduce.
All **1,548 primary calls** are present (129 questions × 4 methods × 3 repetitions), with full
first-stage and pair traces. Top-10 order is unchanged between repetitions. Neural inference
and GPU wall-clock timings were not independently repeated locally.

Cached vectors are finite/unit-normalized; exact top-10 sets agree with the recorded live
queries (maximum cosine difference 4.77e-7). ANN recall is 1.0 at 1/5/10/50 on these small
indexes: **plumbing validation, not evidence of ANN quality at scale**. No recorded primary
pair was truncated (maximum 794 tokens); maximum embedded document is 1,419 tokens.

### Fresh questions: reranking improves ranking

[V2 labels](datasets/phase4_v2/README.md) contain **49 new questions** and 80 legacy regressions.
Requests has 13 new questions (10 tests, 3 symbols), Express 13 (7 tests, 3 examples, 3 symbols),
and Flask 23 (7 tests, 3 examples, 3 symbols, 10 other implementation questions). Symbol lookup
is a subset of implementation, not an additional disjoint stratum. All labels remain
**author-checked; independent review pending**. All **24/24 fresh test-target questions
explicitly contain “test” or “tested”** (10 Requests, 7 Express, 7 Flask). Their strong scores
assess explicit test intent, not ambiguous implementation-versus-demonstration intent.
Fresh means newly authored, not guaranteed
unseen model training data. Some symbol questions target previously assessed code.

All primary policies use the same AST chunks with production headers, dense/BM25 top50,
fixed weights 1.0/0.3 and RRF k=60. Rerank20 scores the fused top20; quota reserves the first
five BM25 chunks and fills a deduplicated pool of twenty from the fused ordering.

| Repo / fresh questions | Dense MRR@10 | Current hybrid MRR@10 | Rerank20 MRR@10 | Quota + rerank20 MRR@10 | Rerank20 − hybrid [95% CI] |
|---|---:|---:|---:|---:|---|
| Requests / 13 | 0.673 | 0.699 | 1.000 | 1.000 | +0.301 [+0.147, +0.455] |
| Express / 13 | 0.621 | 0.635 | 0.949 | 0.949 | +0.314 [+0.167, +0.468] |
| Flask / 23 | 0.651 | 0.739 | 0.957 | 0.957 | +0.217 [+0.072, +0.362] |
| Pooled / 49 | 0.649 | 0.701 | 0.966 | 0.966 | +0.265 [+0.175, +0.355] |

Required ≥90%-complete gold@10 is **1.000** for dense and both reranked policies, **0.980**
for current hybrid. The latter misses Flask fresh-013 at top10; its gold chunk is in the
20-candidate pool and reranking promotes it to first. This run supports a ranking gain from
reranking on the labelled sample. It does not compare chunkers or establish a new AST gain.

| Fresh stratum | Questions | Hybrid MRR@10 | Rerank20 MRR@10 | Paired difference [95% CI] | Hybrid → rerank20 completeness |
|---|---:|---:|---:|---|---|
| Tests | 24 | 0.705 | 1.000 | +0.295 [+0.174, +0.424] | 0.958 → 1.000 |
| Examples | 6 | 0.681 | 0.889 | +0.208 [+0.014, +0.431] | 1.000 → 1.000 |
| Explicit symbol lookup | 9 | 0.667 | 1.000 | +0.333 [+0.130, +0.537] | 1.000 → 1.000 |

Pooled numerical promotion checks pass, including coverage and test/example guards. Requests
has no example stratum, so its per-repo checklist correctly does not pass all guards. Six
example questions are limited evidence, and zero observed coverage loss is not a universal
no-regression guarantee. Intervals resample evidence groups, are pointwise/exploratory and
are not multiplicity-adjusted. Repetitions and strata are not independent experiments.

### Latency and deployment limits

Warm sequential single-query timings include query embedding, retrieval/fusion and real pair
scoring, with CUDA synchronization. Models are resident. Token diagnostics/report writes,
cold loading and optional symbol/top50 diagnostics are excluded. These are per-repo quantiles
from 3 repetitions per fresh question; do not average percentiles across repositories.

| Repo | Hybrid p50 / p95 ms | Rerank20 p50 / p95 ms | Quota + rerank20 p50 / p95 ms |
|---|---:|---:|---:|
| Requests | 61.1 / 63.7 | 925.4 / 1028.2 | 930.0 / 1016.8 |
| Express | 61.5 / 63.3 | 975.6 / 1028.3 | 969.5 / 1008.9 |
| Flask | 61.7 / 63.3 | 888.7 / 944.1 | 888.7 / 945.7 |

Reranking adds roughly **0.83–0.91 seconds at p50** even on the A100. Cold first-pool scoring
is 26.05 / 4.79 / 4.08 seconds in the separate Requests/Express/Flask processes; these are not
standalone model-download or service-startup benchmarks. Shared-process peak allocated CUDA
memory is 12,689,126,400 / 12,579,308,032 / 12,383,095,808 bytes, including resident embedding
and reranking models. Concurrency, production service p95, throughput and laptop speed are unmeasured.

**Production parity limit:** current CLI `--rerank` normally scores 50 candidates, and returns
8 by default. This experiment scores 20 and evaluates top10. Its latency and gains are not a
measurement of the CLI's top50/default-output path. Before deploying rerank20, expose a separate
candidate-depth control and verify that path on intended hardware; no such change was made here.

### What failed, and why

- **A blanket test penalty fails on test intent.** On the 24 fresh test questions, MRR drops
  **0.705 → 0.352**, paired −0.353 [−0.471, −0.237]; completeness drops 0.958 → 0.917.
  It helps implementation/example strata but suppresses valid test answers. Flask fresh-017
  falls from rank1 to outside top10. These supplemental pooled intervals are post-hoc
  diagnostics; the production choice is to keep the penalty at 1.0.
- **A reserved lexical quota adds no fresh benefit.** Both reranked policies have identical
  fresh MRR/completeness, although their candidate sets differ on 39/49 fresh questions.
  On legacy Requests-034, the raise site changes from rank1 to rank2. Its pair score drops
  from 0.94967 to 0.94660 and ties the exception-definition chunk, which appears earlier in
  the quota pool. The CUDA scorer uses bfloat16; different batches can change close scores.
  This trace shows an ordering/tie sensitivity, not evidence that the quota fixes retrieval.
- **Frozen implementation gold can rank below valid demonstrations.** Requests-036 goes from
  first gold rank2 to rank5: the reproducing test scores 0.99924, while the validation chunk
  scores 0.81757. Express-028 goes rank9 → rank10: redirect/location tests outscore the
  `Location` setter. Express-036 improves rank4 → rank3 but still puts tests first. These
  gold chunks are already in the pool; expanding ANN depth does not explain these gold-rank
  losses. The wording does not explicitly exclude demonstrations; a relevant test is not
  inherently a wrong answer. Alternative-evidence adjudication is pending.
- **Pair relevance does not ensure multi-block completeness.** Requests-029 retains its main
  function at rank1 but pushes required helper spans from ranks4/7 to 11/14. Completeness
  falls 1.000 → 0.500. Legacy Requests-036 remains complete despite worse first-hit rank.
  Flask fresh-007 also moves a test above the implementation (rank1 → rank2), with a small
  pair-score margin 0.99782 vs 0.99737. Do not claim reranking never regresses.
- **Symbol routing is promising, still diagnostic.** On nine explicit-name lookups, the
  source-derived diagnostic moves dense MRR 0.587 → 1.000. It scans source outside timed
  calls and is not a production symbol table/router. Broad name resolution belongs in Phase 6.

The four requested 028/036 case traces are in
[case_studies.json](retrieval_results/phase4_v2_full/case_studies.json). Frozen-v1 rescoring is
stored separately in each repo's `v1_regression.json`; revised required/supporting labels
must not disguise old-label regressions. Legacy-v2 MRR hybrid → rerank20 is Requests
0.760 → 0.893 and Express 0.496 → 0.792; legacy questions were already inspected, not fresh.

### Current follow-up: expanded 90-question depth comparison

The next run uses [90 frozen questions](datasets/phase4_comparison_v3/README.md), **30 per repo**,
with **75 additions and 15 retained questions**. The original 15-case set and v1/v2 labels remain
unchanged. The new diagnostic sample contains 36 ambiguous queries without source-type or symbol
cues, 18 multi-block workflows, 18 explicit test targets, 12 implementation queries and 6
example targets. Requests has no comparable indexed examples directory; examples are split
equally between Express and Flask. Evidence spans **58 repo/file pairs and 83 connected groups**.
All evidence hashes validate against clean pinned source; every question has a complete
accepted proof somewhere in the full production corpus. That is answerability, not top10 success.

Source-authored labels/protocol were frozen before expanded retrieval scoring. The same
assistant authored and source-reviewed them; independent review is pending. New wording is
not independence from previously inspected code: the manifest records old-v2 evidence overlap.
Six example cases remain a small guard sample. Ninety cases do not guarantee significance.

The matching next-run artifacts are **notebooks/phase4_comparison_on_colab.ipynb** and
**notebooks/phase4_comparison_source.zip**. Upload the latter to My Drive, select A100 and run
the notebook. Both rerank depths run independently through production search, alongside hybrid,
at fixed weights and first-stage depth. Three rotated repetitions produce **810 timed calls**;
quality uses repetition zero. Reports include direct paired 50-minus-20 intervals, each repo/type,
added75/retained15, complete-proof metrics and per-repo warm p50/p95. Results carry the source
ZIP, question/protocol snapshots and full traces. Labels and weights will not be tuned on this run.

[Expanded BM25 preflight](retrieval_results/comparison_v3_preflight/README.md) is measured:

| Repo | Questions | MRR@10 | Any complete accepted proof@10 |
|---|---:|---:|---:|
| Requests | 30 | 0.5009 | 0.5667 |
| Express | 30 | 0.6346 | 0.7333 |
| Flask | 30 | 0.3479 | 0.5333 |

No expanded GPU/rerank result or latency has been measured. All original 15 BM25 rankings
and metrics reproduce with **zero changes**. Production code/defaults are unchanged by this
expansion; reranking stays opt-in. A100 is experiment-only, deployment hardware undecided.
Phase 4 evaluation remains complete with the rerank-default decision pending; Phase 5 is unstarted.

Verification for the expansion: **350 tests passed, 1 network deselected in 94.44 s**;
Ruff clean, **135 files formatted**; mypy **47 maintained files clean**. The new source ZIP is
**118,440 bytes**, SHA-256 `aa47cbf96a17c445be55726a85138475aee6183434ae25c940d45b943b7322f9`;
the notebook is **7,664 bytes** with no encoded payload. No new dependencies, staging or commits.

### Earlier follow-up: label review, 15-case stress set and production depth

[Source review](PHASE4_LABEL_REVIEW.md) records a second pass on 15 previously evaluated ambiguous/multi-block
cases plus underspecified parameterized demonstrations. It is by the same assistant,
**not independent review**. Multipart, TLS and class-wide gold may require too much optional
context; newline rejection omits the regex proof; some top tests are acceptable behavioural
answers. Frozen v1/v2 gold and measured scores remain unchanged.

[15 new ambiguous-intent questions](datasets/phase4_ambiguous/README.md), five per repo, omit
“test”, explicit symbols and example/tutorial cues. They accept either a sufficient implementation
proof or a complete demonstration, with parameters/fixtures included. Pieces of different
proofs cannot jointly count as a complete answer. These are targeted paraphrases of inspected
behaviours, a **post-hoc stress set**, not an independent fresh holdout. Do not pool with v2.

Production now has `rerank_candidate_depth` / `REPO_RAG_RERANK_CANDIDATE_DEPTH` and CLI
`--rerank-depth 20` or `50`, separately from first-stage `retrieval_k`. The compatibility
default remains **50**; reranking still requires `--rerank`. A reranked request whose result
count exceeds its depth is rejected rather than mixing scored and unscored results.
The new runner calls the actual production path for hybrid, rerank20 and rerank50 and records
real warm p50/p95, pair scores and grouped intervals. Its compact A100 notebook and separate
Drive ZIP were `notebooks/phase4_ambiguous_on_colab.ipynb` and `phase4_ambiguous_source.zip`;
the expanded 90-question pair above is now the next-run workflow.
**The new GPU comparison has not run locally; no new rerank/latency numbers are claimed.**

Deployment remains undecided. Existing GPU opt-in reranking stays available; the laptop auto
path uses hybrid and skips the reranker. A100 is experimental hardware only. Default promotion
requires independent label/alternative-evidence review, the new comparison and timings on
actual chosen deployment hardware. No policy was enabled automatically.

**Measured regression check after the depth-setting change:** all 129 v2 BM25 rows and
rankings are unchanged. For the fresh subsets:

| Repo | BM25 MRR before → after | Required completeness before → after |
|---|---:|---:|
| Requests | 0.365 → 0.365 | 0.769 → 0.769 |
| Express | 0.567 → 0.567 | 0.846 → 0.846 |
| Flask | 0.545 → 0.545 | 0.826 → 0.826 |

Controlled replay through production search, holding logged dense lists and recorded pair
scores fixed, also reproduces all 129 hybrid and rerank20 top10 rankings and candidate order.
This is wiring validation, not new neural/ANN or latency evidence. A rebuilt live index can
change lower-rank ordering; the replay does not assume identical ANN order across builds.

[BM25-only stress preflight](retrieval_results/ambiguous_preflight/README.md) is source-validated
and saved. It does not measure the pending hybrid/reranking comparison. At that checkpoint the
generated source ZIP was **88,802 bytes**, SHA-256
`f34e3a4b0ddf168753de8f86fa4e7cf786db498193ed0b22f64b7fb8cde411a2`;
notebook **5,615 bytes**, with no Base64 cell. Live source ZIPs are regenerated after shared
eval changes; evaluated v1/v2 source snapshots remain separately preserved.

The next roadmap phase remains **Phase 5: commit ingestion and bug-report → past-fix retrieval**;
it has not started. Matched chunk budgets, more languages, ANN scale and generated answer
accuracy remain open limits. HNSW does not supply code call/import edges.

Verification after the depth/stress follow-up: **341 passed, 1 network deselected in
120.30 s**; Ruff clean with **123 files formatted**; mypy **44 maintained files clean**.
Frozen v1/v2 labels and results remain unchanged. Current source intentionally adds the new
production setting and stress harness; old source ZIPs preserve both evaluated versions.
No new dependencies, staging or commits.

## Phase 4: first real GPU run — completeness improves, comparison remains open (2026-10-25)

Historical checkpoint: the mixed-target follow-up and phase decision are now recorded above.

**Status:** the supplied `phase4_results.zip` contains valid full results for Requests and
Express, plus **separate reranker runs for both repos**. Main runs have `rerank: false`;
`rerank_experiment/*/full/run.json` has `rerank: true`. The broader comparison remains unresolved:
gold contains no test/example targets, budgets differ, and these are two repos/two languages.
Phase 4 stays unchecked pending the mixed-target follow-up. Production defaults are unchanged.

**Post-hoc failure research:** [diagnosis and improvement plan](PHASE4_FAILURE_ANALYSIS.md)
traces all original questions through the rankings. For AST/header-on, exact dense top50
contains ≥90% of every gold range on all 80 questions. Ranking, source intent and evidence
selection are immediate priorities. It also identifies overly broad gold in specific cases,
differences between common evaluation and production headers, and the production reranker's
default commit instruction. These diagnostics are not a new held-out benchmark.

Current full reports: [Requests](retrieval_results/colab_full/psf__requests/full/summary.md),
[Express](retrieval_results/colab_full/expressjs__express/full/summary.md).
Separate reranker reports: [Requests](retrieval_results/colab_full/rerank_experiment/psf__requests/full/summary.md),
[Express](retrieval_results/colab_full/rerank_experiment/expressjs__express/full/summary.md).

### Provenance and validation

- Hardware: **NVIDIA A100-SXM4-40GB**, Linux, Python 3.13.16; Qwen3-Embedding-0.6B,
  **512 dimensions, float32**. Chroma 1.5.9, sentence-transformers 5.7.0, tree-sitter 0.26.0,
  NumPy 2.1.3, torch 2.11.0+cu130. Repos remain pinned to Requests `611c6162cbc4` and
  Express `0dee2c52c7c8`; each has 40 questions, 16 dev / 24 test.
- At import, all **34 source hashes per run** matched local source after LF normalization. The
  original matching source is now preserved in `retrieval_results/colab_full/phase4_v1_source.zip`;
  current runtime code intentionally includes the v2 bug fix. Dataset hashes
  also match after CRLF→LF normalization, as used when building the uploaded source ZIP.
- Locally verified all **16 variants / 8,480 question-method rows**: exported chunk ids, text
  and metadata match pinned local chunking; all question metrics and grouped bootstrap
  summaries/strategy differences reproduce. Cached query/document vectors are finite,
  unit-normalized and 512-dimensional; exact dense top-10 and BM25 rankings reproduce locally.
  This validates stored inference outputs; neural inference itself was not re-run locally.
- Input archive SHA-256:
  `4a533beaded0125220e8f377c8b7d718bebd760c17034a083a7ee71998c51a2e`.
  Validation details and supplementary paired comparisons are saved beside the reports.

### Chunking: report completeness separately from ranking

Fixed **exact dense** search, all 40 questions per repo. Header-off was the predeclared primary
chunking comparison; header-on shows the incremental common-context policy. These are views of
one experiment per repo, not independent replications. Gold ≥90% means the question-macro
fraction of gold ranges reaching 90% line coverage in the top 10, not fully answered questions.

| Repo | Chunker | Chunks | Avg body chars | MRR@10, header off | Gold ≥90%@10, off | MRR@10, header on | Gold ≥90%@10, on |
|---|---|---|---|---|---|---|---|
| Requests | char | 331 | 1408 | 0.644 | 0.604 | 0.805 | 0.654 |
| Requests | lc_lang | 377 | 1121 | 0.697 | 0.621 | 0.786 | 0.625 |
| Requests | astchunk | 270 | 1478 | 0.692 | 0.742 | 0.809 | 0.842 |
| Requests | ast | 274 | 1483 | 0.737 | 0.754 | 0.840 | 0.842 |
| Express | char | 477 | 1291 | 0.318 | 0.600 | 0.358 | 0.725 |
| Express | lc_lang | 565 | 1031 | 0.341 | 0.575 | 0.410 | 0.725 |
| Express | ast | 384 | 1427 | 0.397 | 0.850 | 0.434 | 0.825 |

**Header-on exact dense, AST minus char**, paired evidence-group bootstrap intervals:

| Repo | Sample | ΔMRR@10 [95% CI] | ΔGold ≥90%@10 [95% CI] |
|---|---|---|---|
| Requests | all 40 | +0.035 [−0.066, +0.135] | +0.188 [+0.085, +0.292] |
| Requests | test 24 | +0.063 [−0.087, +0.217] | +0.167 [+0.021, +0.327] |
| Express | all 40 | +0.077 [−0.017, +0.176] | +0.100 [+0.000, +0.225] |
| Express | test 24 | +0.173 [+0.040, +0.313] | +0.167 [+0.042, +0.333] |

Header-off AST-minus-char completeness differences are **+0.150 [+0.043, +0.268]** on
Requests and **+0.250 [+0.125, +0.375]** on Express, all questions. Header-off MRR differences
are +0.093 [−0.026, +0.207] and +0.079 [+0.003, +0.159], respectively.

The strongest supported observation is **more complete retrieved evidence under these tested
defaults**, with no consistent ranking advantage. Absence of a significant ranking difference
is not an equivalence test. Reference astchunk matches our header-on Requests completeness
(both 0.842; paired delta 0.000 [−0.038, +0.038]), so this does not establish an advantage for
our implementation over reference cAST. Character budgets count all characters, whereas AST
budgets count non-whitespace characters: the run does not isolate boundary choice causally.
There is no reference astchunk/Express comparison because upstream lacks JavaScript support.

Some hybrid configurations reverse the ranking sign. With headers, BM25 weight 0.3 and test
multiplier 0.8, all-question Requests AST-minus-char MRR is **−0.074 [−0.137, −0.017]**.
Do not write a general “AST ranks better” claim or generalize to Java, Go or TypeScript.

### Retrieval settings: no universal best weight, and the penalty is biased

AST/header-on, all-question MRR: Requests **0.840 dense → 0.781 default hybrid**; Express
**0.434 dense → 0.494 default hybrid**, still below **0.599 BM25 alone**. On Requests the
dense-minus-default-hybrid MRR delta is +0.059 [+0.008, +0.117]; on Express it is
−0.060 [−0.112, −0.021]. The signals trade recall/completeness against early ranking.
Identifier-bearing wording may contribute to Express's lexical strength, but that cause has
not been isolated. Development-selected BM25 weights vary from 0 to 0.5 on just 16 questions.
Keep the current weight provisional; do not infer a universal optimum from these selections.

The 0.8 multiplier was selected for every strategy/header combination, but **all gold lies in
implementation files** (`src/requests/` or `lib/`). Down-weighting test paths is rewarded by
construction on this set, which contains no counterexample where a test/example is the answer.
Express AST/header-on default hybrid MRR rises **0.494 → 0.886**, but this is a biased-target
diagnostic, not production-policy evidence. **Do not adopt this penalty as a default.** Add
about 10 independently reviewed test/example-target questions per repo and report target-type
strata before reconsidering it; retain this original dataset as a frozen historical benchmark.

### HNSW: small-index plumbing check only

Reported ANN neighbour recall@1/5/10 is **1.0** for all 560 main query/configuration cases,
on indexes of **270–565 vectors**. This verifies the small-index retrieval plumbing. It does
not establish ANN recall at scale, demonstrate the best index choice, or show code-call links.
Whether searches traverse HNSW or a buffer is not measured here. A larger-corpus experiment
with explicit search/build settings and latency/recall measurement is still needed.

### Separate code reranker experiment did run

Qwen3-Reranker-4B reranked fixed AST/header-on candidates from default hybrid (BM25 0.3, no
test penalty), with a code-specific instruction. Both depths reuse scores for the same top-50
pool. No separate top-20 latency was measured.

| Repo | Split | Default hybrid MRR | Rerank 20 MRR | Rerank 50 MRR | Gold ≥90%@10: baseline / 20 / 50 |
|---|---|---|---|---|---|
| Requests | all 40 | 0.781 | 0.880 | 0.878 | 0.854 / 0.925 / 0.900 |
| Requests | test 24 | 0.781 | 0.849 | 0.849 | 0.875 / 0.917 / 0.875 |
| Express | all 40 | 0.494 | 0.769 | 0.768 | 0.950 / 1.000 / 1.000 |
| Express | test 24 | 0.592 | 0.740 | 0.738 | 1.000 / 1.000 / 1.000 |

Rerank-20 minus baseline MRR: Requests all +0.099 [−0.020, +0.222], test +0.067
[−0.095, +0.219]; Express all +0.275 [+0.144, +0.405], test +0.148 [−0.019, +0.310].
The held-out intervals span zero. Depth 20 has higher development MRR on Requests and ties
50 on Express; their test MRR differences are tiny. These measurements answer the first code
reranking experiment, but do not justify universal enablement. Keep code reranking opt-in.
Recorded top-50 scoring-call means were 2.346 s/query (Requests) and 2.894 s/query (Express),
potentially including first-load cost. These are not independent timings for each depth or
end-to-end notebook/indexing times.

All intervals above are pointwise/exploratory, with 10,000 group resamples and seed 42;
multiple comparisons are not corrected. Pooled rows are descriptive. The same questions
underlie headers, signals, splits and reranking; neither repeated views nor the separate
reranking run add independent questions. Coverage counts blank lines and partial touched
lines. Gold-author/wording bias and Requests's earlier role in model selection remain limits.

**Next:** freeze these reports, expand the gold set with test/example targets, and rerun the
mixed-target assessment. Add a matched-budget sensitivity study and an ANN scale test before
broader claims. No new GPU run is needed merely to prove that the separate reranker run exists.

## Historical BM25-only review: coverage and uncertainty (2026-10-25)

**Current saved reports:** [Requests](retrieval_results/bm25/psf__requests/summary.md) and
[Express](retrieval_results/bm25/expressjs__express/summary.md), with coverage, confidence
intervals, paired differences and per-question evidence. The original overlap-only snapshot
is archived in `retrieval_results/bm25_legacy/`. The earlier `bm25_review/` copy is a historical
snapshot of this same experiment, not an additional run for inference.

**At this earlier checkpoint, only BM25 had been measured.** Reports include source-line union
coverage, the fraction of gold ranges reaching 90% coverage, all 40 questions for fixed
configurations, and separate dev/test tables. Confidence intervals use **10,000 paired
evidence-group bootstrap resamples**, seed 42, 95% pointwise percentile bounds. Shared-answer
questions stay together: Requests has 32 groups overall / 20 on test; Express has 40 / 24.
Intervals are exploratory, not adjusted for multiple comparisons or evidence of generalization.

Header-on fixed comparisons (each point estimate is question-macro averaged):

| Repo | Strategy | All 40 MRR@10 | All 40 line coverage@10 | All 40 gold ≥90%@10 | Test 24 gold ≥90%@10 |
|---|---|---|---|---|---|
| Requests | char | 0.498 | 0.717 | 0.558 | 0.583 |
| Requests | lc_lang | 0.623 | 0.648 | 0.504 | 0.583 |
| Requests | astchunk | 0.568 | 0.712 | 0.646 | 0.625 |
| Requests | ast | 0.569 | 0.720 | 0.633 | 0.646 |
| Express | char | 0.580 | 0.845 | 0.725 | 0.750 |
| Express | lc_lang | 0.600 | 0.843 | 0.775 | 0.708 |
| Express | ast | 0.599 | 0.900 | 0.900 | 0.833 |

Paired **AST minus baseline** differences, same header-on BM25 rankings:

| Repo | Sample | Baseline | ΔMRR@10 [95% CI] | ΔGold ≥90%@10 [95% CI] |
|---|---|---|---|---|
| Requests | all 40 | char | +0.071 [−0.019, +0.165] | +0.075 [−0.037, +0.188] |
| Requests | test 24 | char | +0.126 [+0.005, +0.262] | +0.062 [−0.071, +0.190] |
| Requests | all 40 | lc_lang | −0.054 [−0.125, +0.012] | +0.129 [+0.008, +0.250] |
| Requests | test 24 | lc_lang | −0.039 [−0.122, +0.040] | +0.062 [−0.083, +0.208] |
| Express | all 40 | char | +0.019 [−0.059, +0.102] | +0.175 [+0.075, +0.300] |
| Express | test 24 | char | −0.020 [−0.100, +0.069] | +0.083 [+0.000, +0.208] |
| Express | all 40 | lc_lang | −0.002 [−0.114, +0.116] | +0.125 [+0.025, +0.250] |
| Express | test 24 | lc_lang | +0.018 [−0.052, +0.122] | +0.125 [+0.000, +0.292] |

Some pointwise intervals exclude zero, so “no result is statistically distinguishable” is too
broad. These are sample-specific associations: the completeness metric was added after review,
the budgets still differ, and Requests informed earlier model selection. The pooled fixed rows
are descriptive; selected/tuned settings continue to be assessed using held-out test only.
The evidence does not establish “AST wins.”

**Express interpretation:** on these BM25 questions and default budgets, AST returns more
complete evidence; it has not demonstrated better ranking. All-40 ≥90% completeness improves
by **+0.175 [ +0.075, +0.300 ]** against char, whereas the held-out-test difference is
**+0.083 [ +0.000, +0.208 ]**, touching zero. Express MRR differences span zero.
Header-on/off variants and dev/test/all tables reuse the same 40 questions. They are views
of one experiment per repository, not independent replications or extra observations.

Before/after verification: **560 question/configuration pairs**, identical retrieved ids/ranks
and source envelopes, **0 changed pre-existing overlap/MRR values**. The new test completeness
values reproduce the review. Coverage excludes overlapping-chunk double credit and missing
reference text. A reference audit found **5 reconstructed-text/source-line mismatches among 270
chunks**: two contain partial source lines, and three contain omitted multiline leaf text.
The adapter now exports represented `source_spans`; blank padding for missing nonblank source
receives no hit/coverage credit. Reference text and split/merge behavior were preserved.

Full current summaries with marginal CIs, paired differences and auditable per-question
evidence are retained in `eval/retrieval_results/bm25/`; original results are preserved in
`eval/retrieval_results/bm25_legacy/`, and the prior review snapshot remains in `bm25_review/`.
Question wording/author independence, matched budgets, and the real GPU run remain open.

## Historical Phase 4 snapshot: initial overlap-only validation (2026-10-25)

This section records the initial report format. Use the current coverage/uncertainty section
above and the main `bm25/` reports for present interpretation. These rows are not additional
experiments; rankings and the original metrics are unchanged.

**Status: full embedding comparison pending; Phase 4 remains incomplete.** The new harness
ran BM25 on the same pinned code corpus for every supported chunker, with common contextual
headers on/off. These are real lexical retrieval measurements, **not** a result for the full
Qwen3/Chroma/hybrid pipeline and **not** evidence that AST chunking generally wins.

Datasets: 40 questions each for Requests (`611c6162cbc4`) and Express (`0dee2c52c7c8`),
16 dev / 24 test per repo. Paraphrases and overlapping gold evidence stay in one split.
Gold is path + inclusive source lines, validated against SHA-256 evidence hashes; scoring
never uses chunk symbols. The table contains **test questions only**. No defaults were tuned.
The 36 Requests seed questions were previously used to choose embeddings, so its test split
is held out for Phase 4 tuning, not for historical model selection. Express questions are new.

| Repo | Strategy | Header | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Chunks | Avg body chars |
|---|---|---|---|---|---|---|---|---|
| Requests | char | off | 0.271 | 0.708 | 0.833 | 0.472 | 331 | 1408 |
| Requests | char | on | 0.271 | 0.708 | 0.812 | 0.467 | 331 | 1408 |
| Requests | lc_lang | off | 0.354 | 0.708 | 0.812 | 0.579 | 377 | 1121 |
| Requests | lc_lang | on | 0.438 | 0.708 | 0.812 | 0.632 | 377 | 1121 |
| Requests | astchunk | off | 0.271 | 0.688 | 0.729 | 0.494 | 270 | 1478 |
| Requests | astchunk | on | 0.312 | 0.688 | 0.729 | 0.548 | 270 | 1478 |
| Requests | ast | off | 0.312 | 0.688 | 0.750 | 0.522 | 274 | 1483 |
| Requests | ast | on | 0.396 | 0.729 | 0.750 | 0.593 | 274 | 1483 |
| Express | char | off | 0.542 | 0.833 | 0.875 | 0.652 | 477 | 1291 |
| Express | char | on | 0.542 | 0.833 | 0.875 | 0.652 | 477 | 1291 |
| Express | lc_lang | off | 0.458 | 0.792 | 0.833 | 0.587 | 565 | 1031 |
| Express | lc_lang | on | 0.500 | 0.792 | 0.833 | 0.614 | 565 | 1031 |
| Express | ast | off | 0.542 | 0.750 | 0.833 | 0.624 | 384 | 1427 |
| Express | ast | on | 0.542 | 0.750 | 0.833 | 0.633 | 384 | 1427 |

`astchunk==0.1.0` has no JavaScript support: its Express rows are N/A, with no substitute
parser or smaller corpus. Its eval adapter changes only coordinate access to avoid a native
crash on this Windows/CPython 3.13/tree-sitter combination; reference split/merge is unchanged.
Character splitters count total characters; AST splitters count non-whitespace characters.
A one-line overlap counts, so retrieval success does not guarantee a complete answer.

Original raw summaries, settings/source hashes and per-question rankings:
`eval/retrieval_results/bm25_legacy/`. Full reproduction and limitations: `eval/README.md`.
GPU entry point: `notebooks/retrieval_eval_on_colab.ipynb`. It measures exact dense vs HNSW,
BM25-weight/test-file-penalty grids, and optional code reranking at depths 20 and 50. We must
review its real results before changing production weights or enabling code reranking.

## T3 rerank: reranker vs bigger embedder (2026-10-22)

**Run:** `eval/embedding_bench/t3_rerank.ipynb`, Colab A100. 144 issues over 4,882 commits,
using the settings adopted after T3 tuning (`symbols_diff800` docs, the shared instruction).
Each first stage's top 50 is reranked. Nothing is tuned, so all 144 queries count. CIs are
paired bootstrap against production (Qwen3-0.6B @512, no rerank). Raw results:
`eval/embedding_bench/rerank_results/`.

**Verdict: Qwen3-Reranker-4B closes the gap and passes the 7B embedder.** On production's
own candidates it scores **0.687** nDCG@10 (from 0.494; gain +0.192, CI [+0.133, +0.251]),
versus 0.596 for nomic-embed-code 7B. Hit@5 is 0.79 vs 0.56.

| First stage | Reranker | Recall@50 | nDCG@10 (all 144) | Hit@5 | Gain vs prod [95% CI] |
|---|---|---|---|---|---|
| Qwen3-4B @512 | Qwen3-Reranker-4B | 0.903 | **0.692** | 0.79 | +0.198 [+0.137, +0.259] |
| Qwen3-4B @2560 | Qwen3-Reranker-4B | 0.910 | 0.691 | 0.80 | +0.197 [+0.135, +0.257] |
| **Qwen3-0.6B @512 (prod)** | **Qwen3-Reranker-4B** | 0.861 | **0.687** | 0.79 | **+0.192 [+0.133, +0.251]** |
| Qwen3-4B @2560 | none | 0.910 | 0.612 | 0.72 | +0.117 [+0.069, +0.166] |
| Qwen3-4B @512 | none | 0.903 | 0.581 | 0.70 | +0.086 [+0.036, +0.139] |
| Qwen3-4B @512 | Qwen3-Reranker-0.6B | 0.903 | 0.569 | 0.67 | +0.074 [+0.021, +0.127] |
| Qwen3-0.6B @512 (prod) | Qwen3-Reranker-0.6B | 0.861 | 0.556 | 0.67 | +0.062 [+0.012, +0.114] |
| **Qwen3-0.6B @512 (prod)** | none | 0.861 | 0.494 | 0.56 | baseline |
| Qwen3-0.6B @512 (prod) | bge-reranker-v2-m3 | 0.861 | 0.373 | 0.44 | **−0.121** [−0.193, −0.052] |

| Reranker | Pairs/s (A100) | Per query, top 50 (A100) | Laptop CPU |
|---|---|---|---|
| Qwen3-Reranker-4B | 21 | 1.83 s | impractical |
| Qwen3-Reranker-0.6B | 64 | 0.58 s | ~23 s for 5 pairs (measured on the dev laptop) |
| bge-reranker-v2-m3 | 315 | 0.16 s | n/a (hurts quality) |

### Findings

1. **The reranker does the work, not the embedder.** With the 4B reranker, upgrading the
   first stage from 0.6B to 4B adds only +0.004. The 4B embedder without reranking (0.612)
   only matches the 7B model.
2. **The 0.6B embedder's candidates are good enough.** Recall@50 is 0.86, and the 4B reranker
   reaches 80% of that ceiling.
3. **Reranker size matters:** 4B gives +0.19, 0.6B only +0.06.
4. **bge-reranker-v2-m3 is harmful here** (−0.12). It's a general passage reranker and
   doesn't understand bug report ↔ code change. It's removed from `config.py`.
5. **Deployment:** the 4B reranker needs a GPU (1.8 s per query on an A100). On a laptop CPU
   reranking isn't viable at all.

### Decision

- **Embedder unchanged:** Qwen3-Embedding-0.6B @512 everywhere (laptop-friendly). Qwen3-4B
  as an embedder is rejected: 3× slower indexing for +0.004 once reranked.
- **Reranker: Qwen3-Reranker-4B over the top 50**, with `reranker_mode="auto"`, which runs it
  only when a CUDA GPU is present. Laptop-only installs run without reranking (0.494 on T3).
  GPU or hosted deployments get 0.687. `retrieval_k` is now 50.
- **Not yet measured:** rerank on code questions (T2r). Phase 3 measures it before enabling
  reranking for routes other than `bug_match`, and checks whether top 20 keeps most of the
  gain at about 40% of the latency.

---

## T3 tuning: instructions × commit doc formats (2026-10-22)

**Run:** `eval/embedding_bench/t3_tuning.ipynb`, Qwen3-0.6B only, Colab A100. 144 issues
split 50/50 into dev (72) and test (72) with a fixed seed. Selection on dev at 512 dims,
reported on test. Raw results: `eval/embedding_bench/t3_results/`.
**Reproducibility:** the baseline at full dims gives 0.488 on all 144, identical to run 2.

**Verdict: tuning does not close the gap.** The dev winner (`code_instruction` +
`diff2000`) gains **+0.017** nDCG@10 on test (0.464 → 0.482), with a paired bootstrap 95% CI
of **[-0.017, +0.050]**, which includes 0 (P(gain ≤ 0) = 0.16). The gap to nomic-embed-code
(0.596) stays at about 0.1.

### What the grid shows

| Doc format (mean over instructions, 512d) | Dev | Test | Median tokens | Index speed |
|---|---|---|---|---|
| `diff2000` (baseline) | 0.497 | 0.460 | 121 | 176 docs/s |
| `symbols_diff800` | 0.492 | 0.464 | 129 | 252 docs/s |
| `symbols` (no diff) | 0.418 | 0.456 | 24 | 582 docs/s |
| `message` only | 0.289 | 0.389 | 7 | 612 docs/s |

| Instruction (mean over formats, 512d) | Dev | Test |
|---|---|---|
| `code_instruction` (same as the code search) | **0.440** | **0.451** |
| `symptom_to_change` | 0.429 | 0.450 |
| `none` | 0.433 | 0.439 |
| `baseline` (run 2) | 0.422 | 0.447 |
| `issue_to_commit` | 0.396 | 0.424 |

1. **The diff carries the signal.** Message-only loses about 0.15. Touched symbols recover
   most of it. 800 chars of diff matches 2,000.
2. **The instruction barely matters** (±0.02), except the most specific one
   (`issue_to_commit`), which is consistently worst. The code-search instruction is slightly
   best on both halves, so one instruction can serve every route.
3. With 72 test queries, differences under about 0.05 are noise. Dev/test disagreements (for
   example `symbols`: 0.418 dev vs 0.456 test) show that.

### Adopted (simplicity and speed, not accuracy)

- **One query instruction for all collections:** `embedding_query_instruction` in `config.py`.
- **Commit doc = message + files + touched symbols + 800 chars of diff:**
  `commit_diff_chars = 800`. Same quality, about 1.4× faster indexing. Phase 5 will use real
  AST symbols instead of hunk headers.

### Next lever for T3

Embedding tweaks are exhausted at this size. Candidates, both on Colab:
1. **A cross-encoder reranker** over the top 50 (e.g. Qwen3-Reranker-0.6B). The plan already
   has a rerank stage, so this tests it.
2. **Qwen3-Embedding-4B** for the `commits` collection only.

---

## Embedding model: final decision (run 2, 2026-10-21)

**Decision: `Qwen/Qwen3-Embedding-0.6B`, stored at 512 dims, for all collections in v1.**
`config.py` already defaults to it.

**Run 2** has the same setup as run 1, plus **T3** (144 real issue → fix-commit pairs over
4,882 commits) and **T2r** (T2 restricted to requests' chunks, as production filters by
repo). Raw results: `eval/embedding_bench/results/`; run 1 is kept in `results_run1/`.
**Reproducibility:** T1 and T2 match run 1 exactly (max nDCG@10 difference 0.0).

### nDCG@10 on the tasks that matter (T2r = questions, T3 = bug → fix)

| Model | T2r questions (n=36) | T3 bug → fix (n=144) | Mean |
|---|---|---|---|
| **Qwen3-0.6B** | **0.736** | 0.488 | **0.612** |
| Qwen3 @ 512d | **0.738** | 0.482 | 0.610 |
| Qwen3 @ 256d | 0.726 | 0.463 | 0.594 |
| Qwen3 + BM25 (RRF) | 0.704 | 0.466 | 0.585 |
| nomic-embed-code 7B + BM25 | 0.576 | 0.524 | 0.550 |
| nomic-embed-code 7B | 0.431 | **0.596** | 0.513 |
| CodeRankEmbed + BM25 | 0.549 | 0.443 | 0.496 |
| BM25 | 0.522 | 0.347 | 0.434 |
| CodeRankEmbed | 0.416 | 0.436 | 0.426 |

Hit@5: on T2r, Qwen3 0.889 vs 0.722 (7B) and 0.556 (CodeRankEmbed). On T3, the 7B model
0.688 vs Qwen3 0.535, a gap of **22 of 144 queries**.

### Findings

1. **The 7B model genuinely wins bug → fix** by 0.108 nDCG@10, which beats the 0.05 threshold
   set beforehand. On 144 queries that's likely real, not noise. It's still last-tier on
   questions (0.431 vs 0.736).
2. **Qwen3 is best overall** (mean 0.612 vs 0.550 for the best 7B variant) and best at what
   users ask most: questions about code.
3. **Filtering by repo matters a lot.** Every model gains 0.13–0.18 nDCG@10 from T2 to T2r.
   Production must filter retrieval by repo, which the plan already does through metadata.
4. **512 dims is free:** T2r 0.738 vs 0.736, T3 0.482 vs 0.488. It halves the index
   (205 MB per 100k chunks).
5. **Plain RRF with BM25 still hurts Qwen3** on both real tasks (0.612 → 0.585 mean). It only
   helps symptom questions on T2r (Hit@5 0.83 → 1.00). Phase 3 must tune **weighted** fusion.

### Why not the 7B model, or a split (7B for commits only)?

It would need a GPU at **query time** (it can't run on a laptop), a 3.5× bigger index and
about 4× slower indexing, all to win one of two tasks. Separate Chroma collections do make a
per-collection model possible later. It isn't worth it before trying the cheaper fixes for T3
below.

### Follow-ups to close the T3 gap (cheap first)

1. Tune Qwen3's T3 instruction (it's instruction-aware; the current one is a first guess).
2. Improve the commit document: message plus touched symbols, with less raw diff.
3. Try Qwen3-Embedding-4B for the `commits` collection only (GPU-friendly, same family).

---

## Embedding model (run 1, 2026-10-21)

**Run:** `eval/embedding_bench/`, Colab A100 40 GB, bf16, max 2,048 tokens per chunk.
**Corpus:** 2,880 Python chunks from psf/requests, pallets/flask and encode/httpx at pinned SHAs.
**Raw results:** `eval/embedding_bench/results/` (summary.md, metrics.csv, speed.csv,
t2_per_question.csv, quality_vs_speed.png).

> **T3 (bug report → fix commit) did not run.** `GITHUB_TOKEN` didn't reach the run. This
> decision is **provisional until run 2 includes T3**.

### Quality: nDCG@10 (Hit@5 in brackets)

| Model | T1 docstring→code (n=499) | T2 dev questions (n=36) |
|---|---|---|
| BM25 | 0.305 (0.37) | 0.418 (0.61) |
| CodeRankEmbed 137M | 0.842 (0.94) | 0.290 (0.36) |
| CodeRankEmbed + BM25 | 0.586 (0.69) | 0.392 (0.53) |
| **Qwen3-Embedding-0.6B** | 0.813 (0.92) | **0.558 (0.75)** |
| Qwen3 @ 512 dims | 0.818 (0.92) | **0.572 (0.72)** |
| Qwen3 @ 256 dims | 0.802 (0.90) | 0.544 (0.69) |
| Qwen3 + BM25 | 0.583 (0.67) | 0.534 (0.69) |
| nomic-embed-code 7B | **0.892 (0.97)** | 0.277 (0.36) |
| nomic-embed-code + BM25 | 0.596 (0.69) | 0.413 (0.56) |

### Speed and cost (A100 unless noted)

| Model | Corpus docs/s | Tokens/s | GPU query p50 | CPU query p50 (Colab CPU) | Peak GPU | Dim | Index / 100k chunks |
|---|---|---|---|---|---|---|---|
| CodeRankEmbed | 194 | 33,398 | 16.8 ms | 67 ms | 26.8 GB* | 768 | 307 MB |
| **Qwen3-0.6B** | **311** | **41,534** | 44.4 ms | 269 ms | 9.4 GB | 1024 (MRL 256/512) | 410 MB (205 MB @ 512d) |
| nomic-embed-code | 77 | 10,242 | 36.8 ms | n/a (7B) | 17.1 GB | 3584 | 1,434 MB |

\* CodeRankEmbed's custom model code doesn't use fused attention, so memory balloons at
batch 64 × 2,048 tokens. That's only an indexing-time concern.

### Findings

1. **The two tasks disagree, and T2 is the one that matters.** The nomic models win T1
   (docstring → code) but collapse on T2 (real developer questions): 0.29 and 0.28 nDCG@10,
   *below BM25*. They look tuned to CodeSearchNet-style docstring queries.
2. **Qwen3 is the only model that handles questions.** Per question on T2, Qwen3 gets 11 that
   both nomic models miss and misses none that either of them gets. A 14-question Hit@5 gap is
   far beyond the ~2.8 points one question is worth. Its edge is largest on bug-symptom
   questions (Hit@5 0.67 vs 0.00).
3. **Equal-weight RRF with BM25 hurts strong dense models.** T1 drops from about 0.84 to 0.59
   for every model, and Qwen3's T2 drops from 0.558 to 0.534. It only helps the weak nomic T2
   results. Production fusion must be **weighted** and tuned (plan.md Phase 3/4), not plain RRF.
4. **Qwen3 at 512 dims costs nothing:** T2 0.572 vs 0.558 at full size, which is within noise,
   at half the index size. 256 dims loses about 0.01–0.015.
5. **Qwen3 is also the fastest to index** (311 docs/s, more than the 137M CodeRankEmbed, whose
   custom code is unoptimized). Its CPU query p50 was 269 ms on Colab's 2-vCPU CPU, inside our
   300 ms budget. A laptop should be faster, but measure it.
6. **9 T2 questions are missed by all models.** Most ask about behaviour httpx implements too
   (redirects, timeouts, streaming, case-insensitive headers, `response.text` charset), and
   the questions don't name the repo. Run 2 adds **T2r**: the same questions searching only
   requests' chunks, the way production filters by repo. A CPU smoke test with MiniLM moved
   from 0.49 to 0.65 nDCG@10 under that filter.

### Decision (provisional)

**Qwen/Qwen3-Embedding-0.6B**, likely stored at **512 dims**. `config.py` now defaults to it
(`embedding_provider="huggingface"`), replacing the CC-BY-NC jina fallback.

- Confirm with run 2: T3 present, T2r added.
- The 7B model is rejected unless it wins T3 by a wide margin. It loses T2 badly, is ~4×
  slower to index, has a 3.5× bigger index and can't run queries on a laptop.
- vLLM stays deferred (it only mattered if the 7B model won).
