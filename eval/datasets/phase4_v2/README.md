# Phase 4 v2 protocol and labels

V1 files in the parent directory are frozen. The matching original runtime source is preserved
at `../../retrieval_results/colab_full/phase4_v1_source.zip`. Do not overwrite v1 or tune on its
old 16-question development subsets. `matrix.json` fixes the next experiment before scoring.

| Repository | Legacy questions | New test | New example | New symbol lookup | Other new implementation | Total |
|---|---:|---:|---:|---:|---:|---:|
| Requests | 40 | 10 | 0 | 3 | 0 | 53 |
| Express | 40 | 7 | 3 | 3 | 0 | 53 |
| Flask | 0 | 7 | 3 | 3 | 10 | 23 |

Flask is pinned to `d086db856be187255b8ec61ef409357393020f32`. The other pins remain unchanged.
Requests has no example-code targets in this corpus; its ten additions target tests. All
questions are source-authored and author-checked; independent review remains outstanding.
New symbol queries sometimes target previously evaluated implementations. Neither new wording
nor a third repo proves independence from model training or author bias.

`gold` means **required evidence**. `supporting_gold` is optional context and is measured only
on questions with supporting labels. `target_type` distinguishes implementation/test/example;
`category=symbol_lookup` marks explicit identifiers. `cohort=fresh` separates the 49 new
questions from previously inspected regression questions. There is no v2 tuning split: all
rows use `split=test`, with connected evidence groups across required and supporting ranges.
This does not make the legacy questions held-out again.

## Review of the eight Requests and two Express incomplete cases

Full source ranges and rationale are recorded in `manifest.json`; this is an author review.

| Question | V2 decision |
|---|---|
| Requests 001 | Required redirect loop/limit/dispatch control flow; remaining function lines supporting |
| Requests 007 | Chunk iterator required; line iterator supporting |
| Requests 014 | Retain file encoding and request-body preparation as complementary required evidence |
| Requests 018 | Adapter lookup required; adapter registration supporting |
| Requests 020 | Retain broad class evidence pending independent minimal-evidence review |
| Requests 024 | Timeout construction and transport call required; remaining send method supporting |
| Requests 032 | Retain decoding/detection/header parsing as the multi-part diagnosis; flag for independent review |
| Requests 034 | Only limit check/raise required; remaining 122-line function supporting |
| Express 010 | Retain compact implementation; observed issue is ranking |
| Express 025 | Retain compact implementation; observed issue is ranking |

001/024/034 required and supporting ranges are disjoint and together cover the original gold.
The GPU runner also rescoring current rankings against unchanged v1 labels in `v1_regression.json`.
Do not describe score changes from revised required evidence as retrieval improvements.

## Fixed GPU matrix

Primary policies: dense HNSW, current weighted hybrid, hybrid-top20 reranking, and reranking a
20-candidate union reserving the first five BM25 hits. Same production AST chunks/headers for
all policies; Qwen3-Embedding-0.6B @512 float32 and Qwen3-Reranker-4B. No weight sweep.
V1's common-header experiment remains separate. BM25-only, a test multiplier of 0.8 and
source-derived explicit-symbol routing are labelled diagnostics, with no production enablement.

Full first-stage lists/scores, fusion scores, candidate provenance and pair scores are logged
for every repetition. IDs 028 and 036 in both original repos also get top50 pair traces. Warm
end-to-end timing includes live query encoding and actual retrieval/reranking, repeated three
times with rotated method order. Cold first-pool timing is separate; token audits/report writes
are outside warm timings. CUDA memory is shared-process allocation, not isolated model memory.

Fresh quality rows are question-weighted and bootstrapped by evidence group; repetitions are
not independent questions. Per-repo and target-type rows must accompany pooled results.
Promotion requires the predeclared fresh win/no-regression checks and explicit label/latency
review. Missing or inconclusive strata are not passes. `decision.json` is a checklist and never
changes production settings. Keeping reranking opt-in is a valid outcome.

## Run

Use `../../../notebooks/phase4_followup_on_colab.ipynb` with the matching separate source ZIP.
Upload `phase4_followup_source.zip` to My Drive root, select an A100, run all cells, and return
`phase4_followup_results.zip`. Dependencies install with uv; there is no embedded Base64 payload.

Local validation/BM25 can run without models:

```powershell
uv run python eval/run_followup_eval.py --repo psf/requests --checkout data/repos/psf__requests --dataset eval/datasets/phase4_v2/requests_qa.jsonl --mode validate
uv run python eval/run_followup_eval.py --repo psf/requests --checkout data/repos/psf__requests --dataset eval/datasets/phase4_v2/requests_qa.jsonl --mode bm25 --output data/v2_bm25_check
```

Choose a new output directory after a completed run; the runner refuses to overwrite run.json.
The full mode requires CUDA and refuses to substitute fake embeddings. Rebuild with
`uv run python notebooks/build_phase4_followup.py` after runtime/data/matrix changes.
