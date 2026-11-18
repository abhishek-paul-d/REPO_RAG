# Ambiguous-intent stress set

Fifteen questions, five each for Requests, Express and Flask, frozen before this new GPU
comparison. Queries omit “test”, “tested”, examples/tutorials, backtick symbols and snake-case
identifiers. They deliberately target previously inspected behaviours: **new wording does not
make this an independent fresh holdout**. Do not pool scores with v2's 49-question estimate.

Each case records two acceptable sufficient evidence sets: implementation and a demonstration
of the requested behaviour. Alternatives are OR; all ranges within one alternative are AND.
Parameters, fixtures and assertions are included where needed. It is valid to return either
file type when the user has not specified one. MRR accepts overlap with either proof;
`any_complete_proof@10` requires ≥90% of every range in at least one proof. Partial pieces
from different alternatives cannot combine into a complete proof. The older `complete_gold`
field remains a diagnostic fraction of completed ranges in the best alternative.

All labels are source-checked by the same assistant that authored the earlier benchmark;
**independent review is outstanding**. The separate [source review](../../PHASE4_LABEL_REVIEW.md)
records ambiguous and multi-block issues without changing frozen v1/v2 labels or scores.
Question text alone goes to retrieval; gold and alternatives are used only for scoring.

The pre-chosen methods are **current hybrid, production rerank20, production rerank50**:
top50 per signal, weights 1.0/0.3, RRF 60, no test penalty/quota, top10 evaluation,
3 rotated warm repetitions. `eval/run_ambiguous_eval.py` calls the actual `HybridRetriever`
method, including real query embedding and reranking. Pair scores, rankings, source/dataset
hashes, model revisions, paired evidence-group CIs and per-repo p50/p95 are saved.
Shared fixture evidence joins the two Requests header-mapping questions into one group.

Use `notebooks/phase4_ambiguous_on_colab.ipynb` with the separate
`notebooks/phase4_ambiguous_source.zip` uploaded to My Drive root. The notebook uses uv and
contains no embedded archive. **A100 is experimental hardware; deployment remains undecided.**
Do not present A100 warm sequential latency as deployment service latency.

No weights or candidate depths will be selected by tuning these questions. Keep the current
50-candidate compatibility default and explicit opt-in; laptop auto mode skips reranking.
GPU results from this stress set and deployment-hardware timings are pending.
