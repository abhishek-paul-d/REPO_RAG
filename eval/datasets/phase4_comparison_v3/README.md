# Expanded comparison: 90 questions, three fixed policies

This replaces the 15-question notebook as the next comparison run. The original dataset and
notebook remain available. Upload the **new** `phase4_comparison_source.zip` with the matching
`phase4_comparison_on_colab.ipynb`; do not mix it with the earlier 15-question ZIP.

| Question type | Requests | Express | Flask | Total |
|---|---:|---:|---:|---:|
| Ambiguous behaviour, without source-type or identifier cues | 12 | 12 | 12 | 36 |
| Implementation workflows requiring multiple blocks | 6 | 6 | 6 | 18 |
| Explicit automated-test targets | 6 | 6 | 6 | 18 |
| Implementation navigation, without symbol names | 6 | 3 | 3 | 12 |
| Runnable example/tutorial targets | 0 | 3 | 3 | 6 |
| **Total** | **30** | **30** | **30** | **90** |

This is a deliberate diagnostic sample covering 58 repository/file combinations. Requests
has no comparable examples directory in its indexed corpus; no example questions are invented
for it. Explicit test/example questions are guard strata, reported separately from the 36
ambiguous queries. The ambiguous queries omit test/tested, example/tutorial, implementation,
backtick symbol and snake-case identifier cues. Domain terms such as HTTP and JSON remain.

There are **75 additions and 15 retained questions**, reported separately. Questions cover
authentication, redirects, parameters, encoding, cookies, connection pools, streaming,
negotiation, cache validators, file transfer, routing, rendering, request/context lifecycles,
configuration and tutorial authentication/authorization. Workflows require their listed
blocks together; ambiguous questions can accept either a sufficient implementation proof or
a complete behavioural demonstration. Source wrappers around external libraries establish
where work is delegated, not the internals of those dependencies.

All evidence is hashed against clean pinned checkouts. Parameterization and behaviour-specific
helper assertions are included where needed. Before freezing, every question was verified to
have at least one complete accepted proof within the full production chunk corpus. This is
an answerability check, not top10 retrieval performance.

Shared evidence across **any** accepted proof is connected into bootstrap groups: **28 Requests,
27 Express, 28 Flask**, 83 total. The model cannot use labels, groups, strata or topics for
retrieval; only question text is searched. A readable question/proof list is in
[QUESTIONS.md](QUESTIONS.md). `manifest.json` records which cases overlap old v2 evidence.

## Fixed comparison and interpretation

The actual production search method runs independently as **hybrid, rerank20 and rerank50**.
All retrieve 50 per signal; weights remain 1.0/0.3, RRF k=60, no path penalty/quota, same AST
corpus and production headers. Each returns top10 for scoring. Query encoding and real pair
scoring are included in the warm measurement; each policy gets three rotated repetitions.
Quality uses the pre-chosen repetition zero. Full candidate/fusion/pair traces are retained.

The report includes paired rerank50-minus-rerank20 intervals as well as each method versus
hybrid, per repo/type, pooled, and retained/addition subsets. MRR checks first relevant overlap;
`any_complete_proof@10` checks ≥90% of every required range in one accepted proof. It cannot
join incomplete fragments from separate alternatives. Pooled all90 is equal-repo weighted
because each repo contributes 30 questions. Group-bootstrap intervals are exploratory and
pointwise, not multiplicity-adjusted. A CI crossing zero is inconclusive, not proof of equality.
See [protocol.json](protocol.json) for the choices fixed before retrieval scoring.

**No expanded GPU quality or latency is measured yet.** The notebook records actual hardware,
model/source/dataset revisions, per-repo warm p50/p95 and repetition stability. A100 remains
experiment-only; eventual deployment timings require chosen deployment hardware.

This larger sample is not a random user-query sample or an independently reviewed holdout.
The same assistant authored/reviewed labels, and prior source overlap is explicitly recorded;
newly authored wording is not guaranteed unseen model-training data. Independent review is
still needed. Six example questions remain a small guard sample. These questions assess
answerable code retrieval, not abstention, generated answers, ANN scale, additional languages
or a new AST-versus-character comparison. No automatic production default change is made.
