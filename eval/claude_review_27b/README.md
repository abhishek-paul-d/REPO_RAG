# Claude's review of the 27B code and docs answers (2026-11-18)

The 52 code and docs answers of one 27B run, read by Claude (an AI reader, not a person) against the
cited lines, and against the pinned `psf/requests` source (`611c6162cb`) where the evidence the judge
tooling extracts was cut off.

- **Answers:** `eval/agent_eval_results/agent_eval_production_history_grounded_extra3_notests_chars24000_in12288_snap_abst1_toolerr1_20261118T101342.json`
  (Qwen3.8-27B, `history_grounded`, citation snapping and the abstain and tool-error retries on; this is
  not the default configuration). Dataset `eval/datasets/agent_eval_v1.jsonl`.
- **Evidence:** `eval/judge_answers.py prepare` (protocol v2) built the cited and reference text; nothing was
  sent to any judge model.
- **Verdicts:** `verdicts.json`, `{id: [faithful|partial|unfaithful, correct|partial|incorrect, note]}`.
- **Also in this folder:** `exact_facts_verdicts.json` (15 exact answers) and `commit_review.json` (113 commit answers).
- **Totals:** faithful 50/52 (2 partial), correct 49/52 (3 partial); code 38/40 correct, docs 11/12.
  Not both top: code-006 and code-032 (text decoding with no charset), code-025 (netrc wording and a
  wrong line), docs-051 (thin answer).
- **Limits:** one AI reader, no human labels; the questions are the 40 + 12 the system was tuned against;
  claims were checked, not every uncited line; "correct" allows omissions that are not asked for.
