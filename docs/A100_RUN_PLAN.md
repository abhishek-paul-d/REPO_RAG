# A100 session plan: section 1 and the fact set on the 27B

Written 2026-11-18. It replaces the 14B plan (first session and "Round 2"), which is kept as the
record of what was measured at `experiments/docs/A100_RUN_PLAN.md`. Do not run that one as written.

Answer model: `cyankiwi/Qwen3.8-27B-AWQ-INT4` on vLLM (CUDA graphs + MTP, 12,288-token input cap),
the default in `config.py`. **Not measured yet on the 27B:** section 1 (32 questions), the fact set,
and the guards and `file_history` tool added after the 14B's section 1 failures. For the 14B the
section 1 criterion was not met (9 of 32 abstained or failed, several more weak) and the facts were
57/60; see `docs/PROJECT_LOG.md` 2026-11-16.

Time, estimated from earlier 27B runs and not timed for this plan: about 4 s per question (the
128-question regression took about 10 minutes), about 4 minutes of vLLM startup, and 20 s of
embedding warm-up on the laptop per process. Section 1 and the fact set are 32 and 60 questions, so
the whole session should fit in about 45 minutes.

The settings the 27B was measured with (`docs_hint_narrow`, 3 extra files, tests skipped, 24,000
tool characters, 12,288 input tokens) are now the defaults, so `section1_check.py` and
`agent_facts.py` pick them up without flags. `agent_eval.py` takes them as flags (run S4).

## Before you start

1. If `src/repo_rag` has changed since the zip was built (2026-11-18 09:03), rebuild it with
   `uv run python notebooks/build_colab_indexer.py` and put **`notebooks/repo_rag_ask_source.zip`**
   at the top of My Drive, replacing the old one. The serve notebook checks its hash.
2. Open **`notebooks/serve_model_27b_on_colab.ipynb`** on an **A100 (40 GB)** and Run all. It prints a
   public URL and a token, and releases the GPU after 10 idle minutes, so run the commands below in
   one sitting. Defaults: CUDA graphs + MTP, `MAX_INPUT_TOKENS = 12288`.
3. Keep the laptop's `data/` as it is (index, manifest, `metadata.db`, the psf/requests clone).

`<URL>` and `<TOKEN>` are the printed values. Never reuse an older token.

## Runs

| Run | Command (laptop) | Looks at |
|---|---|---|
| S1 baseline | `uv run python eval/section1_check.py run --url <URL> --token <TOKEN>` | section 1 on the 27B with production defaults: how many of the 14B's abstentions and failures remain |
| S2 facts | `uv run python eval/agent_facts.py --url <URL> --token <TOKEN>` | the 90% end-to-end gate (14B: 57/60) |
| S3 guards | `uv run python eval/section1_check.py run --url <URL> --token <TOKEN> --citation-snap --abstain-retries 1 --tool-error-retries 1 --variant history_grounded` | only if S1 still shows abstentions or tool failures: the same questions with the guards, snapping and the `file_history` tool |
| S4 regression | `uv run python eval/agent_eval.py run --url <URL> --token <TOKEN> --types code,docs,prsearch,probe,history,refs,kind --variant history_grounded --citation-snap --abstain-retries 1 --tool-error-retries 1 --extra-files 3 --skip-tests --tool-chars 24000 --max-input-tokens 12288` | only if S3 is better: did the guards cost a pass, and do all 5 probes still abstain |

S4's comparison is the existing 27B run (no guards), so it needs no rerun:
`experiments/eval/agent_eval_results/agent_eval_production_docs_hint_narrow_extra3_notests_chars24000_in12288_20261118T024701.json`
(code 39/40, docs 11/12, prsearch 10/10, probes 5/5, history 11/11, refs 10/10, kind 5/5).
The scripts write new results under `eval/`; the earlier 14B results are in `experiments/eval/`.

## Rule, fixed before the runs

- **S1** is read first. `section1_*.md` lists the automatic flags first (missing or impossible
  citations, abstentions, SHAs found in no tool output), but read the unflagged answers too, because
  the checks cannot catch a fluent wrong answer. Phase 7 needs "cited, sensible" for every question
  except the out-of-scope `oos-1`, which should abstain.
- **S2** passes at 54/60 or more (90%). If lower, look at the misses first (the 14B's were
  `blame_line` httpx/_config.py:114 and the tags 0.28.1 and 0.6.4).
- **Keep the guards, snapping and `history_grounded`** only if S4 loses no pass against the baseline
  above in any type, all 5 probes still abstain, and S3 answers hist-1 and hist-2 with citations.
  A one-question difference is a chance flip. Otherwise leave the defaults (all off) and record
  the numbers.
- Do not tune on the 32 section 1 questions beyond what the failures show. They are adapted to one
  repository, and an answer fixed only by naming its subject in the prompt is not a fix.

## After the runs

- Tell me the runs are finished. I read the result files, apply the rule and log the numbers in
  `docs/PROJECT_LOG.md`.
- Still open whatever the result: the judge has not been run on 27B answers, and no person has
  labelled any answer, so quality figures stay AI-checked.
- Stop the Colab session when done (the serve notebook's last cell does it).

## Known limits

- One repository, one answer model; the judge is the same model family.
- Section 1 questions are adapted to psf/requests, and their exact-fact answers are checked only
  for citation form and SHAs, not for correctness.
- The 27B baselines are single deterministic runs. Movement of one or two questions was seen
  between decoding modes (architecture 12 to 13 of 15), so do not read a one-question change as a
  result.
