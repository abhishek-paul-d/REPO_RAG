# One Colab session for everything (2026-11-19)

Work is in the git worktree `D:\Repo_chatbot_quality` (branch `quality/answers`, made from the commit
`2bd1c8e`). Run everything below from that folder. Its `.env` points at the shared
`D:\Repo_chatbot\data`, so the index and `metadata.db` are the same.

## What the session measures, and the question each step answers

| Step | Questions | About | What it tells us |
|---|---|---|---|
| `fresh_default` | 73 new | 6 min | the honest pass rate on questions the system was not tuned on |
| `fresh_guards` | 73 new | 6 min | do the answer guards help? (decides their default) |
| `commit_default` | 113 | 9 min | the extra row fields and appended facts did not change the commit score (was 70) |
| `commit_explain` | 113 | 9 min | the "SHA plus one sentence why" prompt keeps the score and explains |
| `regression_default` | 103 | 8 min | nothing was lost on the tuned questions |
| `architecture_v1`, `architecture_fresh` | 15 + 10 | 7 min | answers to read: architecture has never been reviewed |
| `fresh_repeat_2`, `fresh_repeat_3` | 73 new | 12 min | run-to-run noise on the fresh set |
| `summaries` | 1,143 commits | about 80 min | the input of the commit-summary pilot (resumable) |

About 60 minutes without the summaries, about 2.5 hours with them. The reranker is not measured: it
does not fit next to the 27B on one 40 GB GPU.

## Steps for you

1. Upload to Drive, from the worktree (`D:\Repo_chatbot_quality\notebooks\`):
   `repo_rag_ask_source.zip` and `serve_model_27b_on_colab.ipynb`. Its hash starts `2e86916f`.
2. Start the 27B server notebook. The health line it prints must say `decode_mode: cuda-graphs + mtp3`
   and `max_input_tokens: 12288` (the runner refuses to start otherwise).
3. Run, with the URL and token the notebook prints:

```powershell
cd D:\Repo_chatbot_quality
uv run python eval/colab_session.py --url <url> --token <token>
```

   If the session dies, start the server again and run the same command: finished steps are skipped
   and the summary step resumes. `--only fresh_default commit_explain` runs just those;
   `--no-summaries` leaves out the long step.
4. Read `eval/colab_session_results/latest/report.md` (pass counts, flips, noise) and
   `answers_sample.md` (four answers per type per step), or tell Claude the run finished and it reads
   them and the full answers.
5. After the summary step finishes, in a separate GPU Colab session (the 27B uses the GPU of the first):

```powershell
uv run python eval/commit_summary_pilot.py package
```

   Upload `notebooks/commit_summary_pilot_on_colab.ipynb`, run it, upload
   `eval/commit_summary_pilot/pilot_input.zip` when it asks, download `pilot_vectors.npz`, then:

```powershell
uv run python eval/commit_summary_pilot.py score --vectors <path to pilot_vectors.npz>
```

6. Decide, with the report in front of you: the guards default, the `commit_explain` default, and
   whether the summary pilot justifies summarising all 4,883 commits. Nothing changes a default until then.

## Faster next time: questions in parallel

vLLM batches requests by itself; only our wrapper served one at a time. `serve_model(...,
concurrency=N)` now lets N `/generate` calls overlap (the 27B notebook sets `CONCURRENCY = 8`, shown
in the health line), and `agent_eval.py run --workers N` asks N questions at once, one agent per
thread. Defaults stay 1, so every earlier number is unchanged. Not yet measured live: first run one
step with `--workers 1` and again with `--workers 4` and compare pass counts and wall time before
trusting it. The notebooks were rebuilt for this (zip hash starts `c2a718e2`; upload that zip and the
27B notebook again to get it). The session now running uses the old server and is unaffected.

## Merging the worktree back

Nothing in the worktree is committed. When you are happy, commit there (`git -C D:\Repo_chatbot_quality
add -A`, `commit`) and merge the branch `quality/answers` into `main`, or copy the changed files over.
`tests/`, `experiments/` and `.env` are ignored files: the worktree has its own copies, so new tests
written here (`tests/test_commit_answer.py`, `test_colab_session.py`, `test_fresh_v2.py`) must be copied
to `D:\Repo_chatbot\tests` by hand.
