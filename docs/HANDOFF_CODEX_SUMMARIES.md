# Handoff to Codex: write the remaining commit summaries (2026-11-19)

Work in the git worktree `D:\Repo_chatbot_quality` (branch `quality/answers`). Do not commit anything and
do not touch `D:\Repo_chatbot`. Do not change anything under `src/`, `pyproject.toml` or `uv.lock`
(that would make every generated notebook stale).

## Why this exists

The commit-summary pilot asks one question: does adding a short plain-English `Summary:` line to each
indexed commit make "bug report -> past fix commit" search better? The summaries were meant to be
written by the project's own model (Qwen3.8-27B on Colab). The Colab session ended after **67 of 1,143**
were written, and the model server is gone. You write the other **1,076** by imitating the 27B's style.

**Validity warning (keep it in every report you write).** These summaries will come from two writers: the
27B (67) and you (1,076). The pilot then measures "summaries written by Codex imitating the 27B", which
is not the same as "summaries the shipped 27B would write". Any result must be labelled that way, and
must not be used on its own to justify summarising all 4,883 commits with the 27B. If the 27B server is
up again later, the 27B can finish the job (`uv run python eval/colab_session.py --url ... --token ...
--only summaries` resumes from what is saved); then this handoff is not needed.

## Files (all in `eval/commit_summary_pilot/`)

- `commits.jsonl`: 1,143 rows `{"sha", "text"}`. `text` is the indexed commit text: message, then
  `Files:`, `Symbols:`, then the diff hunks. This is the ONLY input a summary may use.
- `summaries.jsonl`: the 67 finished 27B summaries, rows `{"sha", "summary", "seconds"}`. Read all 67
  first and copy their register: length, tone, what they name.
- `queries.jsonl`: the 141 bug-report test queries and their gold commits.
  **Do not open `queries.jsonl` or `eval/datasets/phase5_bugfix_v2/`.** A summary must be written from the
  commit alone, as the 27B did. Reading the bug reports or the gold commits would leak the answer and
  void the pilot. Do not use `git log`, GitHub, the issue tracker or anything outside `text` either.
  (The gold commits are mixed in among the 1,143, so you cannot and should not try to tell them apart.)

## The exact task the 27B was given (reproduce it)

System: "You write short, plain-English descriptions of git commits for a search index. Use only the
commit shown. Never invent details that are not in it."

Instruction: "Describe this commit in two or three plain sentences, at most 60 words. Say what behaviour
or problem it changes (the symptom a user would notice, if it fixes a bug), and name the main function,
class or module. Do not quote code or diff lines and do not mention issue numbers. Reply with the
description only."

The model saw `text[:6000]` (the first 6,000 characters). Do the same: ignore everything after that.

Measured on the 67 existing summaries: 158 to 330 characters, mean 230. Typical shape: "This commit adds
support for the HTTP PATCH method to the requests library. It updates the `request` function in
`requests/api.py` to accept PATCH requests and exports the new `patch` function."

## Rules for each summary

1. Two or three plain sentences, at most 60 words, one line, no leading "Summary:" label, no `<think>`.
2. Say the behaviour or symptom first; name the main function, class or module (backticks are fine).
3. Use only facts visible in that commit's `text[:6000]`. Never guess a cause that is not in it. If the
   commit is only a docs, typo or version bump, say so plainly.
4. No code or diff lines quoted, and no issue or PR numbers (`#123`, `GH-123`).
5. Do not write in a way that reflects the later bug reports. You have not seen them; keep it that way.

## How to do it

1. Read the 67 existing summaries and a few of their commits to learn the style.
2. Write the remaining 1,076 in batches (for example 50 to 100 per batch), in the order of
   `commits.jsonl`, skipping any `sha` already in `summaries.jsonl`.
3. Append to `summaries.jsonl` rows `{"sha": <full sha>, "summary": <text>, "seconds": 0, "source": "codex"}`.
   Keep the 67 existing rows unchanged. The extra `source` field is ignored by the pilot code
   (`package` and `score` read only `sha` and `summary`), and it lets us separate the two writers later.
4. Pass every summary through the pilot's own cleaner so it is the same shape the 27B's went through:
   `from eval.commit_summary_pilot import clean_summary` (cuts to 700 characters, one line, removes a
   leading label). If you generate summaries with a script that calls a model, use that function; if you
   write them by hand or in-session, apply it before saving.
5. Make no network call to the Colab server; it no longer exists.

## Check before finishing

```powershell
cd D:\Repo_chatbot_quality
uv run python eval/commit_summary_pilot.py package
```

It must print `wrote ...pilot_input.zip (... 1143 commits)`. It refuses with "N commits have no summary
yet" until every commit has one. Also report: the number of Codex rows, duplicate shas (must be 0), rows
with a summary over 60 words or containing `#` followed by digits (should be 0), and a sample of 10 for
the reviewer.

```powershell
uv run pytest tests/test_commit_summary_pilot.py -q --basetemp <a scratch folder>
uv run ruff check eval
```

## What happens next (not for Codex)

Claude reviews a sample of the Codex summaries against their commits (faithful, within the rules, not
leaking), then the user runs `notebooks/commit_summary_pilot_on_colab.ipynb` on a GPU Colab with
`eval/commit_summary_pilot/pilot_input.zip`, downloads `pilot_vectors.npz`, and Claude scores it with
`uv run python eval/commit_summary_pilot.py score --vectors <path>`. The score compares dense and fused
rank of the 141 gold commits with and without the summary. Report it as "summaries by Codex imitating
the 27B".
