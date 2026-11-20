# Handoff to Codex: read ALL 4,883 commit summaries and report which are wrong (2026-11-19)

This replaces `docs/HANDOFF_CODEX_REVIEW.md` (the 208-row sample). If you already finished that one, keep its
files; this job covers every commit, so do not repeat the 208 separately.

Work in the git worktree `D:\Repo_chatbot_quality` (branch `quality/answers`). Do not commit anything, do
not touch `D:\Repo_chatbot`, and do not change anything under `src/`, `pyproject.toml` or `uv.lock`. You are
only a reviewer: do not edit any summary, do not call the Colab server (it may be gone), and do not write a
script or call a model that decides the verdicts. **Read each row yourself.** A script may only be used to
display rows, to write down your own verdicts, or to count them.

## Why this exists

The project added a plain-English `Summary:` line to every indexed commit of `psf/requests` (4,883 commits),
written by the 27B answer model from the commit text alone. Claude read about 140 of them and found 3 wrong
(about 2%), which says nothing about the other 97%. You are asked to read all of them so we know how many
are wrong, which kinds of commit it gets wrong, and whether any should be fixed or dropped before the
summaries go anywhere near the shipped index.

Stay independent: **do not open** the "Validity check of the 4,883 summaries" paragraph in
`docs/PROJECT_LOG.md` (§2, entry "2026-11-19 (evening)"), nor `eval/commit_summary_pilot/validate_flags.json`,
`manual_batch1.json`, `lost_ranks.json`, `agent_query_ranks.json`, `review_sample.jsonl` or anything in
`eval/agent_eval_results/`.

## The input: 56 batch files

`eval/commit_summary_pilot/review_all/batch_001.jsonl` ... `batch_056.jsonl`: 4,883 rows in total, sorted by
sha, 24 to 100 rows per batch (each batch is about 90,000 characters of commit text, so roughly 25,000
tokens). One JSON object per line:

- `n`: global row number, 1..4883
- `sha`: the commit
- `commit_text`: exactly what the model saw (message, `Files:`, `Symbols:`, then the diff hunks, cut to 6,000
  characters). **This is the only evidence you may use.** No `git log`, GitHub, issue tracker, or memory of
  the requests project. If the text does not show it, you cannot confirm it.
- `summary`: the 27B's description to judge

In total this is about 3.4 million characters (roughly 1.1 million tokens) of reading, so expect many
sessions. That is the job; do not shorten it by sampling, skimming, or reading only the summary.

## What the summaries were told to do (so you can judge them fairly)

"Describe this commit in two or three plain sentences, at most 60 words. Say what behaviour or problem it
changes (the symptom a user would notice, if it fixes a bug), and name the main function, class or module.
Do not quote code or diff lines and do not mention issue numbers. Use only the commit shown. Never invent
details that are not in it."

## How to judge each summary

Read the commit message, the `Files:` line and the diff, then read the summary and give one verdict:

- **`ok`**: every claim is supported by the commit text, or is an obvious, safe restatement of it (for example
  `MMXVII` to `MMXVIII` read as 2017 to 2018, `10 * 1024` read as 10,240 bytes).
- **`filler`**: nothing contradicts the commit, but it adds something the text does not show, stated as fact:
  generic benefit language ("bringing bug fixes and improvements", "improves reliability"), a guess about
  motive, calling something "deprecated", "broken" or "invalid" when the text never says so, or quoting diff
  text. Weaknesses, not errors.
- **`wrong`**: it contradicts the commit text or states a specific the text does not support. Typical cases:
  the direction is reversed (adds vs removes, increases vs decreases, from-value and to-value swapped), a
  number, name, file or function is not in the commit, or it describes a different change than the diff.
- **`cannot_tell`**: the part that decides it is outside the 6,000 characters shown. Use sparingly and say why.

Check specifically: the direction of every change (look at the `-` and `+` lines and the `@@ -a,b +c,d @@`
counts: `-0,0 +1,991` means the file was ADDED), every number and version, every function or class named,
and whether the summary describes the change the message says was made. For vendored-library updates and
other very large commits, judge on the message, the file list and the first hunks.

## How to work (resumable)

Work one batch at a time, in order. For each batch `NNN`:

1. Read all rows of `review_all/batch_NNN.jsonl`.
2. Write `eval/commit_summary_pilot/review_all/verdicts_NNN.jsonl`, one row per input row, same order:
   `{"n": 1, "sha": "<full sha>", "verdict": "ok|filler|wrong|cannot_tell", "reason": "<one short sentence;
   required for filler, wrong and cannot_tell, optional for ok>"}`.
3. Only then start the next batch. Never write a verdicts file for rows you have not read. If a session ends
   mid-batch, delete that batch's partial verdicts file and redo the batch.

Progress check at any time:

```powershell
cd D:\Repo_chatbot_quality
uv run python -c "import glob,json,collections; c=collections.Counter(); n=0
for f in sorted(glob.glob('eval/commit_summary_pilot/review_all/verdicts_*.jsonl')):
    for l in open(f,encoding='utf-8'): c[json.loads(l)['verdict']]+=1; n+=1
print(n,'of 4883 read',dict(c))"
```

## What to write at the end

When all 56 batches are done, write `docs/REVIEW_CODEX_ALL_SUMMARIES_REPORT.md` with:

- counts of each verdict over all 4,883, and the number of rows read in full (must be 4,883; if fewer, say so
  and list the batches missing);
- every `wrong` row: `n`, short sha, the sentence that is wrong, the evidence in the commit text that
  contradicts it (quote a few words from the diff, not a whole hunk), and what the commit really does;
- every `cannot_tell` row with the reason (a table is fine);
- the `filler` count and 10 representative examples;
- the error rate (wrong / 4,883) with a plain sentence about reliability, and the same rate split by kind of
  commit where you can see one (docs-only, version bump, vendored library update, test-only, code fix);
- one paragraph on anything systematic: a kind of commit the model gets wrong, a phrase it overuses, a rule
  from the prompt it breaks (for example quoting code, naming a function the diff never touches);
- a recommendation, in your own words, on whether the `wrong` rows should be blanked or rewritten before the
  summaries are used anywhere, and which kind of commit should get no summary at all.

## Rules and cautions

- Do not skim and do not guess an expected error rate. The report must say what you actually saw.
- Read-only for everything except the `verdicts_NNN.jsonl` files and the final report. Do not touch
  `summaries.jsonl`, `summaries_27b.jsonl`, `all_commits.jsonl`, `review_all/batch_*.jsonl`, or the Chroma,
  BM25 and metadata files.
- Make no network call. Do not commit.

## Check before finishing

```powershell
cd D:\Repo_chatbot_quality
uv run python -c "import glob,json; r=[json.loads(l) for f in sorted(glob.glob('eval/commit_summary_pilot/review_all/verdicts_*.jsonl')) for l in open(f,encoding='utf-8')]; print(len(r), sorted({x['verdict'] for x in r}), len({x['sha'] for x in r}), sorted(x['n'] for x in r)==list(range(1,4884)))"
```

It must print 4883 rows, only the four verdict names, 4883 distinct shas and `True`.

## What happens next (not for Codex)

Claude compares your verdicts with its own on the 140 or so it read, spot-checks a sample of your `wrong` and
`ok` rows, and reports the agreement and a combined error rate to the user. Your report is not edited by
Claude; disagreements are listed side by side.
