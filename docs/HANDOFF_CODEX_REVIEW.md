# Handoff to Codex: read 208 commit summaries and report which are wrong (2026-11-19)

Work in the git worktree `D:\Repo_chatbot_quality` (branch `quality/answers`). Do not commit anything,
do not touch `D:\Repo_chatbot`, and do not change anything under `src/`, `pyproject.toml` or `uv.lock`.
You are only a reviewer: do not edit any summary, do not call the Colab server (it may be gone), and do
not write a script or call a model that decides the verdicts. **Read each one yourself.** A script may
only be used to display rows or to count your own verdicts.

## Why this exists

The project added a one-paragraph, plain-English `Summary:` line to every indexed commit of `psf/requests`
(4,883 commits), written by the 27B answer model from the commit text alone. Retrieval improved with them,
but the agent's answers did not, so we want to know how trustworthy the summaries are. Claude read about
140 of them and found 3 wrong (about 2%). This review is a second, independent reader on a different set.
Agreement or disagreement with Claude is the point, so **do not read Claude's findings before you finish**:
do not open the "Validity check of the 4,883 summaries" paragraph in `docs/PROJECT_LOG.md`
(§2, entry "2026-11-19 (evening)"), nor `eval/commit_summary_pilot/validate_flags.json`,
`manual_batch1.json`, `lost_ranks.json` or `agent_query_ranks.json`.

## The input

`eval/commit_summary_pilot/review_sample.jsonl`: 208 rows, one JSON object per line:

- `n`: row number 1..208
- `sha`: the commit
- `commit_text`: exactly what the model saw (message, `Files:`, `Symbols:`, then the diff hunks, cut to 6,000
  characters). **This is the only evidence you may use.** No `git log`, GitHub, issue tracker or memory
  of the requests project. If the text does not show it, you cannot confirm it.
- `summary`: the 27B's description to judge

The set mixes random commits, commits whose summary states a direction or a from-to change, 20 long commits,
and a few the lead already knows about. Do not try to guess which ones are which.

## What the summaries were told to do (so you can judge them fairly)

"Describe this commit in two or three plain sentences, at most 60 words. Say what behaviour or problem it
changes (the symptom a user would notice, if it fixes a bug), and name the main function, class or module.
Do not quote code or diff lines and do not mention issue numbers. Use only the commit shown. Never invent
details that are not in it."

## How to judge each summary

Read the commit message, the `Files:` line and the diff, then read the summary and give one verdict:

- **`ok`**: every claim is supported by the commit text, or is an obvious, safe restatement of it (for example
  `MMXVII` to `MMXVIII` read as 2017 to 2018, `10 * 1024` read as 10,240 bytes).
- **`filler`**: nothing contradicts the commit, but it adds something the text does not show, as a plain
  statement: generic benefit language ("bringing bug fixes and improvements", "improves reliability"), a
  guess about motive, calling something "deprecated", "broken" or "invalid" when the text never says so, or
  quoting diff text. These are weaknesses, not errors.
- **`wrong`**: it contradicts the commit text or states a specific the text does not support. Typical cases:
  the direction is reversed (adds vs removes, increases vs decreases, from-value and to-value swapped), a
  number, name, file or function is not in the commit, or it describes a different change than the diff.
- **`cannot_tell`**: the part that decides it is outside the 6,000 characters shown. Use sparingly and say
  why.

Check specifically: the direction of every change (look at `-` and `+` lines and the `@@ -a,b +c,d @@`
counts: `-0,0 +1,991` means the file was ADDED), every number and version, every function or class named,
and whether the summary describes the change the message says was made.

## What to write

1. `eval/commit_summary_pilot/review_codex.jsonl`: one row per sample row, all 208, in order:
   `{"n": 1, "sha": "<full sha>", "verdict": "ok|filler|wrong|cannot_tell", "reason": "<one short sentence,
   required for filler, wrong and cannot_tell, optional for ok>"}`. Write it in batches of about 25 rows as you
   go (append), so an interrupted session loses little.
2. `docs/REVIEW_CODEX_SUMMARIES_REPORT.md`, with:
   - counts of each verdict, and the number of rows you actually read in full (must be 208, say so honestly if
     it is fewer and which rows are missing);
   - every `wrong` row: `n`, short sha, the summary sentence that is wrong, the evidence in the commit text
     that contradicts it (quote a few words from the diff, not a whole hunk), and what the commit really does;
   - every `cannot_tell` row with the reason;
   - three or four representative `filler` rows;
   - an error-rate estimate for this sample (wrong / read) and an honest sentence on how it should not be
     generalised: the sample deliberately over-represents risky and long commits, so it is not an unbiased
     estimate for all 4,883;
   - one paragraph: anything systematic you noticed (a kind of commit the model gets wrong, a phrase it
     overuses, a rule from the prompt it breaks).

## Rules and cautions

- Do not skim. A sample of 208 is about 230,000 characters in total; most rows are short. The hard ones are
  the long commits (vendored library updates): judge them on the message, the file list and the first hunks,
  and mark `cannot_tell` only when the deciding part is genuinely missing.
- Do not be generous and do not be harsh to match an expected answer: you do not know the expected answer.
  The report must say what you actually saw.
- Do not look up or modify `summaries.jsonl`, `summaries_27b.jsonl`, `all_commits.jsonl`, or the
  Chroma / BM25 / metadata files. Read-only for everything except the two output files above.
- Make no network call.

## Check before finishing

```powershell
cd D:\Repo_chatbot_quality
uv run python -c "import json; r=[json.loads(l) for l in open('eval/commit_summary_pilot/review_codex.jsonl',encoding='utf-8')]; print(len(r), sorted({x['verdict'] for x in r}), len({x['sha'] for x in r}))"
```

It must print 208 rows, only the four verdict names, and 208 distinct shas.

## What happens next (not for Codex)

Claude compares your verdicts with its own on the overlapping rows and with the rows it already knows,
then reports the agreement and a combined error estimate to the user. Codex's report is not edited by
Claude; disagreements are listed side by side.
