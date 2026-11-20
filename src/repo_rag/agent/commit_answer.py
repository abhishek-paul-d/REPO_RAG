"""Facts added to short answers from the tool results: commit details and leads, PR and issue
titles, contributor activity, and the note on commit counts.

The model's commit answer is often just a SHA. This adds, from the search results alone (never
from the model): the commit's date, author and subject, where it ranked, and that the match is a
text match rather than a verified fix. When the model found nothing it can still show the closest
candidates, clearly marked as unverified leads. Nothing here changes `abstain`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

SHA = re.compile(r"\b[0-9a-f]{12}\b")
LEADS = 3  # candidates shown when the model found no confident match
SUBJECT_CHARS = 100
NUMBER_ANSWER_CHARS = 300  # only short, exact-fact answers get number and people details
NUMBER_LINES = 5
COUNT_TOOLS = frozenset({"count_commits", "top_contributors", "change_frequency"})
COUNT_NOTE = "Note: commit counts cover non-merge commits only (merge commits are not indexed)."
MATCH_NOTE = (
    "ranked {rank} of {total} search results; "
    "a text match on the report, not confirmed against the code"
)
LEADS_HEADER = (
    "No confident match. Closest candidates from the commit search "
    "(unverified leads, not confirmed fixes):"
)


def subject_of(row: dict[str, Any]) -> str:
    lines = str(row.get("text") or row.get("subject") or "").splitlines()
    if not lines:
        return ""
    if lines[0].startswith(("Files: ", "Symbols: ")):
        return "(no commit message)"
    return lines[0][:SUBJECT_CHARS]


def describe(row: dict[str, Any]) -> str:
    who = ", ".join(x for x in (str(row.get("date") or ""), str(row.get("author") or "")) if x)
    head = f"{row['sha']} ({who})" if who else str(row["sha"])
    subject = subject_of(row)
    return f"{head}: {subject}" if subject else head


def commit_facts(answer: str, abstain: bool, searches: Sequence[Sequence[dict[str, Any]]]) -> str:
    """The text to append to a commit answer, or "" when there is nothing from the search to add."""
    searches = [rows for rows in searches if rows]
    if not searches:
        return ""
    if abstain:
        lines = [f"- {describe(row)}" for row in searches[-1][:LEADS]]
        return "\n".join([LEADS_HEADER, *lines])
    named: list[str] = []
    for sha in SHA.findall(answer.lower()):
        if sha not in named:
            named.append(sha)
    lines = []
    for sha in named:
        for rows in searches:
            position = next((i for i, row in enumerate(rows) if row["sha"] == sha), None)
            if position is not None:
                note = MATCH_NOTE.format(rank=position + 1, total=len(rows))
                lines.append(f"- {describe(rows[position])}; {note}.")
                break
    return "\n".join(["Matched commit details:", *lines]) if lines else ""


def count_note(tool_names: Sequence[str]) -> str:
    return COUNT_NOTE if COUNT_TOOLS & set(tool_names) else ""


def _values(calls: Sequence[dict[str, Any]], name: str) -> list[Any]:
    return [(c.get("result") or {}).get("value") for c in calls if c.get("name") == name]


def number_facts(answer: str, calls: Sequence[dict[str, Any]]) -> str:
    """Title and state for each PR or issue number a short answer names, from the tool rows.

    A bare "- 5797" says nothing; the rows the model saw carry the title. Only numbers of three or
    more digits (or written #N) count, so a plain count such as "12" is never matched to an issue.
    """
    if len(answer) > NUMBER_ANSWER_CHARS:
        return ""
    known: dict[str, dict[str, Any]] = {}
    for rows in _values(calls, "search_prs"):
        for row in rows if isinstance(rows, list) else []:
            first_line = str(row.get("text") or "").splitlines()[:1]
            known[str(row.get("number"))] = {**row, "title": first_line[0] if first_line else ""}
    for value in _values(calls, "find_issues"):
        for row in (value or {}).get("shown", []) if isinstance(value, dict) else []:
            known[str(row.get("number"))] = {**row, "kind": "issue"}
    for value in _values(calls, "get_pr"):
        if isinstance(value, dict) and value.get("number") is not None:
            known[str(value["number"])] = {**value, "kind": "pull request"}
    lines: list[str] = []
    for token in re.findall(r"#(\d+)|\b(\d{3,6})\b", answer):
        number = token[0] or token[1]
        row = known.get(number)
        if row is None or f"#{number}" in "".join(lines):
            continue
        kind = str(row.get("kind") or "")
        state = str(row.get("state") or "")
        label = ", ".join(x for x in (kind, state.lower()) if x)
        title = str(row.get("title") or "")[:SUBJECT_CHARS]
        lines.append(f"- #{number} ({label}): {title}" if label else f"- #{number}: {title}")
    return "\n".join(["Details from the search results:", *lines[:NUMBER_LINES]]) if lines else ""


def people_facts(answer: str, calls: Sequence[dict[str, Any]]) -> str:
    """Commits and active years for each contributor a short answer names (top_contributors)."""
    if len(answer) > NUMBER_ANSWER_CHARS:
        return ""
    text = answer.lower()
    lines = []
    for value in _values(calls, "top_contributors"):
        for row in (value or {}).get("rows", []) if isinstance(value, dict) else []:
            names = [str(row.get("name") or ""), str(row.get("login") or "")]
            if not any(n and n.lower() in text for n in names):
                continue
            first, last = str(row.get("first_commit") or "")[:10], str(row.get("last_commit") or "")
            lines.append(
                f"- {row['name']}: {row.get('commits')} commits in this history, "
                f"first {first}, most recent {last[:10]}"
            )
    return "\n".join(["Activity from the commit history:", *lines[:NUMBER_LINES]]) if lines else ""


def enrich(
    answer: str,
    abstain: bool,
    searches: Sequence[Sequence[dict[str, Any]]],
    tool_names: Sequence[str],
    calls: Sequence[dict[str, Any]] = (),
) -> str:
    parts = [answer.rstrip()]
    extras = [commit_facts(answer, abstain, searches)]
    if not abstain:
        extras += [count_note(tool_names), number_facts(answer, calls), people_facts(answer, calls)]
    parts += [extra for extra in extras if extra]
    return "\n\n".join(parts)
