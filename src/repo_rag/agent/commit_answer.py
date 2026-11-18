"""Facts and leads added to a commit answer, and the note on commit counts.

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
    return lines[0][:SUBJECT_CHARS] if lines else ""


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


def enrich(
    answer: str,
    abstain: bool,
    searches: Sequence[Sequence[dict[str, Any]]],
    tool_names: Sequence[str],
) -> str:
    parts = [answer.rstrip()]
    for extra in (
        commit_facts(answer, abstain, searches),
        "" if abstain else count_note(tool_names),
    ):
        if extra:
            parts.append(extra)
    return "\n\n".join(parts)
