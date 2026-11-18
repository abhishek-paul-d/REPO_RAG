"""Metrics over distinct source evidence ranges, independent of chunker metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from langchain_core.documents import Document

from eval.retrieval.dataset import Question
from repo_rag.config import get_settings


def source_spans(doc: Document) -> list[tuple[int, int]]:
    meta = doc.metadata
    if "eval_source_spans" in meta:
        return [(int(a), int(b)) for a, b in json.loads(meta["eval_source_spans"])]
    return [(int(meta["start_line"]), int(meta["end_line"]))]


def union_length(intervals: Sequence[tuple[int, int]]) -> int:
    """Inclusive source-line union; overlapping or repeated chunks never add extra credit."""
    total = 0
    previous_end = 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end + 1) + 1)
        previous_end = max(previous_end, end)
    return total


def coverage_scores(
    question: Question, retrieved: Sequence[Document], threshold: float
) -> tuple[float, float]:
    gold_by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
    covered_by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
    complete = 0
    for gold in question.gold:
        gold_by_path[gold.path].append((gold.start_line, gold.end_line))
        intersections = []
        for doc in retrieved:
            meta = doc.metadata
            if meta["path"] == gold.path:
                for begin, stop in source_spans(doc):
                    start = max(gold.start_line, begin)
                    end = min(gold.end_line, stop)
                    if start <= end:
                        intersections.append((start, end))
        covered_by_path[gold.path].extend(intersections)
        complete += union_length(intersections) / (gold.end_line - gold.start_line + 1) >= threshold
    lines = sum(union_length(spans) for spans in gold_by_path.values())
    covered = sum(union_length(spans) for spans in covered_by_path.values())
    return covered / lines, complete / len(question.gold)


def score_question(
    question: Question,
    ranked: Sequence[Document],
    cutoffs: Sequence[int],
    *,
    coverage_threshold: float | None = None,
) -> dict[str, float]:
    if coverage_threshold is None:
        coverage_threshold = get_settings().eval_coverage_threshold
    if not cutoffs or min(cutoffs) < 1 or not 0 < coverage_threshold <= 1:
        raise ValueError("positive cutoffs and a coverage threshold in (0, 1] are required")
    # A second overlapping chunk cannot inflate recall for the same gold range.
    found: set[int] = set()
    first = 0
    scores: dict[str, float] = {}
    seen: set[str] = set()
    unique = []
    for doc in ranked:
        cid = str(doc.metadata["chunk_id"])
        if cid not in seen:
            seen.add(cid)
            unique.append(doc)
    for rank in range(1, max(cutoffs) + 1):
        if rank <= len(unique):
            doc = unique[rank - 1]
            meta = doc.metadata
            matches = {
                i
                for i, gold in enumerate(question.gold)
                if meta["path"] == gold.path
                and any(
                    start <= gold.end_line and end >= gold.start_line
                    for start, end in source_spans(doc)
                )
            }
            if matches and not first:
                first = rank
            found.update(matches)
        if rank in cutoffs:
            scores[f"recall@{rank}"] = len(found) / len(question.gold)
            scores[f"hit@{rank}"] = float(bool(found))
            coverage, complete = coverage_scores(question, unique[:rank], coverage_threshold)
            scores[f"line_coverage@{rank}"] = coverage
            scores[f"complete_gold@{rank}"] = complete
    scores[f"mrr@{max(cutoffs)}"] = 1.0 / first if first else 0.0
    return scores


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}
