"""Ambiguous-intent stress cases with alternative sufficient evidence sets.

Alternatives are OR; ranges within an alternative are AND. Frozen v1/v2 labels are untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval.retrieval.dataset import GoldRange, Question, overlaps, validate_checkout
from eval.retrieval.metrics import score_question
from repo_rag.config import Settings


class AmbiguousCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Question
    accepted_evidence: list[list[GoldRange]] = Field(min_length=1)
    rationale: str
    related_v2_ids: list[str]
    stratum: Literal["ambiguous", "multi_block", "implementation", "test", "example"] = "ambiguous"
    topic: str = ""
    origin_case_id: str | None = None

    @model_validator(mode="after")
    def valid_alternatives(self) -> AmbiguousCase:
        if any(not alternative for alternative in self.accepted_evidence):
            raise ValueError("evidence alternatives must be nonempty")
        if self.question.gold != self.accepted_evidence[0]:
            raise ValueError("question gold must equal the first accepted evidence set")
        return self


def read_cases(path: Path) -> list[AmbiguousCase]:
    cases = [
        AmbiguousCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({c.question.id for c in cases}) != len(cases):
        raise ValueError("empty stress set or duplicate question ids")
    if len({(c.question.repo, c.question.commit_sha) for c in cases}) != 1:
        raise ValueError("one case file must describe one pinned repository")
    for i, left in enumerate(cases):
        for right in cases[i + 1 :]:
            if left.question.group != right.question.group and any(
                overlaps(a, b)
                for proof in left.accepted_evidence
                for a in proof
                for other in right.accepted_evidence
                for b in other
            ):
                raise ValueError("overlapping accepted evidence must share a bootstrap group")
    return cases


def validate_cases(checkout: Path, cases: list[AmbiguousCase]) -> None:
    validate_checkout(
        checkout,
        [
            c.question.model_copy(update={"gold": [g for a in c.accepted_evidence for g in a]})
            for c in cases
        ],
    )


def score_case(case: AmbiguousCase, docs: list[Document], settings: Settings) -> dict:
    alternatives = [
        score_question(
            case.question.model_copy(update={"gold": gold}),
            docs,
            settings.eval_cutoffs,
            coverage_threshold=settings.eval_coverage_threshold,
        )
        for gold in case.accepted_evidence
    ]
    return {
        # These are separate "any acceptable evidence" metrics, not comparable to v2's gold.
        **{key: max(row[key] for row in alternatives) for key in alternatives[0]},
        **{
            f"any_complete_proof@{k}": float(
                any(row[f"complete_gold@{k}"] == 1 for row in alternatives)
            )
            for k in settings.eval_cutoffs
        },
        "accepted_evidence_scores": alternatives,
    }
