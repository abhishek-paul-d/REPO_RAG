"""Cheap, conservative detection of an explicit source-type request in a code question.

Only phrasings that state the target ("which test checks...", "locate the implementation of...")
are matched. Anything else is "other" and keeps the default instruction. The patterns were
written from the failures in the Phase 4 comparison and are NOT validated on other wording.
"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["implementation", "test", "other"]

_TEST = re.compile(
    r"\b(which|what)\s+(?:\w+\s+){0,2}tests?\b"
    r"|\btests?\s+(?:that\s+)?(?:verif|check|ensur|assert|cover|prov)"
    r"|\b(?:is|are) there (?:a|any) tests?\b"
    r"|\b(?:find|show|locate)\s+the\s+(?:\w+\s+){0,3}tests?\b"
    r"|\bwhere\b[^?.]*\btested\b",
    re.IGNORECASE,
)
_IMPLEMENTATION = re.compile(
    r"\b(?:locate|find|show)\b[^?.]*\b(?:implementation|implemented|definition|defined)\b"
    r"|\bwhere (?:is|are)\b[^?.]*\b(?:implemented|defined)\b",
    re.IGNORECASE,
)


def detect_intent(query: str) -> Intent:
    if _TEST.search(query):
        return "test"
    if _IMPLEMENTATION.search(query):
        return "implementation"
    return "other"
