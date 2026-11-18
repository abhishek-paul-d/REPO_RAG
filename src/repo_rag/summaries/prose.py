"""Opt-in transformers summarizer and deterministic lexical grounding audit.

The audit rejects invented code names and invented numbers; it does not prove the paragraph's
semantic claims. Ordinary English words are free, because a check that rejected every word
missing from the facts would discard any natural paragraph and make the experiment uninformative.
No model is loaded until summarize() is called, after an explicit approval flag.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from repo_rag.agent.prompts import SUMMARY_PROMPT
from repo_rag.config import Settings

# Bump when the rules below change: it is part of the summarizer's cache identity.
CHECK_VERSION = "identifiers-numbers-v2"

TOKEN = re.compile(r"[^\W\d]\w*(?:[./][^\W\d]\w*)*")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
BACKTICKED = re.compile(r"`([^`\n]+)`")
SENTENCE_END = ".!?:\n"


def _variants(token: str) -> set[str]:
    """The token and its simple plural/possessive stems ('Sessions' -> 'Session')."""
    found = {token}
    for suffix in ("'s", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            found.add(token[: -len(suffix)])
    return found


def _allowed_tokens(facts: str) -> set[str]:
    allowed = set(TOKEN.findall(facts))
    for token in tuple(allowed):
        boundaries = [m.start() for m in re.finditer(r"[./]", token)]
        starts, ends = [0, *(i + 1 for i in boundaries)], [*boundaries, len(token)]
        allowed.update(token[a:b] for a in starts for b in ends if a < b)
    return allowed


def _at_sentence_start(text: str, start: int) -> bool:
    before = text[:start].rstrip(" \t\"'(*-")
    return not before or before[-1] in SENTENCE_END


def _code_like(token: str, text: str, match: re.Match[str], quoted: set[str]) -> bool:
    """True for names a reader would take as code: they must appear in the facts."""
    if token in quoted or text[match.end() : match.end() + 1] == "(":
        return True
    if any(ch in token for ch in "_./"):
        return True
    # camelCase, PascalCase with a later capital, HTTPAdapter; plain acronyms (HTTP) are not names.
    if token[1:] != token[1:].lower() and token != token.upper():
        return True
    # A capitalised single word is a class or proper name unless it opens a sentence.
    return (
        token[:1].isupper() and token[1:].islower() and not _at_sentence_start(text, match.start())
    )


def violations(prose: str, facts: str) -> list[str]:
    """Code-like names and numbers in `prose` that the facts do not contain.

    Rejected: a dotted, slashed, snake_case, camelCase or backticked name, a name followed by
    `(`, a capitalised word inside a sentence, or any number that is not in the facts. Plain
    lowercase words are free (the model must be able to write English), as are plurals of names
    that are present. Case matters for names. This is lexical grounding, not entailment: a false
    relation between two names that are in the facts passes, as does a spelled-out number.
    """
    allowed = _allowed_tokens(facts)
    lowered = {t.lower() for t in allowed}
    numbers = set(NUMBER.findall(facts))
    quoted = {part for span in BACKTICKED.findall(prose) for part in TOKEN.findall(span)}
    bad: set[str] = set()
    for match in TOKEN.finditer(prose):
        token = match.group()
        if not _code_like(token, prose, match, quoted):
            continue
        if _variants(token) & allowed:
            continue
        if token[:1].isupper() and token[1:].islower() and token.lower() in lowered:
            continue  # 'Python' when the facts say 'python'
        bad.add(token)
    bad.update(n for n in NUMBER.findall(prose) if n not in numbers)
    return sorted(bad)


@dataclass
class ProseAudit:
    paragraphs: int = 0
    rejected: int = 0

    @property
    def violation_rate(self) -> float:
        return self.rejected / self.paragraphs if self.paragraphs else 0.0


class TransformersSummarizer:
    def __init__(self, settings: Settings, *, approved: bool = False, revision: str | None = None):
        if not approved:
            raise PermissionError("Explicit user approval of the summary model is required")
        if not settings.summary_model:
            raise ValueError("summary_model must be explicitly selected")
        self.settings = settings
        self.model_id = settings.summary_model
        self.revision = revision
        prompt_hash = hashlib.sha256((SUMMARY_PROMPT + CHECK_VERSION).encode()).hexdigest()
        self.name = (
            f"{settings.summary_model}@{revision}:{prompt_hash}:{settings.summary_prose_max_tokens}"
        )
        self.model: Any = None
        self.tokenizer: Any = None
        self.audit = ProseAudit()
        self.records: list[dict] = []

    def summarize(self, kind: str, path: str, facts: str) -> str:
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError("Summary prose experiment requires a Colab CUDA GPU")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, torch_dtype=torch.bfloat16
            )
            self.model.to("cuda")
            self.model.eval()
        import torch

        assert self.tokenizer is not None
        prompt = SUMMARY_PROMPT.format(facts=facts)
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.settings.summary_prose_max_tokens
            )
        return self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

    def validate(self, kind: str, path: str, facts: str, paragraph: str) -> str:
        """Runs even on cached raw prose, so every run has a complete audit denominator."""
        bad = violations(paragraph, facts)
        self.audit.paragraphs += 1
        self.audit.rejected += bool(bad)
        self.records.append(
            {
                "kind": kind,
                "path": path,
                "facts_sha256": hashlib.sha256(facts.encode()).hexdigest(),
                "paragraph": paragraph,
                "violations": bad,
                "accepted": not bad,
            }
        )
        return "" if bad else paragraph
