"""Phase 7 answer-quality check: a local open-source LLM judge grades saved agent answers.

    python eval/judge_answers.py prepare --answers A.json --dataset D.jsonl --checkout CLONE --out DIR
    python eval/judge_answers.py judge   --out DIR --model MODEL_ID_OR_PATH
    python eval/judge_answers.py report  --out DIR

Scope: the `code` and `docs` answers of one saved `eval/agent_eval.py` run. Every other question type is
scored exactly by `agent_eval.py` and needs no judge. Evidence comes from the pinned checkout
(`git show <sha>:<path>`), never from the model:

* faithfulness: is every claim in the answer supported by the lines it cites?
* correctness: does the answer agree with reference evidence (the gold chunk for code questions;
  the reference doc files for docs questions)?

The judge is only trusted if it first rejects negative controls (answers paired with another
question's evidence). The gates are frozen in eval/datasets/phase7_answer_judge_v2/protocol.json (v1 = the first run).
The pure functions here run offline; only `judge` needs a GPU and vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(
    os.environ.get("JUDGE_PROTOCOL") or ROOT / "eval/datasets/phase7_answer_judge_v2/protocol.json"
)
CITATION = re.compile(r"([\w./-]+\.\w+):(\d+)-(\d+)")
FAITHFUL_VERDICTS = ("faithful", "partial", "unfaithful")
CORRECT_VERDICTS = ("correct", "partial", "incorrect")

FAITHFULNESS_SYSTEM_V1 = (
    "You are a strict reviewer of answers about a source repository. You are given an answer and the "
    "exact source lines it cites. Decide whether every factual claim in the answer is supported by "
    "those cited lines. A claim about code behaviour that the cited lines do not show is unsupported. "
    "Ignore style. Do not use outside knowledge. Reply with one JSON object only."
)
FAITHFULNESS_USER_V1 = """Question:
{question}

Answer to review:
{answer}

Cited source lines:
{evidence}

Reply with JSON: {{"reasoning": "<two sentences at most>", "unsupported_claims": ["<claim>", ...], \
"verdict": "faithful" | "partial" | "unfaithful"}}
faithful = every claim supported; partial = some claims unsupported; unfaithful = most claims \
unsupported or the lines are unrelated to the answer."""

CORRECTNESS_SYSTEM_V1 = (
    "You are a strict reviewer of answers about a source repository. You are given a question, "
    "reference evidence taken from the repository, and an answer. Decide whether the answer is a "
    "correct response to the question, judged against the reference evidence. Extra true detail is "
    "fine; a wrong or unrelated answer is not. Do not use outside knowledge. Reply with one JSON "
    "object only."
)
CORRECTNESS_USER_V1 = """Question:
{question}

Reference evidence:
{evidence}

Answer to review:
{answer}

Reply with JSON: {{"reasoning": "<two sentences at most>", "verdict": "correct" | "partial" | \
"incorrect"}}
correct = answers the question and agrees with the evidence; partial = right direction but \
incomplete or partly wrong; incorrect = wrong, unrelated or contradicted by the evidence."""


# v2 (2026-11-11): written after reading where the v1 judge and two other reviewers differed.
# Answers are only judged on what they assert; citation line numbers are not content.
FAITHFULNESS_SYSTEM_V2 = (
    "You are a strict reviewer of answers about a source repository. You are given an answer and the "
    "exact source lines it cites. Decide whether every factual claim the answer makes is supported "
    "by those cited lines. A claim about code behaviour that the cited lines do not show is "
    "unsupported, and so is a claim the lines contradict. Judge only what the answer asserts: "
    "leaving something out is not a faithfulness problem, and a cited range that is wider than "
    "needed is fine as long as the lines shown contain what the answer describes. Do not use outside "
    "knowledge. Reply with one JSON object only."
)
FAITHFULNESS_USER_V2 = FAITHFULNESS_USER_V1
CORRECTNESS_SYSTEM_V2 = (
    "You are a strict reviewer of answers about a source repository. You are given a question, "
    "reference evidence taken from the repository, and an answer. Decide whether the answer is a "
    "correct response to the question, judged against the reference evidence. Ignore the file "
    "paths and line numbers the answer cites: judge its content only. Extra true detail is fine, "
    "and leaving out detail the question did not ask for is fine. An answer that leaves out "
    "something the question asks for (a format, a usage example, the actual mechanism) is partial. "
    "A wrong or unrelated answer, or one that contradicts the evidence, is not correct. Do not use "
    "outside knowledge. Reply with one JSON object only."
)
CORRECTNESS_USER_V2 = CORRECTNESS_USER_V1
PROMPTS = {
    "v1": (
        FAITHFULNESS_SYSTEM_V1,
        FAITHFULNESS_USER_V1,
        CORRECTNESS_SYSTEM_V1,
        CORRECTNESS_USER_V1,
    ),
    "v2": (
        FAITHFULNESS_SYSTEM_V2,
        FAITHFULNESS_USER_V2,
        CORRECTNESS_SYSTEM_V2,
        CORRECTNESS_USER_V2,
    ),
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- evidence


def git_lines(checkout: Path, sha: str, path: str) -> list[str] | None:
    """Lines of `path` at `sha`, or None when the file does not exist there."""
    done = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{sha}:{path}"],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", errors="replace").splitlines()


def snippet(lines: list[str], start: int, end: int, max_lines: int) -> str:
    start = max(start, 1)
    end = min(end, len(lines), start + max_lines - 1)
    return "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))


def cited_evidence(
    answer: str, reader: Callable[[str], list[str] | None], limits: dict[str, int]
) -> tuple[str, list[dict[str, Any]]]:
    """Text of every distinct cited range that resolves, plus a record of each citation."""
    seen: set[tuple[str, int, int]] = set()
    blocks: list[str] = []
    records: list[dict[str, Any]] = []
    for path, a, b in CITATION.findall(answer):
        key = (path, int(a), int(b))
        if key in seen:
            continue
        seen.add(key)
        lines = reader(path)
        if lines is None or int(a) > len(lines) or int(b) < int(a):
            records.append({"citation": f"{path}:{a}-{b}", "resolved": False})
            continue
        records.append({"citation": f"{path}:{a}-{b}", "resolved": True})
        if len(blocks) < limits["max_citations"]:
            body = snippet(lines, int(a), int(b), limits["max_lines_per_citation"])
            blocks.append(f"[{path} lines {a}-{b}]\n{body}")
    text = "\n\n".join(blocks)
    return text[: limits["max_evidence_chars"]], records


_STOPWORDS = frozenset(
    [
        "how",
        "do",
        "does",
        "did",
        "i",
        "a",
        "an",
        "the",
        "to",
        "on",
        "in",
        "of",
        "is",
        "are",
        "can",
        "what",
        "where",
        "which",
        "for",
        "with",
        "or",
        "and",
        "my",
        "it",
        "this",
        "that",
        "set",
        "use",
        "using",
        "get",
        "make",
        "when",
        "why",
        "should",
        "would",
    ]
)
_RST_UNDERLINE = re.compile(r"^([=\-~^\"`#*+.])\1{2,}\s*$")


def doc_sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """(heading, first line, last line) per section of an .rst or .md file, 1-based, in file order.

    A heading is a Markdown `#` line or an RST title line followed by an underline at least as
    long. Text before the first heading is one section titled "(start of file)".
    """
    starts: list[tuple[str, int]] = []
    fenced = False
    for n, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        if re.match(r"^#{1,6}\s+\S", line):
            starts.append((line.lstrip("# ").strip(), n))
        elif (
            n < len(lines)
            and line.strip()
            and not line.startswith((" ", "\t"))
            and _RST_UNDERLINE.match(lines[n])
            and len(lines[n].rstrip()) >= len(line.rstrip())
            and not _RST_UNDERLINE.match(line)
        ):
            starts.append((line.strip(), n))
    if not starts or starts[0][1] > 1:
        starts.insert(0, ("(start of file)", 1))
    return [
        (title, first, (starts[k + 1][1] - 1 if k + 1 < len(starts) else len(lines)))
        for k, (title, first) in enumerate(starts)
    ]


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9]+", text.lower())
    return {w[:5] for w in words if w not in _STOPWORDS and len(w) > 2}


def doc_reference(
    lines: list[str], question: str, max_chars: int, max_sections: int = 3
) -> tuple[str, list[str]]:
    """The sections of a doc most about the question, in file order, within `max_chars`.

    Scored by question stems found in a section's heading (weight 3) or body (weight 1), each
    stem weighted by how rare it is across the file's sections. Falls back to the start of the
    file when no stem matches. Deterministic; no model.
    """
    import math

    sections = doc_sections(lines)
    stems = _stems(question)
    heads = [_stems(title) for title, _, _ in sections]
    bodies = [_stems("\n".join(lines[a - 1 : b])) for _, a, b in sections]
    scored: list[tuple[float, int]] = []
    for k in range(len(sections)):
        score = 0.0
        for stem in stems:
            df = sum(stem in body or stem in head for body, head in zip(bodies, heads, strict=True))
            if df:
                weight = math.log(1 + len(sections) / df)
                score += weight * (3 if stem in heads[k] else 1 if stem in bodies[k] else 0)
        scored.append((score, k))
    ranked = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]))
    best = [x for x in ranked if x[0] >= 0.5 * ranked[0][0]][:max_sections] if ranked else []
    if not best:
        return "\n".join(lines)[:max_chars], ["(start of file)"]
    chosen = sorted(k for _, k in best)
    blocks: list[str] = []
    used = 0
    for k in chosen:
        title, first, last = sections[k]
        body = "\n".join(lines[first - 1 : last]).rstrip()
        room = max_chars - used
        if room <= 0:
            break
        blocks.append(f"[lines {first}-{last}: {title}]\n{body[:room]}")
        used += len(blocks[-1])
    return "\n\n".join(blocks), [sections[k][0] for k in chosen]


def reference_evidence(
    record: dict[str, Any],
    reader: Callable[[str], list[str] | None],
    limits: dict[str, int],
    mode: str = "head",
) -> str:
    """Reference text for correctness: gold ranges for code; for docs the named files, either
    the start of each (`head`, v1) or the sections most about the question (`sections`, v2)."""
    expect = record["expect"]
    blocks: list[str] = []
    if record["type"] == "code":
        for gold in expect["gold"]:
            lines = reader(gold["path"])
            if lines is not None:
                body = snippet(
                    lines, gold["start_line"], gold["end_line"], limits["max_reference_lines"]
                )
                blocks.append(
                    f"[{gold['path']} lines {gold['start_line']}-{gold['end_line']}]\n{body}"
                )
    else:
        for path in expect["paths"][: limits["max_reference_files"]]:
            lines = reader(path)
            if lines is None:
                continue
            if mode == "sections":
                body, _ = doc_reference(
                    lines, record["question"], limits["max_reference_file_chars"]
                )
                blocks.append(f"[{path}, sections about the question]\n{body}")
            else:
                body = "\n".join(lines)[: limits["max_reference_file_chars"]]
                blocks.append(f"[{path}, start of file]\n{body}")
    return "\n\n".join(blocks)[: limits["max_evidence_chars"]]


# ----------------------------------------------------------------------------- items


def build_items(
    records: dict[str, dict[str, Any]],
    answers: dict[str, Any],
    reader: Callable[[str], list[str] | None],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    limits = protocol["limits"]
    items: list[dict[str, Any]] = []
    for row in answers["rows"]:
        if row["type"] not in protocol["judged_types"]:
            continue
        record = records[row["id"]]
        base = {
            "id": row["id"],
            "type": row["type"],
            "question": record["question"],
            "answer": row["answer"] if isinstance(row["answer"], str) else "",
            "mechanical_passed": row["passed"],
        }
        if row["failure"] or row["abstain"] or not base["answer"].strip():
            items.append({**base, "skipped": "failed" if row["failure"] else "abstained"})
            continue
        evidence, citations = cited_evidence(base["answer"], reader, limits)
        items.append(
            {
                **base,
                "skipped": "",
                "cited_evidence": evidence,
                "citations": citations,
                "reference_evidence": reference_evidence(
                    record, reader, limits, protocol.get("reference_mode", "head")
                ),
            }
        )
    return items


def build_controls(items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Negative controls: each answer paired with a different question's evidence.

    A derangement within each question type, so no item keeps its own partner. Expected verdicts:
    faithfulness `unfaithful` (the cited lines belong to another answer) and correctness
    `incorrect` (the answer belongs to another question).
    """
    rng = random.Random(seed)
    usable = [i for i in items if not i["skipped"] and i["cited_evidence"]]
    controls: list[dict[str, Any]] = []
    for kind in sorted({i["type"] for i in usable}):
        group = [i for i in usable if i["type"] == kind]
        if len(group) < 2:
            continue
        order = list(range(len(group)))
        for _ in range(100):
            rng.shuffle(order)
            if all(a != b for a, b in enumerate(order)):
                break
        else:  # fall back to a rotation, always a derangement
            order = [(n + 1) % len(group) for n in range(len(group))]
        for n, item in enumerate(group):
            other = group[order[n]]
            controls.append(
                {
                    "id": f"ctl-{item['id']}~{other['id']}",
                    "type": kind,
                    "source": item["id"],
                    "partner": other["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "cited_evidence": other["cited_evidence"],
                    "reference_evidence": other["reference_evidence"],
                    "question_of_partner": other["question"],
                    "answer_of_partner": other["answer"],
                }
            )
    return controls


def _prompts() -> tuple[str, str, str, str]:
    return PROMPTS[load_protocol().get("prompt_version", "v1")]


def faithfulness_messages(question: str, answer: str, evidence: str) -> list[dict[str, str]]:
    system, template, _, _ = _prompts()
    user = template.format(question=question, answer=answer, evidence=evidence)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def correctness_messages(question: str, answer: str, evidence: str) -> list[dict[str, str]]:
    _, _, system, template = _prompts()
    user = template.format(question=question, answer=answer, evidence=evidence)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_requests(
    items: list[dict[str, Any]], controls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every prompt the judge will see, with its key and the verdict a good judge must give."""
    requests: list[dict[str, Any]] = []
    for item in items:
        if item["skipped"]:
            continue
        if item["cited_evidence"]:
            requests.append(
                {
                    "key": f"faith:{item['id']}",
                    "kind": "faithfulness",
                    "role": "answer",
                    "id": item["id"],
                    "messages": faithfulness_messages(
                        item["question"], item["answer"], item["cited_evidence"]
                    ),
                }
            )
        if item["reference_evidence"]:
            requests.append(
                {
                    "key": f"correct:{item['id']}",
                    "kind": "correctness",
                    "role": "answer",
                    "id": item["id"],
                    "messages": correctness_messages(
                        item["question"], item["answer"], item["reference_evidence"]
                    ),
                }
            )
    for control in controls:
        requests.append(
            {
                "key": f"faith:{control['id']}",
                "kind": "faithfulness",
                "role": "control",
                "id": control["id"],
                "messages": faithfulness_messages(
                    control["question"], control["answer"], control["cited_evidence"]
                ),
            }
        )
        requests.append(
            {
                "key": f"correct:{control['id']}",
                "kind": "correctness",
                "role": "control",
                "id": control["id"],
                "messages": correctness_messages(
                    control["question_of_partner"],
                    control["answer"],
                    control["reference_evidence"],
                ),
            }
        )
    return requests


# ----------------------------------------------------------------------------- parsing


def parse_verdict(text: str, kind: str) -> dict[str, Any]:
    """First JSON object in the reply with an allowed verdict; otherwise a parse failure."""
    allowed = FAITHFUL_VERDICTS if kind == "faithfulness" else CORRECT_VERDICTS
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", cleaned)):
        try:
            data, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict in allowed:
            return {"ok": True, "verdict": verdict, "reasoning": str(data.get("reasoning", ""))[:600],
                    "unsupported_claims": data.get("unsupported_claims", [])}  # fmt: skip
    return {"ok": False, "verdict": "", "reasoning": "", "unsupported_claims": []}


def run_judge(
    requests: list[dict[str, Any]],
    generate: Callable[[list[list[dict[str, str]]]], list[str]],
    retry_note: str,
) -> list[dict[str, Any]]:
    """Judge every request once, then re-ask unparsable ones once with a stricter reminder."""
    replies = generate([r["messages"] for r in requests])
    rows = []
    for request, reply in zip(requests, replies, strict=True):
        rows.append(
            {**request, "raw": reply, "retried": False, **parse_verdict(reply, request["kind"])}
        )
    failed = [n for n, row in enumerate(rows) if not row["ok"]]
    if failed:
        again = generate(
            [[*rows[n]["messages"], {"role": "user", "content": retry_note}] for n in failed]
        )
        for n, reply in zip(failed, again, strict=True):
            parsed = parse_verdict(reply, rows[n]["kind"])
            rows[n].update({"raw_retry": reply, "retried": True, **parsed})
    return rows


# ----------------------------------------------------------------------------- reporting


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(
    judgments: list[dict[str, Any]], items: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    gates = protocol["gates"]
    by_item = {i["id"]: i for i in items}
    out: dict[str, Any] = {"controls": {}, "answers": {}, "gates": {}}
    parse_total = len(judgments)
    parse_bad = sum(not j["ok"] for j in judgments)
    out["parse_failures"] = {"failed": parse_bad, "of": parse_total,
                             "rate": _rate(parse_bad, parse_total)}  # fmt: skip

    rejected = {"faithfulness": ("unfaithful",), "correctness": ("incorrect",)}
    for kind, good in rejected.items():
        controls = [j for j in judgments if j["role"] == "control" and j["kind"] == kind]
        hit = sum(j["ok"] and j["verdict"] in good for j in controls)
        # An unparsable control counts as not rejected: the judge did not give a usable verdict.
        out["controls"][kind] = {
            "rejected": hit,
            "n": len(controls),
            "rate": _rate(hit, len(controls)),
        }

    answers = [j for j in judgments if j["role"] == "answer"]
    for kind, top in (("faithfulness", "faithful"), ("correctness", "correct")):
        rows = [j for j in answers if j["kind"] == kind]
        counts = Counter(j["verdict"] if j["ok"] else "unparsed" for j in rows)
        per_type: dict[str, Any] = {}
        for qtype in sorted({by_item[j["id"]]["type"] for j in rows}):
            sub = [j for j in rows if by_item[j["id"]]["type"] == qtype]
            top_n = sum(j["ok"] and j["verdict"] == top for j in sub)
            per_type[qtype] = {"n": len(sub), top: top_n, "rate": _rate(top_n, len(sub))}
        agree = {"mechanical_pass": [0, 0], "mechanical_miss": [0, 0]}
        for j in rows:
            slot = "mechanical_pass" if by_item[j["id"]]["mechanical_passed"] else "mechanical_miss"
            agree[slot][1] += 1
            agree[slot][0] += bool(j["ok"] and j["verdict"] == top)
        out["answers"][kind] = {
            "n": len(rows),
            "verdicts": dict(counts),
            "top_rate": _rate(counts.get(top, 0), len(rows)),
            "by_type": per_type,
            f"{top}_given_mechanical": {k: {top: v[0], "n": v[1]} for k, v in agree.items()},
        }
    out["skipped"] = dict(Counter(i["skipped"] for i in items if i["skipped"]))
    unresolved = [
        {"id": i["id"], "citation": c["citation"]}
        for i in items
        if not i["skipped"]
        for c in i["citations"]
        if not c["resolved"]
    ]
    out["unresolved_citations"] = unresolved

    out["gates"] = {
        "controls_rejected": {
            kind: out["controls"][kind]["rate"] >= gates["control_rejection_at_least"]
            and out["controls"][kind]["n"] >= gates["min_controls_per_kind"]
            for kind in rejected
        },
        "parse_failure_rate_ok": out["parse_failures"]["rate"]
        <= gates["parse_failure_rate_at_most"],
    }
    out["judge_trusted"] = (
        all(out["gates"]["controls_rejected"].values()) and out["gates"]["parse_failure_rate_ok"]
    )
    return out


def render_markdown(summary: dict[str, Any], run: dict[str, Any]) -> str:
    lines = [
        "# Answer-quality judge check",
        "",
        f"Judge: `{run.get('model')}` (quantized: {run.get('quantization')}). "
        f"Answers: `{run.get('answers_file')}`.",
        "",
        f"**Judge trusted by the frozen gates: {summary['judge_trusted']}** "
        "(if False, the answer numbers below are not valid evidence).",
        "",
        "## Negative controls (a good judge rejects all of them)",
        "",
    ]
    for kind, c in summary["controls"].items():
        lines.append(f"- {kind}: rejected {c['rejected']}/{c['n']} ({c['rate']:.0%}); gate "
                     f"{summary['gates']['controls_rejected'][kind]}")  # fmt: skip
    p = summary["parse_failures"]
    lines += [
        f"- unparsable replies after one retry: {p['failed']}/{p['of']}",
        "",
        "## Answers",
        "",
    ]
    for kind, a in summary["answers"].items():
        lines.append(
            f"- {kind}: verdicts {a['verdicts']}; top rate {a['top_rate']:.0%} of {a['n']}"
        )
        for qtype, t in a["by_type"].items():
            top = next(k for k in t if k not in ("n", "rate"))
            lines.append(f"  - {qtype}: {t[top]}/{t['n']}")
    lines += ["", f"Skipped (not judged): {summary['skipped'] or 'none'}",
              f"Citations that did not resolve in the pinned checkout: {len(summary['unresolved_citations'])}"]  # fmt: skip
    lines += [
        "",
        "Raw replies are in `judgments.jsonl`; read the disagreements before quoting a rate.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- commands


def prepare(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    dataset = Path(args.dataset)
    if answers["dataset_sha256"] != sha256_of(dataset):
        raise SystemExit("The answers file was produced from a different question file")
    records = {}
    for line in dataset.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        records[row["id"]] = row
    checkout = Path(args.checkout)
    sha = protocol["repo_commit_sha"]
    cache: dict[str, list[str] | None] = {}

    def reader(path: str) -> list[str] | None:
        if path not in cache:
            cache[path] = git_lines(checkout, sha, path)
        return cache[path]

    items = build_items(records, answers, reader, protocol)
    controls = build_controls(items, protocol["seed"])
    requests = make_requests(items, controls)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(
        json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (out / "requests.json").write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
    (out / "prepare.json").write_text(
        json.dumps(
            {
                "answers_file": Path(args.answers).name,
                "answers_sha256": sha256_of(Path(args.answers)),
                "answer_model": answers.get("llm_model"),
                "prompt_variant": answers.get("prompt_variant"),
                "dataset_sha256": sha256_of(dataset),
                "protocol_sha256": sha256_of(PROTOCOL),
                "repo_commit_sha": sha,
                "items": len(items),
                "judged": sum(not i["skipped"] for i in items),
                "controls": len(controls),
                "requests": len(requests),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{len(items)} items, {len(controls)} controls, {len(requests)} judge prompts -> {out}")


def _vllm_generate(
    model: str, protocol: dict[str, Any]
) -> Callable[[list[list[dict[str, str]]]], list[str]]:
    # FlashInfer's sampler compiles a CUDA kernel on first use and needs `ninja` and `nvcc`; the
    # first Colab run died there ("No such file: ninja"). Greedy decoding gains nothing from it, so
    # switch it off, and put this interpreter's bin directory on PATH for any other tool it needs.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    bin_dir = str(Path(sys.executable).parent)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    cfg = protocol["runtime"]
    llm = LLM(
        model=model,
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        seed=protocol["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        # The model has a vision tower we never use; do not reserve memory for images or video.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])

    def generate(batch: list[list[dict[str, str]]]) -> list[str]:
        outputs = llm.chat(
            batch, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=True
        )
        return [o.outputs[0].text for o in outputs]

    return generate


def judge(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    out = Path(args.out)
    requests = json.loads((out / "requests.json").read_text(encoding="utf-8"))
    config_path = Path(args.model) / "config.json"
    quantization = None
    if config_path.is_file():
        quantization = json.loads(config_path.read_text(encoding="utf-8")).get(
            "quantization_config"
        )
    if protocol["require_quantized"] and not quantization:
        raise SystemExit(
            f"{args.model} has no quantization_config: this protocol allows quantized weights only"
        )
    started = time.perf_counter()
    generate = _vllm_generate(args.model, protocol)
    rows = run_judge(requests, generate, protocol["retry_note"])
    elapsed = round(time.perf_counter() - started, 1)
    with (out / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: v for k, v in row.items() if k != "messages"},
                                    ensure_ascii=False) + "\n")  # fmt: skip
    import vllm  # type: ignore[import-not-found]

    (out / "judge_run.json").write_text(
        json.dumps(
            {
                "model": args.model_id or args.model,
                "model_path": args.model,
                "quantization": quantization.get("quant_method") if quantization else None,
                "quantization_config": quantization,
                "vllm": getattr(vllm, "__version__", "?"),
                "seconds": elapsed,
                "requests": len(requests),
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"judged {len(rows)} prompts in {elapsed}s")


def report(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    out = Path(args.out)
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    judgments = [json.loads(x) for x in (out / "judgments.jsonl").read_text("utf-8").splitlines()]
    prepared = json.loads((out / "prepare.json").read_text(encoding="utf-8"))
    run = {**prepared, **json.loads((out / "judge_run.json").read_text(encoding="utf-8"))}
    summary = summarize(judgments, items, protocol)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    markdown = render_markdown(summary, run)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--answers", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkout", required=True)
    p.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--out", required=True)
    j.add_argument("--model", required=True, help="local path of the downloaded quantized model")
    j.add_argument("--model-id", help="Hugging Face id, recorded in the results")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "judge": judge, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
