"""Mechanical replay/scoring of raw bakeoff turns; never trust cached final/score fields."""

from __future__ import annotations

import math
import re
from typing import Any

from eval.run_facts import matches
from repo_rag.agent.bakeoff import ParseFailure, canonical, parse_reply
from repo_rag.agent.bakeoff_tools import tool_schemas
from repo_rag.agent.prompts import ANSWER_BAKEOFF_PROMPT
from repo_rag.summaries.prose import CHECK_VERSION, violations

PATH_REF = re.compile(r"(?P<path>[\w./-]+):(?P<start>\d+)-(?P<end>\d+)")
SHA_REF = re.compile(r"(?<![\w/])([0-9a-fA-F]{7,40})(?![\w/])")


def replay(
    trace: dict, protocol: dict, prompt: str = ANSWER_BAKEOFF_PROMPT, reminder: str | None = None
) -> tuple[dict | None, str, list[dict]]:
    schemas = tool_schemas()
    messages = [
        {
            "role": "system",
            "content": prompt.replace("{schemas}", canonical(list(schemas.values()))),
        },
        {"role": "user", "content": trace["question"]},
    ]
    executed: list[dict] = []
    failure = "budget"
    final = None
    remaining = protocol["generation"]["tool_result_chars_per_question"]
    turns = trace["turns"]
    if len(turns) > protocol["generation"]["max_generation_turns"]:
        raise ValueError("Too many generation turns")
    for _index, turn in enumerate(turns):
        if turn["messages"] != messages:
            raise ValueError("Prompt/transcript replay mismatch")
        generation = turn.get("generation")
        if generation is None:
            failure = "budget" if turn.get("error") == "budget" else "model"
            break
        try:
            reply = parse_reply(generation["text"], schemas)
        except ParseFailure:
            failure = "budget" if generation["token_cap_hit"] else "parse"
            break
        messages.append({"role": "assistant", "content": generation["text"]})
        if reply["type"] == "final":
            final, failure = reply, ""
            break
        if len(executed) >= protocol["generation"]["max_tool_calls"]:
            failure = "budget"
            break
        if len(trace["calls"]) <= len(executed):
            raise ValueError("Missing executed tool trace")
        call = trace["calls"][len(executed)]
        if {k: call[k] for k in ("name", "arguments")} != {
            "name": reply["name"],
            "arguments": reply["arguments"],
        }:
            raise ValueError("Tool trace differs from generated call")
        executed.append(call)
        if "error" in call:
            failure = "tool"
            break
        if "result" not in call:
            failure = "budget"
            break
        remaining -= len(canonical(call["result"]))
        if remaining < 0:
            raise ValueError("Tool transcript exceeds character budget")
        messages.append({"role": "tool", "content": canonical(call["result"])})
        if reminder is not None and reply["name"] == "search_code":
            messages.append({"role": "user", "content": reminder})
    else:
        _index = len(turns) - 1
    if len(executed) != len(trace["calls"]):
        raise ValueError("Unexplained tool calls")
    if turns and _index != len(turns) - 1:
        raise ValueError("Transcript continues after terminal event")
    return final, failure, executed


def _value(call: dict) -> Any:
    return call.get("result", {}).get("value")


PATH_ARGS = ("path", "path_prefix")
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def corrected_match(fact: dict, answer: Any) -> bool:
    """matches() plus the three registered v4 equivalences (protocol scoring.v4_corrections)."""
    if matches(fact, answer):
        return True
    expected = fact["expected"]
    if fact["type"] == "blame_line":
        return (
            isinstance(answer, str)
            and FULL_SHA.fullmatch(answer) is not None
            and answer.startswith(expected)
        )
    if fact["compare"] == "set" and isinstance(answer, str):
        return len(expected) == 1 and answer == expected[0]
    return False


def _same_args(actual: dict, expected: dict, corrected: bool) -> bool:
    if not corrected:
        return actual == expected

    def norm(args: dict) -> dict:
        return {
            k: v[:-1] if k in PATH_ARGS and isinstance(v, str) and v.endswith("/") else v
            for k, v in args.items()
        }

    return norm(actual) == norm(expected)


def fact_trace(fact: dict, calls: list[dict], corrected: bool = False) -> tuple[bool, bool, Any]:
    """Names, arguments and value projection, including the predeclared release chain."""
    got: Any
    tool, args = fact["tool"], dict(fact["args"])
    if tool == "release_first":
        names = {c["name"] for c in calls}
        for i, first in enumerate(calls):
            if first["name"] != "get_pr" or first["arguments"] != {"number": args["number"]}:
                continue
            value = _value(first)
            sha = value.get("merge_commit_sha") if isinstance(value, dict) else None
            for second in calls[i + 1 :]:
                if (
                    sha
                    and second["name"] == "releases_containing"
                    and second["arguments"] == {"sha": sha}
                ):
                    releases = _value(second)
                    if isinstance(releases, list) and (releases or not second["result"]["omitted"]):
                        return True, True, releases[0]["tag"] if releases else "none"
        return {"get_pr", "releases_containing"} <= names, False, None
    expected_args = {k: v for k, v in args.items() if k != "field" and v is not None}
    if tool == "blame":
        expected_args = {"path": args["path"], "start": args["line"], "end": args["line"]}
    names_ok = any(c["name"] == tool for c in calls)
    args_matched = False
    for call in calls:
        actual = dict(call["arguments"])
        if tool in ("top_contributors", "change_frequency"):
            actual.pop("limit", None)
        if call["name"] != tool or not _same_args(actual, expected_args, corrected):
            continue
        args_matched = True
        value = _value(call)
        if value is None or "error" in call:
            continue
        if tool in ("get_pr", "get_issue_meta"):
            got = str(value[args["field"]])
        elif tool == "prs_closing_issue":
            if call["result"]["omitted"]:
                continue
            got = [str(v["number"]) for v in value]
        elif tool == "blame":
            got = value[0]["sha"][:12] if value else None
        elif tool == "count_commits":
            got = str(value["count"])
        elif tool in ("top_contributors", "change_frequency"):
            rows = value["rows"]
            got = rows[0]["name" if tool == "top_contributors" else "path"] if rows else None
        elif tool == "get_tag":
            got = value["sha"]
        else:
            raise ValueError("Unregistered fact tool")
        return names_ok, True, got
    return names_ok, args_matched, None


def references(
    answer: str, calls: list[dict], repo: str | None = None, pin: str | None = None
) -> tuple[list[dict], list[dict]]:
    code = [row for call in calls if call["name"] == "search_code" for row in (_value(call) or [])]
    code = [
        row
        for row in code
        if (repo is None or row["metadata"].get("repo") == repo)
        and (pin is None or row["metadata"].get("commit_sha") == pin)
    ]
    shas: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("sha", "commit_sha", "merge_commit_sha") and isinstance(item, str):
                    shas.add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for call in calls:
        walk(code if call["name"] == "search_code" else _value(call))
    clean = answer.replace("`", "")
    refs: list[dict] = []
    spans = []
    for ref in PATH_REF.finditer(clean):
        path, start, end = ref["path"], int(ref["start"]), int(ref["end"])
        valid = start <= end and any(
            row["metadata"]["path"] == path
            and start <= int(row["metadata"]["end_line"])
            and end >= int(row["metadata"]["start_line"])
            for row in code
        )
        refs.append({"ref": ref.group(), "valid": valid})
        spans.append(ref.span())
    for ref in SHA_REF.finditer(clean):
        if any(a <= ref.start() < b for a, b in spans):
            continue
        refs.append({"ref": ref.group(), "valid": any(s.startswith(ref.group()) for s in shas)})
    return refs, code


def attach_citations(text: str, code: list[dict]) -> list[str]:
    """v5 answer check: `path:start-end` for each retrieved chunk the answer itself names.

    A chunk counts when its file path occurs in the answer or its symbol's last component
    (at least 4 characters) occurs as a whole word. Nothing is attached for chunks the answer
    does not mention, so the check cannot invent support.
    """
    clean = text.replace("`", "")
    attached: list[str] = []
    for row in code:
        meta = row["metadata"]
        name = str(meta.get("symbol", "")).split(".")[-1]
        named = meta["path"] in clean or (
            len(name) >= 4 and re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", clean) is not None
        )
        ref = f"{meta['path']}:{int(meta['start_line'])}-{int(meta['end_line'])}"
        if named and ref not in attached:
            attached.append(ref)
    return attached


def score_trace(
    trace: dict,
    row: dict,
    protocol: dict,
    prompt: str = ANSWER_BAKEOFF_PROMPT,
    reminder: str | None = None,
) -> dict:
    if trace["question"] != row["question"]:
        raise ValueError("Question identity mismatch")
    final, failure, calls = replay(trace, protocol, prompt, reminder)
    answer = final["answer"] if final else ""
    text = answer if isinstance(answer, str) else " ".join(answer)
    pin = row["label"].get("commit_sha")
    refs, code = references(text, calls, row["repo"], pin)
    phrase = any(
        p in " ".join(text.casefold().split()) for p in protocol["scoring"]["abstention_phrases"]
    )
    abstain = bool(failure or (final and final["abstain"]) or phrase)
    complete = bool(refs) and all(r["valid"] for r in refs) and not failure
    attached = [] if complete or abstain or failure else attach_citations(text, code)
    checked = complete
    if attached:
        checked_refs, _ = references(f"{text} ({' '.join(attached)})", calls, row["repo"], pin)
        checked = bool(checked_refs) and all(r["valid"] for r in checked_refs)
    bare = PATH_REF.sub("", text.replace("`", ""))
    bare = SHA_REF.sub("", bare)
    bad = violations(bare, "\n".join(r["text"] for r in code))
    result = {
        "id": row["id"],
        "suite": row["suite"],
        "repo": row["repo"],
        "failure": failure,
        "citations": len(refs),
        "invalid_citations": sum(not r["valid"] for r in refs),
        "citation_complete": complete,
        "citations_attached": len(attached),
        "citation_complete_checked": checked,
        "abstain": abstain,
        "probe_ok": bool(final and not failure and final["abstain"] and phrase and not refs),
        "lexical_violations": bad,
        "lexical_checker": CHECK_VERSION,
        "tool_calls": len(calls),
        "extra_fact_calls": None,
        "code_evidence_present": bool(code),
        "raw_exact": False,
        "raw_exact_corrected": False,
        "fact_ok_strict": False,
        "arguments_ok_strict": False,
        "fact_ok": False,
        "tool_name_ok": False,
        "arguments_ok": False,
        "seconds": trace["seconds"],
        "memory": trace["memory"],
        "generation_seconds": sum(
            t.get("generation", {}).get("seconds", 0) for t in trace["turns"]
        ),
        "output_tokens": sum(
            t.get("generation", {}).get("output_tokens", 0) for t in trace["turns"]
        ),
    }
    if row["suite"] == "facts":
        fact = row["label"]
        minimum_calls = 2 if fact["tool"] == "release_first" else 1
        result["extra_fact_calls"] = max(0, len(calls) - minimum_calls)
        name_ok, strict_args, got = fact_trace(fact, calls)
        raw = matches(fact, answer)
        _, args_ok, corrected_got = fact_trace(fact, calls, corrected=True)
        raw_corrected = corrected_match(fact, answer)
        clean = not failure and not abstain
        result.update(
            tool_name_ok=name_ok,
            arguments_ok=args_ok,
            arguments_ok_strict=strict_args,
            raw_exact=raw,
            raw_exact_corrected=raw_corrected,
            fact_ok_strict=bool(raw and strict_args and matches(fact, got) and clean),
            fact_ok=bool(raw_corrected and args_ok and matches(fact, corrected_got) and clean),
        )
    else:
        gold = row["label"]
        if row["suite"] == "architecture":
            wanted = set(gold["gold_files"])
            hit = any(r["metadata"]["path"] in wanted for r in code)
        else:
            wanted = gold.get("gold", [])
            hit = any(
                r["metadata"]["path"] == g["path"]
                and int(r["metadata"]["start_line"]) <= int(g["end_line"])
                and int(r["metadata"]["end_line"]) >= int(g["start_line"])
                for r in code
                for g in wanted
            )
        result["retrieval_hit"] = hit
    return result


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]
