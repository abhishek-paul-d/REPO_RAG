"""Question-only, model-agnostic bakeoff controller. No model or label loading here."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

from repo_rag.agent.prompts import (
    ABSTAIN_RETRY,
    ANSWER_BAKEOFF_PROMPT,
    PARSE_RETRY,
    TOOL_ERROR_NEXT,
)

ERROR_DETAIL_CHARS = 300  # how much of a tool error the model is shown on a recovery
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def redact_model_text(text: str) -> str:
    """Redact decoded string tokens without repairing malformed model JSON.

    Scanning escaped JSON directly mistakes `\\n@pytest.mark.skipif` for an email and
    can leave an invalid escape. Preserve every non-email token and all outer syntax.
    """
    pieces = []
    end = 0
    for match in JSON_STRING.finditer(text):
        pieces.append(EMAIL.sub("<email omitted>", text[end : match.start()]))
        token = match.group()
        try:
            decoded = json.loads(token)
            clean = EMAIL.sub("<email omitted>", decoded)
            pieces.append(token if clean == decoded else json.dumps(clean, ensure_ascii=False))
        except (ValueError, TypeError):
            pieces.append(EMAIL.sub("<email omitted>", token))
        end = match.end()
    pieces.append(EMAIL.sub("<email omitted>", text[end:]))
    return "".join(pieces)


def redact(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, str):
        return EMAIL.sub("<email omitted>", value)
    if isinstance(value, dict):
        return {str(k): redact(v) for k, v in value.items() if "email" not in str(k).lower()}
    if isinstance(value, (tuple, list)):
        return [redact(v) for v in value]
    if value is None or type(value) in (int, float, bool):
        return value
    raise TypeError("Tool output must be JSON serializable")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass
class Generation:
    text: str
    output_tokens: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    token_cap_hit: bool = False


class Model(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> Generation: ...

    def start_question(self) -> None: ...

    def finish_question(self) -> dict[str, Any]: ...


class ParseFailure(ValueError):
    pass


class BudgetFailure(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParseFailure("Duplicate JSON key")
        result[key] = value
    return result


def validate_arguments(args: dict, schema: dict) -> None:
    properties = schema["properties"]
    if set(args) - set(properties) or set(schema["required"]) - set(args):
        raise ParseFailure("Unknown or missing argument")
    kinds = {"string": str, "integer": int, "boolean": bool}
    for name, value in args.items():
        spec = properties[name]
        if type(value) is not kinds[spec["type"]]:
            raise ParseFailure("Wrong argument type")
        if "minimum" in spec and value < spec["minimum"]:
            raise ParseFailure("Argument below minimum")
        if "enum" in spec and value not in spec["enum"]:
            raise ParseFailure("Argument outside enumeration")


_OUTER_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*)\n```[ \t]*\Z", re.DOTALL)


def parse_reply(
    text: str, schemas: Mapping[str, dict], *, tolerant: bool = False
) -> dict[str, Any]:
    """Strict whole-output JSON; no fences, repair, duplicate keys, NaN or reasoning.

    `tolerant` (production only; the bake-off stays strict) accepts exactly two harmless things:
    the whole reply wrapped in one ```json fence, and backticks inside the answer string.
    Everything else (duplicate keys, extra keys, reasoning tags, prose around the JSON) still fails.
    It also accepts a flat tool call whose extra keys are all arguments the tool declares
    (`{"type":"tool_call","name":"search_commits","query":"..."}`); the arguments are validated.
    """
    if tolerant:
        fenced = _OUTER_FENCE.match(text.strip())
        if fenced:
            text = fenced.group(1)
    try:
        obj = json.loads(text, object_pairs_hook=_pairs)
        if not isinstance(obj, dict):
            raise ParseFailure("Expected object")
        if obj.get("type") == "final":
            answer = obj.get("answer")
            if set(obj) != {"type", "answer", "abstain"} or type(obj["abstain"]) is not bool:
                raise ParseFailure("Invalid final schema")
            if not isinstance(answer, str) and not (
                isinstance(answer, list) and all(isinstance(x, str) for x in answer)
            ):
                raise ParseFailure("Answer must be string or string list")
        elif obj.get("type") == "tool_call":
            if tolerant and "arguments" not in obj and obj.get("name") in schemas:
                # Flat call: {"type","name","query":...}. Only keys the tool declares are moved.
                declared = set(schemas[obj["name"]]["parameters"]["properties"])
                flat = {k: v for k, v in obj.items() if k not in ("type", "name")}
                if set(flat) <= declared:
                    obj = {"type": "tool_call", "name": obj["name"], "arguments": flat}
            if set(obj) != {"type", "name", "arguments"}:
                raise ParseFailure("Invalid call schema")
            if not isinstance(obj["name"], str) or obj["name"] not in schemas:
                raise ParseFailure("Unknown tool")
            if not isinstance(obj["arguments"], dict):
                raise ParseFailure("Arguments must be object")
            validate_arguments(obj["arguments"], schemas[obj["name"]]["parameters"])
        else:
            raise ParseFailure("Unknown reply type")
        if "<think>" in text or "</think>" in text or ("```" in text and not tolerant):
            raise ParseFailure("Reasoning/fence in output")
        return obj
    except (ValueError, TypeError, KeyError) as exc:
        raise ParseFailure("Invalid model JSON") from exc


def bounded_result(value: Any, budget: int) -> dict[str, Any]:
    """Keep whole records/chunks, including the wrapper in the cumulative char budget."""
    value = redact(value)
    wrapped = {"value": value, "omitted": 0}
    if len(canonical(wrapped)) <= budget:
        return wrapped
    rows = (
        value if isinstance(value, list) else value.get("rows") if isinstance(value, dict) else None
    )
    result: dict[str, Any]
    if isinstance(rows, list):
        kept: list[Any] = []
        for row in rows:
            candidate = [*kept, row]
            new_value = candidate if isinstance(value, list) else {**value, "rows": candidate}
            trial = {"value": new_value, "omitted": len(rows) - len(candidate)}
            if len(canonical(trial)) > budget:
                break
            kept = candidate
        result = {
            "value": kept if isinstance(value, list) else {**value, "rows": kept},
            "omitted": len(rows) - len(kept),
        }
    else:
        result = {"value": None, "omitted": 1}
    if len(canonical(result)) > budget:
        raise BudgetFailure("Tool-result budget exhausted")
    return result


def parse_detail(exc: ParseFailure) -> str:
    """Why a reply was rejected, for the retry message. `parse_reply` re-raises its own checks as
    a generic "Invalid model JSON", so the specific reason is on the cause."""
    cause = exc.__cause__
    if isinstance(cause, ParseFailure):
        return str(cause)
    return "the reply was not valid JSON" if cause is not None else str(exc)


def run_question(
    question: str,
    model: Model,
    schemas: Mapping[str, dict],
    execute: Callable[[str, dict], Any],
    limits: dict,
    prompt: str = ANSWER_BAKEOFF_PROMPT,
    reminder: str | None = None,
    reminder_tools: frozenset[str] = frozenset({"search_code"}),
    tolerant: bool = False,
    parse_retries: int = 0,
    docs_reminder: str | None = None,
    tool_error_retries: int = 0,
    abstain_retries: int = 0,
    abstain_min_calls: int = 2,
) -> dict[str, Any]:
    """The only dataset-derived model input is this string. No row/label argument exists.

    `prompt` and `reminder` select a registered arm (agent.prompts.BAKEOFF_ARMS); the reminder is
    a fixed user message added only after a search_code result, never derived from the question.

    `parse_retries` (production only; the bake-off keeps 0) lets a reply that fails the format
    check be answered once with a fixed message naming what was wrong, instead of ending the run.
    A reply cut off by the output-token cap is never retried. Each retry uses one generation turn.

    `tool_error_retries` (production only): a failing tool call is returned to the model as an
    error message (`TOOL_ERROR_NEXT`) this many times, instead of ending the run.
    `abstain_retries`: an abstention made before `abstain_min_calls` tool calls is answered once
    with `ABSTAIN_RETRY`. Both are off (0) in the bake-off. Each uses a turn and, for the error, a
    tool call from the caps.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    system = prompt.replace("{schemas}", canonical(list(schemas.values())))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": redact(question)},
    ]
    trace: dict[str, Any] = {
        "question": redact(question),
        "turns": [],
        "calls": [],
        "final": None,
        "failure": "",
        "parse_retries": 0,
    }
    remaining = limits["tool_result_chars_per_question"]
    retries_left = parse_retries
    error_retries_left, abstain_left = tool_error_retries, abstain_retries
    trace["tool_error_recoveries"] = 0
    trace["abstain_retries"] = 0
    model.start_question()
    started = time.perf_counter()
    try:
        for _ in range(limits["max_generation_turns"]):
            turn: dict[str, Any] = {"messages": copy.deepcopy(messages)}
            trace["turns"].append(turn)
            try:
                generation = model.generate(copy.deepcopy(messages))
            except BudgetFailure:
                turn["error"] = "budget"
                raise
            except Exception:
                turn["error"] = "model"
                raise
            recorded = asdict(generation)
            recorded["text"] = redact_model_text(generation.text)
            turn["generation"] = recorded
            turn["raw_text_sha256"] = hashlib.sha256(generation.text.encode()).hexdigest()
            try:
                reply = parse_reply(generation.text, schemas, tolerant=tolerant)
            except ParseFailure as exc:
                if generation.token_cap_hit:
                    raise BudgetFailure("Output token cap before valid JSON") from None
                if retries_left <= 0:
                    raise
                retries_left -= 1
                trace["parse_retries"] += 1
                turn["parse_error"] = parse_detail(exc)
                messages.append({"role": "assistant", "content": recorded["text"]})
                messages.append(
                    {"role": "user", "content": PARSE_RETRY.format(error=parse_detail(exc))}
                )
                continue
            reply = redact(reply)
            messages.append({"role": "assistant", "content": recorded["text"]})
            if reply["type"] == "final":
                if (
                    reply.get("abstain")
                    and abstain_left > 0
                    and len(trace["calls"]) < abstain_min_calls
                    and len(trace["calls"]) < limits["max_tool_calls"]
                ):
                    abstain_left -= 1
                    trace["abstain_retries"] += 1
                    messages.append({"role": "user", "content": ABSTAIN_RETRY})
                    continue
                trace["final"] = reply
                break
            if len(trace["calls"]) >= limits["max_tool_calls"]:
                raise BudgetFailure("Tool-call cap")
            call = {"name": reply["name"], "arguments": reply["arguments"]}
            trace["calls"].append(call)
            try:
                result = execute(reply["name"], reply["arguments"])
                call["result"] = bounded_result(result, remaining)
                remaining -= len(canonical(call["result"]))
            except BudgetFailure:
                raise
            except Exception as exc:
                call["error"] = type(exc).__name__
                if error_retries_left <= 0:
                    raise RuntimeError("Tool failed") from exc
                error_retries_left -= 1
                trace["tool_error_recoveries"] += 1
                call["error_detail"] = str(exc)[:ERROR_DETAIL_CHARS]
                messages.append(
                    {
                        "role": "user",
                        "content": TOOL_ERROR_NEXT.format(
                            error=f"{type(exc).__name__}: {call['error_detail']}"
                        ),
                    }
                )
                continue
            messages.append({"role": "tool", "content": canonical(call["result"])})
            if docs_reminder is not None and reply["name"] == "search_docs":
                messages.append({"role": "user", "content": docs_reminder})
            elif reminder is not None and reply["name"] in reminder_tools:
                messages.append({"role": "user", "content": reminder})
        if trace["final"] is None:
            raise BudgetFailure("No final within turn cap")
    except ParseFailure:
        trace["failure"] = "parse"
    except BudgetFailure:
        trace["failure"] = "budget"
    except Exception as exc:
        trace["failure"] = "tool" if trace["calls"] and "error" in trace["calls"][-1] else "model"
        trace["error_class"] = type(exc).__name__
    trace["memory"] = model.finish_question()
    trace["seconds"] = time.perf_counter() - started
    return trace
