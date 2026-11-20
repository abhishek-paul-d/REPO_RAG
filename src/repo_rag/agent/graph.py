"""LangGraph agent: exact-reference pre-step, the evaluated tool loop, citation check, memory.

The answer model sees exactly what it saw in the bake-off: the system prompt of the configured
arm, the standalone question, tool results and (arms 1-2) the fixed reminder. Nothing derived by
the graph (references found, routes) is added to the model input. The one addition is the last
answered turns of the same thread, and only on a follow-up (`with_history`): a first question
reaches the model byte-for-byte as in the bake-off, so the measured single-turn behaviour is the
behaviour that runs. Follow-ups themselves are not measured. The graph then validates every
`path:start-end` against the retrieved chunks (`agent.citations`) before the answer is returned.

    pre-step -> answer (tool loop) -> citation check -> remember

The model is injected (`agent.bakeoff.Model`); building a real local model is a deployment
decision outside this module.
"""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from repo_rag.agent.bakeoff import BudgetFailure, Generation, Model, run_question
from repo_rag.agent.bakeoff_tools import tool_schemas
from repo_rag.agent.citation_snap import Lookup, snap_citations
from repo_rag.agent.citations import check_citations, chunks_from_results, repair_citations
from repo_rag.agent.commit_answer import enrich
from repo_rag.agent.prompts import (
    BAKEOFF_ARMS,
    COMMIT_SCOPE_VARIANTS,
    COMMIT_TOOL_VARIANTS,
    HISTORY_BLOCK,
    HISTORY_TOOL_VARIANTS,
    HISTORY_TURN,
    production_docs_reminder,
    production_prompt,
    production_reminder,
)
from repo_rag.agent.tools import production_schemas
from repo_rag.config import Settings
from repo_rag.retrieval.exact_refs import find_exact_refs

NO_ANSWER = "No answer could be produced: the model's reply was not valid within the limits."
TOOL_FAILED = (
    "No answer could be produced: a tool failed ({detail}). Commit, PR and issue data may not "
    "be indexed for this repository."
)
MODEL_FAILED = "No answer could be produced: the answer model failed ({detail})."
TOOL_ERROR_CHARS = 200
RAW_REPLY_CHARS = 800
_SHA = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")


@dataclass(frozen=True)
class Toolset:
    """What the model is given: tool schemas, system prompt, and when the fixed reminder is sent."""

    schemas: Mapping[str, dict]
    prompt: str
    reminder: str | None
    reminder_tools: frozenset[str] = frozenset({"search_code"})
    chunk_tools: frozenset[str] = frozenset({"search_code"})  # results made of citable chunks
    tolerant: bool = False  # accept a fenced reply and backticks inside the answer (production)
    parse_retries: int = 0  # answer a badly formatted reply with feedback this many times
    docs_reminder: str | None = None  # if set, sent after search_docs instead of `reminder`
    tool_error_retries: int = 0  # give a failing tool's error back to the model this many times
    abstain_retries: int = 0  # ask for another search when it abstains after fewer than 2 calls
    budget_note: bool = False  # say so when a result is empty because the budget is used up


def toolset_for(settings: Settings) -> Toolset:
    """Evaluated: the bake-off setup exactly. Production: adds four unmeasured tools."""
    if settings.agent_toolset == "production":
        both = frozenset({"search_code", "search_docs"})
        variant = settings.agent_prompt_variant
        return Toolset(
            production_schemas(
                search_prs=variant != "no_search_prs",
                file_history=variant in HISTORY_TOOL_VARIANTS,
                get_commit=variant in COMMIT_TOOL_VARIANTS,
                commit_scope=variant in COMMIT_SCOPE_VARIANTS,
            ),
            production_prompt(variant),
            production_reminder(variant),
            both,
            both,
            tolerant=True,
            parse_retries=settings.agent_parse_retries,
            docs_reminder=production_docs_reminder(variant),
            tool_error_retries=settings.agent_tool_error_retries,
            abstain_retries=settings.agent_abstain_retries,
            budget_note=settings.agent_budget_note,
        )
    prompt, reminder = BAKEOFF_ARMS[settings.llm_prompt_arm]
    return Toolset(tool_schemas(), prompt, reminder)


class _RecordingModel:
    """Passes everything to the model and remembers why `generate` failed (not budget caps)."""

    def __init__(self, model: Model) -> None:
        self.model, self.error = model, ""

    def start_question(self) -> None:
        self.model.start_question()

    def finish_question(self) -> dict[str, Any]:
        return self.model.finish_question()

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        try:
            return self.model.generate(messages)
        except BudgetFailure:
            raise
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc)[:TOOL_ERROR_CHARS]}"
            raise


class AgentState(TypedDict, total=False):
    question: str
    refs: list[dict[str, Any]]
    trace: dict[str, Any]
    result: dict[str, Any]
    history: Annotated[list[dict[str, Any]], operator.add]


def with_history(question: str, history: list[dict[str, Any]], settings: Settings) -> str:
    """The question as the model sees it: unchanged when the thread has no answered turn, else
    preceded by the last `agent_history_turns` answered ones. Abstentions are skipped (they carry
    no information), and each answer is cut to `agent_history_answer_chars`."""
    limit = settings.agent_history_turns
    answered = [t for t in history if not t.get("abstain")]
    if limit == 0 or not answered:
        return question
    cap = settings.agent_history_answer_chars
    turns = "\n\n".join(
        HISTORY_TURN.format(question=t["question"], answer=_clip(str(t["answer"]), cap))
        for t in answered[-limit:]
    )
    return HISTORY_BLOCK.format(turns=turns, question=question)


def last_reply(trace: dict[str, Any]) -> str:
    """What the model last said (already redacted), cut to `RAW_REPLY_CHARS`. Only failed runs
    return it, so a `parse` failure can be read instead of guessed at."""
    for turn in reversed(trace.get("turns", [])):
        text = (turn.get("generation") or {}).get("text")
        if text is not None:
            return _clip(text, RAW_REPLY_CHARS)
    return ""


def _clip(text: str, chars: int) -> str:
    return text if len(text) <= chars else text[: chars - 1].rstrip() + "…"


def generation_limits(settings: Settings) -> dict[str, int]:
    return {
        "tool_result_chars_per_question": settings.agent_tool_result_chars,
        "max_generation_turns": settings.agent_max_turns,
        "max_tool_calls": settings.agent_max_tool_calls,
        # Once the model has searched commits it may take more steps (one more turn than calls).
        "commit_max_tool_calls": settings.agent_commit_max_tool_calls,
        "commit_max_generation_turns": settings.agent_commit_max_tool_calls + 1,
    }


class RepoAgent:
    def __init__(
        self,
        model: Model,
        execute: Callable[[str, dict], Any],
        settings: Settings | None = None,
        checkpointer: Any = None,
        toolset: Toolset | None = None,
        snap_lookup: Lookup | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.snap_lookup = snap_lookup
        self.model, self.execute = model, execute
        self.toolset = toolset or toolset_for(self.settings)
        graph = StateGraph(AgentState)
        graph.add_node("prestep", self._prestep)
        graph.add_node("answer", self._answer)
        graph.add_node("check", self._check)
        graph.add_node("remember", self._remember)
        graph.add_edge(START, "prestep")
        graph.add_edge("prestep", "answer")
        graph.add_edge("answer", "check")
        graph.add_edge("check", "remember")
        graph.add_edge("remember", END)
        self.graph = graph.compile(checkpointer=checkpointer or MemorySaver())

    def _prestep(self, state: AgentState) -> dict[str, Any]:
        """Exact references (#N, SHAs, tags, paths) found in the question, for the caller."""
        return {"refs": [asdict(ref) for ref in find_exact_refs(state["question"])]}

    def _answer(self, state: AgentState) -> dict[str, Any]:
        errors: list[str] = []
        commit_searches = 0

        def execute(name: str, arguments: dict) -> Any:
            nonlocal commit_searches
            if name == "search_commits":
                commit_searches += 1
                if self.settings.agent_commit_query == "question":
                    arguments = {**arguments, "query": state["question"]}
                elif self.settings.agent_commit_query == "fused" and commit_searches == 1:
                    arguments = {**arguments, "_question": state["question"]}
            if (
                name == "search_code"
                and self.settings.agent_toolset == "production"
                and (self.settings.agent_code_extra_files or self.settings.agent_code_skip_tests)
            ):
                arguments = {**arguments, "_question": state["question"]}
            try:
                return self.execute(name, arguments)
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:TOOL_ERROR_CHARS]}")
                raise

        model = _RecordingModel(self.model)
        trace = run_question(
            with_history(state["question"], state.get("history", []), self.settings),
            model,
            self.toolset.schemas,
            execute,
            generation_limits(self.settings),
            prompt=self.toolset.prompt,
            reminder=self.toolset.reminder,
            reminder_tools=self.toolset.reminder_tools,
            tolerant=self.toolset.tolerant,
            parse_retries=self.toolset.parse_retries,
            docs_reminder=self.toolset.docs_reminder,
            tool_error_retries=self.toolset.tool_error_retries,
            abstain_retries=self.toolset.abstain_retries,
            budget_note=self.toolset.budget_note,
        )
        if errors:
            trace["tool_error"] = errors[-1]
        if model.error:
            trace["model_error"] = model.error
        return {"trace": trace}

    def _check(self, state: AgentState) -> dict[str, Any]:
        trace = state["trace"]
        calls = [{"name": c["name"], "arguments": c["arguments"]} for c in trace["calls"]]
        final = trace["final"]
        if trace["failure"] or final is None:
            if trace["failure"] == "tool" and trace.get("tool_error"):
                detail = trace["tool_error"]
                text = TOOL_FAILED.format(detail=detail)
            elif trace["failure"] == "model" and trace.get("model_error"):
                detail = trace["model_error"]
                text = MODEL_FAILED.format(detail=detail)
            else:
                detail, text = "", NO_ANSWER
            return {
                "result": {
                    "answer": text,
                    "abstain": True,
                    "failure": trace["failure"] or "no_final",
                    "error": detail,
                    "raw_reply": last_reply(trace),
                    "parse_retries": trace.get("parse_retries", 0),
                    "guards": {
                        "abstain_retries": trace.get("abstain_retries", 0),
                        "tool_error_recoveries": trace.get("tool_error_recoveries", 0),
                    },
                    "sources": [],
                    "unverified_shas": [],
                    "citations": {
                        "valid": 0,
                        "removed": [],
                        "attached": [],
                        "uncited": False,
                        "checked": False,
                    },
                    "tool_calls": calls,
                    "refs": state["refs"],
                }
            }
        answer, abstain = final["answer"], bool(final["abstain"])
        if isinstance(answer, list):  # the parser allows a list of strings; the checks need text
            answer = "\n".join(f"- {item}" for item in answer)
        rows = [
            row
            for call in trace["calls"]
            if call["name"] in self.toolset.chunk_tools
            for row in ((call.get("result") or {}).get("value") or [])
        ]
        chunks = chunks_from_results(rows)
        sources = [f"{c.path}:{c.start_line}-{c.end_line} {c.symbol}".strip() for c in chunks]
        searches = [
            [
                row
                for row in ((call.get("result") or {}).get("value") or [])
                if isinstance(row, dict) and "sha" in row
            ]
            for call in trace["calls"]
            if call["name"] == "search_commits"
        ]
        commits = [row for rows in searches for row in rows]
        # Full rows carry `text`; shortlist rows (agent_commit_k > 5) carry only `subject`.
        for row in commits:
            lines = (row.get("text") or row.get("subject") or "").splitlines()
            sources.append(f"commit {row['sha']} {lines[0][:80] if lines else ''}")
        evidence = json.dumps([c.get("result") for c in trace["calls"]]).lower()
        unverified = (
            sorted({m for m in _SHA.findall(answer.lower()) if m not in evidence})
            if isinstance(answer, str)
            else []
        )
        # `checked` is False when no code or docs chunk was retrieved (a callers, commit, history
        # or count answer) or the model abstained: there was nothing to check, which is not the
        # same as "0 citations match".
        report = {"valid": 0, "removed": [], "attached": [], "uncited": False, "checked": False}
        if isinstance(answer, str) and chunks and not abstain:
            repaired = repair_citations(answer, chunks)
            answer = repaired.answer
            snapped: list[dict[str, str]] = []
            if self.settings.agent_citation_snap and self.snap_lookup is not None:
                try:
                    result = snap_citations(answer, chunks, self.snap_lookup)
                except Exception:  # a symbol-table problem must never cost the user an answer
                    result = None
                if result is not None:
                    answer = result.answer
                    snapped = [{"old": x.old, "new": x.new, "why": x.why} for x in result.snaps]
            report = {
                "checked": True,
                "valid": sum(r.valid for r in check_citations(answer, chunks)),
                "removed": repaired.removed,
                "attached": repaired.attached,
                "uncited": repaired.uncited,
            }
            if self.settings.agent_citation_snap:
                report["snapped"] = snapped
        model_answer = answer
        if isinstance(answer, str) and self.settings.agent_answer_facts:
            answer = enrich(
                answer, abstain, searches, [c["name"] for c in trace["calls"]], trace["calls"]
            )
        return {
            "result": {
                "answer": answer,
                "model_answer": model_answer,
                "abstain": abstain,
                "failure": "",
                "error": "",
                "sources": sources,
                "unverified_shas": unverified,
                "parse_retries": trace.get("parse_retries", 0),
                "guards": {
                    "abstain_retries": trace.get("abstain_retries", 0),
                    "tool_error_recoveries": trace.get("tool_error_recoveries", 0),
                },
                "citations": report,
                "tool_calls": calls,
                "refs": state["refs"],
            }
        }

    def _remember(self, state: AgentState) -> dict[str, Any]:
        result = state["result"]
        return {
            "history": [
                {
                    "question": state["question"],
                    "answer": result["answer"],
                    "abstain": result["abstain"],
                }
            ]
        }

    def ask(self, question: str, thread_id: str = "default") -> dict[str, Any]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke({"question": question}, config)["result"]

    def stream(self, question: str, thread_id: str = "default") -> Iterator[tuple[str, Any]]:
        """`(node, update)` as each graph node finishes; the `check` update carries the result.
        The model is called once per generation turn, not token by token, so this reports steps,
        not partial text."""
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for update in self.graph.stream({"question": question}, config, stream_mode="updates"):
            yield from update.items()

    def history(self, thread_id: str = "default") -> list[dict[str, Any]]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        return list(self.graph.get_state(config).values.get("history", []))
