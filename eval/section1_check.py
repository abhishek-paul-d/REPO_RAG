"""Run the plan.md section 1 sample questions (adapted to psf/requests) through the agent.

    uv run python eval/section1_check.py run --url <colab url> --token <token> \
        [--variant grounded_complete] [--citation-snap]
    uv run python eval/section1_check.py questions        # print the question list, no model

There is no pass/fail: section 1's criterion is "cited, sensible answers". The run writes
`eval/section1_results/section1_<tag>_<stamp>.json` and a `.md` with each answer, its sources and
automatic checks, so a person reads the flagged ones first:

- code or docs question with no resolvable `path:start-end` citation;
- a cited range that does not exist at the pinned commit;
- a backticked name no cited range shows (`citation_support`);
- an abstention on a question that should have an answer, or an answer to an out-of-scope one;
- a failed run (parse error, budget, tool error).

The checks find citation problems, not wrong reasoning: an unflagged answer is not a verified one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import judge_answers as ja  # noqa: E402

from repo_rag.agent.citation_snap import citation_support  # noqa: E402
from repo_rag.store.symbol_table import SymbolTable  # noqa: E402

REPO = "psf/requests"
SHA = "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60"
RESULTS = ROOT / "eval/section1_results"
CITATION = re.compile(r"([\w./-]+\.\w+):(\d+)-(\d+)")

# (id, category, question, kind): kind is "code" (needs a code citation), "docs", "exact"
# (answered from metadata.db or git, no code citation expected) or "out_of_scope" (should abstain).
QUESTIONS: list[tuple[str, str, str, str]] = [
    (
        "arch-1",
        "architecture",
        "Give me a high-level overview of this repo's architecture. "
        "What are the main modules and how do they talk to each other?",
        "code",
    ),
    (
        "arch-2",
        "architecture",
        "How does a request flow from Session.request to the network connection?",
        "code",
    ),
    (
        "arch-3",
        "architecture",
        "Where is authentication handled, and what happens to the request before it is sent?",
        "code",
    ),
    ("arch-4", "architecture", "What design pattern do the transport adapters use?", "code"),
    (
        "explain-1",
        "explain",
        "Explain what `rebuild_auth` does, step by step, and why it might be written this way.",
        "code",
    ),
    (
        "explain-2",
        "explain",
        "Who calls `get_environ_proxies`, and what happens if it raises?",
        "code",
    ),
    (
        "explain-3",
        "explain",
        "What's the difference between `Session.send` and `Session.request`?",
        "code",
    ),
    (
        "explain-4",
        "explain",
        "`super_len` has no docstring. Write one based on how it's used.",
        "code",
    ),
    (
        "bug-1",
        "bug report",
        "Users report that response.text shows garbled characters for a "
        "page with no charset in its headers. Has something like this been fixed before?",
        "exact",
    ),
    (
        "bug-2",
        "bug report",
        "Timeouts are ignored when proxies are set. Find commits that "
        "touched proxy and timeout handling.",
        "exact",
    ),
    (
        "bug-3",
        "bug report",
        "Which commit introduced the redirect behaviour change in v2.3?",
        "exact",
    ),
    (
        "nav-1",
        "navigational",
        "Where is `DEFAULT_REDIRECT_LIMIT` defined and where is it used?",
        "code",
    ),
    ("nav-2", "navigational", "Show me all the places that read environment variables.", "code"),
    ("impact-1", "impact", "What breaks if I change the signature of `Session.send`?", "code"),
    (
        "impact-2",
        "impact",
        "Which third-party libraries does requests depend on, and which versions are pinned?",
        "docs",
    ),
    (
        "hist-1",
        "change history",
        "Show me the history of `prepare_url`: who changed it and what each change did.",
        "exact",
    ),
    (
        "hist-2",
        "change history",
        "What were the three most recent changes to src/requests/adapters.py?",
        "exact",
    ),
    ("people-1", "people", "Who wrote most of src/requests/auth.py?", "exact"),
    ("people-2", "people", "Who last touched lines 40-80 of src/requests/sessions.py?", "exact"),
    ("people-3", "people", "Who knows the sessions code best?", "exact"),
    ("people-4", "people", "Who opened, who reviewed and who merged PR #5856?", "exact"),
    ("pr-1", "pull requests", "What's PR #7272 about? Give me its description.", "exact"),
    (
        "pr-2",
        "pull requests",
        "Which PR switched the chardet dependency to charset_normalizer?",
        "exact",
    ),
    ("pr-3", "pull requests", "What did reviewers push back on in #5856?", "exact"),
    ("issue-1", "issues", "Are there open bugs about timeouts?", "exact"),
    ("rel-1", "releases", "What changed between v2.30.0 and v2.31.0?", "exact"),
    ("rel-2", "releases", "Which release first contained the fix in PR #5856?", "exact"),
    ("rel-3", "releases", "When did the `json=` parameter ship?", "exact"),
    ("stats-1", "stats", "Who are the top 5 contributors?", "exact"),
    ("stats-2", "stats", "Which files change most often?", "exact"),
    ("onboard-1", "onboarding", "How do I install requests and run its tests?", "docs"),
    ("oos-1", "out of scope", "Show me all the places we write to the cache.", "out_of_scope"),
]


def check(row: dict[str, Any], kind: str, read: Any, lookup: Any) -> list[str]:
    """Automatic flags for one answer (see the module docstring)."""
    flags: list[str] = []
    answer = row["answer"]
    if row["failure"]:
        flags.append(f"run failed: {row['failure']}")
    if kind == "out_of_scope":
        if not row["abstain"]:
            flags.append("answered an out-of-scope question")
        return flags
    if row["abstain"]:
        flags.append("abstained")
        return flags
    cites = CITATION.findall(answer)
    if kind in ("code", "docs") and not cites:
        flags.append("no path:start-end citation")
    for path, start, end in cites:
        lines = read(path)
        if lines is None:
            flags.append(f"cites a file that does not exist at the pinned commit: {path}")
        elif int(start) > int(end) or int(end) > len(lines):
            flags.append(f"cited range outside the file: {path}:{start}-{end}")
    for item in citation_support(answer, read, lookup):
        if item.missing:
            flags.append(f"{item.ref}: no cited range shows {', '.join(item.missing)}")
    if row["unverified_shas"]:
        flags.append(f"SHAs not in any tool output: {row['unverified_shas']}")
    return flags


def render(rows: list[dict[str, Any]], header: str) -> str:
    flagged = [r for r in rows if r["flags"]]
    out = [
        f"# Section 1 sample questions: {header}",
        "",
        f"{len(rows)} questions, {len(flagged)} flagged by the automatic checks. Read the flagged "
        "ones first; an unflagged answer has only passed the citation checks, not been verified.",
        "",
        "| id | category | flags |",
        "|---|---|---|",
    ]
    for r in rows:
        out.append(f"| {r['id']} | {r['category']} | {len(r['flags']) or 'none'} |")
    for r in rows:
        out += [
            "",
            f"## {r['id']} ({r['category']}, {r['kind']})",
            "",
            f"**Q:** {r['question']}",
            "",
            f"**A:** {r['answer'] or '(no answer)'}",
            "",
        ]
        if r["flags"]:
            out += ["**Flags:**", *[f"- {f}" for f in r["flags"]], ""]
        if r["sources"]:
            out += ["**Sources shown to the model:**", *[f"- {s}" for s in r["sources"][:8]], ""]
        out.append(f"Tools: {', '.join(r['tools']) or 'none'}; {r['seconds']} s")
    return "\n".join(out) + "\n"


def run(args: argparse.Namespace) -> None:
    from repo_rag.agent.remote_llm import RemoteChatModel
    from repo_rag.agent.runtime import build_agent
    from repo_rag.config import Settings

    settings = Settings().model_copy(
        update={
            "agent_toolset": "production",
            "agent_prompt_variant": args.variant,
            "agent_citation_snap": args.citation_snap,
            "agent_abstain_retries": args.abstain_retries,
            "agent_tool_error_retries": args.tool_error_retries,
        }
    )
    model = RemoteChatModel(args.url, args.token)
    info = model.health()
    agent = build_agent(REPO, model, settings, reranker_factory=model.reranker_factory(info))
    cache: dict[str, list[str] | None] = {}

    def read(path: str) -> list[str] | None:
        if path not in cache:
            cache[path] = ja.git_lines(Path(args.checkout), SHA, path)
        return cache[path]

    rows = []
    with SymbolTable(settings.metadata_db_path) as table:

        def lookup(name: str, path: str) -> list[tuple[str, int, int]]:
            return [
                (r.qualname, r.start_line, r.end_line)
                for r in table.get_symbol(REPO, name, path=path)
            ]

        for n, (qid, category, question, kind) in enumerate(QUESTIONS, 1):
            started = time.perf_counter()
            result = agent.ask(question, thread_id=qid)
            row = {
                "id": qid,
                "category": category,
                "kind": kind,
                "question": question,
                "answer": result["answer"],
                "abstain": result["abstain"],
                "failure": result["failure"],
                "sources": result["sources"],
                "citations": result["citations"],
                "tools": [c["name"] for c in result["tool_calls"]],
                "unverified_shas": result["unverified_shas"],
                "seconds": round(time.perf_counter() - started, 1),
            }
            row["flags"] = check(row, kind, read, lookup)
            rows.append(row)
            print(f"[{n}/{len(QUESTIONS)}] {qid:<10} {len(row['flags'])} flag(s)  {row['tools']}")
    tag = args.variant + ("_snap" if args.citation_snap else "")
    tag += f"_abst{args.abstain_retries}" if args.abstain_retries else ""
    tag += f"_toolerr{args.tool_error_retries}" if args.tool_error_retries else ""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    RESULTS.mkdir(exist_ok=True)
    base = RESULTS / f"section1_{tag}_{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "citation_snap": args.citation_snap,
                "server": info,
                "llm_model": settings.llm_model,
                "rows": rows,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    base.with_suffix(".md").write_text(render(rows, tag), encoding="utf-8")
    print(f"\n{sum(bool(r['flags']) for r in rows)} of {len(rows)} flagged; wrote {base}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("questions")
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--token", required=True)
    r.add_argument("--variant", default="docs_hint_narrow")
    r.add_argument("--citation-snap", action="store_true")
    r.add_argument("--abstain-retries", type=int, default=0)
    r.add_argument("--tool-error-retries", type=int, default=0)
    r.add_argument("--checkout", default=str(ROOT / "data/repos/psf__requests"))
    args = parser.parse_args()
    if args.command == "questions":
        for qid, category, question, kind in QUESTIONS:
            print(f"{qid:<10} {category:<15} {kind:<13} {question}")
    else:
        run(args)


if __name__ == "__main__":
    main()
