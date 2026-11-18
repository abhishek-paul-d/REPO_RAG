"""Phase 7 gate: the Phase 5b fact set (encode/httpx, 60 facts) answered by the agent itself.

    uv run python eval/agent_facts.py --url https://....trycloudflare.com --token TOKEN

`eval/run_facts.py` makes the right tool call for each fact and scored 60/60 on the saved httpx
metadata. Here the model has to choose the call itself. The answer must contain the expected
value (names, numbers, tags, SHAs), checked mechanically, no LLM judge.

Deviations from a full agent, stated so the score is read correctly:
- httpx has no vector index on this machine, so the search tools (`search_code`, `search_docs`,
  `search_commits`, `search_prs`, `find_callers`, ...) that need it raise "not available"; a
  question the model sends to one of them fails as a tool failure. The facts only need the exact
  tools (`get_pr`, `get_issue_meta`, blame, releases, counts, tags), so this tests routing among
  those, not retrieval.
- The metadata is `data/answer_bakeoff_capsule/metadata.db` (500 httpx PRs).
- The check is lenient on extra text: an answer that names the expected value among others passes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FACTS = ROOT / "eval/datasets/httpx_facts.jsonl"
CHECKOUT = ROOT / "data/repos/encode__httpx"
DATA_DIR = ROOT / "data/answer_bakeoff_capsule"
RESULTS_DIR = ROOT / "eval/agent_eval_results"
TARGET = 0.90
_HEX = re.compile(r"[0-9a-f]{7,40}")


class NoIndex:
    """Stands in for the retriever: httpx is not indexed here."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("search is not available in the facts run: httpx has no vector index")


def contains(answer: str, value: str) -> bool:
    if _HEX.fullmatch(value):  # a SHA may be written in full or shortened
        return re.search(rf"(?<![0-9a-zA-Z]){value}", answer.lower()) is not None
    return re.search(rf"(?<![\w.-]){re.escape(value)}(?![\w-])", answer) is not None


def passed(fact: dict, answer: str) -> bool:
    expected = fact["expected"]
    values = expected if isinstance(expected, list) else [expected]
    return bool(values) and all(contains(answer, str(v)) for v in values)


def load_facts(limit: int | None = None) -> list[dict]:
    facts = [json.loads(line) for line in FACTS.read_text(encoding="utf-8").splitlines() if line]
    return facts[:limit] if limit else facts


def make_agent(model: Any, fact: dict, data_dir: Path = DATA_DIR, checkout: Path = CHECKOUT) -> Any:
    from repo_rag.agent.graph import RepoAgent
    from repo_rag.agent.runtime import evaluated_settings
    from repo_rag.agent.tools import ProductionTools
    from repo_rag.config import Settings
    from repo_rag.history.metadata_db import MetadataDB
    from repo_rag.store.symbol_table import SymbolTable

    settings = evaluated_settings(
        Settings(data_dir=data_dir).model_copy(update={"agent_toolset": "production"})
    )
    tools = ProductionTools(
        fact["repo"],
        checkout,
        fact["head"],
        settings,
        NoIndex(),  # type: ignore[arg-type]
        MetadataDB(settings.metadata_db_path),
        SymbolTable(settings.metadata_db_path),
    )
    return RepoAgent(model, tools.execute, settings)


def run(url: str, token: str, limit: int | None) -> None:
    from repo_rag.agent.remote_llm import RemoteChatModel

    facts = load_facts(limit)
    model = RemoteChatModel(url, token)
    info = model.health()
    agent = make_agent(model, facts[0])
    rows = []
    for index, fact in enumerate(facts, 1):
        started = time.perf_counter()
        result = agent.ask(fact["question"], thread_id=fact["id"])
        ok = not result["abstain"] and passed(fact, str(result["answer"]))
        rows.append(
            {
                "id": fact["id"],
                "type": fact["type"],
                "question": fact["question"],
                "expected": fact["expected"],
                "answer": result["answer"],
                "passed": ok,
                "abstain": result["abstain"],
                "failure": result["failure"],
                "error": result["error"],
                "raw_reply": result.get("raw_reply", ""),
                "tools": [c["name"] for c in result["tool_calls"]],
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print(
            f"[{index}/{len(facts)}] {fact['id']:<28} {'pass' if ok else 'MISS'} {rows[-1]['tools']}"
        )
    by_type: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row["passed"])
    total = sum(r["passed"] for r in rows)
    rate = total / len(rows)
    print(f"\nFacts through the agent: {total}/{len(rows)} ({rate:.1%}); target {TARGET:.0%}.")
    for kind, values in sorted(by_type.items()):
        print(f"  {kind:<18}{sum(values)}/{len(values)}")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"agent_facts_{stamp}.json"
    out.write_text(
        json.dumps(
            {"server": info, "target": TARGET, "passed": total, "n": len(rows), "rows": rows},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--limit", type=int, help="only the first N facts (a smoke test)")
    args = parser.parse_args()
    run(args.url, args.token, args.limit)


if __name__ == "__main__":
    main()
