"""Exploratory end-to-end check of the agent on psf/requests, one score per question type.

    uv run python eval/agent_eval.py build
    uv run python eval/agent_eval.py run --url https://....trycloudflare.com --token TOKEN

`build` writes eval/datasets/agent_eval_v1.jsonl from the local index, the synced PR/issue data
and the repo's docs; `run` asks every question through the real agent (retrieval and tools on
this machine, the answer model on the Colab server from notebooks/serve_model_on_colab.ipynb)
and scores each answer mechanically. There are no pre-registered gates and no selection depends
on it: it reports rates and failures per type. The `code` questions and the `commit` pairs were
seen in earlier retrieval evaluations, and nothing here was reviewed by a person.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repo_rag.agent.prompts import PROMPT_VARIANTS  # noqa: E402

REPO = "psf/requests"
SEED = 20261104
DATASET = ROOT / "eval/datasets/agent_eval_v1.jsonl"
RESULTS_DIR = ROOT / "eval/agent_eval_results"
GOLDEN = ROOT / "eval/datasets/requests_qa.jsonl"
ARCH_SOURCE = ROOT / "eval/datasets/architecture_v1/requests.jsonl"
ARCH_DATASET = ROOT / "eval/datasets/agent_eval_architecture_v1.jsonl"
COMMIT_FULL_DATASET = ROOT / "eval/datasets/agent_eval_commit_full_v1.jsonl"
BUGFIX = ROOT / "eval/datasets/phase5_bugfix_v2/repos/psf__requests/pairs.jsonl"
SAMPLE = {
    "pr_fact": 8, "closes": 3, "refs_out": 5, "refs_in": 5, "kind_issue": 3, "kind_pr": 2,
    "prsearch": 5,
}  # fmt: skip
BUG_PAIRS = 10
BUG_CHARS = 1200
# Extra commits accepted for a bug -> fix question, keyed by the gold SHA's first 12 characters.
# Applied when scoring, never written into the dataset file, so the dataset hash recorded in every
# result stays valid. commit-083 (Gittip -> Gratipay, #2279): two commits fix the same issue in the
# same file, and the one that makes the Gittip -> Gratipay change (2014-10-14) is as valid as the
# labelled layout fix. Added on 2026-11-17 after seeing the model's answer, so results that passed
# only through it say so in `detail`.
ALSO_ACCEPTED = {"96efb5ec9638": ["4d498db3ed23"]}

DOCS = [
    ("Which Python versions does requests officially support?",
     ["README.md", "pyproject.toml", ".github/workflows/run-tests.yml"]),
    ("How do I install requests?", ["README.md", "docs/user/install.rst"]),
    ("What are the runtime dependencies of requests?", ["pyproject.toml", "setup.py"]),
    ("How do I run the test suite?",
     ["Makefile", "docs/dev/contributing.rst", "tox.ini", "requirements-dev.txt"]),
    ("How do I set a timeout on a request?",
     ["docs/user/quickstart.rst", "docs/user/advanced.rst"]),
    ("How do I use HTTP basic authentication?",
     ["docs/user/authentication.rst", "docs/user/quickstart.rst"]),
    ("How do I configure proxies?", ["docs/user/advanced.rst"]),
    ("Which Python versions does the CI test against?",
     [".github/workflows/run-tests.yml", "tox.ini"]),
    ("What license is requests released under?", ["LICENSE", "README.md", "pyproject.toml"]),
    ("How do I verify SSL certificates or turn verification off?",
     ["docs/user/advanced.rst", "docs/user/quickstart.rst"]),
    ("What is a Session object and why would I use one?", ["docs/user/advanced.rst"]),
    ("How do I contribute a change to requests?", ["docs/dev/contributing.rst"]),
]  # fmt: skip

PROBES = [
    ("How does requests integrate with Django's ORM?", "django"),
    ("Where does requests implement GraphQL query batching?", "graphql"),
    ("How does requests implement a WebSocket client?", "websocket"),
    ("Which function deploys requests to a Kubernetes cluster?", "kubernetes"),
    ("How does requests map responses onto SQLAlchemy models?", "sqlalchemy"),
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- building


def build_records() -> list[dict[str, Any]]:
    from repo_rag.config import Settings
    from repo_rag.history.metadata_db import MetadataDB

    rng = random.Random(SEED)
    records: list[dict[str, Any]] = []

    def add(qtype: str, question: str, **expect: Any) -> None:
        records.append({"id": f"{qtype}-{len(records) + 1:03d}", "type": qtype,
                        "question": question, "expect": expect})  # fmt: skip

    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        add("code", row["question"], gold=row["gold"])
    for question, paths in DOCS:
        add("docs", question, paths=paths)

    with MetadataDB(Settings().metadata_db_path) as db:
        prs = [p for p in db.find_prs(REPO, state="MERGED", limit=1_000_000) if p.author]
        prs.sort(key=lambda p: p.number)
        for pr in rng.sample(prs, SAMPLE["pr_fact"]):
            if pr.number % 2:
                add("history", f"Who authored PR #{pr.number} in {REPO}?", all=[pr.author])
            else:
                add("history", f"Who merged PR #{pr.number} in {REPO}?", all=[pr.merged_by])
        closing = [p for p in prs if p.closes_issues]
        for pr in rng.sample(closing, SAMPLE["closes"]):
            add("history", f"Which issues does PR #{pr.number} in {REPO} close?",
                numbers=[str(n) for n in pr.closes_issues])  # fmt: skip

        pr_numbers = {p.number for p in prs}
        issue_numbers = sorted(
            r[0] for r in db.connection.execute("SELECT number FROM issues WHERE repo=?", (REPO,))
        )
        every = sorted(pr_numbers | set(issue_numbers))
        outgoing: list[tuple[int, list[int]]] = []
        incoming: list[tuple[int, list[int]]] = []
        for number in rng.sample(every, min(400, len(every))):
            refs = db.references(REPO, number)
            out = [m["number"] for m in refs["mentions"] if m["kind"] != "unknown"]  # type: ignore[index]
            inc = [m["number"] for m in refs["mentioned_by"]]  # type: ignore[index]
            if 1 <= len(out) <= 4 and not refs["truncated"]:
                outgoing.append((number, out))
            if 1 <= len(inc) <= 4 and not refs["truncated"]:
                incoming.append((number, inc))
        for number, found in outgoing[: SAMPLE["refs_out"]]:
            add("refs", f"Which other PRs or issues does #{number} in {REPO} mention in its "
                        "title or body?", numbers=[str(n) for n in found], exclude=str(number))  # fmt: skip
        for number, found in incoming[: SAMPLE["refs_in"]]:
            add("refs", f"Which PRs or issues in {REPO} mention #{number} in their title or "
                        "body?", numbers=[str(n) for n in found], exclude=str(number))  # fmt: skip
        only_issues = [n for n in issue_numbers if n not in pr_numbers]
        for number in rng.sample(only_issues, SAMPLE["kind_issue"]):
            add("kind", f"Is #{number} in {REPO} a pull request or an issue?", kind="issue")
        for pr in rng.sample(prs, SAMPLE["kind_pr"]):
            add(
                "kind",
                f"Is #{pr.number} in {REPO} a pull request or an issue?",
                kind="pull request",
            )

    pairs = [json.loads(line) for line in BUGFIX.read_text(encoding="utf-8").splitlines()]
    tests = sorted((p for p in pairs if p["split"] == "test"), key=lambda p: p["id"])
    for pair in rng.sample(tests, min(BUG_PAIRS, len(tests))):
        gold = pair["gold"]
        shas = ast.literal_eval(gold) if isinstance(gold, str) else gold
        add("commit", "A user reports the bug below. Was something like this fixed before in "
                      "this repository? If so, which commit?\n\n" + pair["query"][:BUG_CHARS],
            shas=[s[:12] for s in shas])  # fmt: skip

    for question, term in PROBES:
        add("probe", question, term=term)

    # prsearch: find a PR or issue from the words of its own title (an easy lookup that checks the
    # search tool end to end; it does not measure paraphrase). Drawn last from a separate
    # generator, so the questions above are unchanged by adding it.
    rng2 = random.Random(SEED + 1)
    with MetadataDB(Settings().metadata_db_path) as db:
        for label, items in (
            ("pull request", db.find_prs(REPO, limit=1_000_000)),
            ("issue", db.find_issues(REPO, limit=1_000_000)),
        ):
            titles: dict[str, list[int]] = defaultdict(list)
            for item in items:
                titles[item.title.strip().lower()].append(item.number)
            usable = sorted(
                (nums[0], title)
                for title, nums in titles.items()
                if len(nums) == 1 and 25 <= len(title) <= 120 and "#" not in title
            )
            for number, title in rng2.sample(usable, SAMPLE["prsearch"]):
                add("prsearch", f"In {REPO}, which {label} is about: \"{title}\"? Give its number.",
                    number=str(number))  # fmt: skip
    return records


# Label corrections from the owner's review of docs/ARCHITECTURE_LABEL_REVIEW.md (2026-11-18), applied
# here because architecture_v1/ is frozen (freeze.json). Each note was checked against psf/requests at
# 611c616. 04: models + sessions alone let "models.py" pass without saying who coordinates, so both are
# needed. 11: adapters.py only passes the finished body on; encoding is models.py (_encode_params,
# _encode_files) and utils.py (to_key_val_list, guess_filename), and both are needed (the reviewer did
# not give a need for the two-file case; 2 is my choice, following their concern for 04).
ARCH_LABEL_REVIEW: dict[str, dict[str, Any]] = {
    "requests-architecture-04": {"need": 2},
    "requests-architecture-11": {
        "gold_files": ["src/requests/models.py", "src/requests/utils.py"],
        "need": 2,
    },
}


def build_architecture_records() -> list[dict[str, Any]]:
    """The 15 psf/requests architecture questions of architecture_v1, as agent questions.

    Kept in its own file so the 103 questions of agent_eval_v1 are untouched. A question passes
    when the answer names at least half of its gold files (rounded up), so it has to span the
    modules, not just find one of them. This is a proxy for "a coherent module-level answer":
    it checks coverage, not whether the explanation is right, so a person must still read them.
    """
    records: list[dict[str, Any]] = []
    for line in ARCH_SOURCE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        review = ARCH_LABEL_REVIEW.get(row["id"], {})
        files = review.get("gold_files", row["gold_files"])
        records.append(
            {
                "id": f"architecture-{len(records) + 1:03d}",
                "type": "architecture",
                "question": row["question"],
                "expect": {
                    "gold_files": files,
                    "need": review.get("need", -(-len(files) // 2)),
                    "source_id": row["id"],
                    "split": row["split"],
                },
            }
        )
    return records


ARCH_FRESH_DATASET = ROOT / "eval/datasets/agent_eval_architecture_fresh_v1.jsonl"
# Fresh architecture questions for psf/requests at 611c616, written 2026-11-10 after the first 15
# scored 5/15, for testing the `architecture_hint` prompt on questions it was not written around.
# Topics differ from architecture_v1 (proxies, netrc, URL preparation, status errors, JSON decoding,
# digest auth, body length, closing, text encoding, user agent). Gold files were checked against the
# source by hand (each named function exists in the file); unreviewed by anyone else.
ARCH_FRESH = [
    ("Which modules decide which proxy a request goes through, and how is it applied to the connection?",
     ["src/requests/sessions.py", "src/requests/utils.py", "src/requests/adapters.py"]),
    ("How do credentials from a netrc file end up on a request when the caller passes no auth?",
     ["src/requests/utils.py", "src/requests/sessions.py", "src/requests/auth.py",
      "src/requests/models.py"]),
    ("Which parts normalise and validate a URL before a request is sent, and what do they raise?",
     ["src/requests/models.py", "src/requests/utils.py", "src/requests/exceptions.py"]),
    ("How does an HTTP error status code become an exception for the caller?",
     ["src/requests/models.py", "src/requests/exceptions.py"]),
    ("How is a response body decoded as JSON, including guessing its encoding, and which errors surface?",
     ["src/requests/models.py", "src/requests/utils.py", "src/requests/exceptions.py"]),
    ("How does digest authentication answer a 401 challenge, and how is it attached to the response?",
     ["src/requests/auth.py", "src/requests/models.py", "src/requests/hooks.py",
      "src/requests/sessions.py"]),
    ("Which parts decide whether a request body is sent with a content length or chunked?",
     ["src/requests/models.py", "src/requests/utils.py", "src/requests/adapters.py"]),
    ("Which components release network connections when a session or a response is closed?",
     ["src/requests/sessions.py", "src/requests/adapters.py", "src/requests/models.py"]),
    ("How is the character encoding of a response chosen when its text is read?",
     ["src/requests/models.py", "src/requests/utils.py", "src/requests/compat.py",
      "src/requests/adapters.py"]),
    ("Where do the default headers such as the user agent come from, and how do they reach a request?",
     ["src/requests/utils.py", "src/requests/sessions.py", "src/requests/__version__.py"]),
]  # fmt: skip


# Owner's review, 2026-11-18 (docs/ARCHITECTURE_LABEL_REVIEW.md, checked against the source at 611c616):
# 002 +models.py (prepare_auth turns the netrc tuple into HTTPBasicAuth), 004 -status_codes.py
# (raise_for_status never uses `codes`; models + exceptions, both needed), 006 +sessions.py
# (Session.send runs the digest hook), 009 +adapters.py (build_response sets the first encoding).
# 005 "optionally also accept compat.py" is not applied. Need stays half rounded up, except 004 (2 files, 2).
ARCH_FRESH_NEED = {4: 2}


def build_architecture_fresh_records() -> list[dict[str, Any]]:
    return [
        {
            "id": f"archfresh-{n:03d}",
            "type": "architecture",
            "question": question,
            "expect": {
                "gold_files": files,
                "need": ARCH_FRESH_NEED.get(n, -(-len(files) // 2)),
                "split": "fresh",
            },
        }
        for n, (question, files) in enumerate(ARCH_FRESH, start=1)
    ]


def build_architecture_fresh() -> None:
    records = build_architecture_fresh_records()
    ARCH_FRESH_DATASET.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(
        f"{len(records)} fresh architecture questions; sha256 {sha256_of(ARCH_FRESH_DATASET)[:12]}"
    )


def build_architecture() -> None:
    records = build_architecture_records()
    ARCH_DATASET.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(f"{len(records)} architecture questions; sha256 {sha256_of(ARCH_DATASET)[:12]}")


def build_commit_full() -> None:
    """Every `test`-split bug -> fix pair of psf/requests as a commit question (113), in pair-id
    order. The frozen 10-question set is a sample of the same pairs; 113 questions can show a
    difference of a few questions that 10 cannot. The dev pairs (28) are not used here, so they stay
    available for choosing a setting."""
    pairs = [json.loads(line) for line in BUGFIX.read_text(encoding="utf-8").splitlines()]
    tests = sorted((p for p in pairs if p["split"] == "test"), key=lambda p: p["id"])
    records = []
    for number, pair in enumerate(tests, start=1):
        gold = pair["gold"]
        shas = ast.literal_eval(gold) if isinstance(gold, str) else gold
        records.append(
            {
                "id": f"commitfull-{number:03d}",
                "type": "commit",
                "question": "A user reports the bug below. Was something like this fixed before "
                "in this repository? If so, which commit?\n\n" + pair["query"][:BUG_CHARS],
                "expect": {"shas": [s[:12] for s in shas]},
            }
        )
    COMMIT_FULL_DATASET.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(f"{len(records)} commit questions; sha256 {sha256_of(COMMIT_FULL_DATASET)[:12]}")


def audit_probes(records: list[dict[str, Any]], clone: Path) -> None:
    """A probe is only valid if its subject appears nowhere in the indexed repository."""
    text = ""
    for part in ("src", "docs"):
        text += "\n".join(
            f.read_text(encoding="utf-8", errors="ignore").lower()
            for f in (clone / part).rglob("*")
            if f.is_file() and f.suffix in {".py", ".rst", ".md", ".txt"}
        )
    text += (clone / "README.md").read_text(encoding="utf-8").lower()
    for record in records:
        if record["type"] == "probe" and record["expect"]["term"] in text:
            raise SystemExit(f"Probe term {record['expect']['term']!r} occurs in the repository")


def build() -> None:
    from repo_rag.config import Settings

    records = build_records()
    audit_probes(records, Settings().data_dir / "repos" / REPO.replace("/", "__"))
    DATASET.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["type"]] += 1
    print(f"{len(records)} questions {dict(counts)}; sha256 {sha256_of(DATASET)[:12]}")


# ----------------------------------------------------------------------------- scoring

CITATION = re.compile(r"([\w./-]+\.\w+):(\d+)-(\d+)")


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def first_kind(answer: str) -> str:
    """Which of 'issue' / 'pull request' (or PR) the answer names first."""
    lowered = answer.lower()
    spots = {
        "issue": lowered.find("issue"),
        "pull request": min(
            (i for i in (lowered.find("pull request"), lowered.find(" pr ")) if i >= 0),
            default=-1,
        ),
    }
    named = {kind: i for kind, i in spots.items() if i >= 0}
    return min(named, key=lambda kind: named[kind]) if named else ""


def score(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Mechanical pass/fail. A failed run (tool or model error) never passes, not even a probe."""
    kind, expect = record["type"], record["expect"]
    # Score the model's own text: facts the agent appends to the answer must never change a score.
    text = result.get("model_answer", result["answer"])
    answer = text if isinstance(text, str) else json.dumps(text)
    ok = not result["failure"]
    detail = ""
    if kind == "probe":
        passed = ok and result["abstain"]
    elif not ok or result["abstain"]:
        passed, detail = False, "abstained" if ok else "failed"
    elif kind == "code":
        gold = expect["gold"]
        cites = [(p, int(a), int(b)) for p, a, b in CITATION.findall(answer)]
        passed = any(
            p == g["path"] and a <= g["end_line"] and b >= g["start_line"]
            for p, a, b in cites
            for g in gold
        )
        detail = "" if passed else "no cited range overlaps the gold range"
    elif kind == "docs":
        passed = any(path in answer for path in expect["paths"])
    elif kind == "history" and "all" in expect:
        passed = all(token.lower() in answer.lower() for token in expect["all"])
    elif kind in {"history", "refs"}:
        found = numbers_in(answer) - {expect.get("exclude", "")}
        missing = [n for n in expect["numbers"] if n not in found]
        extra = sorted(found - set(expect["numbers"]))
        passed, detail = not missing, f"missing {missing}; extra {extra}"
    elif kind == "kind":
        passed = first_kind(answer) == expect["kind"]
    elif kind == "commit":
        passed = any(sha in answer.lower() for sha in expect["shas"])
        if not passed:
            extra_shas = [a for sha in expect["shas"] for a in ALSO_ACCEPTED.get(sha, [])]
            if any(sha in answer.lower() for sha in extra_shas):
                passed, detail = True, "accepted through ALSO_ACCEPTED"
    elif kind == "prsearch":
        passed = expect["number"] in numbers_in(answer)
    elif kind == "architecture":
        named = [path for path in expect["gold_files"] if path in answer]
        passed = len(named) >= expect["need"]
        detail = (
            f"names {len(named)} of {len(expect['gold_files'])} gold files (need {expect['need']})"
        )
    else:
        raise ValueError(f"Unknown question type {kind}")
    return {"passed": bool(passed), "detail": detail}


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = table.setdefault(
            row["type"], {"n": 0, "passed": 0, "abstained": 0, "failed": 0, "missed": []}
        )
        entry["n"] += 1
        entry["passed"] += row["passed"]
        entry["abstained"] += bool(row["abstain"] and not row["failure"])
        entry["failed"] += bool(row["failure"])
        if not row["passed"]:
            entry["missed"].append(row["id"])
    return table


# ----------------------------------------------------------------------------- running


def run(
    url: str,
    token: str,
    types: set[str] | None,
    toolset: str,
    rerank: bool = True,
    variant: str = "baseline",
    dataset: Path = DATASET,
    commit_query: str = "model",
    commit_fix_only: bool = False,
    extra_files: int = 0,
    skip_tests: bool = False,
    citation_snap: bool = False,
    abstain_retries: int = 0,
    tool_error_retries: int = 0,
    tool_chars: int | None = None,
    max_input_tokens: int | None = None,
    commit_k: int | None = None,
    max_tool_calls: int | None = None,
) -> None:
    from repo_rag.agent.remote_llm import RemoteChatModel
    from repo_rag.agent.runtime import build_agent
    from repo_rag.config import Settings

    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    if types:
        records = [r for r in records if r["type"] in types]
    settings = Settings().model_copy(
        update={
            "agent_toolset": toolset,
            "agent_prompt_variant": variant,
            "agent_commit_query": commit_query,
            "agent_commit_fix_only": commit_fix_only,
            "agent_code_extra_files": extra_files,
            "agent_code_skip_tests": skip_tests,
            **({"agent_tool_result_chars": tool_chars} if tool_chars else {}),
            "agent_citation_snap": citation_snap,
            "agent_abstain_retries": abstain_retries,
            "agent_tool_error_retries": tool_error_retries,
            **({"agent_commit_k": commit_k} if commit_k else {}),
            **(
                {"agent_max_tool_calls": max_tool_calls, "agent_max_turns": max_tool_calls + 1}
                if max_tool_calls
                else {}
            ),
        }
    )
    model = RemoteChatModel(url, token)
    info = model.health()
    served_cap = info.get("max_input_tokens")
    if max_input_tokens is not None and str(max_input_tokens) != str(served_cap):
        # The input cap is enforced on the server, so this flag cannot set it, only check it.
        raise SystemExit(
            f"--max-input-tokens {max_input_tokens} but the server reports {served_cap!r}. "
            "Restart the serve notebook with that cap (REPO_RAG_AGENT_MAX_INPUT_TOKENS)."
        )
    factory = model.reranker_factory(info) if rerank else None
    print(f"reranker on the server: {info.get('rerank') or 'none'}; used: {factory is not None}")
    agent = build_agent(REPO, model, settings, reranker_factory=factory)
    tools = getattr(agent.execute, "__self__", None)  # the bound tools object, if any
    rows = []
    for index, record in enumerate(records, 1):
        started = time.perf_counter()
        failures_before = getattr(tools, "rerank_failures", 0)
        result = agent.ask(record["question"], thread_id=record["id"])
        row = {
            "id": record["id"],
            "type": record["type"],
            "question": record["question"][:200],
            "answer": result["answer"],
            "model_answer": result.get("model_answer", result["answer"]),
            "abstain": result["abstain"],
            "failure": result["failure"],
            "error": result["error"],
            "raw_reply": result.get("raw_reply", ""),  # what the model said, for failed runs
            "parse_retries": result.get("parse_retries", 0),
            "guards": result.get("guards", {}),
            "tools": [c["name"] for c in result["tool_calls"]],
            # The arguments the model chose (its search queries), so a miss can be replayed offline.
            "calls": [
                {"name": c["name"], "arguments": c["arguments"]} for c in result["tool_calls"]
            ],
            "unverified_shas": result["unverified_shas"],
            "citations": result["citations"],  # valid/removed/attached, and snapped when on
            "seconds": round(time.perf_counter() - started, 1),
            # Searches in this question where the server's reranker failed and the fused order
            # was used instead; non-zero means the "reranked" result is partly not reranked.
            "rerank_failures": getattr(tools, "rerank_failures", 0) - failures_before,
            **score(record, result),
        }
        rows.append(row)
        mark = "pass" if row["passed"] else "MISS"
        print(
            f"[{index}/{len(records)}] {record['id']:<12} {mark}  {row['seconds']}s  {row['tools']}"
        )
    table = summarize(rows)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "" if variant == "baseline" else f"_{variant}"
    if dataset != DATASET:
        tag += f"_{dataset.stem}"
    if commit_query != "model":
        tag += f"_cq-{commit_query}"
    if commit_fix_only:
        tag += "_fixonly"
    if extra_files:
        tag += f"_extra{extra_files}"
    if skip_tests:
        tag += "_notests"
    if tool_chars:
        tag += f"_chars{tool_chars}"
    if max_input_tokens:
        tag += f"_in{max_input_tokens}"
    if commit_k and commit_k != Settings().agent_commit_k:
        tag += f"_ck{commit_k}"
    if max_tool_calls:
        tag += f"_tc{max_tool_calls}"
    if citation_snap:
        tag += "_snap"
    if abstain_retries:
        tag += f"_abst{abstain_retries}"
    if tool_error_retries:
        tag += f"_toolerr{tool_error_retries}"
    out = RESULTS_DIR / f"agent_eval_{toolset}{tag}_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "dataset": dataset.name,
                "dataset_sha256": sha256_of(dataset),
                "toolset": toolset,
                "prompt_variant": variant,
                "commit_query": commit_query,
                "commit_fix_only": commit_fix_only,
                "extra_files": extra_files,
                "skip_tests": skip_tests,
                "tool_result_chars": tool_chars or settings.agent_tool_result_chars,
                "max_input_tokens": served_cap,
                "commit_k": commit_k or settings.agent_commit_k,
                "max_tool_calls": max_tool_calls or settings.agent_max_tool_calls,
                "citation_snap": citation_snap,
                "abstain_retries": abstain_retries,
                "tool_error_retries": tool_error_retries,
                "server": info,
                "rerank_used": factory is not None,
                "rerank_failures": sum(r["rerank_failures"] for r in rows),
                "llm_model": settings.llm_model,
                "table": table,
                "rows": rows,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n{'type':<9}{'passed':>8}{'of':>5}{'abstained':>11}{'failed':>8}")
    for kind, e in table.items():
        print(f"{kind:<9}{e['passed']:>8}{e['n']:>5}{e['abstained']:>11}{e['failed']:>8}")
    failures = sum(r["rerank_failures"] for r in rows)
    if failures:
        print(
            f"\nWARNING: the reranker failed {failures} time(s); those searches were not reranked"
        )
    print(f"\nwrote {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("build-architecture")
    sub.add_parser("build-commit-full")
    sub.add_parser("build-architecture-fresh")
    runner = sub.add_parser("run")
    runner.add_argument("--url", required=True)
    runner.add_argument("--token", required=True)
    runner.add_argument("--types", help="comma list, e.g. history,refs")
    runner.add_argument("--toolset", choices=["evaluated", "production"], default="production")
    runner.add_argument(
        "--no-rerank", action="store_true", help="do not use the server's reranker for commits"
    )
    runner.add_argument(
        "--variant",
        choices=list(PROMPT_VARIANTS),
        default="baseline",
        help="production prompt variant for an A/B run (see Settings.agent_prompt_variant)",
    )
    runner.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="question file (default agent_eval_v1; agent_eval_fresh_v1 is the fresh set)",
    )
    runner.add_argument(
        "--commit-query",
        choices=["model", "question"],
        default="model",
        help="search_commits searches the model's query (default) or the user's whole question",
    )
    runner.add_argument(
        "--commit-fix-only",
        action="store_true",
        help="search_commits returns only commits classified as fixes (is_fix=true)",
    )
    runner.add_argument(
        "--extra-files",
        type=int,
        default=0,
        help="search_code also shows this many chunks from other files, found with the question",
    )
    runner.add_argument(
        "--skip-tests",
        action="store_true",
        help="search_code leaves test files out unless the question asks about tests",
    )
    runner.add_argument(
        "--citation-snap",
        action="store_true",
        help="re-point or narrow cited ranges using the symbol table (agent_citation_snap)",
    )
    runner.add_argument(
        "--abstain-retries",
        type=int,
        default=0,
        help="ask the model to search again when it abstains before two tool calls (this many times)",
    )
    runner.add_argument(
        "--tool-error-retries",
        type=int,
        default=0,
        help="give a failing tool's error back to the model this many times instead of failing",
    )
    runner.add_argument(
        "--tool-chars",
        type=int,
        default=None,
        help="cumulative tool-result characters per question (default: the evaluated 12000)",
    )
    runner.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="the input-token cap the server must report (set on the server, checked here)",
    )
    runner.add_argument(
        "--commit-k",
        type=int,
        default=None,
        help="commits search_commits returns (default: agent_commit_k; the first 5 in full, the rest one line)",
    )
    runner.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="tool calls allowed per question (default: agent_max_tool_calls, 4)",
    )
    runner.add_argument(
        "--repeat", type=int, default=1, help="run the selected questions this many times"
    )
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "build-commit-full":
        build_commit_full()
    elif args.command == "build-architecture":
        build_architecture()
    elif args.command == "build-architecture-fresh":
        build_architecture_fresh()
    else:
        types = set(args.types.split(",")) if args.types else None
        for _ in range(args.repeat):  # each run builds a fresh agent, so threads never carry over
            run(
                args.url,
                args.token,
                types,
                args.toolset,
                not args.no_rerank,
                args.variant,
                args.dataset,
                args.commit_query,
                args.commit_fix_only,
                args.extra_files,
                args.skip_tests,
                args.citation_snap,
                args.abstain_retries,
                args.tool_error_retries,
                args.tool_chars,
                args.max_input_tokens,
                args.commit_k,
                args.max_tool_calls,
            )


if __name__ == "__main__":
    main()
