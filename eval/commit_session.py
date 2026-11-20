"""Run the commit-search experiments in ONE Colab model-server session, then write one report.

The 2026-11-19 sort of the 35 missed commit questions found that for 23 of them the gold commit
was not in the agent's first shortlist, and for 12 it was but was not named. This session tests the
changes built for each part, against a control that is run twice so the run-to-run noise is
measured, not guessed:

    control_a, control_b   commit_explain prompt, model-written query (the best baseline)
    fused                  the model's query and the question, rankings merged (+ lenient score)
    verify                 the model reads the best candidates with get_commit before naming one
    fused_verify           both
    fused_fix_only         fused, and only commits whose message looks like a fix (see the caveat)
    fused_w05              fused with the question's weight at 0.5 (best MRR offline)
    summary_hidden         the experiment index with summaries (search uses them, the model does not)
    control_c, summary_fused_hidden_b   a third control and a repeat of the best step, to confirm it
    summary_fused_hidden   that index, fused (the best offline retrieval: gold in the top 5, 75 of 113)

Start the 27B server notebook, then:

    uv run python eval/commit_session.py --url <url> --token <token> --workers 8

A step that already finished is skipped on a re-run (`state.json`), so it can be stopped and
started again. `--only fused verify` runs just those. Nothing here changes a default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import commit_lenient_score as lenient  # noqa: E402

BASE_OUT = ROOT / "eval/commit_session_results"
OUT = BASE_OUT
COMMIT_FULL = "eval/datasets/agent_eval_commit_full_v1.jsonl"
DEFAULT_EXPERIMENT_INDEX = ROOT / "data" / "exp_summaries"
EXPERIMENT_INDEX = DEFAULT_EXPERIMENT_INDEX
SUMMARY_INDEX = "@summary"  # replaced by the summary index folder when a step runs
REPO = "psf/requests"
DEFAULT_DATA_DIR = ROOT / "data"
DATA_DIR = DEFAULT_DATA_DIR
BASE_COMMON = (
    "--types", "commit", "--dataset", COMMIT_FULL, "--extra-files", "3", "--skip-tests",
    "--tool-chars", "24000", "--max-input-tokens", "12288",
)  # fmt: skip
COMMON = list(BASE_COMMON)


@dataclass(frozen=True)
class Step:
    name: str
    why: str
    args: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    caveat: str = ""


STEPS = [
    Step("control_a", "the baseline prompt", ("--variant", "commit_explain")),
    Step("control_b", "the same again: run-to-run noise", ("--variant", "commit_explain")),
    Step(
        "fused",
        "model query + the question, rankings merged",
        ("--variant", "commit_explain", "--commit-query", "fused"),
    ),
    Step(
        "verify",
        "read candidates with get_commit before naming one",
        ("--variant", "commit_verify"),
    ),
    Step(
        "fused_verify",
        "both changes",
        ("--variant", "commit_verify", "--commit-query", "fused"),
    ),
    Step(
        "fused_fix_only",
        "fused, and only commits classified as fixes",
        ("--variant", "commit_explain", "--commit-query", "fused", "--commit-fix-only"),
        caveat=(
            "every gold commit of this set passes the is_fix test by construction (the data "
            "protocol requires it), so this number is an upper bound for real bug reports"
        ),
    ),
    Step(
        "fused_w05",
        "fused, the question's ranking at half weight",
        ("--variant", "commit_explain", "--commit-query", "fused", "--fuse-weight", "0.5"),
    ),
    Step(
        "summary_hidden",
        "index with summaries for search; the model sees the raw commit",
        ("--variant", "commit_explain", "--hide-summary"),
        {"REPO_RAG_DATA_DIR": SUMMARY_INDEX},
    ),
    Step(
        "summary_fused_hidden",
        "summary index for search, fused, the model sees the raw commit",
        ("--variant", "commit_explain", "--commit-query", "fused", "--hide-summary"),
        {"REPO_RAG_DATA_DIR": SUMMARY_INDEX},
    ),
    Step("control_c", "a third identical control", ("--variant", "commit_explain")),
    Step(
        "summary_fused_hidden_b",
        "repeat of summary_fused_hidden",
        ("--variant", "commit_explain", "--commit-query", "fused", "--hide-summary"),
        {"REPO_RAG_DATA_DIR": SUMMARY_INDEX},
    ),
    Step(
        "code_fallback",
        "two-hop code -> symbol/path-scoped commit search fallback",
        ("--variant", "commit_code_fallback"),
    ),
    Step(
        "fused_code_fallback",
        "fused + two-hop code -> symbol/path-scoped commit search fallback",
        ("--variant", "commit_code_fallback", "--commit-query", "fused"),
    ),
    Step(
        "summary_fused_code_fallback",
        "summary index + fused + two-hop code -> symbol/path-scoped commit search fallback",
        ("--variant", "commit_code_fallback", "--commit-query", "fused", "--hide-summary"),
        {"REPO_RAG_DATA_DIR": SUMMARY_INDEX},
    ),
]


def sign_test_p(won: int, lost: int) -> float:
    """Exact two-sided p-value of `won` vs `lost` changed questions under a fair coin."""
    n = won + lost
    if n == 0:
        return 1.0
    k = min(won, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def configure(repo: str, data_dir: Path | None, summary_index: Path | None) -> None:
    """Point the session at another repository: its question file, its own results folder, the
    folder holding its index and metadata.db, and its summary index."""
    global OUT, COMMON, EXPERIMENT_INDEX, REPO, DATA_DIR
    REPO = repo
    slug = repo.replace("/", "__")
    dataset = (
        COMMIT_FULL
        if repo == "psf/requests"
        else f"eval/datasets/agent_eval_commit_full_{slug}_v1.jsonl"
    )
    COMMON = [dataset if part == COMMIT_FULL else part for part in BASE_COMMON] + ["--repo", repo]
    OUT = BASE_OUT if repo == "psf/requests" else BASE_OUT / slug
    DATA_DIR = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    EXPERIMENT_INDEX = (
        summary_index
        if summary_index is not None
        else (
            DEFAULT_EXPERIMENT_INDEX
            if repo == "psf/requests"
            else ROOT / "data" / f"exp_summaries_{slug}"
        )
    )


def command(step: Step, url: str, token: str, workers: int) -> list[str]:
    extra = ["--workers", str(workers)] if workers > 1 else []
    return [sys.executable, "eval/agent_eval.py", "run", "--url", url, "--token", token,
            *COMMON, *step.args, *extra]  # fmt: skip


def run_step(step: Step, url: str, token: str, workers: int) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{step.name}.log"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if DATA_DIR != ROOT / "data":
        env["REPO_RAG_DATA_DIR"] = str(DATA_DIR)
    env.update({k: str(EXPERIMENT_INDEX) if v == SUMMARY_INDEX else v for k, v in step.env.items()})
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        done = subprocess.run(
            command(step, url, token, workers), cwd=ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )  # fmt: skip
    text = log.read_text(encoding="utf-8")
    found = re.findall(r"wrote\s+(\S+\.json)", text)
    if done.returncode or not found:
        raise SystemExit(f"step {step.name} failed (exit {done.returncode}); see {log}")
    print(f"{step.name}: done in {(time.perf_counter() - started) / 60:.1f} min -> {found[-1]}")
    return found[-1]


def load_rows(path: str) -> dict[str, dict]:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    return {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}


def report(state: dict[str, str]) -> str:
    db = sqlite3.connect(
        f"file:{(DATA_DIR.resolve() / 'metadata.db').as_posix()}?mode=ro", uri=True
    )
    try:
        lines = ["# Commit session report", ""]
        runs = {name: load_rows(path) for name, path in state.items()}
        if "control_a" not in runs:
            return "\n".join([*lines, "No control run yet."])
        base = runs["control_a"]
        lines += [
            "| step | strict | lenient | abstained | mean search calls | get_commit calls | won | lost | sign-test p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, rows in runs.items():
            path = Path(state[name])
            res = lenient.score(path if path.is_absolute() else ROOT / path, db)
            calls = [c["name"] for r in rows.values() for c in r["calls"]]
            won = sum(1 for i, r in rows.items() if r["passed"] and not base[i]["passed"])
            lost = sum(1 for i, r in rows.items() if not r["passed"] and base[i]["passed"])
            p = "" if name == "control_a" else f"{sign_test_p(won, lost):.3f}"
            w = "" if name == "control_a" else str(won)
            l = "" if name == "control_a" else str(lost)  # noqa: E741
            lines.append(
                f"| {name} | {res['strict']}/{res['n']} | {res['lenient']}/{res['n']} | "
                f"{sum(bool(r['abstain']) for r in rows.values())} | "
                f"{calls.count('search_commits') / len(rows):.2f} | {calls.count('get_commit')} | "
                f"{w} | {l} | {p} |"
            )
        lines += ["", "won/lost are questions passed/failed compared with control_a.", ""]
        lines += [
            f"Caveat, {st.name}: {st.caveat}" for st in STEPS if st.caveat and st.name in runs
        ]
        if "control_b" in runs:
            flips = sum(
                1 for i in base if bool(base[i]["passed"]) != bool(runs["control_b"][i]["passed"])
            )
            lines.append(
                f"Noise: the two identical control runs differ on {flips} of {len(base)} questions."
            )
        return "\n".join(lines)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", help="step names to run")
    parser.add_argument("--repo", default="psf/requests", help="owner/name (default psf/requests)")
    parser.add_argument("--data-dir", type=Path, help="folder with this repository's index")
    parser.add_argument("--summary-index", type=Path, help="folder with the summary-embedded index")
    args = parser.parse_args()
    configure(args.repo, args.data_dir, args.summary_index)
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    wanted = [s for s in STEPS if not args.only or s.name in args.only]
    if any(s.name.startswith("summary_") for s in wanted) and not EXPERIMENT_INDEX.exists():
        raise SystemExit(f"{EXPERIMENT_INDEX} is missing: build it with eval/summarize_commits.py")
    for step in wanted:
        if step.name in state:
            print(f"{step.name}: already done ({state[step.name]})")
            continue
        print(f"{step.name}: {step.why}", flush=True)
        state[step.name] = run_step(step, args.url, args.token, args.workers)
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    text = report(state)
    (OUT / "report.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
