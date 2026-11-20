"""Run every measurement we need in ONE Colab model-server session, then write one report.

Start the 27B server notebook, then:

    uv run python eval/colab_session.py --url <url> --token <token>

Steps run in order, each one logged; a step that already finished is skipped on a re-run
(`state.json` in the output folder), so the script can be stopped and started again. The slow
step (commit summaries, about an hour) is last. Nothing here changes a default: the report only
compares runs. Read `report.md` and `answers_sample.md` afterwards, then decide.

    --only fresh_default commit_explain     run just these steps
    --no-summaries                          leave out the summary step
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AGENT_EVAL = "eval/agent_eval.py"
FRESH = "eval/datasets/agent_eval_fresh_v2.jsonl"
COMMIT_FULL = "eval/datasets/agent_eval_commit_full_v1.jsonl"
TUNED = "eval/datasets/agent_eval_v1.jsonl"
ARCH = "eval/datasets/agent_eval_architecture_v1.jsonl"
ARCH_FRESH = "eval/datasets/agent_eval_architecture_fresh_v1.jsonl"
# agent_eval's own --variant default is "baseline", not the shipped prompt, so it is set here.
SHIPPED_VARIANT = "commit_pr_commits"  # the default until 2026-11-20 (now commit_explain); kept so "default" steps stay comparable
COMMON = [
    "--extra-files", "3", "--skip-tests", "--tool-chars", "24000", "--max-input-tokens", "12288",
    "--variant", SHIPPED_VARIANT,
]  # fmt: skip
# Only the guards: the prompt stays the shipped one, so the comparison isolates them.
GUARDS = ["--citation-snap", "--abstain-retries", "1", "--tool-error-retries", "1"]  # fmt: skip
EXPECTED_DECODE = "cuda-graphs + mtp3"
EXPECTED_INPUT_CAP = "12288"
SAMPLE_PER_TYPE = 4  # answers per type per step written to answers_sample.md


@dataclass(frozen=True)
class Step:
    name: str
    why: str
    args: tuple[str, ...]  # arguments after `eval/agent_eval.py run`; empty for the summary step
    minutes: int


STEPS = [
    Step(
        "fresh_default",
        "new questions, default settings: the honest pass rate",
        ("--dataset", FRESH, *COMMON),
        6,
    ),
    Step(
        "fresh_guards",
        "same questions with the answer guards on: decides the guards default",
        ("--dataset", FRESH, *COMMON, *GUARDS),
        6,
    ),
    Step(
        "commit_default",
        "113 bug reports, default: extra fields and facts did not change the score",
        ("--types", "commit", "--dataset", COMMIT_FULL, *COMMON),
        9,
    ),
    Step(
        "commit_explain",
        "113 bug reports with the commit_explain prompt",
        ("--types", "commit", "--dataset", COMMIT_FULL, *COMMON, "--variant", "commit_explain"),
        9,
    ),
    Step(
        "regression_default",
        "the 103 tuned questions: nothing was lost",
        ("--dataset", TUNED, *COMMON),
        8,
    ),
    Step(
        "architecture_v1",
        "the 15 architecture questions, default: answers to read (never reviewed)",
        ("--dataset", ARCH, *COMMON),
        4,
    ),
    Step(
        "architecture_fresh",
        "the 10 fresh architecture questions, default: answers to read",
        ("--dataset", ARCH_FRESH, *COMMON),
        3,
    ),
    Step(
        "fresh_repeat_2", "run-to-run noise: the fresh set again", ("--dataset", FRESH, *COMMON), 6
    ),
    Step(
        "fresh_repeat_3",
        "run-to-run noise: the fresh set a third time",
        ("--dataset", FRESH, *COMMON),
        6,
    ),
    Step("summaries", "commit summaries for the pilot (resumable, about an hour)", (), 80),
]


def build_command(step: Step, url: str, token: str, workers: int = 1) -> list[str]:
    if step.name == "summaries":
        return [sys.executable, "eval/commit_summary_pilot.py", "summarize", "--url", url,
                "--token", token]  # fmt: skip
    extra = ["--workers", str(workers)] if workers > 1 else []
    return [sys.executable, AGENT_EVAL, "run", "--url", url, "--token", token, *step.args, *extra]


def result_path(log_text: str) -> str | None:
    found = re.findall(r"wrote\s+(\S+\.json)", log_text)
    return found[-1] if found else None


def summarize_result(path: Path) -> dict[str, dict[str, Any]]:
    """Per question type: passed, of, abstained, failed, mean seconds."""
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row)
    out: dict[str, dict[str, Any]] = {}
    for kind, items in sorted(grouped.items()):
        out[kind] = {
            "passed": sum(r["passed"] in (True, "True") for r in items),
            "of": len(items),
            "abstained": sum(bool(r["abstain"]) for r in items),
            "failed": sum(bool(r["failure"]) for r in items),
            "seconds": round(sum(r["seconds"] for r in items) / len(items), 1),
        }
    return out


def outcomes(path: Path) -> dict[str, bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {r["id"]: r["passed"] in (True, "True") for r in rows}


def flips(a: dict[str, bool], b: dict[str, bool]) -> tuple[list[str], list[str]]:
    """(passed in b only, passed in a only) over the questions both runs have."""
    both = a.keys() & b.keys()
    return (
        sorted(i for i in both if b[i] and not a[i]),
        sorted(i for i in both if a[i] and not b[i]),
    )


def render_report(state: dict[str, Any], root: Path) -> str:
    lines = ["# Colab session report", ""]
    done = {n: s for n, s in state["steps"].items() if s.get("result")}
    kinds = sorted({k for s in done.values() for k in s["summary"]})
    lines += ["| step | " + " | ".join(kinds) + " | seconds/question |",
              "|---|" + "---|" * (len(kinds) + 1)]  # fmt: skip
    for name, step in done.items():
        cells = []
        for kind in kinds:
            m = step["summary"].get(kind)
            cells.append(
                f"{m['passed']}/{m['of']} (abst {m['abstained']}, fail {m['failed']})" if m else ""
            )
        times = [m["seconds"] for m in step["summary"].values()]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {sum(times) / len(times):.1f} |")
    lines.append("")
    pairs = [("fresh_default", "fresh_guards"), ("fresh_default", "fresh_repeat_2"),
             ("fresh_repeat_2", "fresh_repeat_3"), ("commit_default", "commit_explain")]  # fmt: skip
    for a, b in pairs:
        if a in done and b in done:
            gained, lost = flips(
                outcomes(root / done[a]["result"]), outcomes(root / done[b]["result"])
            )
            lines.append(
                f"- **{a} -> {b}:** {len(gained)} questions gained {gained}, {len(lost)} lost {lost}"
            )
    repeats = [n for n in ("fresh_default", "fresh_repeat_2", "fresh_repeat_3") if n in done]
    if len(repeats) >= 2:
        counts = [sum(outcomes(root / done[n]["result"]).values()) for n in repeats]
        lines.append(f"- **Noise on the fresh set** ({', '.join(repeats)}): passes {counts}, "
                     f"spread {max(counts) - min(counts)} questions")  # fmt: skip
    return "\n".join(lines) + "\n"


def sample_answers(state: dict[str, Any], root: Path) -> str:
    """A fixed sample of answers per type per step, for a person (or Claude) to read."""
    rng = random.Random(0)
    lines = [
        "# Answer sample",
        "",
        "Read these against the evidence; a pass here is only mechanical.",
        "",
    ]
    for name, step in state["steps"].items():
        if not step.get("result") or name == "summaries":
            continue
        rows = json.loads((root / step["result"]).read_text(encoding="utf-8"))["rows"]
        by_type: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_type[row["type"]].append(row)
        lines += [f"## {name}", ""]
        for _kind, items in sorted(by_type.items()):
            for row in rng.sample(items, min(SAMPLE_PER_TYPE, len(items))):
                answer = str(row.get("answer", "")).strip().replace("\n", " ")[:700]
                verdict = "PASS" if row["passed"] in (True, "True") else "MISS"
                lines.append(
                    f"- `{row['id']}` {verdict} | {str(row.get('question', ''))[:120]!r}\n  - {answer}"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def check_server(url: str, token: str) -> dict[str, Any]:
    from repo_rag.agent.remote_llm import RemoteChatModel

    info = RemoteChatModel(url, token).health()
    problems = []
    if info.get("decode_mode") != EXPECTED_DECODE:
        problems.append(f"decode_mode is {info.get('decode_mode')!r}, expected {EXPECTED_DECODE!r}")
    if str(info.get("max_input_tokens")) != EXPECTED_INPUT_CAP:
        problems.append(
            f"max_input_tokens is {info.get('max_input_tokens')!r}, expected {EXPECTED_INPUT_CAP}"
        )
    if problems:
        raise SystemExit("server is not the profile every earlier run used: " + "; ".join(problems))
    return info


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"steps": {}}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/colab_session_results/latest")
    parser.add_argument("--only", nargs="*", help="step names to run")
    parser.add_argument("--no-summaries", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="questions asked at once in the evaluation steps (needs the concurrent server)",
    )
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = load_state(state_path)
    state["server"] = check_server(args.url, args.token)
    print("server ok:", state["server"], flush=True)
    chosen = [s for s in STEPS if (not args.only or s.name in args.only)
              and not (args.no_summaries and s.name == "summaries")]  # fmt: skip
    print(
        f"{len(chosen)} steps, about {sum(s.minutes for s in chosen)} minutes of server time",
        flush=True,
    )
    for step in chosen:
        if step.name != "summaries" and state["steps"].get(step.name, {}).get("result"):
            print(f"[skip] {step.name} already done", flush=True)
            continue
        print(f"[run ] {step.name}: {step.why}", flush=True)
        started = time.perf_counter()
        done = subprocess.run(build_command(step, args.url, args.token, args.workers), cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")  # fmt: skip
        log = (done.stdout or "") + (done.stderr or "")
        (out / f"{step.name}.log").write_text(log, encoding="utf-8")
        entry: dict[str, Any] = {
            "seconds": round(time.perf_counter() - started),
            "exit": done.returncode,
        }
        found = result_path(log)
        if step.name != "summaries" and found and done.returncode == 0:
            path = Path(found)
            entry["result"] = (
                path.as_posix() if not path.is_absolute() else path.relative_to(ROOT).as_posix()
            )
            entry["summary"] = summarize_result(ROOT / entry["result"])
        state["steps"][step.name] = entry
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(
            f"       exit {done.returncode} in {entry['seconds']} s; log {step.name}.log",
            flush=True,
        )
        if done.returncode and step.name != "summaries":
            print("       this step failed: the later steps still run; check its log", flush=True)
        (out / "report.md").write_text(render_report(state, ROOT), encoding="utf-8")
        (out / "answers_sample.md").write_text(sample_answers(state, ROOT), encoding="utf-8")
    print(f"\ndone. Read {out / 'report.md'} and {out / 'answers_sample.md'}", flush=True)


if __name__ == "__main__":
    main()
