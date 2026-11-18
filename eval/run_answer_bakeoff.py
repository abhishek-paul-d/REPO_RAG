"""Colab execution only: prepare-git, smoke (14B), dev, then frozen test.

No labels are decoded here. Use analyze_answer_bakeoff.py to freeze on dev and score test.
Importing this module never loads a model; the user runs these commands on Colab.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.answer_bakeoff_io import (
    canonical,
    digest,
    identity,
    load,
    question_rows,
    save,
    verify,
    verify_test_freeze,
)
from repo_rag.agent.bakeoff import run_question
from repo_rag.agent.bakeoff_tools import tool_schemas
from repo_rag.agent.prompts import ANSWER_BAKEOFF_WARMUP, BAKEOFF_ARMS


def verify_capsule(capsule: Path) -> None:
    manifest = load(capsule / "capsule_manifest.json")
    for name, sha in manifest.items():
        if digest(ROOT / name) != sha:
            raise ValueError("Capsule hash mismatch")
    for name, sha in load(capsule / "index_manifest.json")["payload_sha256"].items():
        if manifest.get(name) != sha:
            raise ValueError("Index manifest data identity mismatch")


def prepare_git(capsule: Path, work: Path) -> None:
    recipe = load(capsule / "git_recipe.json")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    for item in recipe["repositories"]:
        checkout = work / "repos" / item["repo"].replace("/", "__")
        checkout.parent.mkdir(parents=True, exist_ok=True)

        def git(*args, checkout=checkout):
            return (
                subprocess.check_output(
                    ["git", "-C", str(checkout), *args], env=env, stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

        if not (checkout / ".git").exists():
            subprocess.run(
                ["git", "clone", "--quiet", "--no-tags", item["url"], str(checkout)],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        git("checkout", "--quiet", "--detach", item["pin"])
        if git("rev-parse", "HEAD") != item["pin"]:
            raise ValueError("Git pin mismatch")
        # Fetch only the frozen tags; future tags must not turn frozen 'none' into an answer.
        actual = git("for-each-ref", "--format=%(refname:short)", "refs/tags").splitlines()
        for tag in actual:
            if tag not in item["tags"]:
                git("update-ref", "-d", "refs/tags/" + tag)
        if item["tags"]:
            git(
                "fetch",
                "--quiet",
                "--no-tags",
                "origin",
                *[f"refs/tags/{tag}:refs/tags/{tag}" for tag in item["tags"]],
            )
        for tag, sha in item["tags"].items():
            if git("rev-parse", "refs/tags/" + tag) != sha:
                raise ValueError("Public tag was rewritten since freeze")
    print("Pinned public checkouts and frozen tag objects verified; no Git text sent to models.")


def hardware() -> dict:
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Registered run requires one CUDA GPU")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    if "A100" not in name or not 38 * 2**30 <= total <= 42 * 2**30:
        raise RuntimeError("Registered cost comparison requires an A100 40 GB")
    return {
        "gpu": name,
        "total_device_bytes": total,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }


def execute(action: str, capsule: Path, work: Path, output: Path, revision: int) -> None:
    protocol, candidates = verify()
    verify_capsule(capsule)
    if action == "prepare-git":
        prepare_git(capsule, work)
        return
    if not 0 <= revision <= 2:
        raise ValueError("At most initial prompt plus two shared revisions")
    if protocol.get("models"):  # v5 runs only the registered subset of the approved models
        candidates = [c for c in candidates if c["model_id"] in protocol["models"]]
    split = "test" if action in ("test", "test-baseline") else "dev"
    if split == "test":
        frozen = verify_test_freeze(
            output / "prompt_parser.freeze.json", capsule / "index_manifest.json"
        )
        revision = frozen["selected_revision"]
        if action == "test-baseline":
            if revision == 0:
                raise ValueError("The baseline arm is the selected arm; nothing to add")
            revision = 0  # paired baseline: the unchanged v4 prompt on the same test questions
    elif action == "dev" and (output / "prompt_parser.freeze.json").exists():
        raise ValueError("Dev is closed after test unlock; use a new experiment version")
    if action == "smoke":
        revision = 1  # longest prompt plus the reminder message: also checks the chat template
    run_dir = output / (f"smoke_{revision}" if action == "smoke" else f"{split}_{revision}")
    if run_dir.exists():
        raise ValueError("Refusing to overwrite raw run; preserve failed/incomplete artifacts")
    arm_prompt, arm_reminder = BAKEOFF_ARMS[revision]
    if action != "smoke":
        smoke = load(output / "smoke_1/preflight.json")
        if (
            not smoke["passed"]
            or smoke["identity"] != identity()
            or smoke["index_manifest_sha256"] != digest(capsule / "index_manifest.json")
        ):
            raise ValueError("Run 14B preflight with this exact source/index before full matrix")
    run_dir.mkdir(parents=True)
    meta: dict[str, Any] = {
        "split": split,
        "revision": revision,
        "action": action,
        "identity": identity(),
        "prompt": arm_prompt,
        "reminder": arm_reminder,
        "hardware": hardware(),
        "index_manifest_sha256": digest(capsule / "index_manifest.json"),
        "candidate_revisions": {c["model_id"]: c["revision"] for c in candidates},
        "started_utc": datetime.now(UTC).isoformat(),
        "complete": False,
        "cache_policy": "Shared fixed index; BM25/expansion caches cleared per search; no query cache",
        "label_file_decodes": 0,
        "models": {},
    }
    save(run_dir / "run.json", meta)
    from eval.answer_bakeoff_backend import HFAnswerModel, Runtime

    runtime = Runtime(capsule, work)
    questions = question_rows(split)
    random.Random(42).shuffle(questions)
    if action == "smoke":
        candidates = [c for c in candidates if c["model_id"] == "Qwen/Qwen3-14B"]
        questions = [next(q for q in question_rows("dev") if q["suite"] == "code")]
    for candidate in candidates:
        model = None
        slug = candidate["model_id"].replace("/", "__")
        try:
            model = HFAnswerModel(candidate, protocol["generation"])
            meta["models"][candidate["model_id"]] = {
                "load_seconds": model.load_seconds,
                "model_revision": candidate["revision"],
                "config_sha256": model.config_sha256,
                "load_includes_download_if_uncached": True,
                "template_sha256": candidate["chat_template"]["sha256_utf8"],
            }
            # Identical synthetic warmup, no dataset labels, excluded from scored timing.
            warmup = run_question(
                ANSWER_BAKEOFF_WARMUP,
                model,
                tool_schemas(),
                lambda name, args: [],
                protocol["generation"],
                prompt=arm_prompt,
                reminder=arm_reminder,
            )
            save(run_dir / f"{slug}.warmup.json", warmup)
            with (run_dir / f"{slug}.jsonl").open("w", encoding="utf-8") as stream:
                for row in questions:
                    tools = runtime.tools(row["repo"])
                    # Deliberately never pass row/id/type/split/expected tool/labels to model.
                    trace = run_question(
                        row["question"],
                        model,
                        tool_schemas(),
                        tools.execute,
                        protocol["generation"],
                        prompt=arm_prompt,
                        reminder=arm_reminder,
                    )
                    trace.update({k: row[k] for k in ("id", "repo", "suite", "split")})
                    stream.write(canonical(trace) + "\n")
                    stream.flush()
                    if action == "smoke":
                        record = {
                            "candidate": candidate["model_id"],
                            "revision": candidate["revision"],
                            "memory": trace["memory"],
                            "failure": trace["failure"],
                            "passed": trace["failure"] != "model"
                            and any("generation" in t for t in trace["turns"]),
                            "identity": identity(),
                            "index_manifest_sha256": digest(capsule / "index_manifest.json"),
                        }
                        save(run_dir / "preflight.json", record)
                        print(canonical(record))
                        if not record["passed"]:
                            raise RuntimeError(
                                "14B one-question preflight failed; retain failure, stop before full run"
                            )
        except Exception as exc:
            meta["fatal_error_class"] = type(exc).__name__
            if action == "smoke":
                import torch

                save(
                    run_dir / "preflight.json",
                    {
                        "passed": False,
                        "failure": type(exc).__name__,
                        "identity": identity(),
                        "index_manifest_sha256": digest(capsule / "index_manifest.json"),
                        "memory": {
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        },
                    },
                )
            save(run_dir / "run.json", meta)
            raise
        finally:
            if model is not None:
                model.close()
    runtime.close()
    meta["complete"] = True
    meta["completed_utc"] = datetime.now(UTC).isoformat()
    meta["raw_sha256"] = {p.name: digest(p) for p in sorted(run_dir.glob("*.jsonl"))}
    save(run_dir / "run.json", meta)
    print(
        f"Completed {action}: {len(candidates)} candidates x {len(questions)} questions; no labels decoded."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-git", "smoke", "dev", "test", "test-baseline"))
    parser.add_argument("--capsule", type=Path, default=ROOT / "bakeoff")
    parser.add_argument("--work", type=Path, default=ROOT / "data/answer_bakeoff_runtime")
    parser.add_argument("--output", type=Path, default=ROOT / "answer_bakeoff_results")
    parser.add_argument("--revision", type=int, default=0)
    args = parser.parse_args()
    execute(args.action, args.capsule, args.work, args.output, args.revision)


if __name__ == "__main__":
    main()
