"""Generate the Phase 7 v5 A100 notebook; never instantiate an answer model.

Run prepare_answer_bakeoff_capsule.py first. Rebuild after source/project/lock changes. Source
bytes are fixed once the dev stage has run: the freeze refuses any change, so if a later edit to
`src/` is unavoidable, vendor the dev-time bytes of the changed files under PINNED (v4 needed
this for two files) instead of letting the bundle follow the live files.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.build_colab_indexer import SETUP_TEMPLATE, code, md, runtime_pins  # noqa: E402

NOTEBOOK = ROOT / "notebooks/answer_bakeoff_on_colab.ipynb"
CAPSULE = ROOT / "notebooks/answer_bakeoff_inputs.zip"
BUNDLE = ROOT / "notebooks/answer_bakeoff_v5_bundle.zip"
# Source bytes the dev runs used for files edited later. The freeze refuses any source
# difference from dev, so the bundle carries these exact bytes, not the live files.
PINNED = ROOT / "notebooks/pinned_dev_sources"


def source_payload() -> bytes:
    paths = [
        *ROOT.joinpath("src/repo_rag").rglob("*.py"),
        *ROOT.joinpath("eval").glob("*answer_bakeoff*.py"),
        ROOT / "eval/run_facts.py",
        ROOT / "eval/__init__.py",
        ROOT / "eval/build_dev_extra_facts.py",
        ROOT / "eval/build_v4_fresh_facts.py",
        ROOT / "eval/build_v5_targets.py",
        ROOT / "eval/build_v5_questions.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            name = path.relative_to(ROOT).as_posix()
            pinned = PINNED / name
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (pinned if pinned.exists() else path).read_bytes())
    return result.getvalue()


UNPACK = """
import hashlib, io, pathlib, zipfile, os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
ROOT = pathlib.Path('/content')
# Upload notebooks/answer_bakeoff_v5_bundle.zip (from the project) to this Drive folder first.
BUNDLE = pathlib.Path('/content/drive/MyDrive/answer_bakeoff_v5_bundle.zip')
EXPECTED_SHA256 = '__BUNDLE_SHA__'
assert BUNDLE.exists(), 'Upload answer_bakeoff_v5_bundle.zip to the top of My Drive'
packed = BUNDLE.read_bytes()
assert hashlib.sha256(packed).hexdigest() == EXPECTED_SHA256, 'Bundle differs from this notebook'
with zipfile.ZipFile(io.BytesIO(packed)) as archive:
    for name in archive.namelist():
        target = (ROOT / name).resolve()
        assert target.is_relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
ENV = {**os.environ, 'PYTHONPATH': '/content/src:/content',
       'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
       'GIT_TERMINAL_PROMPT': '0'}
def command(script, *args):
    # A notebook cell shows only the kernel's own output, so capture the subprocess and print
    # the end of its stderr when it fails; v4 hid every real error behind CalledProcessError.
    done = subprocess.run([sys.executable, str(ROOT / 'eval' / script), *map(str, args)],
                          env=ENV, cwd=ROOT, capture_output=True, text=True)
    print(done.stdout[-3000:])
    if done.returncode:
        print(done.stderr[-4000:])
        raise SystemExit(f'{script} {args[0]} failed with exit status {done.returncode}')
command('prepare_answer_bakeoff_v5.py', 'verify')
print('v1-v5 frozen inputs and approval verified; no model has been loaded yet.')
"""

PARAMETERS = """
# Both models are already approved in the embedded approval records. No keys or hosted
# inference APIs are used. Public Hugging Face weights download here only.
RESULTS = pathlib.Path('/content/drive/MyDrive/repo_rag_answer_bakeoff_v5')
RESULTS.mkdir(parents=True, exist_ok=True)
# Run all stops before the freeze. After the dev results are reviewed, set this to True.
CONFIRM_FREEZE = False
# Save to Drive, then disconnect and delete this runtime when a stage ends (saves compute units).
AUTO_DISCONNECT = True
ARMS = (0, 1, 2)  # 0 = unchanged v4 prompt, 1 = R2, 2 = R3 (agent/prompts.py BAKEOFF_ARMS)
COMMON = ['--output', RESULTS]

import json

def run_once(action, revision=0):
    # Run a stage unless it already finished (so Run all can resume after the dev review).
    if action == 'freeze':
        if (RESULTS / 'prompt_parser.freeze.json').exists():
            return print('freeze already saved; skipping')
        return command('analyze_answer_bakeoff.py', 'freeze', '--output', RESULTS)
    if action == 'test':
        freeze = json.loads((RESULTS / 'prompt_parser.freeze.json').read_text())
        revision = freeze['selected_revision']
    if action == 'test-baseline':
        revision = 0
        frozen = json.loads((RESULTS / 'prompt_parser.freeze.json').read_text())
        if frozen['selected_revision'] == 0:
            return print('the selected arm is the baseline; nothing to add')
    folder = RESULTS / ('smoke_1' if action == 'smoke' else f'{action.split("-")[0]}_{revision}')
    if folder.exists():
        done = (folder / 'preflight.json') if action == 'smoke' else (folder / 'run.json')
        info = json.loads(done.read_text()) if done.exists() else {}
        if info.get('passed') or info.get('complete'):
            return print(f'{folder.name} already finished; skipping')
        raise RuntimeError(f'{folder.name} is incomplete; keep it and report it, do not rerun')
    command('run_answer_bakeoff.py', action, *COMMON, '--revision', revision)

def finish_runtime():
    # Everything is already on Drive: flush it, then disconnect and delete this runtime.
    if AUTO_DISCONNECT:
        drive.flush_and_unmount()
        print('Drive flushed. Disconnecting and deleting the runtime.')
        from google.colab import runtime
        runtime.unassign()
"""

EXPORT = """
import shutil
from google.colab import files
# Keep raw failures and all turns. Recompute does not need model weights or a GPU.
path = shutil.make_archive('/content/answer_bakeoff_results', 'zip', RESULTS)
shutil.copyfile(path, '/content/drive/MyDrive/repo_rag_answer_bakeoff_v5_results.zip')
print('Saved MyDrive/repo_rag_answer_bakeoff_v5_results.zip. No default has been changed.')
if not AUTO_DISCONNECT:
    files.download(path)
"""


def bundle_payload() -> bytes:
    """One deterministic ZIP (source + capsule) for Drive; keeps base64 out of the notebook."""
    result = io.BytesIO()
    seen: dict[str, bytes] = {}
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for payload in (source_payload(), CAPSULE.read_bytes()):
            with zipfile.ZipFile(io.BytesIO(payload)) as part:
                for name in sorted(part.namelist()):
                    data = part.read(name)
                    if name in seen:
                        assert seen[name] == data, f"Conflicting bundle entry {name}"
                        continue
                    seen[name] = data
                    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
    return result.getvalue()


def build() -> dict:
    unpack = UNPACK.replace("__BUNDLE_SHA__", hashlib.sha256(bundle_payload()).hexdigest())
    cells = [
        md("""# Phase 7 answer-model bake-off v5: a tuned prompt for Qwen2.5-Coder-14B

Upload `answer_bakeoff_v5_bundle.zip` to the top level of My Drive (no folder), replacing any
older bundle. Use **one A100 40 GB**, then Run all (it stops before the freeze). This notebook is
generated from source; do not edit the uploaded bundle. v1-v4 stay frozen.

v4 chose no model. Qwen2.5-Coder-14B cited 14 of 44 answerable questions and abstained on 6; this
experiment tests whether a tuned prompt fixes that without losing its refusal of unanswerable
questions. Two models only (the subject Coder-14B and the control Qwen3-14B), three prompt arms
(revision 0 = the unchanged v4 prompt, 1 = R2, 2 = R3), all on questions nobody has seen: dev has
15 answerable + 6 probes, test has 45 answerable + 16 probes, plus the 45 v4 test facts as a
regression guard (seen, so they can detect a loss but cannot support a claim).

Order: verify capsules -> fetch pinned Git/tag objects -> one 14B preflight logging the
peak GPU memory of a dev code question (VRAM estimates stay unmeasured until it runs) -> dev for the
three arms and both models -> stop. After you review the dev results, set `CONFIRM_FREEZE = True`
and Run all again: it freezes the dev-selected arm, runs test once for that arm, and once more for
the unchanged baseline, then rebuilds every table from the raw transcripts.
Only the question string, a fixed system prompt, a fixed reminder after search results and the
tool schemas reach a model. No hidden id, split, label or gold path is passed.

Gates (subject, test): at least 43/45 facts, at most 5% invalid citation references, at least
14/16 probes, at most 4/45 answerable abstentions, at least 34/45 answerable with a valid
citation in the raw output. The answer check (attaching `path:start-end` for retrieved chunks the
answer names) is reported separately. If a gate fails nothing is adopted or relaxed.

The architecture split is skewed to Flask (the unused requests pool was small), several
questions touch the same functions as others, and architecture and probe labels are **unreviewed
by a person**. Citation validity does not prove correctness. A100 timings are experiment costs.
Do not edit any source file after the dev stage: the freeze refuses a different implementation.
"""),
        code(SETUP_TEMPLATE.format(pins=" ".join(f'"{p}"' for p in runtime_pins()))),
        code(unpack),
        code(PARAMETERS),
        md(
            "Pinned Git history is needed for typed blame/release tools. "
            "No raw Git text is model context."
        ),
        code("command('run_answer_bakeoff.py', 'prepare-git', *COMMON)"),
        md("14B memory preflight, logged on Drive **before** the full matrix. Failure stops here."),
        code("run_once('smoke')"),
        md(
            "Dev only: both models on all three prompt arms, same seeded order and budgets. "
            "Each arm is its own stage and is skipped on a rerun if it already finished."
        ),
        code("for arm in ARMS:\n    run_once('dev', arm)"),
        md("Dev results are on Drive. Until the freeze is confirmed this ends the runtime."),
        code("if not CONFIRM_FREEZE:\n    finish_runtime()"),
        md(
            "**Stop here.** Review the dev results first. Then set `CONFIRM_FREEZE = True` in the "
            "parameters cell and Run all (finished stages skip)."
        ),
        code(
            "assert CONFIRM_FREEZE, 'Dev review pending: set CONFIRM_FREEZE = True to freeze and "
            "unlock test'"
        ),
        code("run_once('freeze')"),
        md(
            "The saved freeze binds dev transcripts, the prompt arms, parser, source and index "
            "hashes. Only now may test labels be decoded for scoring."
        ),
        code("run_once('test')"),
        md("The unchanged-prompt baseline on the same test questions (skipped if selected)."),
        code("run_once('test-baseline')"),
        code("command('analyze_answer_bakeoff.py', 'report', '--output', RESULTS)"),
        code(EXPORT),
        code("finish_runtime()"),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    BUNDLE.write_bytes(bundle_payload())
    NOTEBOOK.write_text(render(), encoding="utf-8")
    print(f"Wrote {NOTEBOOK.name} and {BUNDLE.name}")
