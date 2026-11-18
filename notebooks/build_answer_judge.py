"""Build the Phase 7 answer-quality judge notebook (Colab A100) and its Drive ZIP.

    uv run python notebooks/build_answer_judge.py

The ZIP carries only the judge script, its frozen protocol, the question file and the saved
agent answers it grades. The notebook installs vLLM, downloads one quantized judge model,
clones psf/requests at the pinned commit, and runs prepare -> judge -> report.
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

from notebooks.build_colab_indexer import code, md  # noqa: E402

ANSWERS = ROOT / "eval/agent_eval_results/agent_eval_production_20261109T013636.json"
SNAPPED_ANSWERS = (
    ROOT / "eval/agent_eval_results/agent_eval_production_20261109T013636_snapped.json"
)
ARCHIVE = ROOT / "notebooks/answer_judge_bundle.zip"
NOTEBOOK = ROOT / "notebooks/answer_judge_on_colab.ipynb"
SNAPPED_ARCHIVE = ROOT / "notebooks/answer_judge_snapped_bundle.zip"
SNAPPED_NOTEBOOK = ROOT / "notebooks/answer_judge_snapped_on_colab.ipynb"
SNAPPED_NOTE = """

**This is the snapped-citations run.** Same judge, protocol (v2) and 50 answers as the earlier v2
run, but every answer's `path:start-end` references went through `agent/citation_snap.py` first
(6 answers changed). Compare its faithfulness and correctness with the v2 run; the file names
differ (`answer_judge_snapped_*`) so the two runs do not overwrite each other on Drive."""


def bundle_files(answers: Path = ANSWERS) -> list[Path]:
    return [
        ROOT / "eval/judge_answers.py",
        ROOT / "eval/datasets/phase7_answer_judge_v2/protocol.json",
        ROOT / "eval/datasets/agent_eval_v1.jsonl",
        answers,
    ]


BUNDLE_FILES = bundle_files()


def bundle(answers: Path = ANSWERS) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_files(answers)):
            item = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            data = path.read_bytes()
            if path.suffix == ".py":  # data files keep their exact bytes: the answers file
                data = data.replace(b"\r\n", b"\n")  # records the sha256 of the question file
            zf.writestr(item, data)
    return buffer.getvalue()


UNPACK = """
import hashlib, pathlib, zipfile
from google.colab import drive

drive.mount("/content/drive")
source_zip = pathlib.Path(SOURCE_ZIP)
if not source_zip.is_file():
    raise FileNotFoundError(f"Upload answer_judge_bundle.zip to My Drive: {{source_zip}}")
if hashlib.sha256(source_zip.read_bytes()).hexdigest() != "{digest}":
    raise ValueError("The ZIP does not match this notebook. Download both again.")
PAYLOAD = pathlib.Path("/content/judge_payload")
with zipfile.ZipFile(source_zip) as package:
    package.extractall(PAYLOAD)
print("Unpacked", source_zip.name)
"""

INSTALL = """
import importlib.util, pathlib, subprocess, sys
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)
if importlib.util.find_spec("uv") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
# vLLM goes into its own virtual environment, never into Colab's system Python: its torch build
# clashed with Colab's preinstalled torchaudio (CUDA 13.2 vs 13.0) the first time. The judge
# script runs with this environment's Python; the notebook's own packages stay untouched.
VENV = pathlib.Path("/content/vllm_env")
VENV_PY = str(VENV / "bin" / "python")
if not VENV.exists():
    subprocess.run([sys.executable, "-m", "uv", "venv", str(VENV), "--python", sys.executable],
                   check=True)
subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", VENV_PY, "-q",
                "--torch-backend=" + TORCH_BACKEND, VLLM_SPEC], check=True)
# Fail here, in seconds, if vLLM cannot be imported, not after the model download.
subprocess.run([VENV_PY, "-c",
                "import vllm, torch, torchaudio; from vllm import LLM, SamplingParams; "
                "print('vllm', vllm.__version__, '| torch', torch.__version__, "
                "'| torchaudio', torchaudio.__version__)"], check=True)
"""

PREFLIGHT = """
# Everything vLLM needs on this machine, checked in seconds, before the 21 GB download.
import json, shutil, subprocess, sys

probe = (
    "import json, sysconfig, pathlib, torch; "
    "free, total = torch.cuda.mem_get_info(); "
    "print(json.dumps({'gpu': torch.cuda.get_device_name(0), "
    "'capability': list(torch.cuda.get_device_capability(0)), "
    "'free_gib': free / 2**30, 'total_gib': total / 2**30, "
    "'python_h': (pathlib.Path(sysconfig.get_paths()['include']) / 'Python.h').exists(), "
    "'torch_cuda': torch.version.cuda}))"
)
info = json.loads(subprocess.run([VENV_PY, "-c", probe], capture_output=True, text=True,
                                 check=True).stdout.strip().splitlines()[-1])
disk_free = shutil.disk_usage("/content").free / 2**30
ram_gib = int(open("/proc/meminfo").readline().split()[1]) / 2**20
need_gpu = protocol_runtime_util * info["total_gib"]
rows = [
    ("GPU", info["gpu"] + " capability " + str(info["capability"]), info["capability"] >= [8, 0]),
    ("GPU memory free", "%.1f of %.1f GiB (vLLM wants %.1f)" % (info["free_gib"],
     info["total_gib"], need_gpu), info["free_gib"] >= need_gpu),
    ("disk free", "%.0f GiB (need about 45)" % disk_free, disk_free >= 45),
    ("CPU RAM", "%.0f GiB (need about 24)" % ram_gib, ram_gib >= 24),
    ("torch CUDA", info["torch_cuda"], True),
]
for name, value, ok in rows:
    print(("ok   " if ok else "FAIL ") + name + ": " + str(value))
soft = [
    ("Python.h (Triton builds its launcher with it)", info["python_h"],
     "apt-get install -y python3-dev"),
    ("gcc", shutil.which("gcc") is not None, "apt-get install -y gcc"),
]
for name, ok, fix in soft:
    print(("ok   " if ok else "WARN ") + name + ("" if ok else ": missing; try " + fix))
if not all(ok for _, _, ok in rows):
    raise RuntimeError("Preflight failed: see the FAIL lines above.")
"""

MODEL = """
# Quantized weights only. The judge script refuses a model with no quantization_config.
import json, pathlib
from huggingface_hub import snapshot_download, model_info

info = model_info(MODEL_ID)
print("repo:", MODEL_ID, "| revision:", info.sha)
MODEL_PATH = pathlib.Path(snapshot_download(MODEL_ID, revision=info.sha))
config = json.loads((MODEL_PATH / "config.json").read_text())
quant = config.get("quantization_config")
if not quant:
    raise SystemExit(MODEL_ID + " has no quantization_config. Pick a quantized repo.")
print("quantization:", quant.get("quant_method"), "| size on disk (GB):",
      round(sum(f.stat().st_size for f in MODEL_PATH.rglob("*") if f.is_file()) / 1e9, 1))
"""

RUN_BODY = """
import os, pathlib, shutil, subprocess, sys

REPO_DIR = pathlib.Path("/content/judge_requests")
RESULTS = pathlib.Path("/content/judge_results")
if RESULTS.exists():
    raise FileExistsError("Results exist. Download them first, or use a fresh runtime.")
protocol = json.loads((PAYLOAD / "eval/datasets/phase7_answer_judge_v2/protocol.json").read_text())
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    "https://github.com/psf/requests", str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", protocol["repo_commit_sha"]],
               check=True)
env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTHONUTF8": "1",
       "VLLM_USE_FLASHINFER_SAMPLER": "0",
       "PATH": str(VENV / "bin") + os.pathsep + os.environ["PATH"]}
script = str(PAYLOAD / "eval/judge_answers.py")
common = ["--out", str(RESULTS)]

def step(*args):
    done = subprocess.run([VENV_PY, script, *args], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(done.stdout[-6000:])  # the child's own output, which a bare subprocess hides in Colab
    if done.returncode:
        raise RuntimeError(args[0] + " failed (exit " + str(done.returncode) + ")")

step("prepare", "--answers", str(PAYLOAD / "eval/agent_eval_results" / ANSWERS_NAME),
     "--dataset", str(PAYLOAD / "eval/datasets/agent_eval_v1.jsonl"),
     "--checkout", str(REPO_DIR), *common)
step("judge", "--model", str(MODEL_PATH), "--model-id", MODEL_ID, *common)
step("report", *common)
shutil.copy(PAYLOAD / "eval/datasets/phase7_answer_judge_v2/protocol.json",
            RESULTS / "protocol.json")
(RESULTS / "judge_revision.txt").write_text(MODEL_PATH.name + chr(10))
"""

# The pipeline runs inside try/except so the next cell (save, then disconnect) always runs,
# even when a step fails: a failed run must not keep an A100 attached.
_RUN_HEAD = """import json, pathlib, traceback
RUN_ERROR = None
RESULTS = pathlib.Path('/content/judge_results')
def pipeline():
"""
_RUN_TAIL = """try:
    pipeline()
except BaseException:
    RUN_ERROR = traceback.format_exc()
    print(RUN_ERROR)
"""
RUN = (
    _RUN_HEAD
    + "".join(
        ("    " + line if line.strip() else "") + chr(10) for line in RUN_BODY.strip().splitlines()
    )
    + _RUN_TAIL
)

DOWNLOAD = """
import time, shutil, zipfile
from google.colab import files, runtime

RESULTS.mkdir(exist_ok=True)
if RUN_ERROR:
    (RESULTS / "RUN_ERROR.txt").write_text(RUN_ERROR)
archive = pathlib.Path("/content/answer_judge_results.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(RESULTS))
# Keep a copy on Drive, so the results survive the disconnect even if the browser download fails.
shutil.copy(archive, "/content/drive/MyDrive/answer_judge_results.zip")
files.download(str(archive))
print("FAILED, partial results saved; see RUN_ERROR.txt in the zip." if RUN_ERROR
      else "Done. Results also copied to My Drive/answer_judge_results.zip.")
time.sleep(60)  # let the browser finish the download before the runtime goes away
runtime.unassign()  # release the A100
"""


def build(answers: Path = ANSWERS) -> dict:
    digest = hashlib.sha256(bundle(answers)).hexdigest()
    return {
        "cells": [
            md("""# Phase 7 answer-quality check (LLM judge)

1. Upload **answer_judge_bundle.zip** to the top level of My Drive.
2. Choose an **A100 GPU** and Run all (approve Drive). Expect 20 to 40 minutes, mostly the
   model download and vLLM start-up; the 194 judge prompts take a few minutes.
3. Return **answer_judge_results.zip**. The last cell saves it to My Drive too, downloads it, waits
   60 seconds and then **disconnects the runtime**, also when a step failed (partial results and the
   error are in the zip).

**What it does.** A quantized open-source model (Qwen3.8-27B; you asked for "Qwen3.7-28B", which
does not exist, and this is the closest) grades the 52 saved `code` and `docs` answers from one
agent run on psf/requests: is each answer **faithful** to the lines it cites, and **correct**
against the reference evidence. Evidence is read from the repository at the pinned commit, not
produced by a model. 96 negative controls (an answer paired with another question's evidence)
must be rejected at 90% or better, otherwise the judge is not trusted and no answer-quality number
is recorded. The protocol is frozen in `eval/datasets/phase7_answer_judge_v2/protocol.json`.

Not measured: agreement with a person, and self-preference (judge and answer model are both Qwen).
Docs correctness is judged against the sections most about the question (v2), picked by a
keyword score."""),
            code(
                'SOURCE_ZIP = "/content/drive/MyDrive/answer_judge_bundle.zip"\n'
                f'ANSWERS_NAME = "{answers.name}"\n'
                "# 4-bit weights (about 21 GB, third-party Qwen3.8-27B build); vLLM runs them\n"
                "# with Marlin kernels on an A100. The official Qwen/Qwen3.8-27B-FP8 is\n"
                "# block-wise FP8 and an A100 has no FP8 hardware: not the default (untested).\n"
                'MODEL_ID = "cyankiwi/Qwen3.8-27B-AWQ-INT4"\n'
                'VLLM_SPEC = "vllm==0.33.0"  # the version of the first Colab try\n'
                "# cu130 resolved cleanly against vllm 0.33.0 (checked with uv pip compile);\n"
                "# cu132 has no solution for it.\n"
                'TORCH_BACKEND = "cu130"\n'
            ),
            code(UNPACK.format(digest=digest)),
            md("## Install vLLM"),
            code(INSTALL),
            md("## Check the machine"),
            code(
                "import json\n"
                "PROTOCOL_JSON = PAYLOAD / 'eval/datasets/phase7_answer_judge_v2/protocol.json'\n"
                "protocol_runtime_util = json.loads(PROTOCOL_JSON.read_text())['runtime'][\n"
                "    'gpu_memory_utilization']" + PREFLIGHT
            ),
            md("## Download the quantized judge"),
            code(MODEL),
            md("## Prepare, judge, report"),
            code(RUN),
            md("## Download results"),
            code("import pathlib, zipfile\n" + DOWNLOAD),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def snapped_build() -> dict:
    """The same notebook for the snapped answers, with its own file names on Drive."""
    nb = build(SNAPPED_ANSWERS)
    renames = {
        "answer_judge_bundle.zip": "answer_judge_snapped_bundle.zip",
        "answer_judge_results.zip": "answer_judge_snapped_results.zip",
    }
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        for old, new in renames.items():
            text = text.replace(old, new)
        if cell["cell_type"] == "markdown" and text.startswith("# Phase 7 answer-quality"):
            text = text.replace("(LLM judge)", "(LLM judge), snapped citations", 1) + SNAPPED_NOTE
        cell["source"] = text.splitlines(keepends=True)
    return nb


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def render_snapped() -> str:
    return json.dumps(snapped_build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    ARCHIVE.write_bytes(bundle())
    NOTEBOOK.write_text(render(), encoding="utf-8")
    SNAPPED_ARCHIVE.write_bytes(bundle(SNAPPED_ANSWERS))
    SNAPPED_NOTEBOOK.write_text(render_snapped(), encoding="utf-8")
    for f in (NOTEBOOK, ARCHIVE, SNAPPED_NOTEBOOK, SNAPPED_ARCHIVE):
        print(f"{f.name}: {f.stat().st_size} bytes")
