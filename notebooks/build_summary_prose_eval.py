"""Build the E3 Colab notebook after prepare_summary_prose.py packages E2 inputs."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.build_colab_indexer import (  # noqa: E402
    SETUP_TEMPLATE,
    UNPACK_TEMPLATE,
    code,
    md,
    runtime_pins,
    source_archive,
)

NOTEBOOK = ROOT / "notebooks/summary_prose_on_colab.ipynb"
CAPSULE = ROOT / "notebooks/summary_prose_inputs.zip"

PARAMETERS = """
# The user approved this exact model for the E3 run in chat on 2026-11-08 (revision
# cdbee75f17c01a7cc42f958dc650907174af0554, pinned in the capsule). A different model needs
# a new approval.
APPROVED_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
if APPROVED_MODEL != "Qwen/Qwen3-4B-Instruct-2507":
    raise RuntimeError("User approval of the proposed summary model is required before this run")
"""

PAYLOAD = """
import base64, hashlib, io, pathlib, zipfile
packed = base64.b64decode("__ARCHIVE__")
assert hashlib.sha256(packed).hexdigest() == "__DIGEST__"
with zipfile.ZipFile(io.BytesIO(packed)) as archive:
    for info in archive.infolist():
        destination = pathlib.Path("/content") / info.filename
        assert destination.resolve().is_relative_to(pathlib.Path("/content"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(info))
print("Frozen facts, labels and code/query vectors unpacked; labels unreviewed by a person.")
"""

RUN = """
import os, subprocess, sys
env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + "/content",
       "REPO_RAG_EMBEDDING_DTYPE": "float32", "PYTHONUNBUFFERED": "1"}
subprocess.run([sys.executable, "/content/eval/run_summary_prose.py",
                "--approved-model", APPROVED_MODEL,
                "--output", "/content/summary_prose_results"], env=env, check=True)
"""

EXPORT = """
import shutil
from google.colab import files
path = shutil.make_archive(
    "/content/summary_prose_results", "zip", "/content/summary_prose_results")
files.download(path)
print("Keep this ZIP: raw paragraphs, cached outputs, violations, rankings and metrics.")
"""


def build() -> dict:
    raw = CAPSULE.read_bytes()
    src, sha = source_archive()
    cells = [
        md("""# Architecture summaries: E3 model prose (approval required)

Architecture labels are **unreviewed by a person**. E1/E2 selected the router and facts
on dev only; this notebook makes no further selection. The model sees only extracted facts.
Every identifier-like token must occur in those facts or the frozen connecting-word list.
Violating paragraphs are dropped; raw outputs and the rejection rate are preserved. This
mechanical check does not establish semantic entailment. No model judge is used.

Proposed: **Qwen3-4B-Instruct-2507**, 4.0B, Apache-2.0. Estimated 8.05 GB bf16 weights;
plan 12-16 GB VRAM including runtime and short-context cache (estimate, not measured).
Use an L4 or A100 GPU. The revision is pinned in the capsule. No new dependency.
The first code cell requires explicit model approval before installing or loading anything.
After approval: set APPROVED_MODEL, run all, and retain the downloaded results ZIP.
"""),
        code(PARAMETERS),
        code(SETUP_TEMPLATE.format(pins=" ".join(f'"{p}"' for p in runtime_pins()))),
        code(UNPACK_TEMPLATE.format(archive=src, digest=sha)),
        code(
            PAYLOAD.replace("__ARCHIVE__", base64.b64encode(raw).decode()).replace(
                "__DIGEST__", hashlib.sha256(raw).hexdigest()
            )
        ),
        code(RUN),
        code(EXPORT),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    NOTEBOOK.write_text(render(), encoding="utf-8")
    print(f"wrote {NOTEBOOK.name} ({NOTEBOOK.stat().st_size / 1024:.0f} KB)")
