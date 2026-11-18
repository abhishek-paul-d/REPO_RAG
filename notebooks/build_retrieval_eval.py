"""Build a compact Phase 4 Colab notebook and its separate Google Drive source ZIP."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.build_colab_indexer import code, md, runtime_pins  # noqa: E402

NOTEBOOK = ROOT / "notebooks/retrieval_eval_on_colab.ipynb"
SOURCE_ARCHIVE = ROOT / "notebooks/phase4_source.zip"


def archive_files() -> list[Path]:
    return sorted(
        [
            *ROOT.joinpath("src/repo_rag").rglob("*.py"),
            *ROOT.joinpath("eval/retrieval").rglob("*.py"),
            *ROOT.joinpath("eval/datasets").glob("*_qa.jsonl"),
            ROOT / "eval/__init__.py",
            ROOT / "eval/run_retrieval_eval.py",
        ]
    )


def source_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in archive_files():
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes().replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def pins() -> list[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    version = next(p["version"] for p in lock["package"] if p["name"] == "astchunk")
    return [*runtime_pins(), f"astchunk=={version}"]


INTRO = """
# Phase 4: does AST chunking improve retrieval?

This notebook runs the project's real chunkers, Qwen3 embeddings, Chroma, BM25 and weighted
RRF on **40 Requests and 40 Express questions**, at pinned repository revisions. It includes
development and held-out test splits. It does not need an LLM or a GitHub token.

1. Upload **phase4_source.zip** to the top level of **My Drive** in Google Drive.
2. Open this notebook in Colab. Runtime → Change runtime type → select a **GPU**.
   The first full run is verified on an A100 40 GB; other GPU types have not been measured.
3. Runtime → Run all, and approve Colab's request to mount your Drive. The notebook loads
   the ZIP from Drive, verifies it matches this notebook, and extracts the project locally.
4. Keep the session open until `phase4_results.zip` downloads, then bring it back to the chat.

If you put the source ZIP in another Drive folder, change `SOURCE_ZIP` in the Parameters cell.
Download the notebook and source ZIP from the same generation. The notebook contains no
embedded project archive or long encoded payload.

The primary comparison runs four chunkers, with and without identical contextual headers.
The reference `astchunk` does not support JavaScript: Express reports that row as N/A.
The optional final experiment reranks the current AST/header-on candidates at depths 20 and
50, using a code-specific instruction. It downloads the 4B reranker and takes extra GPU time.

No scores are known in advance. A result favouring character chunks is a valid outcome.
Results include overlap, union-of-gold-line coverage and ≥90%-complete gold ranges, never our
AST chunker's symbol metadata. Reports include all 40 questions for fixed comparisons plus
separate dev/test tables and paired evidence-group bootstrap intervals. Selected settings are
assessed on test only; pooled summaries are descriptive. Intervals are pointwise/exploratory,
and the character versus non-whitespace budget confound still applies.
"""

PARAMETERS = """
SOURCE_ZIP = "/content/drive/MyDrive/phase4_source.zip"
RUN_RERANK = True  # Set False to skip the 4B reranker experiment on both repos.
DTYPE = "float32"  # Match CPU laptop embeddings.
"""

UNPACK = """
# Load the separate source ZIP that you uploaded to Google Drive.
import hashlib, pathlib, zipfile
from google.colab import drive

drive.mount("/content/drive")
source_zip = pathlib.Path(SOURCE_ZIP)
if not source_zip.is_file():
    raise FileNotFoundError(
        f"Upload phase4_source.zip to My Drive, or update SOURCE_ZIP. Missing: {{source_zip}}"
    )
expected_sha256 = "{digest}"
if hashlib.sha256(source_zip.read_bytes()).hexdigest() != expected_sha256:
    raise ValueError(
        "Source ZIP does not match this notebook. Download the matching ZIP and notebook again."
    )
PAYLOAD = pathlib.Path("/content/phase4_payload")
with zipfile.ZipFile(source_zip) as package:
    package.extractall(PAYLOAD)
print("Loaded project code and question datasets from", source_zip.name)
"""

RUN = """
import json, os, subprocess, sys, time

RESULTS = pathlib.Path("/content/phase4_results")
REPOS = pathlib.Path("/content/phase4_repos")
REPOS.mkdir(exist_ok=True)
ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join([str(PAYLOAD), str(PAYLOAD / "src")]),
    "REPO_RAG_DATA_DIR": "/content/phase4_unused_production_data",
    "REPO_RAG_COMPUTE_DEVICE": "cuda",
    "REPO_RAG_EMBEDDING_DTYPE": DTYPE,
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

def run_eval(repo, checkout, *extra, output=RESULTS):
    subprocess.run([
        sys.executable, str(PAYLOAD / "eval/run_retrieval_eval.py"),
        "--repo", repo, "--checkout", str(checkout), "--output", str(output), *extra,
    ], env=ENV, check=True)

checkouts = {}
for dataset in sorted((PAYLOAD / "eval/datasets").glob("*_qa.jsonl")):
    first = json.loads(dataset.read_text().splitlines()[0])
    repo, sha = first["repo"], first["commit_sha"]
    checkout = REPOS / repo.replace("/", "__")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/" + repo, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", sha], check=True)
    checkouts[repo] = checkout
    run_eval(repo, checkout, "--mode", "validate")

started = time.perf_counter()
for repo, checkout in checkouts.items():
    print("\\nFull comparison:", repo, flush=True)
    run_eval(repo, checkout, "--mode", "full")
print(f"Main comparison elapsed: {time.perf_counter() - started:.1f} seconds")
"""

RERANK = """
if RUN_RERANK:
    started = time.perf_counter()
    for repo, checkout in checkouts.items():
        print("\\nReranker depth experiment:", repo, flush=True)
        run_eval(repo, checkout, "--mode", "full", "--strategies", "ast", "--headers", "on",
                 "--rerank", output=RESULTS / "rerank_experiment")
    print(f"Reranker experiment elapsed: {time.perf_counter() - started:.1f} seconds")
else:
    print("Reranking skipped; its code-search effect remains unmeasured.")
"""

DOWNLOAD = """
# Export human-readable reports, per-question rankings and portable vector caches.
# Exclude Chroma's platform-specific binary index.
from google.colab import files
archive_path = pathlib.Path("/content/phase4_results.zip")
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and "isolated_index" not in path.parts:
            bundle.write(path, path.relative_to(RESULTS))
print(f"Results archive: {archive_path.stat().st_size / 1e6:.1f} MB")
for path in sorted(RESULTS.rglob("summary.md")):
    print(path.relative_to(RESULTS))
files.download(str(archive_path))
"""


def build() -> dict:
    digest = hashlib.sha256(source_archive()).hexdigest()
    setup = """import importlib.util, subprocess, sys
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)
if importlib.util.find_spec("uv") is None:
    # Bootstrap the installer once if this Colab runtime does not include it.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
# uv installs the pinned dependencies into the same Python environment as this notebook.
subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--system",
                "--python", sys.executable, "-q", *PINS], check=True)
""".replace("*PINS", repr(pins())[1:-1])
    return {
        "cells": [
            md(INTRO),
            md("## Parameters"),
            code(PARAMETERS),
            md("## Load project ZIP from Google Drive"),
            code(UNPACK.format(digest=digest)),
            md("## Install dependencies with uv"),
            code(setup),
            md("## Validate and compare"),
            code(RUN),
            md("## Optional reranker experiment"),
            code(RERANK),
            md("## Download results"),
            code(DOWNLOAD),
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


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    SOURCE_ARCHIVE.write_bytes(source_archive())
    NOTEBOOK.write_text(render(), encoding="utf-8")
    print(f"Wrote {NOTEBOOK} ({NOTEBOOK.stat().st_size / 1024:.0f} KB)")
    print(f"Wrote {SOURCE_ARCHIVE} ({SOURCE_ARCHIVE.stat().st_size / 1024:.0f} KB)")
