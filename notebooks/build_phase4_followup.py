"""Build the fixed v2 matrix notebook and a separate Google Drive source ZIP."""

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
from notebooks.build_retrieval_eval import UNPACK, archive_files, pins  # noqa: E402

NOTEBOOK = ROOT / "notebooks/phase4_followup_on_colab.ipynb"
SOURCE_ARCHIVE = ROOT / "notebooks/phase4_followup_source.zip"


def source_archive() -> bytes:
    paths = sorted(
        set(
            [
                *archive_files(),
                ROOT / "eval/run_followup_eval.py",
                ROOT / "eval/summarize_followup.py",
                *ROOT.joinpath("eval/datasets/phase4_v2").glob("*.json*"),
            ]
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            item = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            item.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(item, path.read_bytes().replace(b"\r\n", b"\n"))
    return buffer.getvalue()


INTRO = """
# Phase 4 follow-up: fixed comparison on new questions and Flask

Upload **phase4_followup_source.zip** to the top level of My Drive. Select an **A100 GPU**,
run all cells, approve Drive mounting, and keep the session open until the results ZIP downloads.
Use this notebook and ZIP as a matching pair. There is no embedded archive or Base64 cell.

This runs 49 newly authored questions plus 80 frozen-question regressions across Requests,
Express and Flask. The new labels distinguish required from supporting evidence. They are
source-checked by the author; independent label review is still pending. New symbol lookups
can target previously evaluated implementations, so newly authored is not fully independent.

The matrix is fixed: dense, current hybrid, rerank20 and a reranked 20-candidate union reserving
five BM25 slots. Penalty and symbol routing are diagnostics, not selected production policies.
No weights are tuned. It logs full lists/scores, reranker pairs, separate v1 scores, actual warm
p50/p95 timings, cold first rerank and CUDA memory. Diagnostic top50 scores explain movements
for requests-036 and express-028. Repetitions do not increase the number of independent questions.
The primary comparison uses production chunk headers. The original common-header study remains
a separate benchmark, so its scores must not be treated as a direct before/after improvement.

Bring **phase4_followup_results.zip** back for review. This experiment never changes production
defaults or marks Phase 4 complete automatically.
"""

RUN = """
import json, os, pathlib, subprocess, sys, time
REPOS = pathlib.Path('/content/phase4_followup_repos')
RESULTS = pathlib.Path('/content/phase4_followup_results')
REPOS.mkdir(exist_ok=True)
env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(PAYLOAD), str(PAYLOAD / 'src')]),
       'REPO_RAG_DATA_DIR': '/content/unused_production_data', 'REPO_RAG_COMPUTE_DEVICE': 'cuda',
       'REPO_RAG_EMBEDDING_DTYPE': 'float32', 'PYTHONUNBUFFERED': '1',
       'TOKENIZERS_PARALLELISM': 'false'}
manifest = json.loads((PAYLOAD / 'eval/datasets/phase4_v2/manifest.json').read_text())
started = time.perf_counter()
for entry in manifest['datasets']:
    repo, sha = entry['repo'], entry['commit_sha']
    checkout = REPOS / repo.replace('/', '__')
    if not checkout.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/' + repo, str(checkout)], check=True)
    subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', sha], check=True)
    dataset = PAYLOAD / 'eval/datasets/phase4_v2' / entry['file']
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Dataset hash mismatch: ' + entry['file'])
    command = [sys.executable, str(PAYLOAD / 'eval/run_followup_eval.py'), '--repo', repo,
               '--checkout', str(checkout), '--dataset', str(dataset), '--output', str(RESULTS)]
    subprocess.run(command + ['--mode', 'validate'], env=env, check=True)
    subprocess.run(command + ['--mode', 'full'], env=env, check=True)
print(f'Full matrix elapsed: {time.perf_counter() - started:.1f} seconds')
subprocess.run([sys.executable, str(PAYLOAD / 'eval/summarize_followup.py'),
                '--results', str(RESULTS)], env=env, check=True)
"""

DOWNLOAD = """
from google.colab import files
archive = pathlib.Path('/content/phase4_followup_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file() and 'isolated_index' not in path.parts:
            bundle.write(path, path.relative_to(RESULTS))
print(f'Results: {archive.stat().st_size / 1e6:.1f} MB')
files.download(str(archive))
"""


def build() -> dict:
    setup = """import importlib.util, subprocess, sys
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system', '--python',
                sys.executable, '-q', *PINS], check=True)
""".replace("*PINS", repr(pins())[1:-1])
    unpack = (
        UNPACK.format(digest=hashlib.sha256(source_archive()).hexdigest())
        .replace("phase4_source.zip", "phase4_followup_source.zip")
        .replace("/content/phase4_payload", "/content/phase4_followup_payload")
    )
    return {
        "cells": [
            md(INTRO),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase4_followup_source.zip"\n'),
            md("## Load from Drive"),
            code(unpack),
            md("## Install pinned dependencies with uv"),
            code(setup),
            md("## Run the fixed matrix"),
            code(RUN),
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


if __name__ == "__main__":
    SOURCE_ARCHIVE.write_bytes(source_archive())
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        NOTEBOOK.name,
        NOTEBOOK.stat().st_size,
        "bytes;",
        SOURCE_ARCHIVE.name,
        SOURCE_ARCHIVE.stat().st_size,
        "bytes",
    )
