"""Build the expanded 90-question A100 notebook with a separate, checksummed Drive ZIP."""

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


def source_archive() -> bytes:
    paths = sorted(
        set(
            [
                *archive_files(),
                ROOT / "eval/run_ambiguous_eval.py",
                ROOT / "eval/summarize_comparison.py",
                *ROOT.joinpath("eval/datasets/phase4_comparison_v3").glob("*"),
            ]
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if not path.is_file():
                continue
            item = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(item, path.read_bytes().replace(b"\r\n", b"\n"))
    return buffer.getvalue()


def build() -> dict:
    unpack = (
        UNPACK.format(digest=hashlib.sha256(source_archive()).hexdigest())
        .replace("phase4_source.zip", "phase4_comparison_source.zip")
        .replace("/content/phase4_payload", "/content/phase4_comparison_payload")
    )
    setup = """import importlib.util, subprocess, sys
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                '--python', sys.executable, '-q', *PINS], check=True)
""".replace("*PINS", repr(pins())[1:-1])
    run = """import json, os, pathlib, shutil, subprocess, sys
REPOS = pathlib.Path('/content/comparison_repos')
RESULTS = pathlib.Path('/content/comparison_results')
LABELS = PAYLOAD / 'eval/datasets/phase4_comparison_v3'
if RESULTS.exists():
    raise FileExistsError('Results already exist. Download them before starting a new runtime.')
REPOS.mkdir(exist_ok=True)
RESULTS.mkdir()
shutil.copytree(LABELS, RESULTS / 'labels')
shutil.copyfile(source_zip, RESULTS / 'phase4_comparison_source.zip')
env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(PAYLOAD), str(PAYLOAD / 'src')]),
       'TOKENIZERS_PARALLELISM': 'false'}
manifest = json.loads((LABELS / 'manifest.json').read_text())
for entry in manifest['datasets']:
    repo = entry['repo']
    checkout = REPOS / repo.replace('/', '__')
    if not checkout.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/' + repo, str(checkout)], check=True)
    subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach',
                    entry['commit_sha']], check=True)
    dataset = LABELS / entry['file']
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Dataset checksum mismatch')
    subprocess.run([sys.executable, str(PAYLOAD / 'eval/run_ambiguous_eval.py'),
                    '--dataset', str(dataset), '--checkout', str(checkout),
                    '--output', str(RESULTS / repo.replace('/', '__')),
                    '--protocol', str(LABELS / 'protocol.json'),
                    '--mode', 'full', '--device', 'cuda', '--repetitions', '3'],
                    env=env, check=True)
subprocess.run([sys.executable, str(PAYLOAD / 'eval/summarize_comparison.py'),
                '--results', str(RESULTS), '--manifest', str(RESULTS / 'labels/manifest.json')],
                env=env, check=True)
print((RESULTS / 'aggregate/summary.md').read_text())
"""
    download = """from google.colab import files
archive = pathlib.Path('/content/phase4_comparison_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file() and 'isolated_index' not in path.parts:
            bundle.write(path, path.relative_to(RESULTS))
files.download(str(archive))
"""
    return {
        "cells": [
            md("""# Expanded comparison: hybrid vs rerank20 vs rerank50

1. Upload **phase4_comparison_source.zip** to the root of My Drive.
2. Select an **A100 GPU** in Colab and Run all. Approve Drive mounting.
3. Bring the downloaded **phase4_comparison_results.zip** back for analysis.

Use this notebook and ZIP together. Source is loaded from Drive; no encoded archive cell.
Packages install with uv. Three pinned repos, 30 questions each: **90 total**, including
75 additions and the original 15. Types: 36 ambiguous, 18 multi-block, 18 test, 12 implementation,
6 example. Ambiguous queries omit test/example/symbol cues; explicit target guards stay separate.

Each policy runs independently through production retrieval, with fixed weights and no tuning.
Rerank20 scores 20 candidates; rerank50 scores 50. All use the same first-stage depth of 50
per signal and return 10. Three rotated calls per policy/question: **810 measured calls**.
Warm timings include query encoding and reranking. Initialization/warmup is separate.
Scoring uses the pre-chosen repetition zero; repetitions are not extra questions.

Reports compare MRR, complete accepted proof, paired 50-minus-20 intervals, and per-repo
p50/p95. Inspect each type and the added75/retained15 views, not just the pooled mean.
Labels, protocol, full candidates, scores, source ZIP and revisions travel with the results.

Source-authored diagnostic sample; independent review pending. More questions improve coverage
but do not guarantee significance or independence. Six example cases remain a small guard set.
This does not rerun the AST/character comparison. **A100 is experiments only; deployment is
undecided. Reranking stays opt-in.** See the bundled QUESTIONS.md for label review.
"""),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase4_comparison_source.zip"\n'),
            code(unpack),
            code(setup),
            code(run),
            code(download),
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
    archive = ROOT / "notebooks/phase4_comparison_source.zip"
    notebook = ROOT / "notebooks/phase4_comparison_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{notebook.name}: {notebook.stat().st_size} bytes; "
        f"{archive.name}: {archive.stat().st_size} bytes"
    )
