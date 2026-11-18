"""Small Drive-ZIP notebook for the production-path ambiguous-intent experiment."""

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
                *ROOT.joinpath("eval/datasets/phase4_ambiguous").glob("*.json*"),
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


def build() -> dict:
    unpack = (
        UNPACK.format(digest=hashlib.sha256(source_archive()).hexdigest())
        .replace("phase4_source.zip", "phase4_ambiguous_source.zip")
        .replace("/content/phase4_payload", "/content/phase4_ambiguous_payload")
    )
    setup = """import importlib.util, subprocess, sys
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                '--python', sys.executable, '-q', *PINS], check=True)
""".replace("*PINS", repr(pins())[1:-1])
    run = """import json, os, pathlib, subprocess, sys
REPOS = pathlib.Path('/content/ambiguous_repos')
RESULTS = pathlib.Path('/content/ambiguous_results')
REPOS.mkdir(exist_ok=True)
env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(PAYLOAD), str(PAYLOAD / 'src')]),
       'TOKENIZERS_PARALLELISM': 'false'}
manifest = json.loads((PAYLOAD / 'eval/datasets/phase4_ambiguous/manifest.json').read_text())
for entry in manifest['datasets']:
    repo = entry['repo']
    checkout = REPOS / repo.replace('/', '__')
    if not checkout.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/' + repo, str(checkout)], check=True)
    subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach',
                    entry['commit_sha']], check=True)
    dataset = PAYLOAD / 'eval/datasets/phase4_ambiguous' / entry['file']
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Dataset checksum mismatch')
    subprocess.run([sys.executable, str(PAYLOAD / 'eval/run_ambiguous_eval.py'),
                    '--dataset', str(dataset), '--checkout', str(checkout),
                    '--output', str(RESULTS / repo.replace('/', '__')),
                    '--mode', 'full', '--device', 'cuda', '--repetitions', '3'],
                    env=env, check=True)
"""
    download = """from google.colab import files
archive = pathlib.Path('/content/phase4_ambiguous_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file() and 'isolated_index' not in path.parts:
            bundle.write(path, path.relative_to(RESULTS))
files.download(str(archive))
"""
    return {
        "cells": [
            md(
                """# Ambiguous-intent stress comparison

Upload phase4_ambiguous_source.zip to My Drive root. Choose A100 and run all.
Uses uv, a separate ZIP, and the actual production search path at
hybrid / rerank20 / rerank50.

These 15 questions omit test/symbol cues but target previously inspected behaviours.
Alternatives accept implementation or a complete demonstration. This is a post-hoc
stress set, not an independent holdout. No tuning or default promotion.
Deployment hardware remains undecided; A100 timing is experimental only.
"""
            ),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase4_ambiguous_source.zip"\n'),
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
    archive = ROOT / "notebooks/phase4_ambiguous_source.zip"
    notebook = ROOT / "notebooks/phase4_ambiguous_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{notebook.name}: {notebook.stat().st_size} bytes; "
        f"{archive.name}: {archive.stat().st_size} bytes"
    )
