"""Build the rerank-policy A100 notebook with a separate, checksummed Drive ZIP."""

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
                ROOT / "eval/run_policy_eval.py",
                *ROOT.joinpath("eval/datasets/phase4_intent_v1").glob("*"),
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
        .replace("phase4_source.zip", "phase4_policy_source.zip")
        .replace("/content/phase4_payload", "/content/phase4_policy_payload")
    )
    setup = """import importlib.util, subprocess, sys
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                '--python', sys.executable, '-q', *PINS], check=True)
""".replace("*PINS", repr(pins())[1:-1])
    run = """import json, os, pathlib, shutil, subprocess, sys
REPOS = pathlib.Path('/content/policy_repos')
RESULTS = pathlib.Path('/content/policy_results')
LABELS = PAYLOAD / 'eval/datasets/phase4_intent_v1'
if RESULTS.exists():
    raise FileExistsError('Results already exist. Download them before starting a new runtime.')
REPOS.mkdir(exist_ok=True)
RESULTS.mkdir()
shutil.copytree(LABELS, RESULTS / 'labels')
shutil.copyfile(source_zip, RESULTS / 'phase4_policy_source.zip')
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
    subprocess.run([sys.executable, str(PAYLOAD / 'eval/run_policy_eval.py'),
                    '--dataset', str(dataset), '--checkout', str(checkout),
                    '--output', str(RESULTS / repo.replace('/', '__')),
                    '--protocol', str(LABELS / 'protocol.json'),
                    '--mode', 'full', '--device', 'cuda'], env=env, check=True)
for repo_dir in sorted(RESULTS.iterdir()):
    summary = repo_dir / 'summary.md'
    if summary.is_file():
        print(repo_dir.name)
        print(summary.read_text())
"""
    download = """from google.colab import files
archive = pathlib.Path('/content/phase4_policy_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file() and 'isolated_index' not in path.parts:
            bundle.write(path, path.relative_to(RESULTS))
files.download(str(archive))
"""
    return {
        "cells": [
            md("""# Rerank policy run: tie-break and task instructions

1. Upload **phase4_policy_source.zip** to the root of My Drive.
2. Select an **A100 GPU** in Colab and Run all. Approve Drive mounting.
3. Bring the downloaded **phase4_policy_results.zip** back for analysis.

Five policies are pre-registered; nothing is tuned. Every policy reranks the fused top 20
through the production retriever with one resident Qwen3-Reranker-4B:

| Policy | First-stage tie-break | Task instruction |
|---|---|---|
| plain | off | the single code instruction (the frozen baseline) |
| tiebreak | weight 0.25 | same |
| intent | off | keyword detector: implementation / test / default |
| source_aware | off | one instruction that prefers source unless a test or example is asked for |
| source_aware_tiebreak | weight 0.25 | same as source_aware |

Cases: the frozen 90 comparison questions plus **30 natural-wording paraphrases** of the
implementation and test questions (same gold). The paraphrases check whether the keyword
detector generalises beyond the template wording it was written from. Scores are shared when two
policies use the same instruction, so differences between them come from ordering alone.

Quality only; no timing. Labels are author-checked, not independently reviewed. A100 is for
experiments; deployment is undecided and reranking stays opt-in. Expect roughly 15 to 30 minutes.
"""),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase4_policy_source.zip"\n'),
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
    archive = ROOT / "notebooks/phase4_policy_source.zip"
    notebook = ROOT / "notebooks/phase4_policy_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{notebook.name}: {notebook.stat().st_size} bytes; "
        f"{archive.name}: {archive.stat().st_size} bytes"
    )
