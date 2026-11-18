"""Build the judge comparison (Qwen3-Reranker 4B vs 8B): compact uv notebook + Drive ZIP."""

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
                ROOT / "eval/run_expansion_eval.py",
                ROOT / "eval/run_judge_eval.py",
                ROOT / "eval/analyze_judge_run.py",
                *ROOT.joinpath("eval/datasets/phase4_expansion_v1").glob("*"),
                *ROOT.joinpath("eval/datasets/phase4_judge_v1").glob("*"),
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
        .replace("phase4_source.zip", "phase4_judge_source.zip")
        .replace("/content/phase4_payload", "/content/phase4_judge_payload")
    )
    setup = """import importlib.util, subprocess, sys
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                '--python', sys.executable, '-q', *PINS], check=True)
""".replace("*PINS", repr(pins())[1:-1])
    models = """# Models are downloaded here, once, from Hugging Face into the Colab cache
# (/root/.cache/huggingface). Later cells load them from that cache.
from huggingface_hub import snapshot_download
import pathlib
MODELS = [
    ('Qwen/Qwen3-Embedding-0.6B', 'embeds code chunks and questions (about 1.2 GB)'),
    ('Qwen/Qwen3-Reranker-4B', 'current judge (about 8 GB)'),
    ('Qwen/Qwen3-Reranker-8B', 'candidate stronger judge (about 16 GB)'),
]
for name, why in MODELS:
    print('Downloading', name, '-', why, flush=True)
    path = pathlib.Path(snapshot_download(name))
    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e9
    print(f'  ready: {path.name}  {size:.1f} GB on disk', flush=True)
"""
    run = """import json, os, pathlib, shutil, subprocess, sys
REPOS = pathlib.Path('/content/judge_repos')
RESULTS = pathlib.Path('/content/judge_results')
LABELS = PAYLOAD / 'eval/datasets/phase4_expansion_v1'
if RESULTS.exists():
    raise FileExistsError('Results already exist. Download them before starting a new runtime.')
REPOS.mkdir(exist_ok=True)
RESULTS.mkdir()
JUDGE = PAYLOAD / 'eval/datasets/phase4_judge_v1'
shutil.copytree(LABELS, RESULTS / 'labels')
shutil.copytree(JUDGE, RESULTS / 'judge_protocol')
shutil.copyfile(source_zip, RESULTS / 'phase4_judge_source.zip')
env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(PAYLOAD), str(PAYLOAD / 'src')]),
       'TOKENIZERS_PARALLELISM': 'false'}
manifest = json.loads((LABELS / 'manifest.json').read_text())
repos = {e['repo']: e['commit_sha'] for e in manifest['datasets']}
for repo, sha in repos.items():
    checkout = REPOS / repo.replace('/', '__')
    if not checkout.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/' + repo, str(checkout)], check=True)
    subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', sha], check=True)
    subprocess.run([sys.executable, str(PAYLOAD / 'eval/run_judge_eval.py'),
                    '--labels', str(LABELS), '--protocol', str(JUDGE / 'protocol.json'),
                    '--repo', repo, '--checkout', str(checkout),
                    '--output', str(RESULTS / repo.replace('/', '__')),
                    '--mode', 'full', '--device', 'cuda'], env=env, check=True)
subprocess.run([sys.executable, str(PAYLOAD / 'eval/analyze_judge_run.py'), str(RESULTS),
                str(JUDGE / 'protocol.json')],
                env=env, check=True)
print((RESULTS / 'aggregate/summary.md').read_text())
"""
    download = """from google.colab import files
archive = pathlib.Path('/content/phase4_judge_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file() and 'isolated_index' not in path.parts:
            bundle.write(path, path.relative_to(RESULTS))
files.download(str(archive))
"""
    return {
        "cells": [
            md("""# Judge comparison: Qwen3-Reranker 8B vs the current 4B

1. Upload **phase4_judge_source.zip** to My Drive root.
2. Choose an **A100 GPU** and Run all (approve Drive). Expect 25 to 45 minutes; the model
   download cell fetches about 25 GB the first time.
3. Return **phase4_judge_results.zip**.

**Where the models are used.** No cell names a model except the download cell below. The
code reads the model names from `Settings` (`local_embedding_model` = Qwen3-Embedding-0.6B, and
the two judges listed in `eval/datasets/phase4_judge_v1/protocol.json`). The download cell fetches
all three from Hugging Face; later cells load them from the cache. Progress shows in that cell.

Everything else is fixed: hybrid first stage, rerank the fused top 20, return the top 10, no
expansion, no tie-break, one code instruction. Only the judge changes. The 150 questions are the
frozen expansion set (30 fresh multi-block including 12 on Pallets Click, plus 120 regression).

The adoption rule (pooled MRR gain of at least +0.03 with a positive 95% lower bound, proof not
lower, no stratum losing more than 0.02) is in `eval/datasets/phase4_judge_v1/protocol.json`,
written before the run. Quality and rough timing only; answer quality is not measured. Labels are
author-checked, independent review pending. A100 is experiment-only; deployment undecided.
"""),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase4_judge_source.zip"\n'),
            code(unpack),
            code(setup),
            code(models),
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
    archive = ROOT / "notebooks/phase4_judge_source.zip"
    notebook = ROOT / "notebooks/phase4_judge_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{notebook.name}: {notebook.stat().st_size} bytes; "
        f"{archive.name}: {archive.stat().st_size} bytes"
    )
