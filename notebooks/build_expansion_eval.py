"""Build the frozen expansion quality comparison: compact uv notebook + separate Drive ZIP."""

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
                ROOT / "eval/analyze_expansion_run.py",
                ROOT / "eval/build_expansion_dataset.py",
                ROOT / "eval/build_comparison_dataset.py",
                ROOT / "eval/build_followup_datasets.py",
                *ROOT.joinpath("eval/datasets/phase4_expansion_v1").glob("*"),
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
        .replace("phase4_source.zip", "phase4_expansion_source.zip")
        .replace("/content/phase4_payload", "/content/phase4_expansion_payload")
    )
    setup = """import importlib.util, subprocess, sys
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                '--python', sys.executable, '-q', *PINS], check=True)
""".replace("*PINS", repr(pins())[1:-1])
    run = """import json, os, pathlib, shutil, subprocess, sys
REPOS = pathlib.Path('/content/expansion_repos')
RESULTS = pathlib.Path('/content/expansion_results')
LABELS = PAYLOAD / 'eval/datasets/phase4_expansion_v1'
if RESULTS.exists():
    raise FileExistsError('Results already exist. Download them before starting a new runtime.')
REPOS.mkdir(exist_ok=True)
RESULTS.mkdir()
shutil.copytree(LABELS, RESULTS / 'labels')
shutil.copyfile(source_zip, RESULTS / 'phase4_expansion_source.zip')
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
    subprocess.run([sys.executable, str(PAYLOAD / 'eval/run_expansion_eval.py'),
                    '--labels', str(LABELS), '--repo', repo, '--checkout', str(checkout),
                    '--output', str(RESULTS / repo.replace('/', '__')),
                    '--mode', 'full', '--device', 'cuda'], env=env, check=True)
subprocess.run([sys.executable, str(PAYLOAD / 'eval/analyze_expansion_run.py'), str(RESULTS)],
                env=env, check=True)
print((RESULTS / 'aggregate/summary.md').read_text())
"""
    download = """from google.colab import files
archive = pathlib.Path('/content/phase4_expansion_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file() and 'isolated_index' not in path.parts:
            bundle.write(path, path.relative_to(RESULTS))
files.download(str(archive))
"""
    return {
        "cells": [
            md("""# Fair test of evidence expansion

Upload **phase4_expansion_source.zip** to My Drive root, choose an A100, and Run all.
Return **phase4_expansion_results.zip** for audit. Packages use uv; source stays in a separate
checksummed Drive ZIP. No long encoded source cell.

**30 newly authored multi-block questions**: six each Requests/Express/Flask, twelve Click
(new repository). **120 frozen regression questions** are also run, unchanged.
All fresh proofs require multiple indexed chunks. Labels and success rule were frozen/hashed
before new retrieval; no git commit was created because the user forbids commits.

Three fixed policies independently traverse production search: **plain rerank20**, **rerank20
+ callee**, **rerank20 + both**. First-stage depth 50/signal, returned top10; expansion keeps
5 hits and takes at most 2 additions from each of 3 seeds. No parameter tuning.
One resident scorer shares exact pair scores across policies with identical inputs:
**quality only, no latency/speedup measurement**. 150 questions, 450 policy outputs.

Primary callee must improve new-question complete proof by at least +0.05 with a paired 95%
lower bound above zero, lose no more than 0.02 MRR or proof on any old stratum, and grow
returned text by no more than 10% separately on fresh30 and regression120. Both is secondary.
The analyzer validates hashes, candidates, expansion reconstruction, metrics and text budgets.
On rejection, leave expansion off and move to Phase 5; do not iterate on these questions.

Name matching is not a call graph. Independent label review and answer-generation evaluation
remain pending. A100 is experiment-only; deployment undecided. Expansion stays off by default,
and it never runs on unreranked hybrid/CPU searches. No default is changed automatically.
"""),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase4_expansion_source.zip"\n'),
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
    archive = ROOT / "notebooks/phase4_expansion_source.zip"
    notebook = ROOT / "notebooks/phase4_expansion_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{notebook.name}: {notebook.stat().st_size} bytes; "
        f"{archive.name}: {archive.stat().st_size} bytes"
    )
