"""Build Phase 5's compact Drive ZIP notebook: freeze inputs, compare 4B/8B, export history."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.build_colab_indexer import code, md, runtime_pins  # noqa: E402
from notebooks.build_retrieval_eval import UNPACK  # noqa: E402


def source_archive() -> bytes:
    labels = ROOT / "eval/datasets/phase5_bugfix_v2"
    frozen_files = [labels / "protocol.json", labels / "protocol.freeze.json"]
    declared = [
        "protocol.json",
        "records.jsonl",
        "pairs.jsonl",
        "corpus.jsonl",
        "links.json",
        "issues.json",
        "selection.json",
        "preparation.json",
        "freeze.json",
    ]
    for folder in sorted(labels.glob("repos/*")):
        if not folder.is_dir():
            continue
        for name in declared:
            path = folder / name
            frozen_files.append(path if path.exists() else Path(str(path) + ".gz"))
    paths = sorted(
        [
            *ROOT.joinpath("src/repo_rag").rglob("*.py"),
            *ROOT.joinpath("eval/retrieval").rglob("*.py"),
            *frozen_files,
            ROOT / "eval/__init__.py",
            ROOT / "eval/prepare_bugfix_eval.py",
            ROOT / "eval/run_bugfix_eval.py",
            ROOT / "eval/analyze_bugfix_run.py",
            ROOT / "eval/export_bugfix_index.py",
            ROOT / "eval/refreeze_bugfix_v2.py",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                data = path.read_bytes()
                # Dataset hashes cover original bytes, including Windows line endings.
                # Normalize source code only; never rewrite a frozen JSONL or gzip input.
                if not path.is_relative_to(labels) and path.suffix != ".gz":
                    data = data.replace(b"\r\n", b"\n")
                archive.writestr(info, data)
    return buffer.getvalue()


def build() -> dict:
    unpack = (
        UNPACK.format(digest=hashlib.sha256(source_archive()).hexdigest())
        .replace("phase4_source.zip", "phase5_bugfix_v2_source.zip")
        .replace("/content/phase4_payload", "/content/phase5_payload")
    )
    setup = """import importlib.util, subprocess, sys, os, pathlib, json
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                '--python', sys.executable, '-q', *PINS], check=True)
from tree_sitter_language_pack import download
download(['python', 'javascript', 'typescript', 'tsx', 'java', 'go'])
ENV = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(PAYLOAD), str(PAYLOAD / 'src')]),
       'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false'}
RESULTS = pathlib.Path('/content/phase5_bugfix_v2_results')
RESULTS.mkdir(exist_ok=True)
PROTOCOL = json.loads((PAYLOAD / 'eval/datasets/phase5_bugfix_v2/protocol.json').read_text())
def execute(script, *args, env=ENV):
    subprocess.run([sys.executable, str(PAYLOAD / script), *map(str, args)],
                   env=env, check=True)
""".replace("*PINS", repr(runtime_pins())[1:-1])
    prepare = """# V2 must use these exact corrected, frozen inputs; no live re-selection.
import shutil
sys.path[:0] = [str(PAYLOAD), str(PAYLOAD / 'src')]
from eval.prepare_bugfix_eval import verify_corpus
for repo in PROTOCOL['repos']:
    stem = repo.replace('/', '__')
    bundled = PAYLOAD / 'eval/datasets/phase5_bugfix_v2/repos' / stem
    dataset = RESULTS / 'datasets' / stem
    if not (bundled / 'freeze.json').exists():
        raise FileNotFoundError('Matching v2 source ZIP must include frozen datasets')
    verify_corpus(bundled)
    if not dataset.exists():
        shutil.copytree(bundled, dataset)
    verify_corpus(dataset)
    print(repo, 'v2 corpus verified before retrieval', flush=True)
backup = pathlib.Path('/content/drive/MyDrive/phase5_bugfix_v2_preparation.zip')
shutil.make_archive(str(backup.with_suffix('')), 'zip', RESULTS / 'datasets')
print('Preparation backup:', backup)
"""
    models = """# Explicit download cell: no LLM service or Anthropic key is used.
from huggingface_hub import snapshot_download
for model in ['Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Reranker-4B', 'Qwen/Qwen3-Reranker-8B']:
    print('Downloading', model, flush=True)
    snapshot_download(model)
"""
    run = """# One process per method: the 4B and 8B never occupy GPU memory together.
import shutil
shutil.copy2(source_zip, RESULTS / 'source.zip')
for repo in PROTOCOL['repos']:
    stem = repo.replace('/', '__')
    for method in PROTOCOL['retrieval']['methods']:
        output = RESULTS / 'runs' / stem / method
        if (output / 'run.json').exists():
            print('Already complete:', repo, method)
            continue
        if output.exists():
            raise FileExistsError(f'Incomplete: {output}; preserve it and use a fresh runtime')
        execute('eval/run_bugfix_eval.py', '--dataset', RESULTS / 'datasets' / stem,
                '--method', method, '--output', output, '--device', 'cuda')
execute('eval/analyze_bugfix_run.py', RESULTS)
execute('eval/export_bugfix_index.py', RESULTS)
print((RESULTS / 'aggregate/summary.md').read_text())
"""
    download = """# Ship evidence, metadata and a vector ZIP; skip Chroma runtime files.
import zipfile
archive = pathlib.Path('/content/phase5_bugfix_v2_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(RESULTS.rglob('*')):
        rel = path.relative_to(RESULTS)
        if not path.is_file() or 'index' in rel.parts or 'production_index' in rel.parts:
            continue
        bundle.write(path, rel.as_posix())
    bundle.write(RESULTS / 'production_index/metadata.db', 'metadata.db')
shutil.copy2(archive, '/content/drive/MyDrive/phase5_bugfix_v2_results.zip')
from google.colab import files
files.download(str(archive))
"""
    return {
        "cells": [
            md("""# Phase 5: bug report to past fix commits

Upload **phase5_bugfix_v2_source.zip** to My Drive and open this notebook on an **A100 40 GB**,
and run the cells in order. Bundled frozen datasets need no GitHub key in Colab.
The v2 dataset is complete: no live GitHub selection occurs in this notebook.
No paid LLM key is needed. The separate ZIP keeps notebook cells short; there is no base64 archive.

V2 preserves the 329 v1 queries, gold labels and connected-group splits, and corrects the
commit scrub to remove reference phrases while keeping adjacent descriptions. Inputs and
protocol were re-frozen before v2 retrieval. This is a correction, not a fresh holdout.
It compares full-query BM25, a 2000-character BM25 sensitivity check, dense, hybrid,
4B reranking and 8B reranking at fixed depth 50.
Both rerankers use the bug-to-fix instruction. No weight tuning or test-set selection occurs.

Issue text is used as the query only. Commit documents contain message, files, AST symbols and
800 characters of scrubbed diff. Links are proxy labels; not every closing issue is a bug and
other valid fixes may be unlabelled. Requests queries may overlap the earlier T3 benchmark.
This measures finding an issue's own fix, not similar historical fixes for new bugs.
File Hit@5 is a secondary same-path proxy: the exact fix itself qualifies, and sharing a path
does not prove semantic relevance. Every gold qualifies from the message heuristic alone;
closing links do not determine evaluation pool membership. The broad fix heuristic remains
a limitation.
BM25 sees full queries; the sensitivity row uses 2000 characters. Embedding and reranker
truncation still have different token budgets. Query-length strata are reported separately.
The fixed 8B rule retains pooled MRR gain >=0.02 and a positive paired CI lower bound;
repo/link-path guards apply only with >=30 connected test groups. FastAPI is descriptive,
with no safety conclusion for its 14 test queries. Failure or inconclusive data is valid.

The explicit model-download cell fetches roughly 25 GB of weights. This GPU run's duration
has not been measured.
Preparation is backed up to Drive; keep returned results intact. Frozen source records retain
public historical text (including email trailers); exported metadata.db redacts email-like strings.

The result ZIP contains raw per-question results, full candidate/pair scores, source and input
hashes, a reproducible analysis, a portable commit index, metadata.db and a real CLI citation.
This tests retrieval, not generated answers, scale limits or deployment latency.
"""),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase5_bugfix_v2_source.zip"\n'),
            code(unpack),
            code(setup),
            code(prepare),
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
    archive = ROOT / "notebooks/phase5_bugfix_v2_source.zip"
    notebook = ROOT / "notebooks/phase5_bugfix_v2_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{notebook.name}: {notebook.stat().st_size} bytes; "
        f"{archive.name}: {archive.stat().st_size} bytes"
    )
