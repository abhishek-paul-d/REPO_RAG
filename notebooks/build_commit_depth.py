"""Build the compact Drive ZIP notebook for commit candidate depth and BM25 weight."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks import build_bugfix_eval as base  # noqa: E402
from notebooks.build_colab_indexer import code, md  # noqa: E402
from notebooks.build_retrieval_eval import UNPACK  # noqa: E402

EXPERIMENT = ROOT / "eval/datasets/phase5_commit_depth_v1"


def source_archive() -> bytes:
    entries = {}
    with zipfile.ZipFile(io.BytesIO(base.source_archive())) as archive:
        entries.update({name: archive.read(name) for name in archive.namelist()})
    for path in sorted(EXPERIMENT.rglob("*")):
        if path.is_file():
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("eval/run_commit_depth_eval.py", "eval/analyze_commit_depth.py"):
        entries[name] = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    return result.getvalue()


def build() -> dict:
    old = base.build()
    unpack = (
        UNPACK.format(digest=hashlib.sha256(source_archive()).hexdigest())
        .replace("phase4_source.zip", "phase5_commit_depth_source.zip")
        .replace("/content/phase4_payload", "/content/phase5_commit_depth_payload")
    )
    setup = "".join(old["cells"][3]["source"]).replace(
        "phase5_bugfix_v2_results", "phase5_commit_depth_results"
    )
    prepare = """import shutil, zipfile, hashlib
sys.path[:0] = [str(PAYLOAD), str(PAYLOAD / 'src')]
from eval.retrieval.commit_depth import verify_protocol
EXPERIMENT = RESULTS / 'experiment'
payload_hash = hashlib.sha256(source_zip.read_bytes()).hexdigest()[:12]
checkpoint_path = pathlib.Path('/content/drive/MyDrive') / (
    'phase5_commit_depth_checkpoint_' + payload_hash + '.zip')
if not any(RESULTS.iterdir()) and checkpoint_path.exists():
    with zipfile.ZipFile(checkpoint_path) as saved:
        for name in saved.namelist():
            target = (RESULTS / name).resolve()
            if not target.is_relative_to(RESULTS.resolve()):
                raise ValueError('Unsafe checkpoint path')
        saved.extractall(RESULTS)
if not EXPERIMENT.exists():
    shutil.copytree(PAYLOAD / 'eval/datasets/phase5_commit_depth_v1', EXPERIMENT)
PROTOCOL = json.loads((EXPERIMENT / 'protocol.json').read_text())
for repo in PROTOCOL['repos']:
    stem = repo.replace('/', '__')
    target = RESULTS / 'datasets' / stem
    if not target.exists():
        shutil.copytree(PAYLOAD / 'eval/datasets/phase5_bugfix_v2/repos' / stem, target)
verify_protocol(EXPERIMENT, RESULTS / 'datasets')
shutil.copy2(source_zip, RESULTS / 'source.zip')

def pack(destination):
    pending = destination.with_suffix('.pending')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(RESULTS.rglob('*')):
            rel = path.relative_to(RESULTS)
            if path.is_file() and not {'index', 'production_index'} & set(rel.parts):
                bundle.write(path, rel.as_posix())
        metadata = RESULTS / 'production_index/metadata.db'
        if metadata.exists():
            bundle.write(metadata, 'metadata.db')
    pending.replace(destination)

def job(stage, stem, judge='rerank8b', weight=None):
    folder = 'weights' if stage == 'weights' else 'timing' if stage == 'timing' else (
        judge if stage == 'depth' else f'selected_{weight:g}')
    output = RESULTS / 'runs' / stem / folder
    if (output / 'run.json').exists():
        print('Already complete:', stem, folder)
        return
    if output.exists():
        raise FileExistsError(f'Incomplete output: {output}; preserve it before retrying')
    args = ['--repo', stem, '--stage', stage, '--judge', judge]
    if weight is not None:
        args += ['--weight', str(weight)]
    execute('eval/run_commit_depth_eval.py', RESULTS, '--experiment', EXPERIMENT, *args)
    pack(checkpoint_path)
    print('Saved Drive checkpoint:', stem, folder, flush=True)
pack(checkpoint_path)
print('Frozen protocol verified. Matching completed stages can resume from Drive.')
"""
    weights = """# Selection reads only the 66 dev questions; test rows never choose weights.
for repo in PROTOCOL['repos']:
    job('weights', repo.replace('/', '__'))
execute('eval/run_commit_depth_eval.py', RESULTS, '--experiment', EXPERIMENT, '--stage', 'select')
selection = json.loads((RESULTS / 'selection.json').read_text())
print(selection)
pack(checkpoint_path)
"""
    models = """# Download models, verify revisions, then use offline caches.
from huggingface_hub import snapshot_download
models = [('embedding', PROTOCOL['models']['embedding']),
          ('rerank4b', PROTOCOL['models']['rerankers'][0]),
          ('rerank8b', PROTOCOL['models']['rerankers'][1])]
for key, model in models:
    print('Downloading', model, flush=True)
    snapshot = pathlib.Path(snapshot_download(model))
    if snapshot.name != PROTOCOL['expected_revisions'][key]:
        raise ValueError(f'{model}: main revision changed; preserve the protocol before proceeding')
ENV['HF_HUB_OFFLINE'] = '1'
ENV['TRANSFORMERS_OFFLINE'] = '1'
"""
    quality = """# One 200-deep scored pool per question per judge. Prefixes reuse those scores.
for judge in ['rerank8b', 'rerank4b']:
    for repo in PROTOCOL['repos']:
        job('depth', repo.replace('/', '__'), judge=judge)
if selection['selected'] != selection['baseline']:
    for repo in PROTOCOL['repos']:
        for weight in [selection['baseline'], selection['selected']]:
            job('selected', repo.replace('/', '__'), weight=weight)
"""
    timing = """# Separate real production calls; quality-prefix reuse does NOT measure latency.
for repo in PROTOCOL['repos']:
    job('timing', repo.replace('/', '__'))
"""
    analysis = """if not (RESULTS / 'aggregate/decision.json').exists():
    execute('eval/analyze_commit_depth.py', RESULTS)
if not (RESULTS / 'cli_sample_query.json').exists():
    execute('eval/export_bugfix_index.py', RESULTS)
pack(checkpoint_path)
print((RESULTS / 'aggregate/summary.md').read_text())
"""
    download = """archive = pathlib.Path('/content/phase5_commit_depth_results.zip')
pack(archive)
drive_archive = pathlib.Path('/content/drive/MyDrive/phase5_commit_depth_results.zip')
shutil.copy2(archive, drive_archive)
from google.colab import files
files.download(str(archive))
"""
    disconnect = """# Release the GPU automatically after verifying and syncing the saved results.
from google.colab import drive, runtime
import hashlib

with archive.open('rb') as local_file, drive_archive.open('rb') as saved_file:
    local_hash = hashlib.file_digest(local_file, 'sha256').digest()
    saved_hash = hashlib.file_digest(saved_file, 'sha256').digest()
    if local_hash != saved_hash:
        raise RuntimeError('Drive results do not match; keep the runtime to recover the ZIP.')
print('Syncing results to Drive before releasing the runtime...', flush=True)
drive.flush_and_unmount(timeout_ms=120000)
print('Results saved:', drive_archive, '\\nReleasing the Colab runtime now.', flush=True)
runtime.unassign()
"""
    return {
        **old,
        "cells": [
            md("""# Phase 5: commit reranking depth and BM25 weight

Upload **phase5_commit_depth_source.zip** to My Drive. Use an **A100 40 GB** and run in order.
uv installs the pinned runtime; the explicit model cell downloads the embedding 0.6B and
rerankers 4B/8B. No GitHub or paid model key is needed. There is no Base64 payload.

The same frozen 329 v2 questions are reused: 66 dev and 263 test, including 208 whose gold
touches a non-doc/changelog path. These proxy source paths include tests and configs.
Gold is link-derived, author-checked, with independent review pending. This measures finding
an issue's own fix, not useful similar past fixes for a new bug. It is not a fresh holdout.

**Depth:** one fused 200-deep pool per question/judge, dense 1.0 + BM25 0.3, both lists 200.
Score every pair once. Depths 50/100/200 sort prefixes of the same probabilities; ties keep
fused order. 8B is primary, 4B secondary. Existing v2 depth50 results remain a historical
control: their first-stage lists were shorter, so the new prefix50 can differ.

**Weight:** first-stage depth50 weights 0, 0.1, 0.3, 0.5, 1.0. Select from dev only and keep
0.3 unless gain reaches 0.01 MRR. Test weights are descriptive. If dev selects another
weight, fresh 8B depth50 calls compare it against 0.3. Depth and weight effects stay separate.

The frozen depth rule requires test Hit@5 +0.02 with positive paired lower bound, MRR loss
no worse than 0.005, nonnegative source-subset Hit@5 change, and no >0.02 repo Hit@5 loss
where >=30 connected groups exist. Small strata are descriptive. Prefer 100 unless 200
clearly wins under the separate rule. **No default changes automatically.**

**Timing:** 60 frozen test questions, round-robin across repos, three rotated warm repetitions
using real primary-8B production calls. All depths keep retrieval_k=200, matching the quality
prefixes. Report p50/p95 and ratios per repo, never average repository percentiles. Prefix
quality reuse is not a latency measurement. This is A100 experiment timing, not laptop speed.

Long reports retain the existing unequal query budgets: full BM25, embedding 2048 tokens,
rerank first 2000 characters then 2048 pair tokens. Query-length views and all pool ceilings
are reported with paired exploratory, unadjusted group-bootstrap intervals.

Completed stages checkpoint to Drive; a fresh runtime with this exact source ZIP can resume.
After a successful run, the last cell verifies the results ZIP, flushes Drive writes and
automatically releases the runtime. Retrieve the ZIP from My Drive if the browser download
has not finished. A failed save or sync stops shutdown so the results can be recovered.
The ZIP contains frozen data/control/source hashes, all pairs and prefix rankings, raw timing,
tables/decision, and the CLI's exact bug report beside its cited evidence. Frozen historical
source data retains public text; metadata.db redacts email-like payloads.

Not measured: generated answers, similar-fix usefulness, independent labels, scale limits,
deployment hardware, other repositories/languages or ANN accuracy at scale. No commits.
"""),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase5_commit_depth_source.zip"\n'),
            code(unpack),
            code(setup),
            code(prepare),
            code(models),
            code(weights),
            code(quality),
            code(timing),
            code(analysis),
            code(download),
            code(disconnect),
        ],
    }


if __name__ == "__main__":
    archive = ROOT / "notebooks/phase5_commit_depth_source.zip"
    notebook = ROOT / "notebooks/phase5_commit_depth_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(notebook.name, notebook.stat().st_size, archive.name, archive.stat().st_size)
