"""Build the Drive ZIP notebook for the pre-registered retrieval_k=200 commit confirmation."""

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

EXPERIMENT = ROOT / "eval/datasets/phase5_commit_confirm_v1"
SCRIPTS = (
    "eval/confirm_lib.py",
    "eval/prepare_commit_confirm.py",
    "eval/run_commit_confirm_eval.py",
    "eval/analyze_commit_confirm.py",
)


def source_archive() -> bytes:
    entries = {}
    with zipfile.ZipFile(io.BytesIO(base.source_archive())) as archive:
        entries.update({name: archive.read(name) for name in archive.namelist()})
    for path in sorted(EXPERIMENT.rglob("*")):
        if path.is_file():  # frozen inputs keep their exact bytes
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in SCRIPTS:
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
        .replace("phase4_source.zip", "phase5_commit_confirm_source.zip")
        .replace("/content/phase4_payload", "/content/phase5_commit_confirm_payload")
    )
    setup = (
        "".join(old["cells"][3]["source"])
        .replace("phase5_bugfix_v2_results", "phase5_commit_confirm_results")
        .replace("phase5_bugfix_v2/protocol.json", "phase5_commit_confirm_v1/protocol.json")
    )
    prepare = """import shutil, zipfile, hashlib
sys.path[:0] = [str(PAYLOAD), str(PAYLOAD / 'src')]
from eval.confirm_lib import verify_experiment, stem_of
EXPERIMENT = RESULTS / 'experiment'
payload_hash = hashlib.sha256(source_zip.read_bytes()).hexdigest()[:12]
checkpoint_path = pathlib.Path('/content/drive/MyDrive') / (
    'phase5_commit_confirm_checkpoint_' + payload_hash + '.zip')
if not any(RESULTS.iterdir()) and checkpoint_path.exists():
    with zipfile.ZipFile(checkpoint_path) as saved:
        for name in saved.namelist():
            if not (RESULTS / name).resolve().is_relative_to(RESULTS.resolve()):
                raise ValueError('Unsafe checkpoint path')
        saved.extractall(RESULTS)
BUNDLED = PAYLOAD / 'eval/datasets/phase5_commit_confirm_v1'
if not EXPERIMENT.exists():
    shutil.copytree(BUNDLED, EXPERIMENT, ignore=shutil.ignore_patterns('repos'))
FRESH = list(json.loads((EXPERIMENT / 'freeze.json').read_text())['counts'])
V2DEV = [stem_of(repo) for repo in PROTOCOL['supporting']['repos']]
for stem in FRESH:
    target = RESULTS / 'datasets' / 'fresh' / stem
    if not target.exists():
        shutil.copytree(BUNDLED / 'repos' / stem, target)
for stem in V2DEV:
    target = RESULTS / 'datasets' / 'v2' / stem
    if not target.exists():
        shutil.copytree(PAYLOAD / 'eval/datasets/phase5_bugfix_v2/repos' / stem, target)
verify_experiment(EXPERIMENT, RESULTS / 'datasets')  # frozen hashes and leakage, before any model
shutil.copy2(source_zip, RESULTS / 'source.zip')

def pack(destination):
    pending = destination.with_suffix('.pending')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(RESULTS.rglob('*')):
            rel = path.relative_to(RESULTS)
            if path.is_file() and 'index' not in rel.parts:
                bundle.write(path, rel.as_posix())
    pending.replace(destination)

def job(group, stem, stage):
    output = RESULTS / 'runs' / group / stem / stage
    if (output / 'run.json').exists():
        print('Already complete:', group, stem, stage)
        return
    if output.exists():
        raise FileExistsError(f'Incomplete output: {output}; preserve it before retrying')
    execute('eval/run_commit_confirm_eval.py', RESULTS, '--experiment', EXPERIMENT,
            '--group', group, '--stem', stem, '--stage', stage)
    pack(checkpoint_path)
    print('Saved Drive checkpoint:', group, stem, stage, flush=True)
pack(checkpoint_path)
print('Frozen inputs verified for', FRESH, 'and v2 dev for', V2DEV)
"""
    models = """# Download both models, verify the preregistered revisions, then go offline.
from huggingface_hub import snapshot_download
models = [('embedding', PROTOCOL['models']['embedding']),
          ('rerank8b', PROTOCOL['models']['rerankers'][1])]
for key, model in models:
    print('Downloading', model, flush=True)
    snapshot = pathlib.Path(snapshot_download(model))
    if snapshot.name != PROTOCOL['expected_revisions'][key]:
        raise ValueError(f'{model}: main revision changed; keep the protocol and ask for review')
ENV['HF_HUB_OFFLINE'] = '1'
ENV['TRANSFORMERS_OFFLINE'] = '1'
"""
    quality = """# Primary evidence: the fresh holdout. Arms A/B/C share per-question scores.
for stem in FRESH:
    job('fresh', stem, 'confirm')
# Supporting evidence, reported separately: the 66 v2 dev questions.
for stem in V2DEV:
    job('v2dev', stem, 'confirm')
"""
    timing = """# Real uncached production calls (first 40 fresh questions, 3 rotated repetitions).
for stem in FRESH:
    job('fresh', stem, 'timing')
"""
    analysis = """if not (RESULTS / 'aggregate/decision.json').exists():
    execute('eval/analyze_commit_confirm.py', RESULTS)
pack(checkpoint_path)
print((RESULTS / 'aggregate/summary.md').read_text())
"""
    download = """archive = pathlib.Path('/content/phase5_commit_confirm_results.zip')
pack(archive)
drive_archive = pathlib.Path('/content/drive/MyDrive/phase5_commit_confirm_results.zip')
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
            md("""# Phase 5: confirm retrieval_k=200 for commit search (pre-registered)

Upload **phase5_commit_confirm_source.zip** to My Drive. Use an **A100 40 GB** and run the cells
in order. uv installs the pinned runtime; the explicit model cell downloads the embedding 0.6B
and the reranker 8B (about 17 GB). No GitHub or paid model key is needed: the fresh dataset is
already built and frozen inside the ZIP. There is no Base64 payload.

**Question.** Does building the candidate pool from longer per-signal lists (`retrieval_k=200`)
while still reranking only the fused top 50 raise Hit@5 and MRR@10 for bug report to fix commit
search? An earlier look at the v2 test split suggested +0.015 Hit@5, but that finding came from
data already examined, so it is confirmed here on repositories not used before. The hypothesis,
arms, adoption rule and timing were frozen and hashed before any sampling or retrieval.

**Arms (8B, rerank depth 50, shared scores).** A: `retrieval_k=50` (current default, control).
B: 200 (hypothesis). C: 100 (descriptive). Every arm runs the production search path; one score
cache per question means a (query, document) pair is scored once, so arms differ only by which
candidates enter the reranked top 50. Arm A is scored again from scratch to measure GPU rerun
noise. First-stage-only results and pool recall are reported for every arm.

**Rule.** B is eligible only if Hit@5 is at least +0.01 and MRR@10 at least 0 versus A, both 95%
lower bounds are at least -0.01, no repository with 30+ groups loses more than 0.02 Hit@5, the
rerun noise does not exceed the effect, and the Hit@5 interval is no wider than 0.06. Fewer than
100 fresh questions makes any conclusion tentative. Even a pass changes nothing automatically:
`retrieval_k` is shared by code, docs and commits and only commits were measured, so adoption
would be a separate per-kind setting.

**Supporting.** The 66 v2 dev questions are reported separately; the effect was never examined
there. **Timing.** 40 fresh questions, three rotated warm repetitions of real production calls per
arm, per-repo p50/p95 (A100, not laptop speed).

Labels are link-derived with no independent review, and this measures finding an issue's own fix,
not useful similar past fixes. Completed stages checkpoint to Drive; a fresh runtime with this
exact ZIP resumes. The last cell verifies the results ZIP, flushes Drive and releases the runtime.
Retrieve `phase5_commit_confirm_results.zip` from My Drive if the browser download is unfinished.

Not measured: answer quality, similar-fix usefulness, independent labels, laptop speed, code and
docs retrieval at 200, other languages. Nothing is committed.
"""),
            code('SOURCE_ZIP = "/content/drive/MyDrive/phase5_commit_confirm_source.zip"\n'),
            code(unpack),
            code(setup),
            code(prepare),
            code(models),
            code(quality),
            code(timing),
            code(analysis),
            code(download),
            code(disconnect),
        ],
    }


if __name__ == "__main__":
    archive = ROOT / "notebooks/phase5_commit_confirm_source.zip"
    notebook = ROOT / "notebooks/phase5_commit_confirm_on_colab.ipynb"
    archive.write_bytes(source_archive())
    notebook.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(notebook.name, notebook.stat().st_size, archive.name, archive.stat().st_size)
