"""Generate notebooks/index_on_colab.ipynb: index repos on a Colab GPU, search them locally.

The notebook is self-contained: it embeds the `repo_rag` source (as a base64 zip) and installs
the dependencies pinned to the exact versions in uv.lock, so the Colab run uses the same code
and library versions as the laptop. It runs the real CLI (`repo-rag index`, `export-index`), so
there is no second implementation to drift.

Regenerate after ANY change under src/repo_rag, pyproject.toml or uv.lock:
    uv run python notebooks/build_colab_indexer.py
A test fails if the committed notebook is stale.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).with_name("index_on_colab.ipynb")
SRC_PACKAGE = ROOT / "src" / "repo_rag"


UI_ONLY = {"streamlit"}  # runs on the laptop; Colab never needs it and its pins could clash


def runtime_pins() -> list[str]:
    """`name==version` for every runtime dependency Colab needs, versions taken from uv.lock."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = sorted(
        {re.split(r"[<>=!~;\[ ]", dep, maxsplit=1)[0].lower() for dep in project["dependencies"]}
        - UI_ONLY
    )
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {pkg["name"]: pkg["version"] for pkg in lock["package"]}
    return [f"{name}=={versions[name]}" for name in names]


def source_archive() -> tuple[str, str]:
    """(base64 of a deterministic zip of src/repo_rag/**/*.py, short content hash)."""
    buffer = io.BytesIO()
    digest = hashlib.sha256()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SRC_PACKAGE.rglob("*.py")):
            relative = f"repo_rag/{path.relative_to(SRC_PACKAGE).as_posix()}"
            data = path.read_bytes().replace(b"\r\n", b"\n")  # same bytes on any OS
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            digest.update(relative.encode() + b"\0" + data)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), digest.hexdigest()[:12]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


INTRO = """
# Index a repo on a Colab GPU, search it on your laptop

Embedding a repo with Qwen3-Embedding-0.6B is slow on a CPU (about 6 s per code chunk on the
dev laptop: psf/requests took 36 minutes) but quick on a GPU. This notebook does the heavy part
on Colab and gives you a zip to merge into your local index. **Searching stays on your laptop:**
one query takes about 0.6 s on a CPU.

**Steps**
1. Runtime → Change runtime type → a **GPU**. The 0.6B model needs only a few GB in float32,
   so a free T4 should be enough and an A100 will be faster. (Not yet run on Colab: if a step
   fails, send the output.)
2. Edit the repo list in the *Parameters* cell. Public GitHub repositories only: no token is used.
3. Runtime → Run all. The last cell downloads `repo_rag_index.zip`.
4. On your machine, from the project folder:
   `uv run repo-rag import-index path\\to\\repo_rag_index.zip`, then
   `uv run repo-rag search "your question" --repo owner/name`.

**Why `float32`:** your laptop embeds *queries* in float32. Indexing in float32 too keeps the
stored vectors and your queries numerically consistent. bfloat16 is faster on a GPU and the
difference is small: on a CPU stand-in, bfloat16 vectors had cosine >= 0.9998 with float32 ones
and returned the same top-1 chunk for 36 of 36 test questions. So `DTYPE = "bfloat16"` in the
Parameters cell is a reasonable speed-up for big repos; float32 is just the conservative default.

**Good to know**
- The notebook embeds the exact `repo_rag` source and library versions of the commit it was
  generated from, and runs the same `repo-rag index` command you would run locally.
- Colab wipes `/content` when the session ends. Download the zip before closing the tab.
- The import *replaces* the repo's chunks in your local index, leaves other repos alone, and
  re-embeds nothing. It refuses an index built with a different embedding model or dimension.
"""

PARAMETERS = """
# --- Parameters -------------------------------------------------------------
REPO_URLS = [
    "https://github.com/psf/requests",
    # "https://github.com/pallets/flask",
]
SHALLOW_CLONE = True        # True: latest commit only (fast), no commit history.
PIN_COMMIT = None           # e.g. "611c6162cbc4": index that revision (needs SHALLOW_CLONE = False)
INDEX_HISTORY = False       # True: also embed commits (needs SHALLOW_CLONE = False)
INDEX_PRS = False           # True: also embed synced PRs and issues; you upload data/metadata.db
DTYPE = "float32"           # "float32" matches a CPU laptop's queries; "bfloat16" is faster
EMBED_BATCH = 64            # texts per GPU forward pass (A100 40 GB; use 16 to 32 on a T4)
PR_EMBED_BATCH = 32         # PR and issue texts are capped at 3,000 characters (about 750 tokens)
COMMIT_EMBED_BATCH = 8      # commit documents run to 2,048 tokens: 64 of them ran out of 40 GB
WRITE_BATCH = 256           # chunks embedded per step before they are written to the store
SMOKE_QUERY = "where are errors handled"   # run against the first repo after indexing
"""

SETUP_TEMPLATE = """
!nvidia-smi --query-gpu=name,memory.total --format=csv
!command -v uv >/dev/null || pip install -q uv
# Pinned to the versions in the project's uv.lock. Colab already ships a CUDA build of torch.
!uv pip install --system -q {pins}
"""

UNPACK_TEMPLATE = """
# The repo_rag package, embedded by notebooks/build_colab_indexer.py (content hash {digest}).
# Don't edit this cell: change src/repo_rag and regenerate the notebook.
import base64, io, pathlib, zipfile

ARCHIVE_B64 = "{archive}"

SRC = pathlib.Path("/content/repo_rag_src")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(ARCHIVE_B64))) as archive:
    archive.extractall(SRC)
print(f"unpacked {{len(list(SRC.rglob('*.py')))}} files into {{SRC}} (content hash {digest})")
"""

SECRETS = """
# Optional: a Hugging Face token makes the model download faster and avoids rate limits.
import os

try:
    from google.colab import userdata

    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    print("HF_TOKEN:", "set" if token else "not set")
except Exception as exc:  # no secret, or not running in Colab
    print("HF_TOKEN: not set (", type(exc).__name__, ") - continuing without it")
"""

RUN = """
import collections, os, re, subprocess, sys, time

# The batch sizes in Parameters are sized for a 40 GB A100. On a smaller GPU (a free T4 has
# 15 GB) the same batches run out of memory, so cap them there. The caps are a guess, not tuned.
import torch

if torch.cuda.is_available():
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)}, {gib:.0f} GiB")
    if gib < 30:
        EMBED_BATCH = min(EMBED_BATCH, 8)
        PR_EMBED_BATCH = min(PR_EMBED_BATCH, 8)
        COMMIT_EMBED_BATCH = min(COMMIT_EMBED_BATCH, 4)
        print(
            f"Small GPU: batches capped at {EMBED_BATCH} (code, docs), {PR_EMBED_BATCH} (PRs), "
            f"{COMMIT_EMBED_BATCH} (commits). This will be slower than an A100."
        )
else:
    print("NO GPU: switch the runtime type to a GPU, or this will take hours.")

ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC),
    "REPO_RAG_DATA_DIR": "/content/data",
    "REPO_RAG_EMBEDDING_DTYPE": DTYPE,
    "REPO_RAG_EMBEDDING_BATCH_SIZE": str(EMBED_BATCH),
    "REPO_RAG_INDEX_WRITE_BATCH": str(WRITE_BATCH),
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def repo_rag(*args, **env):
    # Run the project's own CLI, exactly as on your laptop. Output streams live; if it fails, the
    # last lines are put into the exception text so they survive copying the traceback.
    process = subprocess.Popen(
        [sys.executable, "-m", "repo_rag.cli", *args], env={**ENV, **env},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    tail = collections.deque(maxlen=60)
    for line in process.stdout:
        print(line, end="")
        tail.append(line)
    if process.wait():
        raise RuntimeError(f"repo-rag {args[0]} failed (exit {process.returncode}). Last output:"
                           + chr(10) + "".join(tail))


def slug(url):
    return re.search(r"github\\.com/([^/]+/[^/#?]+)", url).group(1).removesuffix(".git")


if INDEX_PRS:
    # The synced PR and issue rows live in your laptop's data/metadata.db (about 40 MB for
    # psf/requests: it also holds commit and symbol tables, all from public data). Colab has no
    # GitHub token, so upload that file instead of syncing again.
    target = "/content/data/metadata.db"
    if os.path.exists(target) and os.path.getsize(target) > 1_000_000:
        print("Using the metadata.db already uploaded:", os.path.getsize(target), "bytes")
    else:
        from google.colab import files

        uploaded = files.upload()   # Colab renames a repeat upload to 'metadata (2).db'
        assert len(uploaded) == 1, "Upload exactly one file: data/metadata.db from the project"
        os.makedirs("/content/data", exist_ok=True)
        with open(target, "wb") as out:
            out.write(next(iter(uploaded.values())))

for url in REPO_URLS:
    started = time.perf_counter()
    print(f"\\n=== {slug(url)} ===")
    repo_rag("index", url, *(["--no-history"] if SHALLOW_CLONE else []),
             *(["--commit", PIN_COMMIT] if PIN_COMMIT else []))
    if INDEX_HISTORY:
        # `index` only clones; a separate command embeds commits, offline (no GitHub token).
        repo_rag("index-history", slug(url), "--checkout",
                 f"/content/data/repos/{slug(url).replace('/', '__')}", "--offline",
                 REPO_RAG_EMBEDDING_BATCH_SIZE=str(COMMIT_EMBED_BATCH))
    if INDEX_PRS:
        repo_rag("index-prs", slug(url), REPO_RAG_EMBEDDING_BATCH_SIZE=str(PR_EMBED_BATCH))
    print(f"({time.perf_counter() - started:.0f} s including clone)")
"""

SMOKE = """
# A quick search on the GPU box, to see the index works before you download it.
repo_rag("search", SMOKE_QUERY, "--repo", slug(REPO_URLS[0]), "-k", "3")
"""

EXPORT = """
repo_rag("export-index", "/content/repo_rag_index.zip")

import os
print(f"repo_rag_index.zip: {os.path.getsize('/content/repo_rag_index.zip') / 1e6:.1f} MB")

from google.colab import files

files.download("/content/repo_rag_index.zip")
"""

OUTRO = """
## On your machine

```bash
uv run repo-rag import-index path\\to\\repo_rag_index.zip
uv run repo-rag search "where are retries configured" --repo psf/requests
```

If the import says the embedding model or dimension doesn't match, the notebook and your local
`config.py` were generated from different versions: regenerate the notebook with
`uv run python notebooks/build_colab_indexer.py` and run it again.
"""


def build() -> dict:
    archive, digest = source_archive()
    cells = [
        md(INTRO),
        code(SETUP_TEMPLATE.format(pins=" ".join(f'"{pin}"' for pin in runtime_pins()))),
        code(UNPACK_TEMPLATE.format(archive=archive, digest=digest)),
        code(SECRETS),
        md("## Parameters"),
        code(PARAMETERS),
        md("## Index"),
        code(RUN),
        code(SMOKE),
        md("## Download"),
        code(EXPORT),
        md(OUTRO),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    NOTEBOOK.write_text(render(), encoding="utf-8")
    print(f"wrote {NOTEBOOK} ({NOTEBOOK.stat().st_size / 1024:.0f} KB)")
