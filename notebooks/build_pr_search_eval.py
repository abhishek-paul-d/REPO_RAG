"""Generate notebooks/pr_search_on_colab.ipynb: embed encode/httpx PRs and issues on a GPU and
measure issue -> fixing-PR search (BM25 vs hybrid).

Self-contained like index_on_colab.ipynb: it embeds the `repo_rag` source, the pinned
dependency versions, the redacted GitHub snapshot in eval/datasets/httpx_pr_snapshot/ and
eval/run_pr_search_eval.py. It runs the real CLI (`repo-rag index-prs`), so no second
implementation exists. No GitHub token and no network access to GitHub are needed.

Regenerate after ANY change under src/repo_rag, pyproject.toml, uv.lock, the snapshot or
eval/run_pr_search_eval.py:
    uv run python notebooks/build_pr_search_eval.py
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.build_colab_indexer import (  # noqa: E402
    SECRETS,
    SETUP_TEMPLATE,
    UNPACK_TEMPLATE,
    code,
    md,
    runtime_pins,
    source_archive,
)

NOTEBOOK = Path(__file__).with_name("pr_search_on_colab.ipynb")
SNAPSHOT = ROOT / "eval/datasets/httpx_pr_snapshot"
SCRIPT = ROOT / "eval/run_pr_search_eval.py"
SLUG = "encode/httpx"


def payload() -> tuple[str, str]:
    """(base64 of a deterministic tar.gz of the snapshot files and the eval script, hash)."""
    entries = {
        f"github_cache/sync/{p.name}": p.read_bytes() for p in sorted(SNAPSHOT.glob("*.json"))
    }
    entries["run_pr_search_eval.py"] = SCRIPT.read_bytes().replace(b"\r\n", b"\n")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    packed = gzip.compress(raw.getvalue(), mtime=0)
    return base64.b64encode(packed).decode("ascii"), hashlib.sha256(packed).hexdigest()[:12]


INTRO = """
# PR / issue search on a GPU: dense + keyword vs keyword only

Question: given a GitHub issue, does search find the pull request that closed it? Pairs come
from GitHub's own closing links (45 closed issues of encode/httpx whose closing PR is in the
500 newest PRs), the query is the issue title plus the first 1,000 characters of its body, and
the corpus is those 500 PRs. A keyword-only (BM25) run on the laptop gave Hit@5 0.600 (95%
interval 0.467 to 0.733). This notebook adds the **hybrid** (Qwen3-Embedding-0.6B dense +
BM25, weighted RRF) number, which needs a GPU: the laptop CPU embeds about 0.1 documents/s.

**Steps:** Runtime → Change runtime type → an **A100** (or L4) GPU → Run all. The last strategy
loads the 8B reranker (Qwen3-Reranker-8B), which does not fit a 16 GB T4; to use a T4 drop
`hybrid_fused_rerank` from STRATEGIES below.
The last cell downloads `pr_search_results.zip` (report, summary) and `pr_index.zip`.

**What it embeds:** 500 PRs + 500 issues of the redacted snapshot in the notebook (emails
removed). No GitHub token or GitHub access. Not yet run on Colab: if a step fails, send the
output.
"""

PARAMETERS = """
# --- Parameters -------------------------------------------------------------
DTYPE = "float32"      # matches CPU laptop queries; "bfloat16" is faster on a GPU
STRATEGIES = "bm25,hybrid,hybrid_fused,hybrid_fused_rerank"
"""

PAYLOAD_TEMPLATE = """
# The redacted GitHub snapshot and the eval script, embedded by notebooks/build_pr_search_eval.py
# (content hash {digest}). Don't edit this cell: regenerate the notebook instead.
import base64, io, pathlib, tarfile

PAYLOAD_B64 = "{archive}"

DATA = pathlib.Path("/content/data")
(DATA / "github_cache" / "sync").mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD_B64)), mode="r:gz") as tar:
    for member in tar.getmembers():
        root = DATA if member.name.startswith("github_cache/") else pathlib.Path("/content")
        target = root / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tar.extractfile(member).read())
print(sorted(str(p.relative_to("/content")) for p in pathlib.Path("/content").rglob("*.json")))
"""

RUN = """
import os, subprocess, sys, time

ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC),
    "REPO_RAG_DATA_DIR": "/content/data",
    "REPO_RAG_EMBEDDING_DTYPE": DTYPE,
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
SLUG = "__SLUG__"


def repo_rag(*args):
    subprocess.run([sys.executable, "-m", "repo_rag.cli", *args], env=ENV, check=True)


# Load the snapshot into metadata.db exactly as `sync-github` would, without the network.
subprocess.run(
    [
        sys.executable,
        "-c",
        "from pathlib import Path\\n"
        "from repo_rag.config import Settings\\n"
        "from repo_rag.history.github_sync import GitHubSync\\n"
        "from repo_rag.history.metadata_db import MetadataDB\\n"
        "s = Settings(data_dir=Path('/content/data'))\\n"
        "g = GitHubSync(s, None)\\n"
        "with MetadataDB(s.metadata_db_path, readonly=False) as db:\\n"
        f"    db.sync_github('{SLUG}', g.pull_requests('{SLUG}'), g.issues('{SLUG}'),"
        " prs_complete=False, issues_complete=False)\\n"
        "print('metadata.db loaded')",
    ],
    env=ENV,
    check=True,
)
started = time.perf_counter()
repo_rag("index-prs", SLUG)
print(f"(index-prs took {time.perf_counter() - started:.0f} s)")
"""

SMOKE = """
repo_rag(
    "search", "retry on connection timeout", "--repo", SLUG, "--kind", "pull_requests", "-k", "3"
)
"""

EVAL = """
subprocess.run(
    [sys.executable, "/content/run_pr_search_eval.py", SLUG, "--data-dir", "/content/data",
     "--strategies", STRATEGIES, "--out", "/content/pr_search_results"],
    env=ENV, check=True,
)
"""

EXPORT = """
import shutil
shutil.make_archive("/content/pr_search_results", "zip", "/content/pr_search_results")
repo_rag("export-index", "/content/pr_index.zip")
from google.colab import files

files.download("/content/pr_search_results.zip")
files.download("/content/pr_index.zip")
"""

OUTRO = """
## Afterwards

Send `pr_search_results.zip`'s `summary.md` back (or unzip it into
`eval/facts_results/httpx_pr_search_gpu/`). `pr_index.zip` can be merged locally with
`uv run repo-rag import-index pr_index.zip` so `search --kind pull_requests` uses the real
embeddings on the laptop.
"""


def build() -> dict:
    archive, digest = payload()
    src_archive, src_digest = source_archive()
    cells = [
        md(INTRO),
        code(SETUP_TEMPLATE.format(pins=" ".join(f'"{pin}"' for pin in runtime_pins()))),
        code(UNPACK_TEMPLATE.format(archive=src_archive, digest=src_digest)),
        code(PAYLOAD_TEMPLATE.format(archive=archive, digest=digest)),
        code(SECRETS),
        md("## Parameters"),
        code(PARAMETERS),
        md("## Embed the PRs and issues"),
        code(RUN.replace("__SLUG__", SLUG)),
        code(SMOKE),
        md("## Measure"),
        code(EVAL),
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
