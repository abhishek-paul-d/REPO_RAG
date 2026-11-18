"""Generate notebooks/commit_summary_pilot_on_colab.ipynb: embed the pilot commits with and
without LLM-written summaries on a Colab GPU and return the vectors.

Self-contained like index_on_colab.ipynb: it embeds the `repo_rag` source (so the embeddings are
the ones the laptop uses), the pinned dependency versions, and `eval/commit_summary_pilot.py`
(the one place that defines how a summary is added to a commit's text and how ranks are
compared). The pilot data itself (commits, summaries, queries) is NOT embedded: you upload
`eval/commit_summary_pilot/pilot_input.zip`, which `commit_summary_pilot.py package` writes.

Regenerate after ANY change under src/repo_rag, pyproject.toml, uv.lock or
eval/commit_summary_pilot.py:
    uv run python notebooks/build_commit_summary_pilot.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
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

NOTEBOOK = Path(__file__).with_name("commit_summary_pilot_on_colab.ipynb")
SCRIPT = ROOT / "eval/commit_summary_pilot.py"

INTRO = """
# Pilot: do commit summaries help find the right fix commit?

Many commits have a bare message, so a bug report written in English matches only their diff.
Before summarising all 4,883 commits, this pilot embeds about 1,100 of them twice on a GPU:
once as indexed today, once with a short plain-English `Summary:` line (written by the 27B model
from the commit's own message and diff, never from a bug report). It then ranks the 141 bug
reports' gold fix commits both ways.

**Before you start (on your laptop)**
1. `uv run python eval/commit_summary_pilot.py prepare --random 1000`
2. `uv run python eval/commit_summary_pilot.py summarize --url <27B server> --token <token>`
   (about an hour; resumable)
3. `uv run python eval/commit_summary_pilot.py package` writes
   `eval/commit_summary_pilot/pilot_input.zip`

**Here**
1. Runtime → Change runtime type → a **GPU** (an A100 is fastest; a T4 works but is slow).
2. Run all. When asked, upload `pilot_input.zip`.
3. The last cell prints a first answer (dense search only) and downloads `pilot_vectors.npz`.

**Back on your laptop:** `uv run python eval/commit_summary_pilot.py score --vectors
path/to/pilot_vectors.npz` adds the hybrid (dense + BM25) comparison the agent really uses.

Ranks here are within about 1,100 commits, not the full 4,883: they show direction, not the
final score. A gain still has to be confirmed through the agent.
"""

PARAMETERS = """
# --- Parameters -------------------------------------------------------------
DTYPE = "float32"      # matches the laptop's queries; "bfloat16" is faster and nearly identical
DOC_BATCH = 8          # commit texts run to 2,048 tokens: large batches ran out of 40 GB before
"""

UPLOAD = """
import io, json, os, zipfile

from google.colab import files

uploaded = files.upload()   # pilot_input.zip from `commit_summary_pilot.py package`
assert len(uploaded) == 1, "Upload exactly one file: pilot_input.zip"
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    commits = [json.loads(x) for x in archive.read("commits.jsonl").decode().splitlines() if x]
    queries = [json.loads(x) for x in archive.read("queries.jsonl").decode().splitlines() if x]
assert all(c.get("summary") for c in commits), "some commits have no summary: rerun summarize"
print(f"{len(commits)} commits, {len(queries)} queries")
print("example summary:", commits[0]["summary"][:300])
"""

SCRIPT_CELL = """
# eval/commit_summary_pilot.py, embedded by notebooks/build_commit_summary_pilot.py
# (content hash {digest}). Don't edit this cell: edit the script and regenerate the notebook.
import base64, pathlib

pathlib.Path("/content/commit_summary_pilot.py").write_bytes(base64.b64decode("{script}"))
"""

EMBED = """
import os, sys, time

os.environ["PYTHONPATH"] = str(SRC)
os.environ["REPO_RAG_EMBEDDING_DTYPE"] = DTYPE
os.environ["REPO_RAG_EMBEDDING_BATCH_SIZE"] = str(DOC_BATCH)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path[:0] = [str(SRC), "/content"]

import numpy as np
import torch

from repo_rag.config import Settings
from repo_rag.store.embeddings import LocalEmbeddings

import commit_summary_pilot as pilot

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (this will take hours)"
print("GPU:", gpu)
embedder = LocalEmbeddings(Settings())
shas = [c["sha"] for c in commits]
texts = {
    "baseline": [c["text"] for c in commits],
    "summary": [pilot.with_summary(c["text"], c["summary"]) for c in commits],
}
vectors = {}
for name, batch in texts.items():
    started = time.perf_counter()
    vectors[name] = np.asarray(embedder.embed_documents(batch), dtype=np.float32)
    print(f"{name}: {vectors[name].shape} in {time.perf_counter() - started:.0f} s")
started = time.perf_counter()
query_vectors = np.asarray([embedder.embed_query(q["query"]) for q in queries], dtype=np.float32)
print(f"queries: {query_vectors.shape} in {time.perf_counter() - started:.0f} s")
"""

FIRST_ANSWER = """
# A first answer on the GPU box: dense search only, the gold commit's rank inside this corpus.
ranks = {}
for name in ("baseline", "summary"):
    ranks[name] = [
        pilot.ranks_for(pilot.dense_order(vectors[name], qv), shas, q["gold"])
        for q, qv in zip(queries, query_vectors)
    ]
    print(f"{name:9s}", pilot.summarize_ranks(ranks[name]))
print("summary vs baseline:", pilot.compare_ranks(ranks["baseline"], ranks["summary"]))
"""

SAVE = """
np.savez_compressed(
    "/content/pilot_vectors.npz",
    shas=np.array(shas),
    queries=query_vectors,
    vectors_baseline=vectors["baseline"],
    vectors_summary=vectors["summary"],
)
print(f"pilot_vectors.npz: {os.path.getsize('/content/pilot_vectors.npz') / 1e6:.1f} MB")

from google.colab import files

files.download("/content/pilot_vectors.npz")
"""

OUTRO = """
## On your machine

```bash
uv run python eval/commit_summary_pilot.py score --vectors path\\to\\pilot_vectors.npz
```

It prints hit@1/5/10/20 and MRR with and without the summary, for dense search and for the
hybrid search the agent uses, and how many of the gold commits moved up or down. Colab wipes
`/content` when the session ends, so download the vectors before closing the tab.
"""


def script_payload() -> tuple[str, str]:
    data = SCRIPT.read_bytes().replace(b"\r\n", b"\n")
    return base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()[:12]


def build() -> dict:
    archive, digest = source_archive()
    script, script_digest = script_payload()
    cells = [
        md(INTRO),
        code(SETUP_TEMPLATE.format(pins=" ".join(f'"{pin}"' for pin in runtime_pins()))),
        code(UNPACK_TEMPLATE.format(archive=archive, digest=digest)),
        code(SCRIPT_CELL.format(script=script, digest=script_digest)),
        code(SECRETS),
        md("## Parameters"),
        code(PARAMETERS),
        md("## Upload the pilot data"),
        code(UPLOAD),
        md("## Embed with and without the summary"),
        code(EMBED),
        code(FIRST_ANSWER),
        md("## Download"),
        code(SAVE),
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
