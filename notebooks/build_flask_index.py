"""Generate notebooks/index_flask_on_colab.ipynb: index pallets/flask (code, docs and its commit
history) on a Colab GPU, sized for a free T4.

It is `index_on_colab.ipynb` with the Parameters set for the second repository of the commit
experiments: pallets/flask pinned at the commit the Phase 5 bug -> fix pairs were built on, the
full history cloned and embedded (`INDEX_HISTORY`), float32 (a T4 has no fast bfloat16, and the
laptop embeds queries in float32), and no PRs or issues (the commit questions do not need them).
Everything else (the embedded source, the pins, the batch caps for small GPUs) is the same code.

Regenerate after ANY change under src/repo_rag, pyproject.toml or uv.lock:
    uv run python notebooks/build_flask_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks import build_colab_indexer as base  # noqa: E402

NOTEBOOK = Path(__file__).with_name("index_flask_on_colab.ipynb")
REPO = "pallets/flask"
PIN = "d086db856be187255b8ec61ef409357393020f32"  # phase5_bugfix_v2 protocol pin for pallets/flask

INTRO = f"""
# Index pallets/flask on a Colab T4, use it on your laptop

This builds the second repository for the commit-search experiments: Flask's code, docs and its
commit history (about 3,800 non-merge commits up to `{PIN[:12]}`), embedded with
Qwen3-Embedding-0.6B in float32 so the vectors match your laptop's queries.

**Steps**
1. Runtime -> Change runtime type -> **T4 GPU** (an A100 is faster but not needed).
2. Runtime -> Run all. Keep the tab open: the commit step is the slow one (batches are capped
   at 4 documents of up to 2,048 tokens on a 16 GB card; expect roughly an hour, not measured).
3. The last cell downloads `repo_rag_index.zip`.
4. On your laptop, into a SEPARATE data folder so the shipped psf/requests index is untouched
   (see the end of this notebook).

**Why a T4 works:** the notebook reads the GPU size and caps its batches below 30 GiB. A T4
already indexed psf/requests's PRs and issues and embedded the 1,143-commit summary pilot.
**Why float32:** a T4 has no fast bfloat16, and keeping documents and queries in the same
precision keeps the stored vectors consistent with the laptop.
"""

PARAMETERS = f"""
# --- Parameters -------------------------------------------------------------
REPO_URLS = [
    "https://github.com/{REPO}",
]
SHALLOW_CLONE = False       # the commit history is needed
PIN_COMMIT = "{PIN}"
INDEX_HISTORY = True        # embed the commits (the point of this run)
INDEX_PRS = False           # the commit questions do not need PRs or issues
DTYPE = "float32"           # a T4 has no fast bfloat16; matches the laptop's queries
EMBED_BATCH = 8             # texts per forward pass for code and docs (capped again on a small GPU)
PR_EMBED_BATCH = 8
COMMIT_EMBED_BATCH = 4      # commit documents run to 2,048 tokens
WRITE_BATCH = 256
SMOKE_QUERY = "how does Flask push the request context"   # run against Flask after indexing
"""

SOURCE_CLONE = r"D:\Repo_chatbot\data\phase5_repos\pallets__flask"  # untouched Phase 5 clone
ZIP = "path" + chr(92) + "to" + chr(92) + "repo_rag_index.zip"
DATA = r"D:\Repo_chatbot\data_flask"  # a separate folder: the shipped psf/requests index stays
CHECKOUT = rf"{DATA}\repos\pallets__flask"

OUTRO = f"""
## On your machine (a separate data folder)

```powershell
$env:REPO_RAG_DATA_DIR = "{DATA}"
git clone --no-hardlinks {SOURCE_CLONE} {CHECKOUT}
git -C {CHECKOUT} checkout {PIN}
uv run repo-rag import-index {ZIP}
uv run repo-rag index-history {REPO} --checkout {CHECKOUT} --ref {PIN} --offline
uv run repo-rag index-symbols {REPO} --checkout {CHECKOUT}
uv run repo-rag search "how does Flask push the request context" --repo {REPO} --kind commits
```

`git clone` + `checkout` is needed because `phase5_repos\\pallets__flask` has no working tree on
disk (it was cloned with `--no-checkout`, so `index-symbols` would see 0 files there), and
`build_agent` reads `{CHECKOUT}` when the imported manifest has no clone path.
`index-history` here only writes the commit tables into `metadata.db` (the vectors came in the
zip; unchanged commits are not embedded again). If the import says the embedding model or
dimension does not match, regenerate this notebook with
`uv run python notebooks/build_flask_index.py` and run it again.
"""


def build() -> dict:
    archive, digest = base.source_archive()
    cells = [
        base.md(INTRO),
        base.code(
            base.SETUP_TEMPLATE.format(pins=" ".join(f'"{pin}"' for pin in base.runtime_pins()))
        ),
        base.code(base.UNPACK_TEMPLATE.format(archive=archive, digest=digest)),
        base.code(base.SECRETS),
        base.md("## Parameters"),
        base.code(PARAMETERS),
        base.md("## Index"),
        base.code(base.RUN),
        base.code(base.SMOKE),
        base.md("## Download"),
        base.code(base.EXPORT),
        base.md(OUTRO),
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
