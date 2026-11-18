"""Generate embedding_bench.ipynb (for Colab) from bench.py.

bench.py stays the single source of truth; the notebook writes it out with %%writefile.
Run: uv run python eval/embedding_bench/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


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
# Embedding model benchmark: code retrieval for repo-rag

Compares **CodeRankEmbed (137M)**, **Qwen3-Embedding-0.6B** and **nomic-embed-code (7B)** on the
same chunks, with BM25 and BM25 hybrids as reference lines. `voyage-code-3` (our current config
default) is added automatically if `VOYAGE_API_KEY` is set.

**Tasks**
- **T1 docstring → code**: ~500 auto-generated queries; docstrings stripped from the corpus.
- **T2 developer questions**: 36 hand-written questions about `psf/requests` with verified gold symbols.
- **T3 bug report → fix commit**: GitHub issue text → the commit that fixed it. Needs `GITHUB_TOKEN`.

**Setup**
1. Runtime → Change runtime type → **A100 GPU**.
2. Optional, in the 🔑 Secrets panel (left sidebar), with notebook access enabled:
   - `GITHUB_TOKEN`: any token with public read access. Without it, T3 is skipped.
   - `VOYAGE_API_KEY`: adds the paid voyage-code-3 reference row.
   - `HF_TOKEN`: faster, rate-limit-free model downloads.
3. Runtime → Run all. It takes roughly 15-25 minutes; most of that is the 7B model.
4. The last cell downloads `results.zip`. Put its contents in `eval/embedding_bench/results/`.
"""

SETUP = """
!nvidia-smi --query-gpu=name,memory.total --format=csv
# uv resolves and installs much faster than pip. --system targets Colab's own Python (no venv).
!command -v uv >/dev/null || pip install -q uv
# transformers<5: CodeRankEmbed's remote code needs a helper that transformers 5 removed.
!uv pip install --system -q "sentence-transformers>=5.1" "transformers>=4.51,<5" rank-bm25 einops tabulate voyageai
"""

SECRETS = """
import os

try:
    from google.colab import userdata

    for name in ("GITHUB_TOKEN", "VOYAGE_API_KEY", "HF_TOKEN"):
        try:
            value = userdata.get(name)
            reason = ""
        except Exception as exc:  # SecretNotFoundError, NotebookAccessError, ...
            value, reason = None, f" ({type(exc).__name__})"
        if value:
            os.environ[name] = value
        print(f"{name}: {'set' if value else 'NOT SET' + reason}")
    if not os.environ.get("GITHUB_TOKEN"):
        print(
            "\\n>>> T3 (bug report -> fix commit) will be SKIPPED. Add GITHUB_TOKEN in the key "
            "panel on the left and switch on 'Notebook access' for it, then re-run this cell."
        )
except ImportError:
    print("Not running in Colab; using existing environment variables.")
"""

RUN = """
import os

models = ["coderankembed", "qwen3-0.6b", "nomic-embed-code"]
if os.environ.get("VOYAGE_API_KEY"):
    models.append("voyage-code-3")

!python bench.py --data-dir /content/bench_data --out /content/results --models {" ".join(models)}
"""

SUMMARY = """
from pathlib import Path

from IPython.display import Markdown, display

display(Markdown(Path("/content/results/summary.md").read_text(encoding="utf-8")))
"""

CHART = """
import matplotlib.pyplot as plt
import pandas as pd

metrics = pd.read_csv("/content/results/metrics.csv")
speed = pd.read_csv("/content/results/speed.csv")

quality = (
    metrics[metrics.variant.isin(["dense", "hybrid_bm25"])]
    .groupby(["model", "variant"])["ndcg@10"].mean().unstack()
)
merged = speed.set_index("model").join(quality)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
for model, row in merged.iterrows():
    ax.scatter(row.docs_per_s, row.dense, s=60)
    ax.annotate(model, (row.docs_per_s, row.dense), textcoords="offset points", xytext=(6, 4))
ax.set_xscale("log")
ax.set_xlabel("Corpus encoding throughput (docs/s, A100, log scale)")
ax.set_ylabel("Mean nDCG@10 across tasks (dense)")
ax.set_title("Quality vs indexing speed")
ax.grid(alpha=0.3)

per_task = metrics[metrics.variant == "dense"].pivot(index="task", columns="model", values="ndcg@10")
per_task.plot.bar(ax=axes[1], rot=0)
axes[1].set_ylabel("nDCG@10 (dense)")
axes[1].set_title("Quality by task")
axes[1].grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/content/results/quality_vs_speed.png", dpi=150)
plt.show()
"""

FAILURES = """
# Which developer questions does each model miss? (Hit@5 = 0)
pq = pd.read_csv("/content/results/t2_per_question.csv")
misses = pq[(pq.task == "T2r_dev_questions_in_repo") & (pq.variant == "dense") & (pq["hit@5"] == 0)]
misses.groupby("model")["q"].apply(list).to_dict()
"""

DOWNLOAD = """
!cd /content && zip -qr results.zip results
from google.colab import files

files.download("/content/results.zip")
"""


T3_INTRO = """
# T3 tuning: closing Qwen3's bug report -> fix commit gap

Run 2 had Qwen3-Embedding-0.6B at nDCG@10 0.488 on T3 vs 0.596 for nomic-embed-code (7B).
This notebook runs **only Qwen3** and tries 5 query instructions x 4 commit document formats
(message only / + files + touched symbols / + short diff / the run-2 baseline with a long diff).

To keep it honest, the 144 issues are split 50/50 into dev and test. The winner is chosen on
**dev** and reported on **test**, with a paired bootstrap CI against the baseline.

**Setup:** an A100 runtime (an L4 or T4 also works, just slower), plus the `GITHUB_TOKEN`
secret with notebook access. It takes about 5 minutes. The last cell downloads
`t3_results.zip`.
"""

T3_RUN = """
!python t3_tuning.py --data-dir /content/bench_data --out /content/t3_results
"""

T3_SUMMARY = """
from pathlib import Path

from IPython.display import Markdown, display

display(Markdown(Path("/content/t3_results/t3_summary.md").read_text(encoding="utf-8")))
"""

T3_DOWNLOAD = """
!cd /content && zip -qr t3_results.zip t3_results
from google.colab import files

files.download("/content/t3_results.zip")
"""


RERANK_INTRO = """
# T3 rerank experiment: reranker vs bigger embedder for bug report -> fix commit

After T3 tuning, Qwen3-Embedding-0.6B still trails nomic-embed-code (7B) by about 0.1
nDCG@10 on T3. This tests the two remaining levers on the same 144 issues:

- **First stage:** Qwen3-Embedding-0.6B @512d (production) vs **Qwen3-Embedding-4B**
  (@512d and full 2,560d).
- **Rerank the top 50** with **Qwen3-Reranker-0.6B**, **Qwen3-Reranker-4B** or
  **bge-reranker-v2-m3** (the reranker currently in config.py).

Results are reported on all 144 issues and on the held-out test half from t3_tuning, with
paired bootstrap CIs against the production baseline. It also reports rerank latency per query.

**Setup:** an **A100** runtime and the `GITHUB_TOKEN` secret with notebook access. It takes
roughly 15-25 minutes; the 4B reranker is the slow part. The last cell downloads
`rerank_results.zip`.
"""

RERANK_RUN = """
!python t3_rerank.py --data-dir /content/bench_data --out /content/rerank_results
"""

RERANK_SUMMARY = """
from pathlib import Path

from IPython.display import Markdown, display

display(Markdown(Path("/content/rerank_results/t3_rerank_summary.md").read_text(encoding="utf-8")))
"""

RERANK_DOWNLOAD = """
!cd /content && zip -qr rerank_results.zip rerank_results
from google.colab import files

files.download("/content/rerank_results.zip")
"""


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def _source_cell(name: str) -> dict:
    source = (HERE / name).read_text(encoding="utf-8")
    return code(f"%%writefile {name}\n" + source)


def build() -> dict:
    return _notebook(
        [
            md(INTRO),
            code(SETUP),
            code(SECRETS),
            md(
                "## Benchmark code\nGenerated from `eval/embedding_bench/bench.py`. "
                "Edit that file, not this cell."
            ),
            _source_cell("bench.py"),
            md("## Run"),
            code(RUN),
            md("## Results"),
            code(SUMMARY),
            code(CHART),
            code(FAILURES),
            md("## Download"),
            code(DOWNLOAD),
        ]
    )


def build_t3() -> dict:
    return _notebook(
        [
            md(T3_INTRO),
            code(SETUP),
            code(SECRETS),
            md(
                "## Code\nGenerated from `bench.py` and `t3_tuning.py`. "
                "Edit those files, not these cells."
            ),
            _source_cell("bench.py"),
            _source_cell("t3_tuning.py"),
            md("## Run"),
            code(T3_RUN),
            code(T3_SUMMARY),
            md("## Download"),
            code(T3_DOWNLOAD),
        ]
    )


def build_rerank() -> dict:
    return _notebook(
        [
            md(RERANK_INTRO),
            code(SETUP),
            code(SECRETS),
            md(
                "## Code\nGenerated from `bench.py`, `t3_tuning.py` and `t3_rerank.py`. "
                "Edit those files, not these cells."
            ),
            _source_cell("bench.py"),
            _source_cell("t3_tuning.py"),
            _source_cell("t3_rerank.py"),
            md("## Run"),
            code(RERANK_RUN),
            code(RERANK_SUMMARY),
            md("## Download"),
            code(RERANK_DOWNLOAD),
        ]
    )


if __name__ == "__main__":
    for name, nb in (
        ("embedding_bench.ipynb", build()),
        ("t3_tuning.ipynb", build_t3()),
        ("t3_rerank.ipynb", build_rerank()),
    ):
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out}")
