"""Generate notebooks/flask_pr_search_on_colab.ipynb: the PR-search measurement on pallets/flask.

Same notebook as build_pr_search_eval.py (httpx) with a different repo and snapshot; it exists to
check that the httpx result holds on a second repository.

    uv run python notebooks/build_flask_pr_search_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks import build_pr_search_eval as base  # noqa: E402

NOTEBOOK = Path(__file__).with_name("flask_pr_search_on_colab.ipynb")
base.SNAPSHOT = ROOT / "eval/datasets/flask_pr_snapshot"
base.SLUG = "pallets/flask"
base.INTRO = base.INTRO.replace("encode/httpx", "pallets/flask").replace(
    "(45 closed issues of pallets/flask whose closing PR is in the\n500 newest PRs)",
    "(closed issues of pallets/flask whose closing PR is in the 500 newest PRs)",
)
base.INTRO = (
    base.INTRO.split("A keyword-only")[0]
    + (
        "A keyword-only run on the laptop gave Hit@5 0.793 (58 pairs). This is the **second-repo\n"
        "check** of the httpx result (hybrid_fused 0.867, hybrid 0.800, BM25 0.600): run\n"
        "bm25, hybrid and hybrid_fused (full query + title-only query) and compare.\n"
    )
    + "\n"
    + "**Steps:**"
    + base.INTRO.split("**Steps:**", 1)[1]
)

if __name__ == "__main__":
    NOTEBOOK.write_text(base.render(), encoding="utf-8")
    print(f"wrote {NOTEBOOK} ({NOTEBOOK.stat().st_size / 1024:.0f} KB)")
