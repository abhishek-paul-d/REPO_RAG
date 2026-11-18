"""Resume notebook for the partial Phase 5 commit depth run (checkpoint 869432f8c494).

The Drive checkpoint was written by the source ZIP whose SHA-256 starts with 869432f8c494, and
every stored run records that source's file hashes. Source changed after that run, so the current
`phase5_commit_depth_source.zip` cannot resume it. The checkpoint embeds its own `source.zip`; it is
vendored here byte for byte as `phase5_commit_depth_resume_source.zip`. This notebook is the
current depth notebook with that payload, so the unchanged weights, depth and selected-weight
stages are skipped and only timing, analysis and the CLI sample remain.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks import build_commit_depth as depth  # noqa: E402
from notebooks.build_colab_indexer import md  # noqa: E402

RESUME_ZIP = ROOT / "notebooks/phase5_commit_depth_resume_source.zip"
NOTEBOOK = ROOT / "notebooks/phase5_commit_depth_resume_on_colab.ipynb"
CHECKPOINT_PREFIX = "869432f8c494"
DRIVE_NAME = "phase5_commit_depth_resume_source.zip"

INTRO = f"""# Phase 5: finish the commit depth run (resume from checkpoint {CHECKPOINT_PREFIX})

Put these two files at the top level of My Drive (no folder):
**{DRIVE_NAME}** (from the project's notebooks folder) and
**phase5_commit_depth_checkpoint_{CHECKPOINT_PREFIX}.zip** (the checkpoint you already have).
Use an **A100 40 GB** and run in order.

The checkpoint holds the finished weight, 4B/8B depth-200 and selected-weight stages for all
three repos, produced by the source in {DRIVE_NAME}. That source is used unchanged so the
stored hashes still match. Those stages are skipped ("Already complete"). What is left:
the 8B production **timing** (60 frozen test questions, three rotated warm repetitions, depths
50/100/200), the analysis with the frozen decision rule, and the CLI sample that records its
exact bug report beside the cited commits.

Everything else follows the original notebook: the frozen protocol, no default changes, A100
experiment timing only, labels link-derived with independent review pending. The last cell
checks that the Drive results ZIP matches, flushes Drive and releases the runtime.
Return **phase5_commit_depth_results.zip**.
"""


def build() -> dict:
    original = depth.source_archive
    depth.source_archive = lambda: RESUME_ZIP.read_bytes()  # type: ignore[assignment]
    try:
        notebook = depth.build()
    finally:
        depth.source_archive = original  # type: ignore[assignment]
    text = json.dumps(notebook).replace("phase5_commit_depth_source.zip", DRIVE_NAME)
    notebook = json.loads(text)
    notebook["cells"][0] = md(INTRO)
    return notebook


if __name__ == "__main__":
    if hashlib.sha256(RESUME_ZIP.read_bytes()).hexdigest()[:12] != CHECKPOINT_PREFIX:
        raise SystemExit("Resume payload does not match checkpoint " + CHECKPOINT_PREFIX)
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(NOTEBOOK.name, NOTEBOOK.stat().st_size)
