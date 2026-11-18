"""Package dev-selected facts and precomputed code/query vectors for the E3 Colab run.

No new model run. Call after the hybrid E2 evaluation finishes.
"""

from __future__ import annotations

import hashlib

# ruff: noqa: E402
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_architecture_dataset import OUT, REPOS, digest, verify_frozen, write
from eval.run_architecture_eval import DiskEmbeddings, NoEmbeddings, rows
from repo_rag.agent.prompts import SUMMARY_PROMPT
from repo_rag.config import Settings
from repo_rag.store.vectorstore import VectorStore
from repo_rag.summaries.prose import CHECK_VERSION

CAPSULE = ROOT / "notebooks/summary_prose_inputs.zip"
RESULTS = ROOT / "eval/facts_results/architecture_v1_hybrid"
PROPOSAL = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "parameters": "4.0 billion",
    "license": "Apache-2.0",
    "source": "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
    "estimated_vram": "8.05 GB bf16 weights; plan 12-16 GB including short-context KV cache and runtime; estimate, not measured",
    "reason": "non-thinking instruct model; existing transformers supports it; small enough for Colab without quantization or a new dependency",
    "decoding": {"do_sample": False, "max_new_tokens": 160},
    "approval_status": "pending explicit user approval; neither downloaded nor run",
    "label_review": "unreviewed by a person",
    "supplements": "architecture_v1/protocol.json E3; model choice requires explicit approval before run",
    "check_version": CHECK_VERSION,
    "check_rules": (
        "reject a paragraph that contains a code-like name (dotted, slashed, snake_case, camelCase, "
        "backticked, followed by '(', or a capitalised word inside a sentence) or a number that "
        "is not in the facts; plain lowercase words are free"
    ),
    "prompt_sha256": hashlib.sha256(SUMMARY_PROMPT.encode()).hexdigest(),
    "amendment": (
        "v2, before any model run: v1 rejected every word absent from the facts, which discards "
        "faithful natural English and ignores invented numbers (tested 2026-11-01); see "
        "model_proposal_v1.json"
    ),
    "supersedes_sha256": "633f206709181bd01defccd2b381fb04f42c5b080bff6496bead380ffc132dc8",
}


def prepare() -> None:
    verify_frozen()
    proposed = json.loads((OUT / "model_proposal.json").read_text(encoding="utf-8"))
    proposed_freeze = json.loads((OUT / "model_proposal.freeze.json").read_text(encoding="utf-8"))
    if (
        proposed != PROPOSAL
        or digest(OUT / "model_proposal.json") != proposed_freeze["sha256"]["model_proposal.json"]
    ):
        raise ValueError("Frozen model proposal changed")
    selection = json.loads((RESULTS / "selection_E2.json").read_text(encoding="utf-8"))
    if selection["only_split"] != "dev":
        raise ValueError("E3 requires a dev-only E2 selection")
    frozen_settings = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))["settings"]
    settings = Settings(**{**frozen_settings, "data_dir": ROOT / "data/architecture_v1/hybrid"})
    embeddings = DiskEmbeddings(settings.data_dir / "cache", settings, NoEmbeddings())
    store = VectorStore(settings, embeddings)
    entries = {
        f"eval/datasets/architecture_v1/{p.name}": p.read_bytes() for p in OUT.glob("*.json*")
    }
    entries["selection.json"] = json.dumps(selection).encode()
    entries["model_proposal.json"] = json.dumps(PROPOSAL, indent=2).encode()
    for name, (slug, _, _) in REPOS.items():
        facts = RESULTS / selection["variant"] / f"{name}_summaries.json"
        entries[f"facts/{name}.json"] = facts.read_bytes()
        ids, texts, metas, vectors = store.export_records("code", slug)
        entries[f"code/{name}.json"] = json.dumps(
            {"ids": ids, "texts": texts, "metas": metas}
        ).encode()
        buf = io.BytesIO()
        np.save(buf, vectors, allow_pickle=False)
        entries[f"code/{name}.npy"] = buf.getvalue()
    questions = [q for name in REPOS for q in rows(OUT / f"{name}.jsonl")]
    questions += rows(OUT / "requests_code_regression.jsonl")
    query_vectors = {}
    for q in questions:
        found = embeddings.db.execute(
            "SELECT value FROM vectors WHERE key=?", (embeddings.key(q["question"], True),)
        ).fetchone()
        if found is None:
            raise ValueError(f"Query not embedded: {q['id']}")
        query_vectors[q["question"]] = np.frombuffer(found[0], dtype=np.float32).tolist()
    entries["queries.json"] = json.dumps(query_vectors).encode()
    sources = [
        "eval/__init__.py",
        "eval/retrieval/__init__.py",
        "eval/retrieval/statistics.py",
        "eval/build_architecture_dataset.py",
        "eval/run_architecture_eval.py",
        "eval/run_summary_prose.py",
    ]
    for source in sources:
        entries[source] = (ROOT / source).read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
    entries["capsule_manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    with zipfile.ZipFile(CAPSULE, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    write(RESULTS / "model_proposal.json", PROPOSAL)
    write(
        RESULTS / "prose_capsule.json",
        {
            "sha256": digest(CAPSULE),
            "selection": selection,
            "facts_source": str(RESULTS.relative_to(ROOT)),
            "approval": "pending",
            "label_review": "unreviewed by a person",
        },
    )
    print(f"Wrote {CAPSULE.name}: {digest(CAPSULE)}")


if __name__ == "__main__":
    prepare()
