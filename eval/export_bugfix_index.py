"""Export the already embedded Phase 5 candidates and capture one real CLI citation.

No re-embedding, LLM call, or GitHub lookup. metadata.db is delivered separately from
the portable vector ZIP; code/docs collections in this history-only bundle are empty.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.prepare_bugfix_eval import read_lines, verify_frozen  # noqa: E402
from eval.run_bugfix_eval import settings_for  # noqa: E402
from repo_rag.history.issue_linker import Issue, IssueLink  # noqa: E402
from repo_rag.history.metadata_db import MetadataDB, record_from_dict  # noqa: E402
from repo_rag.store.bm25 import Bm25Index, bm25_path  # noqa: E402
from repo_rag.store.embeddings import get_embeddings  # noqa: E402
from repo_rag.store.manifest import Manifest, RepoEntry  # noqa: E402
from repo_rag.store.transfer import export_index  # noqa: E402
from repo_rag.store.vectorstore import VectorStore  # noqa: E402


def save_cli_sample(root: Path, sample: dict, output: str) -> None:
    """Keep the exact report with its evidence, so relevance is independently reviewable."""
    query = {key: sample[key] for key in ("id", "repo", "query")}
    (root / "cli_sample_query.json").write_text(
        json.dumps(query, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "cli_sample.txt").write_text(
        f"Repository: {sample['repo']}\nQuestion ID: {sample['id']}\n\n"
        f"Bug report:\n{sample['query']}\n\nRetrieved commit evidence:\n{output}",
        encoding="utf-8",
    )


def export(root: Path) -> None:
    datasets = sorted((root / "datasets").iterdir())
    protocol = json.loads((datasets[0] / "protocol.json").read_text(encoding="utf-8"))
    settings = settings_for(protocol, root / "production_index", "hybrid", "cuda")
    store = VectorStore(settings, get_embeddings(settings))
    for dataset in datasets:
        freeze = verify_frozen(dataset)
        repo = freeze["repo"]
        local = settings.model_copy(update={"data_dir": root / "runs" / dataset.name / "index"})
        source = VectorStore(local, get_embeddings(local))
        ids, texts, metas, vectors = source.export_records("commits", repo)
        store.replace_records("commits", repo, ids, texts, metas, vectors)
        docs = store.repo_documents("commits", repo)
        Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, repo, "commits"))
        with MetadataDB(settings.metadata_db_path, readonly=False) as db:
            records = [record_from_dict(row) for row in read_lines(dataset / "records.jsonl")]
            links = [
                IssueLink(**row)
                for row in json.loads((dataset / "links.json").read_text(encoding="utf-8"))
            ]
            issues = [
                Issue(**row)
                for row in json.loads((dataset / "issues.json").read_text(encoding="utf-8"))
            ]
            db.sync_history(repo, records, links, issues)
        Manifest(settings.manifest_path).put(
            RepoEntry(
                slug=repo,
                commit_sha=freeze["pin"],
                embedding_id=settings.embedding_id,
                indexed_at=datetime.now(UTC).isoformat(),
                history_sha=freeze["pin"],
                chunks={"commits": len(docs)},
            )
        )
    export_index(root / "phase5_commit_index.zip", store=store, settings=settings)
    # The sample comes from the frozen Requests test split. This is a CLI smoke test,
    # not generated-answer quality or a new accuracy observation.
    requests = root / "datasets" / "psf__requests"
    pairs = read_lines(requests / "pairs.jsonl")
    sample = next(row for row in pairs if row["split"] == "test")
    env = {
        **os.environ,
        "REPO_RAG_DATA_DIR": str(settings.data_dir),
        "REPO_RAG_COMPUTE_DEVICE": "cuda",
        "REPO_RAG_EMBEDDING_DTYPE": "float32",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repo_rag.cli",
            "ask",
            sample["query"],
            "--bug",
            "--repo",
            "psf/requests",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    save_cli_sample(root, sample, result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    export(parser.parse_args().root)
