"""Build a redacted metadata/code capsule without loading or downloading any model.

Git itself cannot be email-redacted while preserving SHAs. Bundle a pinned public clone/tag
recipe instead of Git objects; the notebook clones it, and typed tools redact all results.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.answer_bakeoff_io import DATASET, PROPOSAL, V2, V3, V4, digest, load, save, verify
from repo_rag.agent.bakeoff import EMAIL, canonical, redact
from repo_rag.config import Settings
from repo_rag.history.commit_indexer import CommitRecord, FileChange, is_fix_message
from repo_rag.history.facts_index import index_git_facts
from repo_rag.history.github_sync import GitHubSync
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.ingest.symbol_indexer import index_symbols

CAPSULE = ROOT / "notebooks/answer_bakeoff_inputs.zip"
WORK = ROOT / "data/answer_bakeoff_capsule"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CHECKOUTS = {
    "encode/httpx": ROOT / "data/repos/encode__httpx",
    "psf/requests": ROOT / "data/repos/psf__requests",
    "pallets/flask": ROOT / "data/eval_repos/pallets__flask",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], encoding="utf-8", errors="replace"
    )


def history(root: Path, pin: str, cap: int) -> list[CommitRecord]:
    text = git(
        root,
        "log",
        "--no-merges",
        "--full-history",
        "--no-renames",
        "--numstat",
        f"--max-count={cap}",
        "--format=%x1e%H%x1f%an%x1f%cn%x1f%aI%x1f%cI%x1f%s",
        pin,
    )
    result = []
    for block in text.split("\x1e"):
        if not block.strip():
            continue
        header, *lines = block.strip().splitlines()
        sha, author, committer, authored, committed, message = header.split("\x1f", 5)
        files = []
        for line in lines:
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            added, removed, path = parts
            files.append(
                FileChange(
                    path,
                    path,
                    "M",
                    int(added) if added.isdigit() else 0,
                    int(removed) if removed.isdigit() else 0,
                    "",
                    False,
                    added == "-",
                    (),
                    (),
                )
            )
        result.append(
            CommitRecord(
                "encode/httpx",
                sha,
                redact(message),
                redact(author),
                redact(committer),
                authored,
                committed,
                is_fix_message(message),
                tuple(files),
            )
        )
    return result


def sanitized_database(source: Path) -> bytes:
    """Rebuild, rather than overwrite, so SQLite free pages cannot retain old emails."""
    with tempfile.TemporaryDirectory(dir=WORK) as directory:
        dest = Path(directory) / "metadata.db"
        with closing(sqlite3.connect(source)) as db:
            sql = "\n".join(db.iterdump())
        sql = EMAIL.sub("<email omitted>", sql)
        with closing(sqlite3.connect(dest)) as db:
            db.executescript(sql)
            db.execute("VACUUM")
            db.commit()
            if EMAIL.search("\n".join(db.iterdump())):
                raise ValueError("Email remains in metadata text")
        data = dest.read_bytes()
        return data


def build() -> Path:
    protocol, _ = verify()
    WORK.mkdir(parents=True, exist_ok=True)
    pins = load(DATASET / "splits.json")["repo_pins"]
    for repo, root in CHECKOUTS.items():
        if git(root, "rev-parse", "HEAD").strip() != pins[repo]:
            raise ValueError("Checkout differs from frozen pin")
    settings = Settings(data_dir=WORK, embedding_dtype="float32")
    # The original local facts database is absent. Reconstruct from raw frozen GitHub and
    # pinned Git, never from expected labels or tool answers. No AST/history embeddings.
    snapshot_files = ROOT / "eval/datasets/httpx_pr_snapshot"
    for path in snapshot_files.glob("*.json"):
        # Use a dedicated scratch cache, never alter production or the frozen snapshot.
        target = WORK / "github_cache/sync" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    sync = GitHubSync(settings, cast(Any, None))  # Cached readers only; never call network sync.
    records = history(CHECKOUTS["encode/httpx"], pins["encode/httpx"], settings.max_commits)
    with MetadataDB(settings.metadata_db_path, readonly=False) as db:
        db.sync_history("encode/httpx", records, [], [])
        db.sync_github(
            "encode/httpx",
            sync.pull_requests("encode/httpx"),
            sync.issues("encode/httpx"),
            prs_complete=False,
            issues_complete=False,
        )
    index_git_facts("encode/httpx", CHECKOUTS["encode/httpx"], settings, ref=pins["encode/httpx"])
    symbols = {
        repo: vars(index_symbols(repo, CHECKOUTS[repo], settings))
        for repo in ("psf/requests", "pallets/flask")
    }
    entries: dict[str, bytes] = {
        "bakeoff/metadata.db": sanitized_database(settings.metadata_db_path)
    }
    # Reuse already measured original whole-code vectors. Do not regenerate the summary job.
    original = ROOT / "notebooks/summary_prose_inputs.zip"
    with zipfile.ZipFile(original) as archive:
        for name in archive.namelist():
            if name.startswith("code/"):
                entries["bakeoff/" + name] = archive.read(name)
    counts = {}
    for short, repo in (("requests", "psf/requests"), ("flask", "pallets/flask")):
        raw = json.loads(entries[f"bakeoff/code/{short}.json"])
        if any(m["commit_sha"] != pins[repo] or m["repo"] != repo for m in raw["metas"]):
            raise ValueError("Cached code vectors differ from source pin")
        counts[repo] = len(raw["ids"])
        # Redact before indexing, preserving vector order and recording this textual view.
        redacted = redact(raw)
        entries[f"bakeoff/code/{short}.json"] = canonical(redacted).encode()
    recipes = []
    for repo, checkout in CHECKOUTS.items():
        tags = git(checkout, "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/tags")
        recipes.append(
            {
                "repo": repo,
                "url": f"https://github.com/{repo}.git",
                "pin": pins[repo],
                "tags": dict(line.split() for line in tags.splitlines()),
            }
        )
    recipe = {
        "repositories": recipes,
        "note": "No raw Git objects in capsule: they contain author emails. Clone/fetch on Colab, verify exact commit and tag objects, expose only redacted typed tools.",
    }
    entries["bakeoff/git_recipe.json"] = canonical(recipe).encode()
    provenance = {
        "protocol_sha256": digest(DATASET / "protocol.json"),
        "pins": pins,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_dtype": "float32",
        "embedding_dims": 512,
        "embedding_max_tokens": 2048,
        "embedding_query_instruction": settings.embedding_query_instruction,
        "dense_weight": 1.0,
        "bm25_weight": 0.3,
        "rrf_k": 60,
        "retrieval_k": 50,
        "top_k": 5,
        "rerank": False,
        "evidence_expansion": "none",
        "code_counts": counts,
        "symbol_counts": symbols,
        "history_commits": len(records),
        "history_source": "pinned git log --no-merges --numstat; typed MetadataDB loader",
        "source_vector_capsule_sha256": digest(original),
        "redaction": "Email strings removed in metadata and transmitted code text; existing original vectors preserved. No new embedding run.",
    }
    # The dev-selected index hash must bind the database, Git/tag recipe and exact
    # redacted text/vectors, not only their settings/counts. No self-referential hash.
    provenance["payload_sha256"] = {
        name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items())
    }
    entries["bakeoff/index_manifest.json"] = canonical(provenance).encode()
    # Freeze verifier inputs are byte-exact, including original CRLFs/preparer hashes.
    files = [
        *DATASET.glob("*.json*"),
        *V2.glob("*.json*"),
        *V3.glob("*.json*"),
        *V4.glob("*.json*"),
        *PROPOSAL.glob("*.json*"),
    ]
    for folder in (DATASET, V4, V3, V2, PROPOSAL):
        files += [ROOT / name for name in load(folder / "inputs.freeze.json")["source_files"]]
    files += [
        ROOT / "eval/prepare_answer_bakeoff.py",
        ROOT / "eval/prepare_answer_bakeoff_v2.py",
        ROOT / "eval/prepare_answer_bakeoff_v3.py",
        ROOT / "eval/prepare_answer_bakeoff_v4.py",
        ROOT / "eval/prepare_answer_bakeoff_v5.py",
        ROOT / "eval/build_v5_targets.py",
        ROOT / "eval/build_v5_questions.py",
        ROOT / "eval/build_v4_fresh_facts.py",
        ROOT / "eval/build_dev_extra_facts.py",
    ]
    for path in files:
        entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
    entries["bakeoff/capsule_manifest.json"] = canonical(hashes).encode()
    with zipfile.ZipFile(CAPSULE, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    save(
        WORK / "preparation.json",
        {
            "capsule_sha256": digest(CAPSULE),
            "counts": counts,
            "symbols": symbols,
            "history_commits": len(records),
            "answer_models_loaded": 0,
            "protocol": protocol["version"],
        },
    )
    print(f"Capsule prepared: {counts}; {len(records)} history commits; zero model loads.")
    return CAPSULE


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
