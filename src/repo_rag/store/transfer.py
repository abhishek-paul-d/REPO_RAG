"""Move an index between machines: build on a GPU (Colab), search on a laptop.

The archive is a zip of plain data, not Chroma's files (HNSW binaries and pickles can differ
across OS and library versions):

    export.json                  format version, embedding id, the manifest entries
    <kind>/<owner>__<repo>.jsonl one {"id", "text", "meta"} per chunk, sorted by id
    <kind>/<owner>__<repo>.npy   float32 vectors, row i belongs to line i

Importing replaces the repo's chunks with exactly these records (nothing is re-embedded, other
repos are untouched) and rebuilds the BM25 index from the chunk text. Vectors from a different
embedding space are refused, because mixing them silently ruins search.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from repo_rag.config import Settings, get_settings
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.manifest import Manifest, RepoEntry
from repo_rag.store.vectorstore import SEARCH_COLLECTIONS, VectorStore

FORMAT_VERSION = 1
_META_FILE = "export.json"


class TransferError(Exception):
    """The archive can't be used (wrong format, wrong embedding space, missing parts)."""


@dataclass
class TransferResult:
    repos: list[str] = field(default_factory=list)
    chunks: dict[str, dict[str, int]] = field(default_factory=dict)  # repo -> kind -> count
    removed: int = 0  # stale chunks deleted on import


def _stem(repo: str) -> str:
    return repo.replace("/", "__")


def export_index(
    out: Path,
    *,
    store: VectorStore,
    settings: Settings | None = None,
    repos: list[str] | None = None,
) -> TransferResult:
    """Write the named repos (default: all indexed ones) to a zip."""
    settings = settings or get_settings()
    entries = Manifest(settings.manifest_path).entries()
    wanted = repos or sorted(entries)
    missing = [slug for slug in wanted if slug not in entries]
    if missing:
        raise TransferError(f"not indexed: {', '.join(missing)}")
    if not wanted:
        raise TransferError("nothing is indexed yet")
    stale = [slug for slug in wanted if entries[slug].embedding_id != settings.embedding_id]
    if stale:
        raise TransferError(
            f"{', '.join(stale)} were indexed with a different embedding than the current "
            f"{settings.embedding_id}; re-index them first"
        )

    result = TransferResult(repos=list(wanted))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        meta = {
            "format": FORMAT_VERSION,
            "embedding_id": settings.embedding_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "repos": {slug: asdict(entries[slug]) for slug in wanted},
        }
        archive.writestr(_META_FILE, json.dumps(meta, indent=2))
        for slug in wanted:
            kinds = [kind for kind in SEARCH_COLLECTIONS if kind in entries[slug].chunks]
            for kind in kinds:
                ids, texts, metas, vectors = store.export_records(kind, slug)
                lines = (
                    json.dumps({"id": i, "text": t, "meta": m}, ensure_ascii=False)
                    for i, t, m in zip(ids, texts, metas, strict=True)
                )
                archive.writestr(f"{kind}/{_stem(slug)}.jsonl", "\n".join(lines))
                buffer = io.BytesIO()
                np.save(buffer, vectors, allow_pickle=False)
                archive.writestr(f"{kind}/{_stem(slug)}.npy", buffer.getvalue())
                result.chunks.setdefault(slug, {})[kind] = len(ids)
    return result


def archive_repos(archive_path: Path) -> list[str]:
    """The repos an export zip holds, read from its metadata without extracting anything.

    Raises `TransferError` for a file that is not a readable repo-rag export, so a caller can
    refuse it before touching the index."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if _META_FILE not in archive.namelist():
                raise TransferError(f"{archive_path} is not a repo-rag index export")
            return sorted(json.loads(archive.read(_META_FILE))["repos"])
    except (zipfile.BadZipFile, FileNotFoundError, KeyError, ValueError) as exc:
        raise TransferError(f"can't open {archive_path}: {exc}") from exc


def import_index(
    archive_path: Path,
    *,
    store: VectorStore,
    settings: Settings | None = None,
    repos: list[str] | None = None,
) -> TransferResult:
    """Merge an exported zip into the local index."""
    settings = settings or get_settings()
    try:
        archive = zipfile.ZipFile(archive_path)
    except (zipfile.BadZipFile, FileNotFoundError) as exc:
        raise TransferError(f"can't open {archive_path}: {exc}") from exc

    with archive:
        if _META_FILE not in archive.namelist():
            raise TransferError(f"{archive_path} is not a repo-rag index export")
        meta = json.loads(archive.read(_META_FILE))
        if meta.get("format") != FORMAT_VERSION:
            raise TransferError(
                f"unsupported export format {meta.get('format')!r} (this version reads "
                f"{FORMAT_VERSION})"
            )
        if meta["embedding_id"] != settings.embedding_id:
            raise TransferError(
                f"the export was built with {meta['embedding_id']}, but the current setting is "
                f"{settings.embedding_id}. Set the matching embedding model/dims, or re-export."
            )
        available = meta["repos"]
        unknown = [slug for slug in repos or [] if slug not in available]
        if unknown:
            raise TransferError(f"not in the archive: {', '.join(unknown)}")

        result = TransferResult()
        manifest = Manifest(settings.manifest_path)
        for slug in repos or sorted(available):
            counts: dict[str, int] = {}
            kinds = [kind for kind in SEARCH_COLLECTIONS if kind in available[slug]["chunks"]]
            for kind in kinds:
                docs, vectors = _read_records(archive, kind, slug)
                ids = [str(d.metadata["chunk_id"]) for d in docs]
                result.removed += store.replace_records(
                    kind,
                    slug,
                    ids,
                    [d.page_content for d in docs],
                    [d.metadata for d in docs],
                    vectors,
                )
                Bm25Index.build(docs).save(bm25_path(settings.bm25_dir, slug, kind))
                counts[kind] = len(docs)
            entry = RepoEntry(**available[slug])
            entry.clone_path = ""  # the clone stayed on the other machine
            previous = manifest.get(slug)
            if previous and previous.embedding_id == settings.embedding_id:
                counts = {**previous.chunks, **counts}
                if "commits" not in kinds:
                    entry.history_sha = previous.history_sha
                elif "code" not in kinds and "docs" not in kinds:
                    # A history-only import does not move the code to a new revision.
                    previous.history_sha = entry.history_sha
                    entry = previous
            entry.chunks = counts
            manifest.put(entry)
            result.repos.append(slug)
            result.chunks[slug] = counts
    return result


def _read_records(
    archive: zipfile.ZipFile, kind: str, slug: str
) -> tuple[list[Document], np.ndarray]:
    stem = f"{kind}/{_stem(slug)}"
    names = set(archive.namelist())
    for member in (f"{stem}.jsonl", f"{stem}.npy"):
        if member not in names:
            raise TransferError(f"the archive is missing {member}")
    raw = archive.read(f"{stem}.jsonl").decode("utf-8")
    vectors = np.load(io.BytesIO(archive.read(f"{stem}.npy")), allow_pickle=False)
    records = [json.loads(line) for line in raw.split("\n") if line]
    if len(records) != len(vectors):
        raise TransferError(
            f"{stem}: {len(records)} chunks but {len(vectors)} vectors; the archive is corrupt"
        )
    docs = [Document(page_content=r["text"], metadata=r["meta"]) for r in records]
    return docs, vectors.astype(np.float32, copy=False)
