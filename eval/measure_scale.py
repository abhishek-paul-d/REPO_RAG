"""Phase 5 manual scale tier: time, memory and store size of the CPU indexing stages on a big repo.

Not an accuracy test and not part of CI. It measures what runs without a GPU: the blobless
clone, commit reading (diffs fetch blobs lazily), commit documents, the BM25 index, the
metadata.db history rows, and the symbol table. Embedding is **not** run (CPU embedding is
~0.16 chunks/s, see CLAUDE.md); the number of documents and characters that would be embedded is
reported instead.

    uv run python eval/measure_scale.py pandas-dev/pandas --work <scratch dir> [--max-commits 5000]
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repo_rag.config import Settings
from repo_rag.history.commit_indexer import commit_document, iter_commits
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.ingest.symbol_indexer import index_symbols
from repo_rag.store.bm25 import Bm25Index, bm25_path


def peak_working_set_bytes() -> int:
    """Peak resident memory of this process (Windows); 0 where unavailable."""
    if sys.platform != "win32":
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return 0
    return int(counters.PeakWorkingSetSize)


def folder_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="owner/name")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--max-commits", type=int, default=5000)
    parser.add_argument(
        "--full-clone",
        action="store_true",
        help="Fetch all blobs up front (no --filter=blob:none).",
    )
    args = parser.parse_args()
    slug = args.repo
    work: Path = args.work
    work.mkdir(parents=True, exist_ok=True)
    checkout = work / slug.replace("/", "__")
    settings = Settings(data_dir=work / "data", max_commits=args.max_commits)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "repo": slug,
        "max_commits": args.max_commits,
        "clone": "full" if args.full_clone else "blobless",
        "stages": {},
    }

    def stage(name: str, started: float, **extra: object) -> None:
        result["stages"][name] = {
            "seconds": round(time.perf_counter() - started, 2),
            "peak_working_set_mib_so_far": round(peak_working_set_bytes() / 2**20, 1),
            **extra,
        }
        print(name, json.dumps(result["stages"][name]), flush=True)

    started = time.perf_counter()
    if not (checkout / ".git").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                *([] if args.full_clone else ["--filter=blob:none"]),
                f"https://github.com/{slug}.git",
                str(checkout),
            ],
            check=True,
        )
    stage(
        "clone_full" if args.full_clone else "clone_blobless",
        started,
        git_dir_mib=round(folder_bytes(checkout / ".git") / 2**20, 1),
        worktree_mib=round((folder_bytes(checkout) - folder_bytes(checkout / ".git")) / 2**20, 1),
    )

    started = time.perf_counter()
    records = list(iter_commits(checkout, slug, settings))
    stage(
        "read_commits_with_diffs",
        started,
        commits=len(records),
        files_touched=sum(len(r.files) for r in records),
        patch_chars=sum(len(f.patch) for r in records for f in r.files),
    )

    started = time.perf_counter()
    docs = [commit_document(r, settings) for r in records]
    stage(
        "build_commit_documents",
        started,
        documents=len(docs),
        chars_to_embed=sum(len(d.page_content) for d in docs),
        avg_chars=round(sum(len(d.page_content) for d in docs) / max(1, len(docs))),
    )

    started = time.perf_counter()
    index = Bm25Index.build(docs)
    path = bm25_path(settings.bm25_dir, slug, "commits")
    index.save(path)
    stage("bm25_commits", started, file_mib=round(path.stat().st_size / 2**20, 2))

    started = time.perf_counter()
    with MetadataDB(settings.metadata_db_path, readonly=False) as db:
        db.sync_history(slug, records, [], [])
    stage(
        "metadata_db_history",
        started,
        db_mib=round(settings.metadata_db_path.stat().st_size / 2**20, 2),
    )

    started = time.perf_counter()
    symbols = index_symbols(slug, checkout, settings)
    stage(
        "symbol_table",
        started,
        files=symbols.files,
        definitions=symbols.definitions,
        calls=symbols.calls,
        db_mib_after=round(settings.metadata_db_path.stat().st_size / 2**20, 2),
    )

    result["not_measured"] = (
        "Embedding the documents (needs a GPU) and any live GitHub lookups. At the documented CPU "
        "rate of 0.16 chunks/s the commit documents would take "
        f"{round(len(docs) / 0.16 / 3600, 1)} hours."
    )
    out = ROOT / "eval/scale_results"
    out.mkdir(exist_ok=True)
    (out / f"{slug.replace('/', '__')}_{result['clone']}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"written {out}")


if __name__ == "__main__":
    main()
