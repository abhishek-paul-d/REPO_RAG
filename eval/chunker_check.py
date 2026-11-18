"""Real-repo chunker check: run every chunker invariant over whole repositories.

For each repo (a local git checkout) and each language it reports:
  * files, chunks, time, chunker mix and **why** files fell back (from the chunker's log)
  * invariant violations: chunk text != source lines, lost (uncovered) lines, overlapping
    chunks, duplicate chunk ids, and **definitions within budget that were split**
    (checked independently by re-parsing every file)
  * size: median / p90 / max, and the chunks over budget (listed, so they can be judged)
  * naming: `<anonymous>` symbols and AST chunks without any symbol

Usage:
  uv run python eval/chunker_check.py psf/requests=data/repos/psf__requests [slug=path ...]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from tree_sitter_language_pack import get_parser

from repo_rag.chunking.ast_chunker import AstChunker
from repo_rag.chunking.languages import AST_LANGUAGES
from repo_rag.config import Settings
from repo_rag.ingest.file_walker import walk_repo

_HEADER = re.compile(r"^# (path|in class|in function|signature|defines|section): ")
_MATCHER = AstChunker(Settings())
_WS = re.compile(r"\s")


def body(doc: Document) -> str:
    lines = doc.page_content.split("\n")
    while lines and _HEADER.match(lines[0]):
        lines.pop(0)
    return "\n".join(lines)


def nonws(text: str) -> int:
    return len(_WS.sub("", text))


class _FallbackLog(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.reasons: dict[str, str] = {}

    def emit(self, record: logging.LogRecord) -> None:
        path = str(record.args[0]) if record.args else "?"
        reason = "exception" if record.exc_info else "parse errors"
        if record.exc_info and record.exc_info[1] is not None:
            reason = f"exception: {type(record.exc_info[1]).__name__}: {record.exc_info[1]}"
        self.reasons[path] = reason


@dataclass
class LangStats:
    files: int = 0
    chunks: int = 0
    chunkers: Counter = field(default_factory=Counter)
    kinds: Counter = field(default_factory=Counter)
    sizes: list[int] = field(default_factory=list)
    defs_checked: int = 0
    split_defs: list[str] = field(default_factory=list)
    text_mismatch: list[str] = field(default_factory=list)
    lost_lines: list[str] = field(default_factory=list)
    overlaps: list[str] = field(default_factory=list)
    anonymous: list[str] = field(default_factory=list)
    unnamed_ast: list[str] = field(default_factory=list)
    over_budget: list[tuple[int, str]] = field(default_factory=list)
    fallbacks: dict[str, str] = field(default_factory=dict)


def check_file(
    rel: str, source: str, language: str, docs: list[Document], stats: LangStats, budget: int
) -> None:
    lines = source.split("\n")
    spans = []
    for d in docs:
        m = d.metadata
        start, end = m["start_line"], m["end_line"]
        spans.append((start, end))
        where = f"{rel}:{start}-{end}"
        if m["chunker"] != "fallback" and body(d) != "\n".join(lines[start - 1 : end]):
            stats.text_mismatch.append(where)
        size = nonws(body(d))
        stats.sizes.append(size)
        if size > budget:
            first = body(d).split("\n")[0].strip()[:70]
            stats.over_budget.append((size, f"{where} [{m['symbol_kind']}] {first}"))
        names = [m["symbol"], *m["symbols"].split(",")]
        if any("<anonymous>" in n for n in names):
            stats.anonymous.append(f"{where} {m['symbol']}")
        if (
            m["chunker"] == "ast"
            and m["symbol_kind"] in {"function", "method", "class"}
            and not m["symbol"]
        ):
            stats.unnamed_ast.append(where)
        stats.chunkers[m["chunker"]] += 1
        stats.kinds[m["symbol_kind"]] += 1

    covered = {n for s, e in spans for n in range(s, e + 1)}
    lost = [i for i, ln in enumerate(lines, 1) if ln.strip() and i not in covered]
    if lost:
        stats.lost_lines.append(f"{rel}: {len(lost)} lines, first {lost[:3]}")
    if all(d.metadata["chunker"] == "ast" for d in docs):
        ordered = sorted(spans)
        for (_, prev_end), (start, _) in itertools.pairwise(ordered):
            if start <= prev_end:
                stats.overlaps.append(f"{rel}:{start}")

    # Re-parse and make sure no within-budget definition was split. Definitions are recognised
    # with the chunker's own matcher, so bindings like `res.send = function () {}` count too;
    # the containment check itself is independent of the chunker.
    if language in AST_LANGUAGES and docs and docs[0].metadata["chunker"] == "ast":
        config = AST_LANGUAGES[language]
        stack = [get_parser(language).parse(source.encode()).root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            match = _MATCHER._match(node, config)
            if match is None or match[0].id != node.id:  # skip wrappers; check the def itself
                continue
            if nonws((node.text or b"").decode("utf-8", "replace")) > budget:
                continue
            stats.defs_checked += 1
            s, e = node.start_point[0] + 1, node.end_point[0] + 1
            if not any(cs <= s and ce >= e for cs, ce in spans):
                stats.split_defs.append(f"{rel}:{s}-{e} {node.type}")


def check_repo(
    slug: str, root: Path, settings: Settings
) -> tuple[dict[str, LangStats], float, Counter]:
    chunker = AstChunker(settings)
    handler = _FallbackLog()
    log = logging.getLogger("repo_rag.chunking.ast_chunker")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    by_lang: dict[str, LangStats] = defaultdict(LangStats)
    ids: Counter = Counter()
    t0 = time.perf_counter()
    try:
        for f in walk_repo(root, max_file_bytes=settings.max_file_bytes):
            if f.kind != "code" or f.language is None:
                continue
            source = f.abs_path.read_text(encoding="utf-8", errors="replace")
            docs = chunker.chunk(
                source, repo=slug, commit_sha="check", path=f.path, language=f.language
            )
            stats = by_lang[f.language]
            stats.files += 1
            stats.chunks += len(docs)
            ids.update(d.metadata["chunk_id"] for d in docs)
            check_file(f.path, source, f.language, docs, stats, settings.max_chunk_chars)
            if f.path in handler.reasons:
                stats.fallbacks[f.path] = handler.reasons.pop(f.path)
    finally:
        log.removeHandler(handler)
    return by_lang, time.perf_counter() - t0, ids


def report(slug: str, by_lang: dict[str, LangStats], secs: float, ids: Counter, budget: int) -> str:
    dup = sum(1 for n in ids.values() if n > 1)
    out = [f"## {slug}  ({secs:.1f} s, duplicate chunk ids: {dup})", ""]
    out.append(
        "| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lang, s in sorted(by_lang.items()):
        sizes = sorted(s.sizes) or [0]
        out.append(
            f"| {lang} | {s.files} | {s.chunks} | {s.chunkers['ast']} / {s.chunkers['fallback']} "
            f"| {s.defs_checked} | **{len(s.split_defs)}** | {len(s.text_mismatch)} | {len(s.lost_lines)} "
            f"| {len(s.overlaps)} | {statistics.median(sizes):.0f} / {sizes[int(len(sizes) * 0.9)]} / {sizes[-1]} "
            f"| {len(s.over_budget)} | {len(s.anonymous)} |"
        )
    out.append("")
    for lang, s in sorted(by_lang.items()):
        details = [
            ("split definitions", s.split_defs),
            ("text mismatches", s.text_mismatch),
            ("lost lines", s.lost_lines),
            ("overlaps", s.overlaps),
            ("fallbacks", [f"{p}: {r}" for p, r in s.fallbacks.items()]),
            ("anonymous symbols", s.anonymous),
            ("unnamed AST definitions", s.unnamed_ast),
            ("over budget (largest)", [w for _, w in sorted(s.over_budget, reverse=True)]),
        ]
        for title, items in details:
            if items:
                out.append(f"- **{lang} {title}** ({len(items)}): " + "; ".join(items[:6]))
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repos", nargs="+", help="slug=path pairs")
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = parser.parse_args()
    settings = Settings()
    sections = [f"# Chunker check (budget {settings.max_chunk_chars} non-ws chars)", ""]
    for pair in args.repos:
        slug, path = pair.split("=", 1)
        by_lang, secs, ids = check_repo(slug, Path(path), settings)
        sections.append(report(slug, by_lang, secs, ids, settings.max_chunk_chars))
    text = "\n".join(sections)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
