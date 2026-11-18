"""Opt-in F1/F2/F3/F4 facts. Deterministic source extraction, without a model."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path, PurePosixPath

from repo_rag.config import Settings
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.ingest.file_walker import RepoFile

FEATURES = {"docstrings", "runtime", "history", "readme"}
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def history_ancestors(slug: str, checkout: Path, settings: Settings) -> tuple[set[str], Path]:
    """Use the SOURCE pin even when ancestry is obtained from a separate full clone."""
    pin = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    source = (
        settings.summary_history_repos_dir / slug.replace("/", "__")
        if settings.summary_history_repos_dir
        else checkout
    )
    ancestors = subprocess.check_output(["git", "-C", str(source), "rev-list", pin], text=True)
    return set(ancestors.splitlines()), source


def runtime_facts(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    raises: set[str] = set()
    constants: set[str] = set()
    config: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(target, (ast.Name, ast.Attribute)):
                raises.add(ast.unparse(target))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id.isupper():
            constants.add(node.id)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"[A-Z][A-Z_0-9]+", node.value)
        ):
            config.add(node.value)
    return [
        f"{label}: {', '.join(sorted(values))}"
        for label, values in (
            ("Raises", raises),
            ("Constants defined", constants),
            ("Uppercase string literals (config candidates)", config),
        )
        if values
    ]


def documentation_sections(files: list[RepoFile]) -> list[tuple[str, str]]:
    """Markdown and reStructuredText sections, preserving source path and heading."""
    sections: list[tuple[str, str]] = []
    for f in sorted(files, key=lambda f: f.path):
        if f.kind != "doc":
            continue
        text = f.abs_path.read_text(encoding="utf-8", errors="replace")
        # Both ATX Markdown headings and RST underlined headings.
        parts = re.split(r"(?m)(?=^#{1,6}\s+|^[^\n]+\n[=~^-]{3,}\s*$)", text)
        sections.extend((f.path, part.strip()) for part in parts if part.strip())
    return sections


def matching_sections(path: str, sections: list[tuple[str, str]], settings: Settings) -> list[str]:
    module = path.removesuffix(".py").removeprefix("src/").replace("/", ".")
    stem = PurePosixPath(path).stem
    # Bare short words such as 'app' and 'api' are too ambiguous; require code markup
    # or a dotted module/path. Longer stems still require a whole word.
    names = [re.escape(module), re.escape(path), rf"`{{1,2}}{re.escape(stem)}`{{1,2}}"]
    if len(stem) >= 5 and stem != "__init__":
        names.append(rf"\b{re.escape(stem)}\b")
    pattern = re.compile("|".join(names), re.I)
    return [
        f"Documentation [{source}]: "
        f"{EMAIL.sub('<email omitted>', text)[: settings.summary_section_chars]}"
        for source, text in sections
        if pattern.search(text)
    ][: settings.summary_sections_per_file]


class FactContext:
    """Read history once, no writes and no network. Missing history is recorded by callers."""

    def __init__(self, slug: str, checkout: Path, files: list[RepoFile], settings: Settings):
        self.slug, self.settings = slug, settings
        unknown = set(settings.summary_fact_features) - FEATURES
        if unknown:
            raise ValueError(f"Unknown summary facts: {sorted(unknown)}")
        self.sections = (
            documentation_sections(files) if "readme" in settings.summary_fact_features else []
        )
        self.allowed: set[str] = set()
        self.db: MetadataDB | None = None
        if "history" in settings.summary_fact_features and settings.metadata_db_path.exists():
            self.allowed, _ = history_ancestors(slug, checkout, settings)
            self.db = MetadataDB(settings.metadata_db_path, readonly=True)

    def extra(self, path: str, source: str, docs: dict[str, str]) -> list[str]:
        parts: list[str] = []
        features = self.settings.summary_fact_features
        # F1: every documented function/method, including nested definitions. Baseline's
        # capped class member list and top-level docstrings remain unchanged.
        if "docstrings" in features:
            parts.extend(f"Definition {name}: {line}" for name, line in sorted(docs.items()))
        # F2: static syntax facts only, never inferred runtime exceptions or values.
        if "runtime" in features:
            parts.extend(runtime_facts(source))
        # F3: typed SQLite lookup; no author data, and no commits after checkout HEAD.
        if "history" in features and self.db:
            titles = self.db.file_summary_titles(
                self.slug, path, self.allowed, limit=self.settings.summary_history_limit
            )
            parts.extend(f"Commit subject: {t}" for t in titles["commits"])
            parts.extend(f"Merged PR title: {t}" for t in titles["prs"])
        # F4: only sections that explicitly mention the module; source kept in facts.
        if "readme" in features:
            parts.extend(matching_sections(path, self.sections, self.settings))
        return [EMAIL.sub("<email omitted>", part) for part in parts]

    def close(self) -> None:
        if self.db:
            self.db.connection.close()
