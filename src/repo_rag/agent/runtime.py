"""Build a `RepoAgent` over an indexed repository with the retrieval setup that was evaluated.

The Phase 7 bake-offs ran `search_code` as top-5 of a fused top-50 list with dense weight 1.0,
BM25 weight 0.3, RRF k 60, no reranker and no evidence expansion, on float32 query vectors.
`evaluated_settings` pins exactly that, so the agent retrieves the way the model was measured.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from repo_rag.agent.bakeoff import Model
from repo_rag.agent.bakeoff_tools import BoundTools
from repo_rag.agent.graph import RepoAgent
from repo_rag.agent.tools import ProductionTools
from repo_rag.config import Settings
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.retrieval.retrievers import HybridRetriever
from repo_rag.store.embeddings import get_embeddings
from repo_rag.store.manifest import Manifest
from repo_rag.store.symbol_table import SymbolTable
from repo_rag.store.vectorstore import VectorStore


def evaluated_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "retrieval_k": 50,
            "reranker_mode": "off",
            "evidence_expansion": "none",
            "dense_weight": 1.0,
            "bm25_weight": 0.3,
            "rrf_k": 60,
            "embedding_dtype": "float32",
        }
    )


def build_agent(
    slug: str,
    model: Model,
    settings: Settings | None = None,
    checkout: Path | None = None,
    embeddings: Any = None,
    reranker_factory: Any = None,
) -> RepoAgent:
    """Git tools (blame, releases) read a clone: `checkout`, else the manifest's clone path, else
    `<data_dir>/repos/<owner>__<name>` (an index imported from Colab has no clone path).

    `reranker_factory` (from `RemoteChatModel.reranker_factory`) lets commit search re-score its
    top 50 on the model server's GPU. Without it nothing is reranked, as in the evaluated setup."""
    settings = evaluated_settings(settings or Settings())
    if reranker_factory is not None and settings.agent_toolset == "production":
        settings = settings.model_copy(update={"reranker_mode": "on"})
    entry = Manifest(settings.manifest_path).get(slug)
    if entry is None:
        raise ValueError(f"{slug} is not indexed; run `repo-rag index` or `import-index` first")
    path = checkout or (
        Path(entry.clone_path)
        if entry.clone_path
        else settings.data_dir / "repos" / slug.replace("/", "__")
    )
    if not path.exists():
        raise ValueError(f"No git checkout for {slug}; pass checkout= (git tools need one)")
    retriever = HybridRetriever(
        VectorStore(settings, embeddings or get_embeddings(settings)),
        settings,
        reranker_factory=reranker_factory,
    )
    bound = ProductionTools if settings.agent_toolset == "production" else BoundTools
    symbols = SymbolTable(settings.metadata_db_path)
    tools = bound(
        slug,
        path,
        entry.commit_sha,
        settings,
        retriever,
        MetadataDB(settings.metadata_db_path),
        symbols,
    )

    def snap_lookup(name: str, file: str) -> list[tuple[str, int, int]]:
        try:
            found = symbols.get_symbol(slug, name, path=file)
        except sqlite3.Error:  # symbols not indexed for this database: nothing to snap to
            return []
        return [(row.qualname, row.start_line, row.end_line) for row in found]

    return RepoAgent(model, tools.execute, settings, snap_lookup=snap_lookup)
