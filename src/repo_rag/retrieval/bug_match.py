"""Bug report to past-fix search. Returns evidence, not an LLM-generated diagnosis."""

from __future__ import annotations

from repo_rag.retrieval.retrievers import HybridRetriever, SearchHit


def search_fixes(
    retriever: HybridRetriever,
    repo: str,
    report: str,
    *,
    k: int | None = None,
    rerank: bool = False,
) -> list[SearchHit]:
    if not report.strip():
        raise ValueError("bug report must not be blank")
    return retriever.search(
        repo, report, kind="commits", k=k, filters={"is_fix": "true"}, rerank=rerank
    )
