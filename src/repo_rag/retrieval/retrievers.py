"""Hybrid retrieval: dense (Chroma) + BM25, fused by weighted reciprocal-rank fusion.

    score(doc) = dense_weight / (rrf_k + dense_rank) + bm25_weight / (rrf_k + bm25_rank)

Equal weights hurt the strong dense model in the embedding benchmark, so BM25 gets a smaller
vote; Phase 4 leaves the default weights provisional. An optional cross-encoder
reranker re-scores the fused candidates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from langchain_core.documents import Document

from repo_rag.config import Settings, get_settings
from repo_rag.retrieval.expand import ChunkIndex, expand
from repo_rag.retrieval.intent import detect_intent
from repo_rag.retrieval.rerank import (
    QwenReranker,
    Reranker,
    combine_with_first_stage,
    reranking_enabled,
)
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.vectorstore import VectorStore

log = logging.getLogger(__name__)

# Metadata filters Chroma can apply exactly; `path_prefix` is applied after retrieval.
EXACT_FILTERS = ("language", "symbol_kind", "doc_kind", "is_fix", "state")
_PREFIX_OVERFETCH = 8
TITLE_FUSION_KINDS = ("pull_requests", "issues")


@dataclass
class SearchHit:
    document: Document
    score: float  # the fused score, or the reranker's probability when reranked
    dense_rank: int | None = None  # 1-based; None if dense search didn't return it
    bm25_rank: int | None = None
    dense_similarity: float | None = None
    rerank_score: float | None = None
    expansion_mode: str | None = None
    """Added structural evidence is unscored (score=0), not a model probability."""

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata["chunk_id"])


def weighted_rrf(
    rankings: Mapping[str, list[str]], weights: Mapping[str, float], k: int
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Returns (id, score) best first; ties keep first-seen order."""
    scores: dict[str, float] = {}
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        settings: Settings | None = None,
        reranker: Reranker | None = None,
        *,
        reranker_factory: Callable[[Settings, str], Reranker] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self._reranker = reranker
        self._reranker_factory = reranker_factory or QwenReranker
        self._task_rerankers: dict[str, Reranker] = {}
        self._bm25: dict[tuple[str, str], Bm25Index | None] = {}
        self._chunk_indexes: dict[tuple, ChunkIndex] = {}

    def _bm25_index(self, repo: str, kind: str) -> Bm25Index | None:
        key = (repo, kind)
        if key not in self._bm25:
            self._bm25[key] = Bm25Index.load(bm25_path(self.settings.bm25_dir, repo, kind))
        return self._bm25[key]

    def forget_cached_indexes(self) -> None:
        """Call after reindexing to invalidate both lexical and expansion lookups."""
        self._bm25.clear()
        self._chunk_indexes.clear()

    def search(
        self,
        repo: str,
        query: str,
        *,
        kind: str = "code",
        k: int | None = None,
        filters: Mapping[str, str] | None = None,
        rerank: bool = False,
        paths: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """Top `k` chunks for `query` in one repo.

        `filters` may hold language, symbol_kind, doc_kind (exact) and path_prefix.
        `paths` restricts both signals to a union of exact files or directories (p or p/...).
        An empty union returns no hits. Dense retrieval overfetches once, then post-filters;
        this is bounded candidate retrieval, not an exhaustive search of the selected files.
        `rerank=True` asks for the cross-encoder; it still obeys `reranker_mode`.

        PR and issue searches with a multi-line query (an issue's title plus its body) also search
        the first line alone and fuse the two rankings: long bodies dilute the query, and this
        measured higher Hit@5 and MRR on httpx and flask (docs/PROJECT_LOG.md, 2026-10-30). With
        `rerank=True` the fused top `rerank_candidate_depth` is what the cross-encoder re-scores.
        """
        title = query.split("\n", 1)[0].strip()
        if (
            kind in TITLE_FUSION_KINDS
            and self.settings.title_query_fusion
            and title
            and title != query.strip()
        ):
            settings = self.settings
            top = settings.rerank_top_n if k is None else k
            use_reranker = rerank and reranking_enabled(settings)
            if use_reranker and top > settings.rerank_candidate_depth:
                raise ValueError("k cannot exceed rerank_candidate_depth when reranking is enabled")
            depth = max(settings.retrieval_k, top, settings.rerank_candidate_depth)
            full = self._search_once(repo, query, kind, depth, filters, False, paths)
            short = self._search_once(repo, title, kind, depth, filters, False, paths)
            fused = weighted_rrf(
                {"full": [h.chunk_id for h in full], "title": [h.chunk_id for h in short]},
                {},
                settings.rrf_k,
            )
            by_id = {h.chunk_id: h for h in short} | {h.chunk_id: h for h in full}
            hits = [
                replace(by_id[cid], score=score, dense_rank=None, bm25_rank=None)
                for cid, score in fused
            ]
            if use_reranker and hits:
                hits = self._rerank(query, hits[: settings.rerank_candidate_depth], kind)
            return hits[:top]
        return self._search_once(repo, query, kind, k, filters, rerank, paths)

    def _search_once(
        self,
        repo: str,
        query: str,
        kind: str,
        k: int | None,
        filters: Mapping[str, str] | None,
        rerank: bool,
        paths: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        settings = self.settings
        k = settings.rerank_top_n if k is None else k
        if k < 1:
            raise ValueError("k must be positive")
        scopes = None if paths is None else tuple(sorted(set(p.rstrip("/") for p in paths)))
        if scopes is not None and not scopes:
            return []
        if scopes is not None and any(
            not p or p.startswith("/") or "\\" in p or ".." in p.split("/") for p in scopes
        ):
            raise ValueError("paths must be nonempty repository-relative POSIX paths")
        use_reranker = rerank and reranking_enabled(settings)
        if use_reranker and k > settings.rerank_candidate_depth:
            raise ValueError("k cannot exceed rerank_candidate_depth when reranking is enabled")
        candidates = max(
            settings.retrieval_k, k, settings.rerank_candidate_depth if use_reranker else 0
        )
        filters = {key: value for key, value in (filters or {}).items() if value}
        prefix = filters.pop("path_prefix", "")
        exact = {key: value for key, value in filters.items() if key in EXACT_FILTERS}
        unknown = set(filters) - set(EXACT_FILTERS)
        if unknown:
            raise ValueError(f"unknown filters: {sorted(unknown)}")

        # Dense. A path prefix can't be expressed in Chroma, so over-fetch and filter.
        def in_scope(path: str) -> bool:
            return path.startswith(prefix) and (
                scopes is None or any(path == p or path.startswith(p + "/") for p in scopes)
            )

        n_dense = candidates * (_PREFIX_OVERFETCH if prefix or scopes is not None else 1)
        dense = self.store.search(kind, repo, query, n_dense, exact)
        if prefix or scopes is not None:
            dense = [(d, s) for d, s in dense if in_scope(str(d.metadata["path"]))]
        dense = dense[:candidates]
        dense_ids = [str(d.metadata["chunk_id"]) for d, _ in dense]
        docs: dict[str, Document] = {str(d.metadata["chunk_id"]): d for d, _ in dense}
        similarity = {str(d.metadata["chunk_id"]): s for d, s in dense}

        # BM25 (skipped, with a warning, if the index file is missing).
        bm25_ids: list[str] = []
        index = self._bm25_index(repo, kind)
        if index is None:
            log.warning("no BM25 index for %s/%s; using dense search only", repo, kind)
        elif settings.bm25_weight > 0:

            def accept(meta: dict[str, str]) -> bool:
                return in_scope(meta["path"]) and all(
                    meta.get(key) == value for key, value in exact.items()
                )

            bm25_ids = [cid for cid, _ in index.search(query, candidates, accept)]
            missing = [cid for cid in bm25_ids if cid not in docs]
            docs.update(self.store.get_documents(kind, repo, missing))
            bm25_ids = [cid for cid in bm25_ids if cid in docs]

        fused = weighted_rrf(
            {"dense": dense_ids, "bm25": bm25_ids},
            {"dense": settings.dense_weight, "bm25": settings.bm25_weight},
            settings.rrf_k,
        )
        dense_rank = {cid: r for r, cid in enumerate(dense_ids, start=1)}
        bm25_rank = {cid: r for r, cid in enumerate(bm25_ids, start=1)}
        hits = [
            SearchHit(
                document=docs[cid],
                score=score,
                dense_rank=dense_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
                dense_similarity=similarity.get(cid),
            )
            for cid, score in fused
            if cid in docs
        ]

        if use_reranker and hits:
            hits = self._rerank(query, hits[: settings.rerank_candidate_depth], kind)
        ranked = hits[:k]
        if (
            use_reranker
            and kind == "code"
            and settings.evidence_expansion != "none"
            and len(ranked) > settings.expansion_keep
        ):
            key = (
                repo,
                kind,
                tuple(sorted(exact.items())),
                prefix,
                scopes,
                settings.expansion_max_definers,
            )
            if key not in self._chunk_indexes:
                corpus = [
                    d
                    for d in self.store.repo_documents(kind, repo, exact)
                    if in_scope(str(d.metadata["path"]))
                ]
                self._chunk_indexes[key] = ChunkIndex(corpus, settings.expansion_max_definers)
            expanded = expand(
                [h.document for h in ranked],
                self._chunk_indexes[key],
                mode=settings.evidence_expansion,
                keep=settings.expansion_keep,
                seeds=settings.expansion_seeds,
                per_seed=settings.expansion_per_seed,
            )
            originals = {h.chunk_id: h for h in ranked}
            return [
                originals.get(str(d.metadata["chunk_id"]))
                or SearchHit(document=d, score=0.0, expansion_mode=settings.evidence_expansion)
                for d in expanded
            ]
        return ranked

    def _rerank(self, query: str, hits: list[SearchHit], kind: str = "code") -> list[SearchHit]:
        settings = self.settings
        instruction = {
            "code": settings.code_rerank_instruction,
            "docs": settings.docs_rerank_instruction,
            "pull_requests": settings.pr_rerank_instruction,
        }.get(kind, settings.rerank_instruction)
        if kind == "code" and settings.rerank_instruction_mode == "intent":
            instruction = {
                "implementation": settings.code_rerank_instruction_implementation,
                "test": settings.code_rerank_instruction_test,
            }.get(detect_intent(query), instruction)
        elif kind == "code" and settings.rerank_instruction_mode == "source_aware":
            instruction = settings.code_rerank_instruction_source_aware
        if self._reranker is not None:
            reranker = self._reranker
        else:
            if instruction not in self._task_rerankers:
                self._task_rerankers[instruction] = self._reranker_factory(
                    self.settings, instruction
                )
            reranker = self._task_rerankers[instruction]
        scores = reranker.score(query, [h.document.page_content for h in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = score
            hit.score = score
        if kind == "code" and settings.rerank_first_stage_weight > 0:
            # `hits` is in fused order, so list position is the first-stage rank.
            keys = combine_with_first_stage(scores, settings.rerank_first_stage_weight)
            order = sorted(range(len(hits)), key=lambda i: (-keys[i], i))
            return [hits[i] for i in order]
        return sorted(hits, key=lambda h: h.score, reverse=True)
