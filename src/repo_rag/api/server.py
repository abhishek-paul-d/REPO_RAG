"""FastAPI backend: list indexed repos, ask questions (streamed), import an index built on Colab.

    uv run uvicorn repo_rag.api.server:app

The answer model stays on the remote GPU (`llm_server_url` / `llm_server_token` in the settings,
served by `notebooks/serve_model_on_colab.ipynb`); retrieval, tools and the citation check run in
this process against the local index. The model token never leaves the server: callers of this API
do not send it.

There is **no authentication** here. Run it on localhost (uvicorn's default) and do not expose it
to a network: anyone who can reach `/ask` spends your GPU session, and `/repos/import` writes to
your index. Indexing a repository needs a GPU (about 0.16 chunks/s on a laptop CPU), so there is no
endpoint that indexes a URL: build the index with `notebooks/index_on_colab.ipynb` and upload the
zip to `POST /repos/import`.

An import rewrites the named repos' index files one after another, so while it runs `/ask` about
those repos is refused with 409 ("being replaced"); other repos, `/repos` and `/health` keep
working, and a second import is refused. A question already streaming when an import starts is not
interrupted and may see a mix of old and new chunks.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from repo_rag.config import Settings, get_settings
from repo_rag.store.manifest import Manifest
from repo_rag.store.transfer import TransferError, archive_repos, import_index
from repo_rag.store.vectorstore import VectorStore

AgentFactory = Callable[[str], Any]
StoreFactory = Callable[[], VectorStore]

MAX_QUESTION_CHARS = 4_000
MAX_THREAD_CHARS = 64


class ModelServerNotConfigured(Exception):
    """`llm_server_url` / `llm_server_token` are not set."""


class AskRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=200, description="owner/name, as listed by /repos")
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    thread_id: str | None = Field(
        default=None,
        max_length=MAX_THREAD_CHARS,
        description="Reuse it to continue a conversation; omit it to start one.",
    )


def sse(event: str, data: Any) -> str:
    """One server-sent event. `json.dumps` escapes newlines, so `data:` stays on one line."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def default_agent_factory(settings: Settings) -> AgentFactory:
    embeddings: list[Any] = []  # loaded once, on the first question

    def make(slug: str) -> Any:
        from repo_rag.agent.remote_llm import RemoteChatModel
        from repo_rag.agent.runtime import build_agent

        if not settings.llm_server_url or not settings.llm_server_token:
            raise ModelServerNotConfigured(
                "Set REPO_RAG_LLM_SERVER_URL and REPO_RAG_LLM_SERVER_TOKEN to the values printed "
                "by notebooks/serve_model_on_colab.ipynb."
            )
        if not embeddings:
            from repo_rag.store.embeddings import get_embeddings

            embeddings.append(get_embeddings(settings))
        model = RemoteChatModel(
            settings.llm_server_url, settings.llm_server_token.get_secret_value()
        )
        info = model.health()  # fails fast on a wrong URL, token or a dead tunnel
        factory = model.reranker_factory(info) if settings.agent_commit_rerank else None
        return build_agent(
            slug, model, settings, embeddings=embeddings[0], reranker_factory=factory
        )

    return make


def default_store_factory(settings: Settings) -> StoreFactory:
    def make() -> VectorStore:
        from repo_rag.store.embeddings import get_embeddings

        return VectorStore(settings, get_embeddings(settings))

    return make


def create_app(
    settings: Settings | None = None,
    *,
    agent_factory: AgentFactory | None = None,
    store_factory: StoreFactory | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    make_agent = agent_factory or default_agent_factory(settings)
    make_store = store_factory or default_store_factory(settings)
    agents: dict[str, Any] = {}
    agents_lock = threading.Lock()
    import_lock = threading.Lock()
    replacing: set[str] = set()  # repos an import is rewriting right now
    replacing_lock = threading.Lock()
    app = FastAPI(title="Repository assistant API")

    def manifest() -> Manifest:
        return Manifest(settings.manifest_path)

    def get_agent(slug: str) -> Any:
        with agents_lock:
            if slug not in agents:
                agents[slug] = make_agent(slug)
            return agents[slug]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/repos")
    def list_repos() -> dict[str, Any]:
        """Indexed repos. `searchable` is false when a repo was indexed with another embedding
        model than the one configured; `repo-rag index` or an import must redo it."""
        current = settings.embedding_id
        return {
            "embedding_id": current,
            "repos": [
                {
                    "repo": entry.slug,
                    "commit_sha": entry.commit_sha,
                    "history_sha": entry.history_sha,
                    "indexed_at": entry.indexed_at,
                    "chunks": entry.chunks,
                    "files": entry.files,
                    "searchable": entry.embedding_id == current,
                }
                for entry in sorted(manifest().entries().values(), key=lambda e: e.slug)
            ],
        }

    @app.post("/ask")
    def ask(body: AskRequest) -> StreamingResponse:
        """Server-sent events: `start`, one `step` per finished graph node (prestep, answer,
        check, remember), then `result` (the agent's result dict) or `error`. The model answers
        in whole turns, so events report progress, not partial text."""
        entry = manifest().get(body.repo)
        if entry is None:
            raise HTTPException(404, f"{body.repo} is not indexed; see GET /repos")
        if entry.embedding_id != settings.embedding_id:
            raise HTTPException(
                409,
                f"{body.repo} was indexed with {entry.embedding_id}, not {settings.embedding_id}",
            )
        with replacing_lock:
            busy = body.repo in replacing
        if busy:
            raise HTTPException(
                409, f"{body.repo} is being replaced by an import; ask again when it finishes"
            )
        try:
            agent = get_agent(body.repo)
        except ModelServerNotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
        thread = body.thread_id or uuid.uuid4().hex

        def events() -> Iterator[str]:
            yield sse("start", {"thread_id": thread})
            result: Any = None
            try:
                for node, update in agent.stream(body.question, thread):
                    yield sse("step", {"node": node})
                    if isinstance(update, dict) and "result" in update:
                        result = update["result"]
            except Exception as exc:
                yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
                return
            if result is None:
                yield sse("error", {"message": "the agent finished without a result"})
                return
            yield sse("result", result)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/repos/import")
    async def import_repos(
        request: Request, repo: Annotated[list[str] | None, Query()] = None
    ) -> dict[str, Any]:
        """Body: the raw bytes of a zip from `repo-rag export-index` or the Colab indexing
        notebook (`Content-Type: application/zip`, not multipart). `?repo=owner/name`, repeatable,
        limits which repos in the zip are imported. Nothing is re-embedded."""
        limit = settings.api_max_import_bytes
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise HTTPException(413, f"the upload is larger than {limit} bytes")
        if not import_lock.acquire(blocking=False):
            raise HTTPException(409, "another import is already running")
        path = settings.data_dir / "tmp" / f"import-{uuid.uuid4().hex}.zip"
        names: list[str] = []
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with path.open("wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"the upload is larger than {limit} bytes")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "the request body is empty; send the zip's bytes")
            try:
                names = list(repo) if repo else archive_repos(path)
                with replacing_lock:
                    replacing.update(names)
                result = await run_in_threadpool(_import, path, repo)
            except TransferError as exc:
                raise HTTPException(400, str(exc).replace(str(path), "the upload")) from exc
        finally:
            with replacing_lock:
                replacing.difference_update(names)
            path.unlink(missing_ok=True)
            import_lock.release()
        with agents_lock:
            for slug in result.repos:
                agents.pop(slug, None)  # their retrievers cache the replaced index
        return {"repos": result.repos, "chunks": result.chunks, "removed": result.removed}

    def _import(path: Path, repos: list[str] | None) -> Any:
        return import_index(path, store=make_store(), settings=settings, repos=repos)

    return app


def __getattr__(name: str) -> Any:
    """`uvicorn repo_rag.api.server:app` builds the app on first use, so importing this module
    (tests, tooling) does not read the settings or the manifest."""
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(name)
