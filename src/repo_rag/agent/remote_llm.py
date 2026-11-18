"""Run the answer model on another machine (a Colab GPU) and call it from the laptop.

`serve_model` wraps any `agent.bakeoff.Model` in a small HTTP server (stdlib only):

    GET  /health    -> {"model": ..., "hardware": ..., "rerank": <reranker name, if loaded>}
    POST /generate  -> {"messages": [...]} in, the `Generation` fields out
    POST /rerank    -> {"query", "texts", "instruction"} in, {"scores": [...]} out (optional)

`RemoteChatModel` is the matching client and is itself a `Model`, so `RepoAgent` runs unchanged:
retrieval, tools and the citation check stay on the laptop; only token generation is remote. The
server does the work of `LocalChatModel.generate` (chat template, hash check, input and output
caps), so the model sees the same input as in the bake-offs. Every request needs the shared secret.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from repo_rag.agent.bakeoff import BudgetFailure, Generation


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> Generation: ...


class InstructionReranker(Protocol):
    """`QwenReranker` shape: one scorer whose task instruction can be set per request."""

    instruction: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_BYTES = 1_000_000
MAX_RERANK_TEXTS = 100
RERANK_TEXT_CHARS = 8_000  # the reranker truncates by tokens anyway; this bounds the request
BUDGET_STATUS = 413


def _authorized(header: str | None, token: str) -> bool:
    return hmac.compare_digest((header or "").encode(), f"Bearer {token}".encode())


def make_handler(model: Any, token: str, info: dict[str, str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep prompts out of logs
            return

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self) -> bool:
            if _authorized(self.headers.get("Authorization"), token):
                return True
            self._send(401, {"error": "bad or missing token"})
            return False

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send(404, {"error": "not found"})
            elif self._allowed():
                self._send(200, info)

        def _body(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_REQUEST_BYTES:
                self._send(400, {"error": "request too large"})
                return None
            try:
                body = json.loads(self.rfile.read(length))
                assert isinstance(body, dict)
                return body
            except Exception:
                self._send(400, {"error": "expected a JSON object"})
                return None

        def _drain(self) -> None:
            """Read and drop the request body before refusing it. Replying and closing with the
            body unread can reset the connection, and the client then sees a reset (WinError
            10053 on Windows) instead of the 401 or 404."""
            length = int(self.headers.get("Content-Length") or 0)
            if 0 < length <= MAX_REQUEST_BYTES:
                self.rfile.read(length)

        def do_POST(self) -> None:
            if self.path not in {"/generate", "/rerank"}:
                self._drain()
                self._send(404, {"error": "not found"})
                return
            if not _authorized(self.headers.get("Authorization"), token):
                self._drain()
                self._send(401, {"error": "bad or missing token"})
                return
            body = self._body()
            if body is None:
                return
            if self.path == "/rerank":
                self._rerank(body)
                return
            messages = body.get("messages")
            if not (
                isinstance(messages, list)
                and all(
                    isinstance(m, dict)
                    and isinstance(m.get("role"), str)
                    and isinstance(m.get("content"), str)
                    for m in messages
                )
            ):
                self._send(400, {"error": "expected {'messages': [{'role', 'content'}, ...]}"})
                return
            try:
                self._send(200, asdict(model.generate(messages)))
            except BudgetFailure as exc:
                self._send(BUDGET_STATUS, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})

        def _rerank(self, body: dict[str, Any]) -> None:
            if not getattr(model, "can_rerank", False):
                self._send(404, {"error": "this server has no reranker"})
                return
            query, texts = body.get("query"), body.get("texts")
            instruction = body.get("instruction")
            if not (
                isinstance(query, str)
                and isinstance(instruction, str)
                and isinstance(texts, list)
                and 0 < len(texts) <= MAX_RERANK_TEXTS
                and all(isinstance(t, str) for t in texts)
            ):
                self._send(400, {"error": "expected {'query', 'texts' (1..100), 'instruction'}"})
                return
            try:
                self._send(200, {"scores": model.rerank(query, texts, instruction)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    return Handler


def serve_model(
    model: Generator,
    token: str,
    info: dict[str, str],
    host: str = "127.0.0.1",
    port: int = 8000,
    reranker: InstructionReranker | None = None,
) -> ThreadingHTTPServer:
    """Build the server; the caller runs `serve_forever()` (usually on a thread).

    GPU work is not re-entrant, so requests are served one at a time by serializing `generate`
    and `rerank` behind one lock. A `reranker` adds `POST /rerank`.
    """
    if len(token) < 16:
        raise ValueError("Use a token of at least 16 characters")
    serialized = _Serialized(model, reranker)
    server = ThreadingHTTPServer((host, port), make_handler(serialized, token, info))
    server.idle_seconds = serialized.idle_seconds  # type: ignore[attr-defined]
    return server


class _Serialized:
    def __init__(self, model: Generator, reranker: InstructionReranker | None = None) -> None:
        self.model, self.reranker, self.lock = model, reranker, threading.Lock()
        self.can_rerank = reranker is not None
        self._count_lock = threading.Lock()
        self._busy = 0
        self._last = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last generate or rerank started or finished; 0 while one is running
        (waiting for the lock counts). /health does not count, so polling cannot keep a session
        alive."""
        return 0.0 if self._busy else time.monotonic() - self._last

    def _track(self, start: bool) -> None:
        with self._count_lock:
            self._busy += 1 if start else -1
            self._last = time.monotonic()

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        self._track(True)
        try:
            with self.lock:
                return self.model.generate(messages)
        finally:
            self._track(False)

    def rerank(self, query: str, texts: Sequence[str], instruction: str) -> list[float]:
        assert self.reranker is not None
        self._track(True)
        try:
            with self.lock:
                self.reranker.instruction = instruction
                return [float(x) for x in self.reranker.score(query, texts)]
        finally:
            self._track(False)


class RemoteChatModel:
    """`Model` that generates on a `serve_model` server."""

    def __init__(self, url: str, token: str, timeout: float = 300.0) -> None:
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def _call(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read(MAX_RESPONSE_BYTES))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read(MAX_RESPONSE_BYTES)).get("error", "")
            except Exception:
                detail = ""
            if exc.code == BUDGET_STATUS:
                raise BudgetFailure(detail or "Input token cap") from exc
            raise RuntimeError(f"Model server returned {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Model server unreachable: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Model server sent an unexpected reply")
        return payload

    def health(self) -> dict[str, Any]:
        return self._call("/health")

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        fields = self._call("/generate", {"messages": messages})
        try:
            return Generation(
                str(fields["text"]),
                int(fields.get("output_tokens", 0)),
                float(fields.get("seconds", 0.0)),
                int(fields.get("input_tokens", 0)),
                bool(fields.get("token_cap_hit", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Model server sent an unexpected reply") from exc

    def reranker_factory(self, info: dict[str, Any]) -> Callable[[Any, str], RemoteReranker] | None:
        """For `HybridRetriever(reranker_factory=...)`; None when the server has no reranker."""
        if not info.get("rerank"):
            return None
        return lambda _settings, instruction: RemoteReranker(self, instruction)

    def start_question(self) -> None:
        return None

    def finish_question(self) -> dict[str, Any]:
        return {"hardware": "remote"}


class RemoteReranker:
    """`Reranker` that scores on the server's GPU. Raises `RuntimeError` if the server fails."""

    def __init__(self, client: RemoteChatModel, instruction: str) -> None:
        self.client, self.instruction = client, instruction

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        fields = self.client._call(
            "/rerank",
            {
                "query": query,
                "texts": [t[:RERANK_TEXT_CHARS] for t in texts],
                "instruction": self.instruction,
            },
        )
        scores = fields.get("scores")
        if not (isinstance(scores, list) and len(scores) == len(texts)):
            raise RuntimeError("Model server sent an unexpected rerank reply")
        return [float(x) for x in scores]
