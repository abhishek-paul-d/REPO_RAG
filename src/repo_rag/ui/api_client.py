"""A small client for the FastAPI backend (`repo_rag.api.server`), standard library only.

    client = ApiClient("http://127.0.0.1:8000")
    client.repos()                                  # GET /repos
    client.ask("psf/requests", "How are redirects followed?", thread, on_step=print)

`ask` reads the server-sent events of `POST /ask`: it calls `on_step(node)` for every finished
graph node and returns the `result` dict. An `error` event, or a refusal such as 404/409/503, is
raised as `ApiError` carrying the server's own message.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any


class ApiError(Exception):
    """The API refused the request or the run failed; the message is the server's."""


def parse_events(lines: Iterator[bytes]) -> Iterator[tuple[str, Any]]:
    """`(event, data)` pairs from the lines of a text/event-stream body."""
    event, data = "message", ""
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads(data)
            event, data = "message", ""
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data += line[len("data:") :].lstrip()
    if data:
        yield event, json.loads(data)


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 900.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout  # a question can take minutes: each step is a GPU call

    def _open(self, request: urllib.request.Request) -> Any:
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            with contextlib.suppress(Exception):  # a body that is not JSON: report the status
                detail = json.loads(exc.read()).get("detail", "")
            raise ApiError(f"{exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"API unreachable at {self.base_url}: {exc.reason}") from exc

    def repos(self) -> dict[str, Any]:
        with self._open(urllib.request.Request(f"{self.base_url}/repos")) as response:
            body: dict[str, Any] = json.loads(response.read())
        return body

    def ask(
        self,
        repo: str,
        question: str,
        thread_id: str | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        payload = {"repo": repo, "question": question}
        if thread_id:
            payload["thread_id"] = thread_id
        request = urllib.request.Request(
            f"{self.base_url}/ask",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with self._open(request) as response:
            for event, data in parse_events(iter(response)):
                if event == "step" and on_step:
                    on_step(data.get("node", ""))
                elif event == "error":
                    raise ApiError(data.get("message", "the run failed"))
                elif event == "result":
                    result: dict[str, Any] = data
                    return result
        raise ApiError("the stream ended without a result")
