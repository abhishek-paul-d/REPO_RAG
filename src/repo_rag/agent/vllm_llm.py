"""Answer model served by vLLM, spoken to over HTTP on this machine.

For models our pinned `transformers` cannot load (Qwen3.8-27B, architecture `qwen3_5`). vLLM runs in
its own virtual environment as an OpenAI-compatible server on localhost; `VllmChatModel` has the
same `Model` shape as `LocalChatModel`, so `serve_model` and the agent do not change. Greedy
decoding, thinking switched off, the same input and output token caps from `Settings`.

Not the evaluated setup (that is Qwen2.5-Coder-14B in bf16 through transformers). Importing this
module loads nothing and needs no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from repo_rag.agent.bakeoff import BudgetFailure, Generation
from repo_rag.config import Settings

SERVER_START_TIMEOUT_SECONDS = 1200.0
REQUEST_TIMEOUT_SECONDS = 600.0
HEALTH_POLL_SECONDS = 5.0
LOG_TAIL_CHARS = 3000


def server_command(
    python: str,
    model_path: str,
    served_name: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    *,
    enforce_eager: bool = True,
    speculative_tokens: int = 0,
    max_num_seqs: int | None = None,
) -> list[str]:
    """`vllm` OpenAI server on localhost. No images or video (the model has a vision tower we do
    not use) and a fixed seed. Eager mode (no CUDA-graph capture memory) is the measured default;
    `enforce_eager=False` allows CUDA graphs. `speculative_tokens > 0` turns on the model's own
    multi-token-prediction head as the draft (`"method": "mtp"`, as in the vLLM Qwen3.8 recipe).
    `max_num_seqs` caps concurrent requests (vLLM's default is 256). With CUDA graphs this model
    needs it low: vLLM refused to start with "max_num_seqs (256) exceeds available Mamba cache
    blocks (179)". The agent sends one request at a time."""
    command = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        "0",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0}),
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    if max_num_seqs is not None:
        command += ["--max-num-seqs", str(max_num_seqs)]
    if speculative_tokens > 0:
        config = {"method": "mtp", "num_speculative_tokens": speculative_tokens}
        command += ["--speculative-config", json.dumps(config)]
    return command


def chat_payload(
    served_name: str, messages: list[dict[str, str]], max_new_tokens: int
) -> dict[str, Any]:
    return {
        "model": served_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_completion(body: Mapping[str, Any], max_new_tokens: int, seconds: float) -> Generation:
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    text = (choice.get("message") or {}).get("content") or ""
    output_tokens = int(usage.get("completion_tokens", 0))
    return Generation(
        text,
        output_tokens,
        seconds,
        int(usage.get("prompt_tokens", 0)),
        choice.get("finish_reason") == "length" or output_tokens >= max_new_tokens,
    )


def _is_input_too_long(error_body: str) -> bool:
    lowered = error_body.lower()
    return "maximum context length" in lowered or "max_model_len" in lowered


class VllmChatModel:
    def __init__(
        self,
        base_url: str,
        served_name: str,
        settings: Settings | None = None,
        process: subprocess.Popen[Any] | None = None,
        hardware: str = "",
        load_seconds: float = 0.0,
        log_path: Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_name = served_name
        self.settings = settings or Settings()
        self.process = process
        self.hardware = hardware
        self.load_seconds = load_seconds
        self.log_path = log_path

    def start_question(self) -> None:
        return None

    def finish_question(self) -> dict[str, Any]:
        return {"hardware": self.hardware, "device_level_peak_sampled": False}

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        limit = self.settings.agent_max_new_tokens
        data = json.dumps(chat_payload(self.served_name, messages, limit)).encode()
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            if error.code == 400 and _is_input_too_long(detail):
                raise BudgetFailure("Input token cap") from error
            raise RuntimeError(f"vLLM returned {error.code}: {detail[:300]}") from error
        result = parse_completion(body, limit, time.perf_counter() - started)
        if result.input_tokens > self.settings.agent_max_input_tokens:
            raise BudgetFailure("Input token cap")
        return result

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()


def wait_until_ready(
    base_url: str,
    process: subprocess.Popen[Any],
    log_path: Path,
    timeout: float = SERVER_START_TIMEOUT_SECONDS,
) -> float:
    """Poll `/health` until the server answers; returns the seconds it took. Raises with the log's
    tail if the process exits or the time runs out, so the failure shows in the cell."""
    started = time.monotonic()
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited ({process.returncode}):\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=5) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() - started > timeout:
            process.terminate()
            raise RuntimeError(f"vLLM not ready after {timeout:.0f} s:\n{_tail(log_path)}")
        time.sleep(HEALTH_POLL_SECONDS)


_ERROR_LINE = re.compile(
    r"^(?:\(\w+ pid=\d+\) )?(?:ERROR \S+ \S+ \[[^\]]*\] )?(\w+(?:Error|Exception): .+)$"
)


def root_cause(text: str) -> str:
    """The last `SomeError: message` line of a vLLM log, or "". vLLM prints the engine's real
    error in the middle of the log and ends with a generic "see root cause above", skipped here."""
    found = ""
    for line in text.splitlines():
        match = _ERROR_LINE.match(line.strip())
        if match and "See root cause above" not in line:
            found = match.group(1)
    return found[:600]


def _tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no log)"
    cause = root_cause(text)
    head = f"root cause: {cause}\n" if cause else ""
    return head + text[-LOG_TAIL_CHARS:]
