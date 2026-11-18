"""Generate notebooks/serve_model_on_colab.ipynb: host the answer model for the Streamlit UI.

The UI, retrieval, tools and index stay on the laptop. This notebook only loads
Qwen2.5-Coder-14B on a Colab GPU and serves `/generate` (agent/remote_llm.py) behind a
Cloudflare quick tunnel, protected by a random token printed at the end. The package source is the
same ZIP as the ask notebook (`repo_rag_ask_source.zip` on Drive). Regenerate after any change
under src/repo_rag.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.build_ask_colab import DRIVE_SOURCE, UNPACK, source_payload  # noqa: E402
from notebooks.build_colab_indexer import SETUP_TEMPLATE, code, md, runtime_pins  # noqa: E402

NOTEBOOK = ROOT / "notebooks/serve_model_on_colab.ipynb"
CLOUDFLARED_VERSION = "2026.9.1"
CLOUDFLARED_SHA256 = "03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc"
NOTEBOOK_27B = ROOT / "notebooks/serve_model_27b_on_colab.ipynb"
# Qwen3.8-27B (architecture qwen3_5) is not loadable by the pinned transformers 4.57.6, so vLLM
# serves it from its own environment. Same 4-bit build, vLLM version and torch backend as the
# Phase 7 judge run on this A100 (194 requests, 0 failures); revision pinned.
MODEL_27B = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
REVISION_27B = "e3a700906e6a73ba4290b1873dc2ba8198527ab5"
VLLM_SPEC = "vllm==0.33.0"
TORCH_BACKEND = "cu130"
GPU_MEMORY_UTILIZATION = 0.88
MAX_INPUT_TOKENS_27B = 12288  # the default is 8192; 12288 is the value under test

INSTALL_VLLM = f"""
import importlib.util, pathlib, subprocess, sys
if importlib.util.find_spec("uv") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
# vLLM goes into its own virtual environment, never into Colab's system Python (its torch build
# clashed with Colab's torchaudio the first time). The reranker-free model server below talks to it
# over localhost HTTP.
VENV = pathlib.Path("/content/vllm_env")
VENV_PY = str(VENV / "bin" / "python")
if not VENV.exists():
    subprocess.run([sys.executable, "-m", "uv", "venv", str(VENV), "--python", sys.executable],
                   check=True)
subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", VENV_PY, "-q",
                "--torch-backend={TORCH_BACKEND}", "{VLLM_SPEC}"], check=True)
subprocess.run([VENV_PY, "-c", "import vllm; print('vllm', vllm.__version__)"], check=True)
"""

DOWNLOAD_27B = f"""
# 4-bit weights, about 21 GB, pinned revision.
import json, pathlib
from huggingface_hub import snapshot_download

MODEL_ID = "{MODEL_27B}"
MODEL_PATH = pathlib.Path(snapshot_download(MODEL_ID, revision="{REVISION_27B}"))
quant = json.loads((MODEL_PATH / "config.json").read_text()).get("quantization_config")
assert quant, MODEL_ID + " has no quantization_config"
print("downloaded", MODEL_ID, "| quantization:", quant.get("quant_method"))
"""

LOAD_27B = f"""
import os, subprocess, time
from repo_rag.agent.vllm_llm import VllmChatModel, server_command, wait_until_ready
from repo_rag.config import Settings

# The agent's input cap (default 8192). The 24k-character run failed after two large results,
# probably on it; raise it here to test that, and run the eval with --max-input-tokens to match.
MAX_INPUT_TOKENS = {MAX_INPUT_TOKENS_27B}
os.environ['REPO_RAG_AGENT_MAX_INPUT_TOKENS'] = str(MAX_INPUT_TOKENS)
settings = Settings().model_copy(update={{'llm_model': MODEL_ID}})
VLLM_PORT = 8001
# Prompt and answer both fit in this: the agent's input cap plus its output cap.
max_model_len = settings.agent_max_input_tokens + settings.agent_max_new_tokens
log_path = pathlib.Path('/content/vllm_server.log')
path = str(VENV / 'bin') + os.pathsep + os.environ['PATH']
env = {{**os.environ, 'VLLM_USE_FLASHINFER_SAMPLER': '0', 'PATH': path}}
# Speed switches. Default: CUDA graphs plus the model's own MTP draft head, measured at about 4 s
# per question (eager mode, ENFORCE_EAGER = True and MTP_TOKENS = 0: about 23 s, 7 output tokens per
# second). Startup compiles for several minutes. The mode is shown in the health line.
ENFORCE_EAGER = False
MTP_TOKENS = 3
DECODE_MODE = ('eager' if ENFORCE_EAGER else 'cuda-graphs') + (
    f' + mtp{{MTP_TOKENS}}' if MTP_TOKENS else ''
)
command = server_command(
    VENV_PY, str(MODEL_PATH), 'qwen', VLLM_PORT, max_model_len, {GPU_MEMORY_UTILIZATION},
    enforce_eager=ENFORCE_EAGER, speculative_tokens=MTP_TOKENS,
    max_num_seqs=None if ENFORCE_EAGER else 8,   # CUDA graphs need it low on this model
)
process = subprocess.Popen(
    command,
    env=env, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT,
)
base_url = f'http://127.0.0.1:{{VLLM_PORT}}'
load_seconds = wait_until_ready(base_url, process, log_path)
import torch
hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU'
model = VllmChatModel(base_url, 'qwen', settings, process, hardware, load_seconds, log_path)
reranker = None   # no reranker next to the 27B model: only bug -> past-fix commit search uses one
print(f'serving {{MODEL_ID}} through vLLM ({{DECODE_MODE}}) in {{load_seconds:.0f}} s '
      f'on {{hardware}}')
"""

LOAD = """
from repo_rag.agent.local_llm import LocalChatModel
from repo_rag.config import LOCAL_14B_MODEL, LOCAL_14B_REVISION, Settings

# The default answer model is the 27B; this notebook loads the 14B locally.
settings = Settings().model_copy(
    update={'llm_model': LOCAL_14B_MODEL, 'llm_model_revision': LOCAL_14B_REVISION}
)
model = LocalChatModel(settings)
print(f'loaded {settings.llm_model} in {model.load_seconds:.0f}s on {model.hardware}')

# A second model on the same GPU re-scores commit search results (bug -> past fix). The 8B
# reranker would not fit next to the 14B answer model on 40 GB (parameter-count estimate, not
# measured), so the 4B one is used. If it does not load, the server simply runs without it.
RERANKER_MODEL = 'Qwen/Qwen3-Reranker-4B'   # set to None to serve without a reranker
reranker = None
if RERANKER_MODEL:
    import torch
    from repo_rag.retrieval.rerank import QwenReranker
    try:
        reranker = QwenReranker(settings.model_copy(update={'reranker_model': RERANKER_MODEL}))
        reranker.score('warm-up', ['warm-up'])   # loads the weights now, so a failure shows here
        print('reranker loaded:', RERANKER_MODEL)
    except Exception as error:
        reranker = None
        print('RERANKER NOT LOADED, serving without it:', type(error).__name__, str(error)[:200])
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f'GPU memory in use: {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB')
"""

SERVE = """
import secrets, threading
from repo_rag.agent.remote_llm import serve_model

PORT = 8000
TOKEN = secrets.token_urlsafe(24)   # new every session; only someone holding it can generate
info = {'model': settings.llm_model, 'hardware': model.hardware,
        'max_input_tokens': str(settings.agent_max_input_tokens),
        'decode_mode': globals().get('DECODE_MODE', 'transformers')}
if reranker is not None:
    info['rerank'] = RERANKER_MODEL
server = serve_model(model, TOKEN, info, port=PORT, reranker=reranker)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f'model server listening on 127.0.0.1:{PORT}')
"""

TUNNEL = """
import re, time, urllib.request
CLOUDFLARED_URL = ('https://github.com/cloudflare/cloudflared/releases/download/'
                   '__VERSION__/cloudflared-linux-amd64')
CLOUDFLARED_SHA256 = '__SHA256__'
binary = pathlib.Path('/content/cloudflared')
if not binary.exists():
    data = urllib.request.urlopen(CLOUDFLARED_URL).read()
    assert hashlib.sha256(data).hexdigest() == CLOUDFLARED_SHA256, 'cloudflared differs from pin'
    binary.write_bytes(data)
    binary.chmod(0o755)
tunnel = subprocess.Popen(
    [str(binary), 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{PORT}'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
URL, deadline, log = None, time.time() + 90, []
for line in tunnel.stdout:
    log.append(line)
    found = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', line)
    if found:
        URL = found.group(0)
        break
    if time.time() > deadline:
        break
assert URL, 'No tunnel URL after 90 s. Last log lines:\\n' + ''.join(log[-15:])
threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()  # drain

from repo_rag.agent.remote_llm import RemoteChatModel
for attempt in range(30):   # a new tunnel name can take a few seconds to resolve
    try:
        print('reachable through the tunnel:', RemoteChatModel(URL, TOKEN, timeout=10).health())
        break
    except RuntimeError as error:
        last = error
        time.sleep(2)
else:
    raise last
print()
print('Paste these into the Streamlit sidebar (or .env):')
print('  URL  ', URL)
print('  Token', TOKEN)
"""

KEEPALIVE = """
# Leave this cell running while you use the UI. When no question or rerank has arrived for
# IDLE_MINUTES (a request still running counts as activity), or MAX_HOURS have passed since
# this cell started, it shuts the server and tunnel down and releases the GPU by itself.
# Stop the cell (square button) to take over manually; the server then stays up until the next cell.
import time
IDLE_MINUTES = 10
MAX_HOURS = 4

started = time.monotonic()
reason = None
try:
    while reason is None:
        time.sleep(30)
        if server.idle_seconds() >= IDLE_MINUTES * 60:
            reason = f'no request for {IDLE_MINUTES} minutes'
        elif time.monotonic() - started >= MAX_HOURS * 3600:
            reason = f'{MAX_HOURS} hours reached'
except KeyboardInterrupt:
    print('stopped waiting; the server is still up until you run the next cell')
if reason:
    print('Releasing the GPU:', reason)
    from google.colab import runtime
    server.shutdown()
    tunnel.terminate()
    model.close()
    runtime.unassign()
"""

FINISH = """
# Shut everything down and release the GPU (saves compute units).
from google.colab import runtime
server.shutdown()
tunnel.terminate()
model.close()
runtime.unassign()
"""


def build(variant: str = "14b") -> dict:
    unpack = UNPACK.replace("__DRIVE_SOURCE__", DRIVE_SOURCE).replace(
        "__SOURCE_SHA__", hashlib.sha256(source_payload()).hexdigest()
    )
    tunnel = TUNNEL.replace("__VERSION__", CLOUDFLARED_VERSION).replace(
        "__SHA256__", CLOUDFLARED_SHA256
    )
    cells = [
        md(f"""# Host the answer model for the Streamlit UI

Use a **GPU runtime with at least 40 GB** (A100 40 GB). Put `{DRIVE_SOURCE}` (the project's
`notebooks/{DRIVE_SOURCE}` as built, same name) at the top of My Drive, then run the cells in
order. The last output gives a **URL and a token**: paste them into the UI on your laptop
(`uv run streamlit run src/repo_rag/ui/app.py`).

Only text generation runs here. The index, retrieval, tools and citation check run on your
laptop, so no index needs uploading. Prompts and retrieved code travel through a public
Cloudflare quick tunnel to this machine; every request needs the token, but do not use this for
a private repository. The tunnel program (cloudflared {CLOUDFLARED_VERSION}) is downloaded from
GitHub and checked against a pinned SHA-256 before it runs. The model is **provisional**.
"""),
        code(SETUP_TEMPLATE.format(pins=" ".join(f'"{p}"' for p in runtime_pins()))),
        code(unpack),
        *(
            [
                md("## Install vLLM and download the 27B model (about 21 GB)"),
                code(INSTALL_VLLM),
                code(DOWNLOAD_27B),
                md("Start vLLM and wait for it (several minutes)."),
                code(LOAD_27B),
            ]
            if variant == "27b"
            else [md("Load the model once (a few minutes)."), code(LOAD)]
        ),
        md("Start the model server on this machine."),
        code(SERVE),
        md("Open a tunnel so the laptop can reach it, and print the URL and token."),
        code(tunnel),
        md(
            "Keep this running while you chat. It releases the GPU by itself after 10 minutes with "
            "no request (or 4 hours in total); change `IDLE_MINUTES` and `MAX_HOURS` in the cell "
            "to alter that."
        ),
        code(KEEPALIVE),
        code(FINISH),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(variant: str = "14b") -> str:
    return json.dumps(build(variant), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    NOTEBOOK.write_text(render(), encoding="utf-8")
    NOTEBOOK_27B.write_text(render("27b"), encoding="utf-8")
    print(f"Wrote {NOTEBOOK.name} and {NOTEBOOK_27B.name}")
