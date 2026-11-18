"""Generate notebooks/ask_on_colab.ipynb: ask questions about an indexed repo on a Colab GPU.

The answer model (Qwen2.5-Coder-14B in bf16, about 31 GB) cannot run on the laptop, so the agent
runs in a Colab session: it imports an index made by `repo-rag export-index`, loads the model
once, and answers questions in a cell. The package source comes from a small ZIP on Drive
(no base64 in the notebook), checked against the hash recorded here. Regenerate after any change
under src/repo_rag.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.build_colab_indexer import SETUP_TEMPLATE, code, md, runtime_pins  # noqa: E402

NOTEBOOK = ROOT / "notebooks/ask_on_colab.ipynb"
DRIVE_SOURCE = "repo_rag_ask_source.zip"
SOURCE_ZIP = ROOT / "notebooks" / DRIVE_SOURCE  # same name as on Drive: upload it without renaming
DRIVE_INDEX = "repo_rag_index.zip"


def source_payload() -> bytes:
    """Deterministic ZIP of src/repo_rag/**/*.py (LF-normalised) under repo_rag/."""
    package = ROOT / "src/repo_rag"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*.py")):
            info = zipfile.ZipInfo(
                f"repo_rag/{path.relative_to(package).as_posix()}", (1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes().replace(b"\r\n", b"\n"))
    return buffer.getvalue()


UNPACK = """
import hashlib, io, os, pathlib, subprocess, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')
SRC = pathlib.Path('/content/repo_rag_src')
SOURCE = pathlib.Path('/content/drive/MyDrive/__DRIVE_SOURCE__')
EXPECTED_SHA256 = '__SOURCE_SHA__'
assert SOURCE.exists(), 'Upload __DRIVE_SOURCE__ (from the project notebooks folder) to My Drive'
packed = SOURCE.read_bytes()
assert hashlib.sha256(packed).hexdigest() == EXPECTED_SHA256, 'Source ZIP differs from notebook'
with zipfile.ZipFile(io.BytesIO(packed)) as archive:
    archive.extractall(SRC)
os.environ.update(
    REPO_RAG_DATA_DIR='/content/data', REPO_RAG_COMPUTE_DEVICE='cuda',
    REPO_RAG_EMBEDDING_DTYPE='float32', TOKENIZERS_PARALLELISM='false',
)
sys.path.insert(0, str(SRC))
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN') or ''
except Exception:
    pass  # optional: a Hugging Face token only makes downloads faster
print('unpacked', len(list(SRC.rglob('*.py'))), 'files')
"""

PARAMETERS = """
REPO = 'psf/requests'          # owner/name, as indexed
INDEX_ZIP = pathlib.Path('/content/drive/MyDrive/__DRIVE_INDEX__')  # from `repo-rag export-index`
BUILD_HISTORY = False          # True also embeds commit history (slow; tools work without it)
"""

IMPORT_INDEX = """
from repo_rag.config import LOCAL_14B_MODEL, LOCAL_14B_REVISION, Settings
from repo_rag.store.embeddings import get_embeddings
from repo_rag.store.manifest import Manifest
from repo_rag.store.transfer import import_index
from repo_rag.store.vectorstore import VectorStore

assert INDEX_ZIP.exists(), 'Upload __DRIVE_INDEX__ (repo-rag export-index) to My Drive'
# The default answer model is the 27B; this notebook loads the 14B locally.
settings = Settings().model_copy(
    update={'llm_model': LOCAL_14B_MODEL, 'llm_model_revision': LOCAL_14B_REVISION}
)
store = VectorStore(settings, get_embeddings(settings))
print(import_index(INDEX_ZIP, store=store, settings=settings, repos=[REPO]))
entry = Manifest(settings.manifest_path).get(REPO)
assert entry is not None, 'The index ZIP does not contain ' + REPO
print('indexed commit', entry.commit_sha)
"""

CHECKOUT = """
# Git tools (blame, releases) read a clone pinned to the indexed commit.
CHECKOUT = pathlib.Path('/content/checkout') / REPO.replace('/', '__')
if not CHECKOUT.exists():
    subprocess.run(['git', 'clone', '--quiet', '--filter=blob:none',
                    f'https://github.com/{REPO}.git', str(CHECKOUT)], check=True)
subprocess.run(['git', '-C', str(CHECKOUT), 'checkout', '--quiet', entry.commit_sha], check=True)

from repo_rag.ingest.symbol_indexer import index_symbols
print(index_symbols(REPO, CHECKOUT, settings))   # definitions and call sites, no GPU needed
if BUILD_HISTORY:
    from repo_rag.history.index import index_history
    print(index_history(REPO, CHECKOUT, settings, store, ref=entry.commit_sha))
"""

LOAD = """
from repo_rag.agent.local_llm import LocalChatModel
from repo_rag.agent.runtime import build_agent

model = LocalChatModel(settings)
print(f'loaded {settings.llm_model} in {model.load_seconds:.0f}s on {model.hardware}')
agent = build_agent(REPO, model, settings, checkout=CHECKOUT)

def ask(question, thread='colab'):
    result = agent.ask(question, thread_id=thread)
    print('ABSTAINED' if result['abstain'] else 'ANSWER', '|', result['failure'] or 'ok',
          '|', result['error'])
    print(result['answer'])
    cites = result['citations']
    print('citations valid:', cites['valid'], '| removed:', cites['removed'],
          '| attached:', cites['attached'], '| uncited:', cites['uncited'])
    print('tools:', [call['name'] for call in result['tool_calls']],
          '| exact refs:', [ref['value'] for ref in result['refs']])
    if result['sources']:
        print('retrieved:', *result['sources'], sep='\\n  ')
    return result

from repo_rag.agent.runtime import evaluated_settings
from repo_rag.retrieval.retrievers import HybridRetriever
_retriever = HybridRetriever(VectorStore(evaluated_settings(settings), get_embeddings(settings)),
                             evaluated_settings(settings))

def peek(query, k=10):
    # What search_code would see for this query (the agent shows the model only the top 5).
    for rank, hit in enumerate(_retriever.search(REPO, query, kind='code', k=k, rerank=False), 1):
        meta = hit.document.metadata
        where = f"{meta['path']}:{meta['start_line']}-{meta['end_line']}"
        print(rank, where, meta.get('symbol', ''))
"""

ASK = """
QUESTION = 'How are redirects followed, and where is the maximum number of redirects enforced?'
result = ask(QUESTION)
# If an answer abstains, check retrieval vs the model: are the right files in the top 10?
# peek('How does a call to requests.get() reach the network?')
"""

LOOP = """
# Optional: keep asking; an empty line stops. Earlier questions are remembered but not shown
# to the model.
while True:
    question = input('question> ').strip()
    if not question:
        break
    ask(question)
"""

FINISH = """
# Release the GPU and end the session when you are done (saves compute units).
from google.colab import runtime
model.close()
runtime.unassign()
"""


def build() -> dict:
    unpack = UNPACK.replace("__DRIVE_SOURCE__", DRIVE_SOURCE).replace(
        "__SOURCE_SHA__", hashlib.sha256(source_payload()).hexdigest()
    )
    cells = [
        md(f"""# Ask a repository: Qwen2.5-Coder-14B on a Colab GPU

Use a **GPU runtime with at least 40 GB** (A100 40 GB). Put two files at the top of My Drive:
`{DRIVE_SOURCE}` (from the project's `notebooks/` folder) and `{DRIVE_INDEX}`
(made on the laptop with `repo-rag export-index {DRIVE_INDEX}`). Then run the cells in order.

The agent is the one evaluated in the Phase 7 bake-offs: the same prompt arm, tools, limits and
retrieval settings, plus a check that every `path:start-end` citation matches a chunk that was
actually retrieved. The model is **provisional**: it passed 4 of 5 pre-registered gates and missed
the invalid-citation gate by one reference. Answers are cited but not verified for correctness.
Without commit history indexed, questions about commits, PRs and issues will have little to use.
"""),
        code(SETUP_TEMPLATE.format(pins=" ".join(f'"{p}"' for p in runtime_pins()))),
        code(unpack),
        code(PARAMETERS.replace("__DRIVE_INDEX__", DRIVE_INDEX)),
        md("Import the index (nothing is re-embedded)."),
        code(IMPORT_INDEX.replace("__DRIVE_INDEX__", DRIVE_INDEX)),
        md("Clone the repository at the indexed commit and build the symbol table."),
        code(CHECKOUT),
        md("Load the model once (a few minutes) and build the agent."),
        code(LOAD),
        md("Ask a question."),
        code(ASK),
        code(LOOP),
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


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    SOURCE_ZIP.write_bytes(source_payload())
    NOTEBOOK.write_text(render(), encoding="utf-8")
    print(f"Wrote {NOTEBOOK.name} and {SOURCE_ZIP.name}")
