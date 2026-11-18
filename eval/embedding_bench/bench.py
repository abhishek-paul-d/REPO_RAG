"""Embedding model benchmark for code retrieval.

Compares embedding models on the retrieval tasks this project needs:

  T1  docstring -> code      (auto-generated, NL description to function; docstrings stripped)
  T2  developer questions    (hand-written questions about psf/requests with verified gold symbols)
  T3  bug report -> fix      (GitHub issue text to the commit that fixed it; needs GITHUB_TOKEN)

Every model sees exactly the same chunks, so only the embedding model varies. The corpus
spans three similar HTTP/web repos (requests, flask, httpx), so httpx's look-alike code acts as
hard negatives for requests questions.

The chunker here is a fixed stdlib `ast` function/class chunker used only for this experiment.
It is NOT the Phase 2 tree-sitter chunker.

Run: python bench.py --data-dir ./bench_data --out ./results --models coderankembed qwen3-0.6b
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

REPOS: dict[str, str] = {
    "psf/requests": "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
    "pallets/flask": "d086db856be187255b8ec61ef409357393020f32",
    "encode/httpx": "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
}
BUG_REPO = "psf/requests"

MAX_SEQ_TOKENS = 2048  # same cap for every model, so speed numbers are comparable
MAX_CHUNK_CHARS = 8000
TOP_K = 100  # ranking depth kept per query (for metrics and hybrid fusion)
RRF_K = 60

CODE_QUERY_INSTRUCTION = (
    "Instruct: Given a developer question or a description of behaviour, "
    "retrieve the code that implements it\nQuery:"
)
COMMIT_QUERY_INSTRUCTION = "Instruct: Given a bug report, retrieve the commit that fixed it\nQuery:"
NOMIC_QUERY_PREFIX = "Represent this query for searching relevant code: "


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    params: str
    license: str
    max_context: int
    code_query_prompt: str = ""
    commit_query_prompt: str = ""
    trust_remote_code: bool = False
    revision: str | None = None
    left_padding: bool = False
    mrl_dims: tuple[int, ...] = ()
    laptop_cpu_ok: bool = True
    batch_size: int = 64
    provider: str = "hf"  # "hf" (sentence-transformers) or "voyage"


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="coderankembed",
            hf_id="nomic-ai/CodeRankEmbed",
            params="137M",
            license="MIT",
            max_context=8192,
            code_query_prompt=NOMIC_QUERY_PREFIX,
            commit_query_prompt=NOMIC_QUERY_PREFIX,
            trust_remote_code=True,
            # Pin the remote modeling code we actually ran, rather than whatever is newest.
            revision="3c4b60807d71f79b43f3c4363786d9493691f8b1",
            batch_size=64,
        ),
        ModelSpec(
            key="qwen3-0.6b",
            hf_id="Qwen/Qwen3-Embedding-0.6B",
            params="0.6B",
            license="Apache-2.0",
            max_context=32768,
            code_query_prompt=CODE_QUERY_INSTRUCTION,
            commit_query_prompt=COMMIT_QUERY_INSTRUCTION,
            left_padding=True,
            mrl_dims=(256, 512),
            batch_size=32,
        ),
        ModelSpec(
            key="qwen3-4b",
            hf_id="Qwen/Qwen3-Embedding-4B",
            params="4B",
            license="Apache-2.0",
            max_context=32768,
            code_query_prompt=CODE_QUERY_INSTRUCTION,
            commit_query_prompt=COMMIT_QUERY_INSTRUCTION,
            left_padding=True,
            mrl_dims=(512, 1024),
            laptop_cpu_ok=False,
            batch_size=16,
        ),
        ModelSpec(
            key="nomic-embed-code",
            hf_id="nomic-ai/nomic-embed-code",
            params="7B",
            license="Apache-2.0",
            max_context=32768,
            code_query_prompt=NOMIC_QUERY_PREFIX,
            commit_query_prompt=NOMIC_QUERY_PREFIX,
            trust_remote_code=True,
            left_padding=True,
            laptop_cpu_ok=False,
            batch_size=8,
        ),
        # Optional paid reference: our current config default. Runs only if VOYAGE_API_KEY is set.
        ModelSpec(
            key="voyage-code-3",
            hf_id="voyage-code-3",
            params="n/a (API)",
            license="Proprietary API",
            max_context=32000,
            mrl_dims=(256, 512),
            provider="voyage",
            batch_size=64,
        ),
        # Tiny model used only to smoke-test this script on a CPU.
        ModelSpec(
            key="minilm-smoke",
            hf_id="sentence-transformers/all-MiniLM-L6-v2",
            params="22M",
            license="Apache-2.0",
            max_context=256,
            batch_size=64,
        ),
    ]
}

# T2: questions about psf/requests @ REPOS["psf/requests"]. Gold = "path::Qualified.name"; a hit
# is any chunk whose qualname equals a gold or is nested under it (e.g. a class's methods).
DEV_QUESTIONS: list[dict[str, object]] = [
    {
        "type": "nl",
        "q": "How are redirects followed, and where is the maximum number of redirects enforced?",
        "gold": ["src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"],
    },
    {
        "type": "nl",
        "q": "Where is the Authorization header removed when a redirect goes to a different host?",
        "gold": [
            "src/requests/sessions.py::SessionRedirectMixin.rebuild_auth",
            "src/requests/sessions.py::SessionRedirectMixin.should_strip_auth",
        ],
    },
    {
        "type": "nl",
        "q": "How does a POST request become a GET after a 303 See Other redirect?",
        "gold": ["src/requests/sessions.py::SessionRedirectMixin.rebuild_method"],
    },
    {
        "type": "nl",
        "q": "How are automatic retries configured for the connection pool?",
        "gold": ["src/requests/adapters.py::HTTPAdapter.__init__"],
    },
    {
        "type": "nl",
        "q": "Where are SSL certificates verified and the CA bundle chosen?",
        "gold": ["src/requests/adapters.py::HTTPAdapter.cert_verify"],
    },
    {
        "type": "nl",
        "q": "How is the response body decoded into text when the server sends no charset?",
        "gold": [
            "src/requests/models.py::Response.text",
            "src/requests/models.py::Response.apparent_encoding",
        ],
    },
    {
        "type": "nl",
        "q": "How does streaming a download in chunks work?",
        "gold": [
            "src/requests/models.py::Response.iter_content",
            "src/requests/models.py::Response.iter_lines",
        ],
    },
    {
        "type": "nl",
        "q": "How is JSON parsed from a response, and what error is raised if the body is not valid JSON?",
        "gold": ["src/requests/models.py::Response.json"],
    },
    {
        "type": "nl",
        "q": "Where is HTTP Digest authentication implemented?",
        "gold": ["src/requests/auth.py::HTTPDigestAuth"],
    },
    {
        "type": "nl",
        "q": "How is the value of the Basic auth header built?",
        "gold": ["src/requests/auth.py::_basic_auth_str"],
    },
    {
        "type": "nl",
        "q": "How does the library decide whether a URL should bypass the proxy?",
        "gold": ["src/requests/utils.py::should_bypass_proxies"],
    },
    {
        "type": "nl",
        "q": "Where are proxy settings read from environment variables?",
        "gold": [
            "src/requests/utils.py::get_environ_proxies",
            "src/requests/sessions.py::Session.merge_environment_settings",
            "src/requests/utils.py::resolve_proxies",
        ],
    },
    {
        "type": "nl",
        "q": "How are query parameters URL-encoded into the request URL?",
        "gold": [
            "src/requests/models.py::RequestEncodingMixin._encode_params",
            "src/requests/models.py::PreparedRequest.prepare_url",
        ],
    },
    {
        "type": "nl",
        "q": "How are multipart file uploads encoded?",
        "gold": [
            "src/requests/models.py::RequestEncodingMixin._encode_files",
            "src/requests/models.py::PreparedRequest.prepare_body",
        ],
    },
    {
        "type": "nl",
        "q": "How is the Content-Length of a request body computed?",
        "gold": [
            "src/requests/models.py::PreparedRequest.prepare_content_length",
            "src/requests/utils.py::super_len",
        ],
    },
    {
        "type": "nl",
        "q": "How are cookies from a response stored in the cookie jar?",
        "gold": ["src/requests/cookies.py::extract_cookies_to_jar"],
    },
    {
        "type": "nl",
        "q": "How are session-level settings merged with per-request settings?",
        "gold": [
            "src/requests/sessions.py::merge_setting",
            "src/requests/sessions.py::Session.prepare_request",
        ],
    },
    {
        "type": "nl",
        "q": "How does a session pick which transport adapter handles a URL?",
        "gold": [
            "src/requests/sessions.py::Session.get_adapter",
            "src/requests/sessions.py::Session.mount",
        ],
    },
    {
        "type": "nl",
        "q": "Where are default headers such as User-Agent set?",
        "gold": [
            "src/requests/utils.py::default_headers",
            "src/requests/utils.py::default_user_agent",
        ],
    },
    {
        "type": "nl",
        "q": "How are HTTP header names handled case-insensitively?",
        "gold": ["src/requests/structures.py::CaseInsensitiveDict"],
    },
    {
        "type": "nl",
        "q": "How are event hooks such as the response hook dispatched?",
        "gold": ["src/requests/hooks.py::dispatch_hook"],
    },
    {
        "type": "nl",
        "q": "Where is the Link header parsed into a dictionary?",
        "gold": [
            "src/requests/utils.py::parse_header_links",
            "src/requests/models.py::Response.links",
        ],
    },
    {
        "type": "nl",
        "q": "How does raising for HTTP error status codes build its error message?",
        "gold": ["src/requests/models.py::Response.raise_for_status"],
    },
    {
        "type": "nl",
        "q": "How are connect and read timeouts passed down to urllib3?",
        "gold": ["src/requests/adapters.py::HTTPAdapter.send"],
    },
    {
        "type": "nl",
        "q": "Where are credentials loaded from the user's netrc file?",
        "gold": ["src/requests/utils.py::get_netrc_auth"],
    },
    {
        "type": "ident",
        "q": "What does `prepare_url` do with internationalized (IDNA) host names?",
        "gold": [
            "src/requests/models.py::PreparedRequest.prepare_url",
            "src/requests/models.py::PreparedRequest._get_idna_encoded_host",
        ],
    },
    {
        "type": "ident",
        "q": "What does `requote_uri` do?",
        "gold": ["src/requests/utils.py::requote_uri", "src/requests/utils.py::unquote_unreserved"],
    },
    {
        "type": "ident",
        "q": "How does `rewind_body` work?",
        "gold": ["src/requests/utils.py::rewind_body"],
    },
    {
        "type": "ident",
        "q": "What does `get_connection_with_tls_context` do?",
        "gold": [
            "src/requests/adapters.py::HTTPAdapter.get_connection_with_tls_context",
            "src/requests/adapters.py::HTTPAdapter.build_connection_pool_key_attributes",
        ],
    },
    {
        "type": "ident",
        "q": "Explain `CaseInsensitiveDict.lower_items`.",
        "gold": ["src/requests/structures.py::CaseInsensitiveDict.lower_items"],
    },
    {
        "type": "symptom",
        "q": "My Authorization header disappears after the server redirects me. Why?",
        "gold": [
            "src/requests/sessions.py::SessionRedirectMixin.rebuild_auth",
            "src/requests/sessions.py::SessionRedirectMixin.should_strip_auth",
        ],
    },
    {
        "type": "symptom",
        "q": "response.text shows garbled characters for a page that has no charset in its headers",
        "gold": [
            "src/requests/models.py::Response.text",
            "src/requests/models.py::Response.apparent_encoding",
            "src/requests/utils.py::get_encoding_from_headers",
        ],
    },
    {
        "type": "symptom",
        "q": "Uploading a file object that was partially read sends the wrong Content-Length",
        "gold": [
            "src/requests/utils.py::super_len",
            "src/requests/models.py::PreparedRequest.prepare_content_length",
        ],
    },
    {
        "type": "symptom",
        "q": "I get a TooManyRedirects exception. Where is it raised?",
        "gold": ["src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"],
    },
    {
        "type": "symptom",
        "q": "Requests ignores my NO_PROXY environment variable",
        "gold": [
            "src/requests/utils.py::should_bypass_proxies",
            "src/requests/utils.py::get_environ_proxies",
        ],
    },
    {
        "type": "symptom",
        "q": "A header value containing a newline raises InvalidHeader",
        "gold": [
            "src/requests/utils.py::check_header_validity",
            "src/requests/utils.py::_validate_header_part",
        ],
    },
]

# --------------------------------------------------------------------------------------------
# Repos
# --------------------------------------------------------------------------------------------


def _git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def clone_repos(data_dir: Path, repos: dict[str, str] = REPOS) -> dict[str, Path]:
    """Full clones (T3 needs diffs) checked out at the pinned SHAs."""
    dirs = {}
    for slug, sha in repos.items():
        dest = data_dir / slug.replace("/", "__")
        if not dest.exists():
            subprocess.run(
                ["git", "clone", "--quiet", f"https://github.com/{slug}.git", str(dest)], check=True
            )
        _git(dest, "checkout", "--quiet", sha)
        dirs[slug] = dest
    return dirs


# --------------------------------------------------------------------------------------------
# Chunking (fixed, stdlib ast; identical input for every model)
# --------------------------------------------------------------------------------------------

SKIP_DIR_PARTS = {"node_modules", "vendor", "build", "dist", ".venv", "venv", "site-packages"}


@dataclass
class Chunk:
    id: str
    repo: str
    path: str
    qualname: str
    kind: str  # function | method | class | module
    start: int
    end: int
    text: str  # header + code
    text_nodoc: str  # header + code without its docstring (T1 corpus)
    docstring: str = ""


def _is_overload(node: ast.AST) -> bool:
    decorators = getattr(node, "decorator_list", [])
    return any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in decorators
    )


def _header(path: str, qualname: str) -> str:
    return f"# path: {path}\n# symbol: {qualname}\n"


def _docstring_span(node: ast.AST) -> tuple[int, int] | None:
    body = getattr(node, "body", [])
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].lineno, body[0].end_lineno or body[0].lineno
    return None


def chunk_python_source(repo: str, path: str, source: str) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines()
    chunks: list[Chunk] = []

    def emit(qualname: str, kind: str, start: int, end: int, node: ast.AST | None) -> None:
        code_lines = lines[start - 1 : end]
        code = "\n".join(code_lines)
        if not code.strip():
            return
        doc, nodoc = "", code
        span = _docstring_span(node) if node is not None else None
        if span:
            doc = ast.get_docstring(node) or ""  # type: ignore[arg-type]
            kept = [ln for i, ln in enumerate(code_lines, start) if not span[0] <= i <= span[1]]
            nodoc = "\n".join(kept)
        header = _header(path, qualname)
        chunks.append(
            Chunk(
                id=f"{repo}:{path}::{qualname}@{start}",
                repo=repo,
                path=path,
                qualname=qualname,
                kind=kind,
                start=start,
                end=end,
                text=(header + code)[:MAX_CHUNK_CHARS],
                text_nodoc=(header + nodoc)[:MAX_CHUNK_CHARS],
                docstring=doc,
            )
        )

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if _is_overload(node):
                continue
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                kind = "method" if in_class else "function"
                emit(prefix + node.name, kind, start, node.end_lineno or node.lineno, node)
            elif isinstance(node, ast.ClassDef):
                qual = prefix + node.name
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                members = [
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                ]
                first_member = min(
                    (min([m.lineno] + [d.lineno for d in m.decorator_list]) for m in members),
                    default=None,
                )
                end = first_member - 1 if first_member else (node.end_lineno or node.lineno)
                emit(qual, "class", start, end, node)
                visit(node.body, qual + ".", in_class=True)

    visit(tree.body, "", in_class=False)

    # Module-level code that isn't a def/class (constants, imports, module docstring).
    rest = [
        n
        for n in tree.body
        if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    if rest:
        segments = ["\n".join(lines[n.lineno - 1 : (n.end_lineno or n.lineno)]) for n in rest]
        code = "\n".join(segments)
        if code.count("\n") >= 2:
            header = _header(path, "<module>")
            chunks.append(
                Chunk(
                    id=f"{repo}:{path}::<module>",
                    repo=repo,
                    path=path,
                    qualname="<module>",
                    kind="module",
                    start=rest[0].lineno,
                    end=rest[-1].end_lineno or rest[-1].lineno,
                    text=(header + code)[:MAX_CHUNK_CHARS],
                    text_nodoc=(header + code)[:MAX_CHUNK_CHARS],
                )
            )
    return chunks


def build_corpus(repo_dirs: dict[str, Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for slug, root in repo_dirs.items():
        for rel in _git(root, "ls-files", "*.py").splitlines():
            if any(part in SKIP_DIR_PARTS for part in rel.split("/")):
                continue
            source = (root / rel).read_text(encoding="utf-8", errors="replace")
            chunks.extend(chunk_python_source(slug, rel, source))
    return chunks


# --------------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------------


@dataclass
class Task:
    name: str
    queries: list[str]
    gold: list[set[str]]  # gold doc ids per query
    doc_ids: list[str]
    doc_texts: list[str]
    query_kind: str  # "code" or "commit": selects the query prompt
    query_meta: list[dict[str, object]] = field(default_factory=list)
    # Restrict retrieval to these docs (e.g. one repo, as production filters by repo).
    doc_mask: np.ndarray | None = None


def task_docstring_to_code(chunks: list[Chunk], max_queries: int = 500, seed: int = 0) -> Task:
    """T1: first docstring paragraph -> its function, searched over docstring-free code."""
    seen: set[str] = set()
    candidates = []
    for c in chunks:
        if c.kind not in {"function", "method"} or not c.docstring:
            continue
        name = c.qualname.rsplit(".", 1)[-1]
        if name.startswith("test") or (name.startswith("__") and name.endswith("__")):
            continue
        summary = " ".join(c.docstring.strip().split("\n\n")[0].split())
        if len(summary.split()) < 4 or summary.lower() in seen:
            continue
        seen.add(summary.lower())
        candidates.append((summary, c.id))
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:max_queries]
    return Task(
        name="T1_docstring_to_code",
        queries=[q for q, _ in candidates],
        gold=[{cid} for _, cid in candidates],
        doc_ids=[c.id for c in chunks],
        doc_texts=[c.text_nodoc for c in chunks],
        query_kind="code",
    )


def _gold_ids(chunks: list[Chunk], repo: str, golds: Sequence[str]) -> set[str]:
    ids = set()
    for gold in golds:
        path, qual = gold.split("::", 1)
        for c in chunks:
            if (
                c.repo == repo
                and c.path == path
                and (c.qualname == qual or c.qualname.startswith(qual + "."))
            ):
                ids.add(c.id)
    return ids


def task_dev_questions(chunks: list[Chunk]) -> Task:
    """T2: hand-written developer questions about psf/requests."""
    gold = []
    for item in DEV_QUESTIONS:
        ids = _gold_ids(chunks, "psf/requests", item["gold"])  # type: ignore[arg-type]
        if not ids:
            raise ValueError(f"No chunk matches gold for: {item['q']}")
        gold.append(ids)
    return Task(
        name="T2_dev_questions",
        queries=[str(item["q"]) for item in DEV_QUESTIONS],
        gold=gold,
        doc_ids=[c.id for c in chunks],
        doc_texts=[c.text for c in chunks],
        query_kind="code",
        query_meta=[{"type": item["type"]} for item in DEV_QUESTIONS],
    )


def task_dev_questions_in_repo(t2: Task, chunks: list[Chunk]) -> Task:
    """T2r: the same questions, searching only psf/requests chunks.

    The questions don't name the repo, so in T2 a model that returns httpx's look-alike
    implementation is scored as a miss. Production filters by repo, which is what this models.
    It shares T2's doc list, so the corpus is encoded only once.
    """
    return replace(
        t2,
        name="T2r_dev_questions_in_repo",
        doc_mask=np.array([c.repo == "psf/requests" for c in chunks]),
    )


_CLOSING_REF = re.compile(r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b:?\s+#(\d+)", re.I)
_ANY_REF = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?|see|refs?)\b:?\s+#\d+|\(?#\d+\)?"
    r"|https?://github\.com/\S+/(?:issues|pull)/\d+",
    re.I,
)
_MERGE_PR = re.compile(r"^Merge pull request \S+ from \S+", re.M)


def _clean_commit_text(text: str) -> str:
    text = _MERGE_PR.sub("Merge pull request", text)
    return re.sub(r"\(\s*\)", "", _ANY_REF.sub("", text))


# Text markers, not control bytes: requests' history has diffs containing raw NUL bytes.
_COMMIT_START = "<<<repo-rag-commit:7f3a9c>>>"
_COMMIT_END = "<<<repo-rag-end:7f3a9c>>>"


def _commit_records(repo_dir: Path) -> list[dict[str, object]]:
    """Every commit (incl. merges) with message, parents and, for non-merges, a short diff.

    No pathspec: a pathspec turns on history simplification, which drops PR-branch commits.
    """
    raw = _git(
        repo_dir,
        "log",
        f"--format={_COMMIT_START}%H%n%P%n%B{_COMMIT_END}",
        "-p",
        "--unified=1",
        "--no-color",
        "HEAD",
    )
    records = []
    for block in raw.split(_COMMIT_START)[1:]:
        header, _, diff = block.partition(_COMMIT_END)
        sha, parents, message = [*header.split("\n", 2), "", ""][:3]
        records.append(
            {"sha": sha, "parents": parents.split(), "message": message.strip(), "diff": diff}
        )
    return records


VENDORED_PREFIXES = ("requests/packages/",)


# Commit document formats compared by t3_tuning.py. "diff2000" is the run 1/2 baseline.
COMMIT_DOC_FORMATS = ("diff2000", "symbols_diff800", "symbols", "message")

_HUNK_CONTEXT = re.compile(r"^@@ [^@]* @@ ?(.*)$", re.M)
_DEF_NAME = re.compile(r"\b(?:def|class|func|function)\s+([A-Za-z_]\w*)")


def _touched_symbols(path: str, file_block: str) -> list[str]:
    """Functions/classes a diff touches, from git's hunk headers (`@@ ... @@ def foo(...)`).

    A cheap stand-in for Phase 5's AST hunk -> symbol mapping.
    """
    names: list[str] = []
    for context in _HUNK_CONTEXT.findall(file_block):
        names += _DEF_NAME.findall(context)
    for line in file_block.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            names += _DEF_NAME.findall(line)
    module = path.rsplit("/", 1)[-1]
    return [f"{module}::{n}" for n in dict.fromkeys(names)]


def _commit_doc(record: dict[str, object], fmt: str = "diff2000") -> str:
    """Cleaned message + changed files (+ touched symbols) (+ short diff); vendored files skipped."""
    files: list[str] = []
    symbols: list[str] = []
    diff_lines: list[str] = []
    for file_block in ("\n" + str(record["diff"])).split("\ndiff --git a/")[1:]:
        path = file_block.split(" ", 1)[0]
        if path.startswith(VENDORED_PREFIXES):
            continue
        files.append(path)
        symbols += _touched_symbols(path, file_block)
        diff_lines += [
            ln
            for ln in file_block.splitlines()
            if ln.startswith(("+", "-", "@@")) and not ln.startswith(("+++", "---"))
        ]
    message = _clean_commit_text(str(record["message"]))
    if fmt == "message":
        return message
    parts = [message, "files: " + ", ".join(files[:20])]
    if fmt in {"symbols", "symbols_diff800"}:
        parts.append("touched: " + ", ".join(symbols[:30]))
    if fmt == "diff2000":
        parts.append("\n".join(diff_lines)[:2000])
    elif fmt == "symbols_diff800":
        parts.append("\n".join(diff_lines)[:800])
    elif fmt != "symbols":
        raise ValueError(f"unknown commit doc format: {fmt}")
    return "\n\n".join(parts)


def commit_corpus(
    repo_dir: Path, fmt: str = "diff2000"
) -> tuple[list[str], list[str], dict[int, set[str]]]:
    """Offline half of T3: (doc_ids, doc_texts, issue number -> fix commit shas)."""
    ids, texts, links = commit_corpus_formats(repo_dir, (fmt,))
    return ids, texts[fmt], links


def commit_corpus_formats(
    repo_dir: Path, fmts: Sequence[str]
) -> tuple[list[str], dict[str, list[str]], dict[int, set[str]]]:
    """Like commit_corpus, but builds several doc formats from one `git log` pass.

    Docs are non-merge commits. A merge's fix commits are the PR's own commits.
    """
    records = _commit_records(repo_dir)
    non_merge = [r for r in records if len(r["parents"]) == 1]  # type: ignore[arg-type]
    corpus_shas = {str(r["sha"]) for r in non_merge}

    links: dict[int, set[str]] = {}
    for r in records:
        numbers = {int(n) for n in _CLOSING_REF.findall(str(r["message"]))}
        if not numbers:
            continue
        parents: list[str] = r["parents"]  # type: ignore[assignment]
        if len(parents) == 1:
            fix_shas = {str(r["sha"])}
        else:
            fix_shas = set(_git(repo_dir, "rev-list", f"{parents[0]}..{parents[1]}").split())
        for n in numbers:
            links.setdefault(n, set()).update(fix_shas & corpus_shas)
    links = {n: shas for n, shas in links.items() if shas}
    texts = {fmt: [_commit_doc(r, fmt) for r in non_merge] for fmt in fmts}
    return [str(r["sha"]) for r in non_merge], texts, links


def _github_issue(repo: str, number: int, token: str) -> dict[str, object] | None:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{number}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        # Missing, transferred or deleted issues are skipped.
        return None


def bug_queries(
    links: dict[int, set[str]], token: str, cache_path: Path, max_pairs: int = 200
) -> tuple[list[str], list[set[str]], list[int]]:
    """Issue title+body queries for linked issues: (queries, gold shas, issue numbers)."""
    cache: dict[str, object] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    queries: list[str] = []
    gold: list[set[str]] = []
    numbers: list[int] = []
    for number in sorted(links)[-max_pairs * 2 :]:  # the most recent linked issues
        key = str(number)
        if key not in cache:
            cache[key] = _github_issue(BUG_REPO, number, token)
        issue = cache[key]
        if not isinstance(issue, dict) or "pull_request" in issue:
            continue
        body = str(issue.get("body") or "")[:3000]
        queries.append(f"{issue['title']}\n\n{body}".strip())
        gold.append(links[number])
        numbers.append(number)
        if len(queries) >= max_pairs:
            break
    cache_path.write_text(json.dumps(cache))
    return queries, gold, numbers


def task_bug_to_fix(repo_dir: Path, token: str, cache_path: Path, max_pairs: int = 200) -> Task:
    """T3: issue title+body -> the commit(s) that fixed it. Issue refs are scrubbed from commits."""
    doc_ids, doc_texts, links = commit_corpus(repo_dir)
    queries, gold, _ = bug_queries(links, token, cache_path, max_pairs)
    return Task(
        name="T3_bug_to_fix",
        queries=queries,
        gold=gold,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_kind="commit",
    )


# --------------------------------------------------------------------------------------------
# Retrieval + metrics
# --------------------------------------------------------------------------------------------

_SPLIT = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")


def code_tokens(text: str) -> list[str]:
    """BM25 tokens: whole identifiers plus their camelCase/snake_case parts, lowercased."""
    tokens = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        low = word.lower()
        tokens.append(low)
        parts = [p.lower() for piece in word.split("_") for p in _SPLIT.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def bm25_rankings(task: Task, k: int = TOP_K) -> list[list[int]]:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([code_tokens(t) for t in task.doc_texts])
    rankings = []
    for q in task.queries:
        scores = bm25.get_scores(code_tokens(q))
        if task.doc_mask is not None:
            scores = np.where(task.doc_mask, scores, -np.inf)
        rankings.append(list(np.argsort(-scores)[: _depth(k, task.doc_mask, len(scores))]))
    return rankings


def _depth(k: int, mask: np.ndarray | None, n_docs: int) -> int:
    return min(k, int(mask.sum()) if mask is not None else n_docs)


def dense_rankings(
    q_emb: np.ndarray, d_emb: np.ndarray, k: int = TOP_K, mask: np.ndarray | None = None
) -> list[list[int]]:
    sims = q_emb @ d_emb.T
    if mask is not None:
        sims = np.where(mask[None, :], sims, -np.inf)
    k = _depth(k, mask, sims.shape[1])
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
    return [list(row) for row in np.take_along_axis(top, order, axis=1)]


def rrf(*rankings: list[list[int]], k: int = RRF_K, depth: int = TOP_K) -> list[list[int]]:
    fused = []
    for per_query in zip(*rankings, strict=True):
        scores: dict[int, float] = {}
        for ranking in per_query:
            for rank, doc in enumerate(ranking):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        fused.append(sorted(scores, key=scores.__getitem__, reverse=True)[:depth])
    return fused


def score(task: Task, rankings: list[list[int]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_query = []
    for ranking, gold in zip(rankings, task.gold, strict=True):
        ids = [task.doc_ids[i] for i in ranking[:10]]
        first = next((r for r, d in enumerate(ids) if d in gold), None)
        dcg = sum(1 / math.log2(r + 2) for r, d in enumerate(ids) if d in gold)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gold), 10)))
        per_query.append(
            {
                "hit@1": float(first is not None and first < 1),
                "hit@5": float(first is not None and first < 5),
                "hit@10": float(first is not None),
                "mrr@10": 0.0 if first is None else 1 / (first + 1),
                "ndcg@10": dcg / idcg if idcg else 0.0,
            }
        )
    keys = per_query[0].keys() if per_query else []
    agg = {key: statistics.fmean(p[key] for p in per_query) for key in keys}
    return agg, per_query


def truncate_dims(emb: np.ndarray, dims: int) -> np.ndarray:
    cut = emb[:, :dims]
    return cut / np.linalg.norm(cut, axis=1, keepdims=True).clip(min=1e-12)


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def _patch_extended_attention_mask(model: object) -> None:
    """Restore `get_extended_attention_mask` for remote modeling code that still calls it.

    CodeRankEmbed's `modeling_hf_nomic_bert.py` uses this transformers helper, which newer
    transformers releases removed. The notebook pins transformers<5; this is the fallback if
    that pin is overridden. Same semantics as the old encoder-only helper: an additive mask
    of 0 for real tokens and dtype-min for padding.
    """
    import torch

    def get_extended_attention_mask(
        self, attention_mask, input_shape=None, device=None, dtype=None
    ):  # type: ignore[no-untyped-def]
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        else:
            extended = attention_mask[:, None, None, :]
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    for module in model.modules():  # type: ignore[attr-defined]
        cls = type(module)
        if cls.__name__.startswith("NomicBert") and not hasattr(
            module, "get_extended_attention_mask"
        ):
            cls.get_extended_attention_mask = get_extended_attention_mask  # type: ignore[attr-defined]


class Encoder:
    """Uniform encode(texts, prompt) for sentence-transformers and Voyage."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.spec = spec
        self.device = device
        if spec.provider == "voyage":
            import voyageai

            self.client = voyageai.Client()
            self.model = None
            return
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, object] = {
            "trust_remote_code": spec.trust_remote_code,
            "revision": spec.revision,
            "device": device,
        }
        if device == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        if spec.left_padding:
            kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
        self.model = SentenceTransformer(spec.hf_id, **kwargs)
        self.model.max_seq_length = min(MAX_SEQ_TOKENS, spec.max_context)
        _patch_extended_attention_mask(self.model)

    @property
    def max_tokens(self) -> int:
        if self.model is None:
            return self.spec.max_context
        return int(self.model.max_seq_length)

    @property
    def dim(self) -> int:
        if self.model is None:
            return 1024
        get_dim = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        return int(get_dim())

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        if self.model is None:
            return [len(t) // 4 for t in texts]  # rough estimate for API models
        enc = self.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in enc["input_ids"]]

    def encode(
        self,
        texts: Sequence[str],
        prompt: str = "",
        is_query: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if self.model is None:
            out = []
            for i in range(0, len(texts), 64):
                res = self.client.embed(
                    list(texts[i : i + 64]),
                    model=self.spec.hf_id,
                    input_type="query" if is_query else "document",
                    truncation=True,
                )
                out.extend(res.embeddings)
            emb = np.asarray(out, dtype=np.float32)
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return self.model.encode(
            list(texts),
            prompt=prompt or None,
            batch_size=batch_size or self.spec.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


def _query_prompt(spec: ModelSpec, task: Task) -> str:
    return spec.commit_query_prompt if task.query_kind == "commit" else spec.code_query_prompt


def _latency_ms(fn: Callable[[], object], runs: int) -> tuple[float, float]:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return statistics.median(times), times[max(0, math.ceil(0.95 * len(times)) - 1)]


def _evaluate_on_device(
    spec: ModelSpec,
    tasks: list[Task],
    bm25: dict[str, list[list[int]]],
    device: str,
    speed: dict[str, object],
) -> tuple[list[dict], list[dict]]:
    """Load the model, time corpus/query encoding and score every task.

    The model lives only inside this function, so its memory is freed when it returns.
    """
    metric_rows: list[dict] = []
    per_question: list[dict] = []
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    enc = Encoder(spec, device)
    speed["load_s"] = round(time.perf_counter() - t0, 1)
    speed["dim"] = enc.dim

    # Encode each distinct corpus once (T1 and T2 share chunks but T1 strips docstrings).
    corpus_cache: dict[int, np.ndarray] = {}
    enc.encode(tasks[0].doc_texts[:16])  # warm-up
    total_docs, total_tokens, total_s = 0, 0, 0.0
    truncated, measured = 0, 0
    for task in tasks:
        key = id(task.doc_texts)
        if key in corpus_cache:
            continue
        lengths = enc.token_lengths(task.doc_texts)
        truncated += sum(n > enc.max_tokens for n in lengths)
        measured += len(lengths)
        t0 = time.perf_counter()
        corpus_cache[key] = enc.encode(task.doc_texts)
        total_s += time.perf_counter() - t0
        total_docs += len(task.doc_texts)
        total_tokens += sum(min(n, enc.max_tokens) for n in lengths)
    speed["corpus_docs"] = total_docs
    speed["corpus_encode_s"] = round(total_s, 1)
    speed["docs_per_s"] = round(total_docs / total_s, 1)
    speed["tokens_per_s"] = round(total_tokens / total_s)
    speed["truncated_pct"] = round(100 * truncated / max(measured, 1), 2)

    sample_q = tasks[0].queries[0]
    prompt = _query_prompt(spec, tasks[0])
    p50, p95 = _latency_ms(lambda: enc.encode([sample_q], prompt, is_query=True), runs=30)
    speed[f"query_ms_p50_{device}"] = round(p50, 1)
    speed[f"query_ms_p95_{device}"] = round(p95, 1)
    if device == "cuda":
        import torch

        speed["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    for task in tasks:
        d_emb = corpus_cache[id(task.doc_texts)]
        q_emb = enc.encode(task.queries, _query_prompt(spec, task), is_query=True)
        variants = {"dense": dense_rankings(q_emb, d_emb, mask=task.doc_mask)}
        for dims in spec.mrl_dims:
            if dims < q_emb.shape[1]:
                variants[f"dense@{dims}d"] = dense_rankings(
                    truncate_dims(q_emb, dims), truncate_dims(d_emb, dims), mask=task.doc_mask
                )
        variants["hybrid_bm25"] = rrf(variants["dense"], bm25[task.name])
        for variant, rankings in variants.items():
            agg, per_q = score(task, rankings)
            metric_rows.append(
                {
                    "model": spec.key,
                    "variant": variant,
                    "task": task.name,
                    "n": len(task.queries),
                    **{k: round(v, 4) for k, v in agg.items()},
                }
            )
            if task.name.startswith("T2") and variant in {"dense", "hybrid_bm25"}:
                for i, pq in enumerate(per_q):
                    per_question.append(
                        {
                            "task": task.name,
                            "model": spec.key,
                            "variant": variant,
                            "q": task.queries[i],
                            **task.query_meta[i],
                            **pq,
                        }
                    )
    return metric_rows, per_question


def _cpu_query_latency(spec: ModelSpec, query: str, prompt: str) -> dict[str, float]:
    """Single-query latency on CPU, fp32: what a laptop would pay per question."""
    enc = Encoder(spec, "cpu")
    enc.encode([query], prompt, is_query=True)  # warm-up
    p50, p95 = _latency_ms(lambda: enc.encode([query], prompt, is_query=True), runs=15)
    return {"query_ms_p50_cpu": round(p50, 1), "query_ms_p95_cpu": round(p95, 1)}


def run_model(
    spec: ModelSpec,
    tasks: list[Task],
    bm25: dict[str, list[list[int]]],
    device: str,
    cpu_latency: bool,
) -> tuple[list[dict], dict, list[dict]]:
    speed: dict[str, object] = {"model": spec.key}
    metric_rows, per_question = _evaluate_on_device(spec, tasks, bm25, device, speed)
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
    if cpu_latency and device != "cpu" and spec.laptop_cpu_ok and spec.provider == "hf":
        speed.update(_cpu_query_latency(spec, tasks[0].queries[0], _query_prompt(spec, tasks[0])))
        gc.collect()
    return metric_rows, speed, per_question


def factors_row(spec: ModelSpec, dim: int) -> dict[str, object]:
    return {
        "model": spec.key,
        "hf_id": spec.hf_id,
        "params": spec.params,
        "license": spec.license,
        "commercial_use": spec.license in {"MIT", "Apache-2.0"} or "API" in spec.license,
        "dim": dim,
        "mrl_dims": ",".join(map(str, spec.mrl_dims)) or "-",
        "max_context": spec.max_context,
        "laptop_cpu_query_ok": spec.laptop_cpu_ok,
        "index_mb_per_100k_chunks_fp32": round(dim * 4 * 100_000 / 1e6),
    }


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("bench_data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--models", nargs="+", default=["coderankembed", "qwen3-0.6b", "nomic-embed-code"]
    )
    parser.add_argument("--repos", nargs="+", default=list(REPOS))
    parser.add_argument("--t1-queries", type=int, default=500)
    parser.add_argument("--t3-pairs", type=int, default=200)
    parser.add_argument("--no-cpu-latency", action="store_true")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    repo_dirs = clone_repos(args.data_dir, {slug: REPOS[slug] for slug in args.repos})
    chunks = build_corpus(repo_dirs)
    print(f"corpus: {len(chunks)} chunks from {len(repo_dirs)} repos")

    tasks = [task_docstring_to_code(chunks, args.t1_queries)]
    if "psf/requests" in repo_dirs:
        t2 = task_dev_questions(chunks)
        tasks += [t2, task_dev_questions_in_repo(t2, chunks)]
    token = os.environ.get("GITHUB_TOKEN", "")
    if token and BUG_REPO in repo_dirs:
        tasks.append(
            task_bug_to_fix(
                repo_dirs[BUG_REPO], token, args.data_dir / "issues_cache.json", args.t3_pairs
            )
        )
    else:
        print("GITHUB_TOKEN not set: skipping T3 (bug report -> fix commit)")
    for t in tasks:
        print(f"{t.name}: {len(t.queries)} queries over {len(t.doc_texts)} docs")

    bm25 = {t.name: bm25_rankings(t) for t in tasks}
    metric_rows, speed_rows, factor_rows, per_question = [], [], [], []
    for t in tasks:
        agg, _ = score(t, bm25[t.name])
        metric_rows.append(
            {
                "model": "bm25",
                "variant": "lexical",
                "task": t.name,
                "n": len(t.queries),
                **{k: round(v, 4) for k, v in agg.items()},
            }
        )

    for key in args.models:
        spec = MODELS[key]
        if spec.provider == "voyage" and not os.environ.get("VOYAGE_API_KEY"):
            print(f"skip {key}: VOYAGE_API_KEY not set")
            continue
        print(f"\n=== {key} ({spec.hf_id}) ===")
        rows, speed, pq = run_model(spec, tasks, bm25, device, not args.no_cpu_latency)
        metric_rows += rows
        speed_rows.append(speed)
        per_question += pq
        factor_rows.append(factors_row(spec, int(speed["dim"])))
        print(json.dumps(speed))
        _save(args.out, metric_rows, speed_rows, factor_rows, per_question, chunks, tasks)

    _save(args.out, metric_rows, speed_rows, factor_rows, per_question, chunks, tasks)
    print(f"\nResults written to {args.out}")


def _save(
    out: Path,
    metrics: list[dict],
    speed: list[dict],
    factors: list[dict],
    per_question: list[dict],
    chunks: list[Chunk],
    tasks: list[Task],
) -> None:
    import pandas as pd

    m = pd.DataFrame(metrics)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(speed).to_csv(out / "speed.csv", index=False)
    pd.DataFrame(factors).to_csv(out / "factors.csv", index=False)
    pd.DataFrame(per_question).to_csv(out / "t2_per_question.csv", index=False)
    meta = {
        "repos": REPOS,
        "max_seq_tokens": MAX_SEQ_TOKENS,
        "corpus_chunks": len(chunks),
        "tasks": {t.name: {"queries": len(t.queries), "docs": len(t.doc_texts)} for t in tasks},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    lines = ["# Embedding benchmark results", ""]
    if not m.empty:
        pivot = m.pivot_table(index=["model", "variant"], columns="task", values="ndcg@10")
        pivot["mean"] = pivot.mean(axis=1)
        lines += ["## nDCG@10 by task", "", pivot.round(3).to_markdown(), ""]
        hits = m.pivot_table(index=["model", "variant"], columns="task", values="hit@5")
        lines += ["## Hit@5 by task", "", hits.round(3).to_markdown(), ""]
    if speed:
        lines += ["## Speed and memory", "", pd.DataFrame(speed).to_markdown(index=False), ""]
    if factors:
        lines += ["## Other factors", "", pd.DataFrame(factors).to_markdown(index=False), ""]
    if per_question:
        pq = pd.DataFrame(per_question)
        by_type = pq.pivot_table(index=["task", "model", "variant"], columns="type", values="hit@5")
        lines += ["## T2 Hit@5 by question type", "", by_type.round(3).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
