"""Author source-grounded architecture labels, then freeze them BEFORE retrieval.

Labels are authored by Codex, unreviewed by a person. Never overwrite a freeze.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/datasets/architecture_v1"
REPOS = {
    "requests": (
        "psf/requests",
        "data/repos/psf__requests",
        "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
    ),
    "flask": (
        "pallets/flask",
        "data/eval_repos/pallets__flask",
        "d086db856be187255b8ec61ef409357393020f32",
    ),
}

# group, split, type, question, {module: evidence anchors}. No retrieval authored these.
SEEDS = {
    "requests": [
        (
            "public",
            "dev",
            "layout",
            "How is the public HTTP interface separated from persistent client state and network transport?",
            {"__init__": "", "api": "request", "sessions": "Session", "adapters": "HTTPAdapter"},
        ),
        (
            "public",
            "dev",
            "flow",
            "Trace a one-off HTTP call from a convenience entry point to the network connection pool.",
            {"api": "request", "sessions": "Session.request", "adapters": "HTTPAdapter.send"},
        ),
        (
            "public",
            "dev",
            "dependency",
            "Which internal layers must a convenience HTTP entry point rely on to prepare and send a request?",
            {
                "api": "request",
                "sessions": "Session.request",
                "models": "PreparedRequest",
                "adapters": "HTTPAdapter",
            },
        ),
        (
            "representations",
            "dev",
            "responsibility",
            "Which part owns request preparation and response representation, and which part coordinates their use?",
            {"models": "PreparedRequest", "sessions": "Session.prepare_request"},
        ),
        (
            "representations",
            "dev",
            "flow",
            "How is a low-level transport response converted into the response object returned to the caller?",
            {
                "adapters": "HTTPAdapter.build_response",
                "models": "Response",
                "sessions": "Session.send",
            },
        ),
        (
            "state",
            "test",
            "responsibility",
            "Which components jointly retain cookies between calls and attach them to outgoing requests?",
            {
                "cookies": "RequestsCookieJar",
                "models": "PreparedRequest.prepare_cookies",
                "sessions": "Session.prepare_request",
            },
        ),
        (
            "state",
            "test",
            "flow",
            "Trace the redirect cycle, including rebuilding credentials and preserving cookies across the next send.",
            {
                "sessions": "SessionRedirectMixin.resolve_redirects",
                "cookies": "extract_cookies_to_jar",
                "models": "PreparedRequest",
            },
        ),
        (
            "state",
            "test",
            "dependency",
            "How does persistent client configuration depend on the case-insensitive header container and default settings helpers?",
            {
                "sessions": "Session.__init__",
                "structures": "CaseInsensitiveDict",
                "utils": "default_headers",
            },
        ),
        (
            "security",
            "test",
            "responsibility",
            "Where do authentication strategies live, and how do request preparation and redirect handling invoke them?",
            {
                "auth": "AuthBase",
                "models": "PreparedRequest.prepare_auth",
                "sessions": "SessionRedirectMixin.rebuild_auth",
            },
        ),
        (
            "security",
            "test",
            "dependency",
            "Which pieces cooperate to choose certificate verification defaults and configure TLS for transport?",
            {
                "adapters": "HTTPAdapter.cert_verify",
                "certs": "",
                "utils": "DEFAULT_CA_BUNDLE_PATH",
                "sessions": "Session.merge_environment_settings",
            },
        ),
        (
            "payload",
            "test",
            "responsibility",
            "Which modules cooperate to encode form and multipart upload bodies before network transmission?",
            {
                "models": "RequestEncodingMixin",
                "utils": "to_key_val_list",
                "adapters": "HTTPAdapter.send",
            },
        ),
        (
            "payload",
            "test",
            "flow",
            "Trace streamed response data from the raw connection through content decoding and iteration.",
            {
                "adapters": "HTTPAdapter.build_response",
                "models": "Response.iter_content",
                "utils": "stream_decode_response_unicode",
            },
        ),
        (
            "payload",
            "test",
            "responsibility",
            "How are user hooks registered and dispatched around the response lifecycle?",
            {"hooks": "dispatch_hook", "models": "RequestHooksMixin", "sessions": "Session.send"},
        ),
        (
            "runtime",
            "test",
            "layout",
            "How are compatibility shims, dependency checks and the public exception hierarchy separated?",
            {"compat": "", "__init__": "check_compatibility", "exceptions": "RequestException"},
        ),
        (
            "runtime",
            "test",
            "dependency",
            "Which layers translate connection and timeout failures from the HTTP backend into library-specific errors?",
            {"adapters": "HTTPAdapter.send", "exceptions": "ConnectionError"},
        ),
    ],
    "flask": [
        (
            "layout",
            "dev",
            "layout",
            "How is the framework divided between reusable setup machinery and the concrete WSGI application?",
            {"app": "Flask", "sansio/app": "App", "sansio/scaffold": "Scaffold"},
        ),
        (
            "layout",
            "dev",
            "dependency",
            "Which internal layers supply the public application object, HTTP wrappers and setup base classes?",
            {"__init__": "", "app": "Flask", "wrappers": "Request", "sansio/app": "App"},
        ),
        (
            "layout",
            "dev",
            "responsibility",
            "Where are method-based views adapted into callable endpoints and registered as routing rules?",
            {
                "views": "View.as_view",
                "sansio/app": "App.add_url_rule",
                "sansio/scaffold": "Scaffold.route",
            },
        ),
        (
            "templates",
            "dev",
            "responsibility",
            "Which components connect application template configuration, loader dispatch and rendering helpers?",
            {
                "templating": "DispatchingJinjaLoader",
                "app": "Flask.create_jinja_environment",
                "sansio/scaffold": "Scaffold.jinja_loader",
            },
        ),
        (
            "templates",
            "dev",
            "flow",
            "How do a template rendering call and its default context reach the template environment?",
            {
                "templating": "render_template",
                "app": "Flask.update_template_context",
                "globals": "",
            },
        ),
        (
            "blueprints",
            "test",
            "layout",
            "How is reusable route grouping divided between its public object, shared setup logic and application registration?",
            {
                "blueprints": "Blueprint",
                "sansio/blueprints": "Blueprint",
                "sansio/app": "App.register_blueprint",
            },
        ),
        (
            "blueprints",
            "test",
            "flow",
            "Trace deferred route registration from a reusable route group into the application's URL map.",
            {
                "sansio/blueprints": "Blueprint.register",
                "sansio/app": "App.register_blueprint",
                "sansio/scaffold": "Scaffold.route",
            },
        ),
        (
            "blueprints",
            "test",
            "dependency",
            "Which common setup layer is shared by the application and its reusable route groups?",
            {"sansio/scaffold": "Scaffold", "sansio/app": "App", "sansio/blueprints": "Blueprint"},
        ),
        (
            "context",
            "test",
            "responsibility",
            "Which components implement context-local application, request and session access and context cleanup?",
            {"globals": "", "ctx": "AppContext", "app": "Flask.do_teardown_request"},
        ),
        (
            "context",
            "test",
            "flow",
            "How does the request lifecycle open a signed cookie session and save it back on the response?",
            {
                "ctx": "AppContext._get_session",
                "sessions": "SecureCookieSessionInterface",
                "app": "Flask.process_response",
            },
        ),
        (
            "dispatch",
            "test",
            "flow",
            "Trace an incoming WSGI request through context creation, view dispatch and response conversion.",
            {"app": "Flask.wsgi_app", "ctx": "AppContext.from_environ", "wrappers": "Response"},
        ),
        (
            "dispatch",
            "test",
            "responsibility",
            "Where do error-handler registration, exception dispatch and failure signals fit together?",
            {
                "sansio/scaffold": "Scaffold.register_error_handler",
                "sansio/app": "App._find_error_handler",
                "app": "Flask.handle_exception",
                "signals": "",
            },
        ),
        (
            "dispatch",
            "test",
            "dependency",
            "Which components connect URL-building helpers to the active application and its routing map?",
            {
                "helpers": "url_for",
                "app": "Flask.url_for",
                "globals": "",
                "sansio/app": "App.add_url_rule",
            },
        ),
        (
            "configuration",
            "test",
            "responsibility",
            "How are application defaults, configuration loading and request-specific limits divided?",
            {"app": "Flask", "config": "Config", "wrappers": "Request.max_content_length"},
        ),
        (
            "configuration",
            "test",
            "flow",
            "Trace command-line application discovery into running the development server.",
            {"cli": "ScriptInfo.load_app", "app": "Flask.run"},
        ),
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evidence(checkout: Path, path: str, anchor: str) -> dict:
    source = (checkout / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                found[name] = (child.lineno, child.end_lineno or child.lineno)
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree)
    if anchor and anchor not in found:
        # Constants are evidence too; find an exact assignment name.
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == anchor]
        if not matches:
            raise ValueError(f"Missing anchor {path}:{anchor}")
        start = end = matches[0].lineno
    else:
        start, end = found[anchor] if anchor else (1, min(100, len(source.splitlines())))
    return {
        "path": path,
        "anchor": anchor or "module imports / declarations",
        "start_line": start,
        "end_line": end,
        "sha256": digest(checkout / path),
    }


def verify_frozen(folder: Path = OUT) -> dict:
    frozen = json.loads((folder / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in frozen["sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return frozen


def main() -> None:
    if (OUT / "freeze.json").exists():
        raise ValueError("Refusing to overwrite frozen architecture inputs")
    protocol = {
        "version": "architecture_v1",
        "questions": ["Q0 architecture labels", "Q1 router", "Q2 richer facts", "Q3 prose"],
        "label_review": "unreviewed by a person",
        "author": "Codex",
        "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B", "dims": 512, "dtype": "float32"},
        "arms": [
            "chunks",
            "summaries",
            "fused",
            "router_file_1",
            "router_file_2",
            "router_file_3",
            "router_dir_1",
            "router_dir_2",
            "router_dir_3",
        ],
        "choice": "pooled DEV Recall@5, then MRR@10; remaining ties use listed arm order; E1 router chosen from router arms only; E2 facts chosen by same dev rule with fixed E1 router; no test or code labels for selection",
        "fact_variants": {
            "base": [],
            "F1": ["docstrings"],
            "F2": ["runtime"],
            "F3": ["history"],
            "F4": ["readme"],
            "F1F2": ["docstrings", "runtime"],
            "F1F2F3": ["docstrings", "runtime", "history"],
            "F1F2F3F4": ["docstrings", "runtime", "history", "readme"],
        },
        "metrics": ["recall@5", "recall@10", "hit@3", "mrr@10", "directory_hit@3"],
        "bootstrap": {
            "unit": "repo-qualified question group",
            "samples": 10000,
            "seed": 42,
            "confidence": 0.95,
        },
        "success": {
            "pooled_test_recall5_delta_min": 0.05,
            "paired_lower_bound_strictly_above": 0,
            "each_repo_delta_min": 0,
            "code_mrr10_drop_max": 0.01,
        },
        "baseline": "chunks hybrid, same depth and weights; file collapse after chunk ranking",
        "depth": 50,
        "path_overfetch": 8,
        "empty_router": "empty ranking, no fallback",
        "summary_caps": {"base": 3000, "facts": 16000},
        "limitations": [
            "30 authored questions; human review pending",
            "pointwise exploratory intervals, no multiplicity correction",
            "identifier check measures lexical grounding, not semantic entailment",
        ],
        "E3": {
            "approval_required_before_run": True,
            "model": None,
            "proposal": "separate model_proposal.json before run; no model loading without explicit approval",
            "selection": "best fact variant on dev only",
            "check": "every identifier token must occur in facts or frozen small stop list; reject paragraph on any violation",
        },
        "default_code_search": "unchanged; summaries excluded",
    }
    write(OUT / "protocol.json", protocol)
    for key, seeds in SEEDS.items():
        slug, location, pin = REPOS[key]
        checkout = ROOT / location
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != pin:
            raise ValueError(f"Wrong pin for {slug}")
        rows = []
        for i, (group, split, kind, question, gold) in enumerate(seeds, 1):
            labels = [
                evidence(checkout, f"src/{key}/{module}.py", anchor)
                for module, anchor in gold.items()
            ]
            # Exact filenames are never query cues, even when the concept shares a stem.
            assert not any(label["path"] in question for label in labels)
            rows.append(
                {
                    "id": f"{key}-architecture-{i:02}",
                    "repo": slug,
                    "commit_sha": pin,
                    "group": f"{slug}:{group}",
                    "split": split,
                    "type": kind,
                    "question": question,
                    "gold_files": [label["path"] for label in labels],
                    "gold_directories": sorted(
                        {str(Path(label["path"]).parent.as_posix()) for label in labels}
                    ),
                    "label_sources": labels,
                    "author": "Codex (source inspection, no retrieval)",
                    "label_review": "unreviewed by a person",
                }
            )
        (OUT / f"{key}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
    (OUT / "requests_code_regression.jsonl").write_bytes(
        (ROOT / "eval/datasets/requests_qa.jsonl").read_bytes()
    )
    names = ["protocol.json", "requests.jsonl", "flask.jsonl", "requests_code_regression.jsonl"]
    write(
        OUT / "freeze.json",
        {
            "stage": "frozen before any architecture retrieval or model loading",
            "created_at": datetime.now(UTC).isoformat(),
            "sha256": {n: digest(OUT / n) for n in names},
        },
    )
    print(json.dumps(verify_frozen(), indent=2))


if __name__ == "__main__":
    main()
