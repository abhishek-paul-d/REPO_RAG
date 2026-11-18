"""Source-authored Phase 4 v2 questions. Never overwrite frozen v1 datasets."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Literal, cast

from git import Repo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval.anchors import definitions, javascript_parser  # noqa: E402
from eval.retrieval.dataset import (  # noqa: E402
    GoldRange,
    Question,
    evidence_hash,
    overlaps,
    read_questions,
    validate_checkout,
    write_json,
)

REPOS = {
    "requests": (
        "psf/requests",
        "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
        "data/repos/psf__requests",
    ),
    "express": (
        "expressjs/express",
        "0dee2c52c7c873498f8ed7d8ee4ddd0a04d3ada8",
        "data/eval_repos/expressjs__express",
    ),
    "flask": (
        "pallets/flask",
        "d086db856be187255b8ec61ef409357393020f32",
        "data/eval_repos/pallets__flask",
    ),
}

# Questions and source anchors are specified before running the new retrieval comparison.
# tuples: target type, question category, question, file, independent source anchor.
REQUESTS = [
    (
        "test",
        "nl",
        "Which test checks that a proxy request URL with leading slashes is made safe?",
        "tests/test_adapters.py",
        "test_request_url_handles_leading_path_separators",
    ),
    (
        "test",
        "nl",
        "Find the test showing response hooks run in sequence and can return None without replacing the value.",
        "tests/test_hooks.py",
        "test_hooks",
    ),
    (
        "test",
        "nl",
        "Which test verifies the initial response-hook list is empty?",
        "tests/test_hooks.py",
        "test_default_hooks",
    ),
    (
        "test",
        "nl",
        "Find the test asserting header dictionary deletion works with different key casing.",
        "tests/test_structures.py",
        "TestCaseInsensitiveDict.test_delitem",
    ),
    (
        "test",
        "nl",
        "Which test checks a case-insensitive dictionary copy is equal but is a different object?",
        "tests/test_structures.py",
        "TestCaseInsensitiveDict.test_copy",
    ),
    (
        "test",
        "nl",
        "Where is equality against a differently cased header mapping tested?",
        "tests/test_structures.py",
        "TestCaseInsensitiveDict.test_instance_equality",
    ),
    (
        "test",
        "nl",
        "Find the test for a missing LookupDict attribute raising AttributeError.",
        "tests/test_structures.py",
        "TestLookupDict.test_getattr",
    ),
    (
        "test",
        "nl",
        "Which test ensures an exception inside the environment-variable context is propagated?",
        "tests/test_utils.py",
        "test_set_environ_raises_exception",
    ),
    (
        "test",
        "nl",
        "Find the test that an extracted zipped CA path contains the expected bytes.",
        "tests/test_utils.py",
        "TestExtractZippedPaths.test_zipped_paths_extracted",
    ),
    (
        "test",
        "nl",
        "Which test checks an invalid UNC certificate path is left untouched?",
        "tests/test_utils.py",
        "TestExtractZippedPaths.test_invalid_unc_path",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Locate the definition of `guess_json_utf`.",
        "src/requests/utils.py",
        "guess_json_utf",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Find the definition of `RequestsCookieJar.get_dict`.",
        "src/requests/cookies.py",
        "RequestsCookieJar.get_dict",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Locate `Session.get_adapter` in the implementation.",
        "src/requests/sessions.py",
        "Session.get_adapter",
    ),
]

EXPRESS = [
    (
        "test",
        "nl",
        "Which test checks that HEAD falls back to the GET route?",
        "test/app.head.js",
        "it:should default to GET",
    ),
    (
        "test",
        "nl",
        "Find the test comparing response headers for HEAD and GET while ignoring Date.",
        "test/app.head.js",
        "it:should output the same headers as GET requests",
    ),
    (
        "test",
        "nl",
        "Which test proves an explicit HEAD handler takes precedence over GET?",
        "test/app.head.js",
        "it:should override",
    ),
    (
        "test",
        "nl",
        "Find the test that clearCookie ignores a supplied maxAge.",
        "test/res.clearCookie.js",
        "it:should ignore maxAge",
    ),
    (
        "test",
        "nl",
        "Which test verifies clearing a cookie ignores a user-provided expiration date?",
        "test/res.clearCookie.js",
        "it:should ignore user supplied expires param",
    ),
    (
        "test",
        "nl",
        "Find the response status test rejecting a fractional status number.",
        "test/res.status.js",
        "it:should raise error for non-integer status codes",
    ),
    (
        "test",
        "nl",
        "Which test checks reading a header without supplying its name throws?",
        "test/req.get.js",
        "it:should throw missing header name",
    ),
    (
        "example",
        "nl",
        "Where does the static-files example mount the same public directory at /static?",
        "examples/static-files/index.js",
        "lines:30:30",
    ),
    (
        "example",
        "nl",
        "Find the content-negotiation example returning users as HTML, plain text and JSON.",
        "examples/content-negotiation/index.js",
        "lines:9:27",
    ),
    (
        "example",
        "nl",
        "Where does the web-service example reject missing or invalid API keys?",
        "examples/web-service/index.js",
        "lines:32:44",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Locate the definition of `res.append`.",
        "lib/response.js",
        "res.append",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Find the implementation of `app.engine`.",
        "lib/application.js",
        "app.engine",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Locate `req.range` in the request implementation.",
        "lib/request.js",
        "req.range",
    ),
]

FLASK = [
    (
        "implementation",
        "nl",
        "How is a timedelta value constructed when configuration provides a numeric number of seconds?",
        "src/flask/app.py",
        "_make_timedelta",
    ),
    (
        "implementation",
        "nl",
        "How is the default OPTIONS response populated with allowed HTTP methods?",
        "src/flask/app.py",
        "Flask.make_default_options_response",
    ),
    (
        "implementation",
        "nl",
        "How can a before-request callback stop further preprocessing by returning a value?",
        "src/flask/app.py",
        "Flask.preprocess_request",
    ),
    (
        "implementation",
        "nl",
        "How are after-request functions applied before the session is saved?",
        "src/flask/app.py",
        "Flask.process_response",
    ),
    (
        "implementation",
        "nl",
        "Where are the before-render and template-rendered signals emitted around rendering?",
        "src/flask/templating.py",
        "_render",
    ),
    (
        "implementation",
        "nl",
        "How does loading a mapping into application configuration ignore lowercase keys?",
        "src/flask/config.py",
        "Config.from_mapping",
    ),
    (
        "implementation",
        "nl",
        "How does configuration namespace extraction optionally strip and lowercase the prefix?",
        "src/flask/config.py",
        "Config.get_namespace",
    ),
    (
        "implementation",
        "nl",
        "How are fallback secret keys added when constructing the secure-cookie signer?",
        "src/flask/sessions.py",
        "SecureCookieSessionInterface.get_signing_serializer",
    ),
    (
        "implementation",
        "nl",
        "How does the JSON provider choose compact versus pretty response serialization?",
        "src/flask/json/provider.py",
        "DefaultJSONProvider.response",
    ),
    (
        "implementation",
        "nl",
        "How are flashed messages filtered by category and optionally returned with their categories?",
        "src/flask/helpers.py",
        "get_flashed_messages",
    ),
    (
        "test",
        "nl",
        "Which test checks automatically generated OPTIONS responses advertise GET and HEAD?",
        "tests/test_basic.py",
        "test_options_work",
    ),
    (
        "test",
        "nl",
        "Find the test proving an early before-request return skips later callbacks and the view.",
        "tests/test_basic.py",
        "test_request_preprocessing_early_return",
    ),
    (
        "test",
        "nl",
        "Which test checks supplying methods to a method-specific route decorator is rejected?",
        "tests/test_basic.py",
        "test_method_route_no_methods",
    ),
    (
        "test",
        "nl",
        "Where is loading a TOML configuration file tested?",
        "tests/test_config.py",
        "test_config_from_file_toml",
    ),
    (
        "test",
        "nl",
        "Find the test for nested configuration keys supplied through prefixed environment variables.",
        "tests/test_config.py",
        "test_from_prefixed_env_nested",
    ),
    (
        "test",
        "nl",
        "Which test checks blueprint names cannot contain a dot?",
        "tests/test_blueprints.py",
        "test_dotted_name_not_allowed",
    ),
    (
        "test",
        "nl",
        "Find the test proving a Decimal value is serialized as a JSON string.",
        "tests/test_json.py",
        "test_json_decimal",
    ),
    (
        "example",
        "nl",
        "Where does the tutorial reuse a SQLite connection from g within a request?",
        "examples/tutorial/flaskr/db.py",
        "get_db",
    ),
    (
        "example",
        "nl",
        "Find the tutorial decorator redirecting anonymous users to the login page.",
        "examples/tutorial/flaskr/auth.py",
        "login_required",
    ),
    (
        "example",
        "nl",
        "Where does the tutorial register database cleanup and the init-db CLI command?",
        "examples/tutorial/flaskr/db.py",
        "init_app",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Locate `Flask.test_cli_runner`.",
        "src/flask/app.py",
        "Flask.test_cli_runner",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Find the definition of `has_request_context`.",
        "src/flask/ctx.py",
        "has_request_context",
    ),
    (
        "implementation",
        "symbol_lookup",
        "Locate `stream_template_string`.",
        "src/flask/templating.py",
        "stream_template_string",
    ),
]


def gold_range(root: Path, path: str, anchor: str) -> GoldRange:
    source = (root / path).read_text(encoding="utf-8")
    if anchor.startswith("lines:"):
        _, a, b = anchor.split(":")
        start, end = int(a), int(b)
    elif anchor.startswith("it:"):
        encoded = source.encode()
        tree = javascript_parser().parse(encoded)
        stack = [tree.root_node]
        matches = []
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                fn = node.child_by_field_name("function")
                args = node.child_by_field_name("arguments")
                if fn and fn.text == b"it" and args and args.named_children:
                    label = (args.named_children[0].text or b"").decode()[1:-1]
                    if label == anchor[3:]:
                        matches.append(node)
            stack.extend(node.named_children)
        if len(matches) != 1:
            raise ValueError(f"expected one test anchor: {path}::{anchor}, got {len(matches)}")
        start, end = matches[0].start_point[0] + 1, matches[0].end_point[0] + 1
    else:
        anchors = definitions(source, "python" if path.endswith(".py") else "javascript")
        matches = [a for a in anchors if a.symbol == anchor]
        if len(matches) != 1:
            raise ValueError(f"expected one definition: {path}::{anchor}, got {len(matches)}")
        start, end = matches[0].start, matches[0].end
    return GoldRange(
        path=path,
        start_line=start,
        end_line=end,
        source_sha256=evidence_hash(source, start, end),
        symbol=anchor,
    )


def review_legacy(rows: list[Question], root: Path) -> list[dict]:
    reviews = []
    for q in rows:
        q.schema_version = 2
        q.split = "test"  # v2 has no tuning split; cohort keeps legacy distinct from fresh.
        old = list(q.gold)
        if q.id in {"requests-001", "requests-024", "requests-034"}:
            path, symbol = old[0].path, old[0].symbol
            required, supporting = {
                "requests-001": (
                    [(202, 209), (216, 219), (240, 247), (287, 307)],
                    [(186, 201), (210, 215), (220, 239), (248, 286)],
                ),
                "requests-024": ([(681, 707)], [(634, 680), (708, 748)]),
                "requests-034": ([(216, 219)], [(186, 215), (220, 307)]),
            }[q.id]
            q.gold = [
                gold_range(root, path, f"lines:{a}:{b}").model_copy(update={"symbol": symbol})
                for a, b in required
            ]
            q.supporting_gold = [
                gold_range(root, path, f"lines:{a}:{b}").model_copy(update={"symbol": symbol})
                for a, b in supporting
            ]
            q.gold_review = "Required: requested control flow/branch/call site; supporting: remaining implementation context. Source-authored review, not independent human review."
        elif q.id == "requests-007":
            q.gold, q.supporting_gold = old[:1], old[1:]
            q.gold_review = "The question asks for chunk streaming; line iteration is supporting rather than required."
        elif q.id == "requests-018":
            q.gold, q.supporting_gold = old[:1], old[1:]
            q.gold_review = "Adapter selection requires get_adapter; mount explains registration order as supporting context."
        elif q.id == "requests-014":
            q.gold_review = "Retained both ranges: file encoding and request-body preparation together explain multipart construction and integration."
        elif q.id == "requests-020":
            q.gold_review = "Retained class range: the broad question concerns dictionary behaviour across storage, lookup and updates; minimal branch labels need independent review."
        elif q.id == "requests-032":
            q.gold_review = "Retained decoding, detection and header parsing as multi-part encoding diagnosis; distinguish optional branches during independent review."
        elif q.id in {"express-010", "express-025"}:
            q.gold_review = "Retained compact implementation range: observed miss is ranking, not excessively broad gold."
        if q.gold_review:
            reviews.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "v1_gold": [g.model_dump() for g in old],
                    "required": [g.model_dump() for g in q.gold],
                    "supporting": [g.model_dump() for g in q.supporting_gold],
                    "rationale": q.gold_review,
                    "review_status": q.review_status,
                }
            )
    return reviews


def build() -> None:
    output = ROOT / "eval/datasets/phase4_v2"
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "tuning": "none; matrix fixed before scoring",
        "review": "author_checked; independent review outstanding",
        "datasets": [],
        "gold_reviews": [],
    }
    for name, seeds in (("requests", REQUESTS), ("express", EXPRESS), ("flask", FLASK)):
        repo, sha, folder = REPOS[name]
        root = ROOT / folder
        if Repo(root).head.commit.hexsha != sha:
            raise ValueError(f"wrong pinned checkout for {repo}")
        legacy_path = ROOT / "eval/datasets" / f"{name}_qa.jsonl"
        rows = read_questions(legacy_path) if name != "flask" else []
        manifest["gold_reviews"].extend(review_legacy(rows, root))
        for i, (target, category, question, path, anchor) in enumerate(seeds, 1):
            rows.append(
                Question(
                    id=f"{name}-fresh-{i:03}",
                    repo=repo,
                    commit_sha=sha,
                    question=question,
                    category=category,
                    split="test",
                    group="pending",
                    provenance="v2 source-authored before follow-up scoring; not independently reviewed",
                    gold=[gold_range(root, path, anchor)],
                    schema_version=2,
                    cohort="fresh",
                    target_type=cast(Literal["implementation", "test", "example"], target),
                )
            )
        components: list[list[Question]] = []
        for row in rows:
            joined = [
                group
                for group in components
                if any(
                    overlaps(a, b)
                    for q in group
                    for a in [*q.gold, *q.supporting_gold]
                    for b in [*row.gold, *row.supporting_gold]
                )
            ]
            components = [group for group in components if group not in joined]
            components.append([row, *(q for group in joined for q in group)])
        for component in components:
            group = hashlib.sha256(",".join(sorted(q.id for q in component)).encode()).hexdigest()[
                :12
            ]
            for q in component:
                q.group = group
        validate_checkout(root, rows)
        output_path = output / f"{name}_qa.jsonl"
        output_path.write_text(
            "".join(q.model_dump_json() + "\n" for q in rows), encoding="utf-8", newline="\n"
        )
        read_questions(output_path)
        manifest["datasets"].append(
            {
                "repo": repo,
                "commit_sha": sha,
                "file": output_path.name,
                "questions": len(rows),
                "fresh": len(seeds),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "v1_sha256": hashlib.sha256(
                    legacy_path.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                if name != "flask"
                else None,
            }
        )
        print(repo, len(rows), "questions;", len(seeds), "fresh", flush=True)
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    build()
