"""Source-author a frozen, stratified 90-question depth comparison; never rewrite older sets."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_followup_datasets import REPOS, gold_range  # noqa: E402
from eval.retrieval.ambiguous import AmbiguousCase, read_cases, validate_cases  # noqa: E402
from eval.retrieval.dataset import Question, overlaps, read_questions, write_json  # noqa: E402
from eval.retrieval.followup import production_corpus  # noqa: E402
from eval.retrieval.metrics import coverage_scores  # noqa: E402
from repo_rag.config import Settings  # noqa: E402


# Proof anchors: source ranges or complete Python definitions (including parameter decorators).
# Every question is authored from pinned code before running the expanded retrieval comparison.
def L(path: str, start: int, end: int) -> tuple:
    return path, f"lines:{start}:{end}"


def F(path: str, symbol: str) -> tuple:
    return path, symbol


def T(path: str, symbol: str) -> tuple:
    return path, f"python-test:{symbol}"


def J(path: str, start: int, end: int) -> tuple:
    return L(path, start, end)


# stratum, question, alternative sufficient proofs, topic.
REQUESTS: list[tuple] = [
    (
        "ambiguous",
        "How are byte-string credentials encoded for HTTP basic authentication?",
        [
            [L("src/requests/auth.py", 65, 75)],
            [T("tests/test_requests.py", "test_basicauth_encodes_byte_strings")],
        ],
        "basic-auth",
    ),
    (
        "ambiguous",
        "What failure occurs when JSON is requested from a response with no content?",
        [
            [L("src/requests/models.py", 1119, 1124), L("src/requests/models.py", 1070, 1071)],
            [T("tests/test_requests.py", "test_response_json_when_content_is_None")],
        ],
        "empty-json",
    ),
    (
        "ambiguous",
        "Why does looking up a cookie by name fail when several stored cookies match it?",
        [
            [L("src/requests/cookies.py", 437, 452)],
            [
                T(
                    "tests/test_requests.py",
                    "test_cookie_duplicate_names_raises_cookie_conflict_error",
                )
            ],
        ],
        "cookie-conflicts",
    ),
    (
        "ambiguous",
        "How do extra query parameters stay ahead of a URL fragment?",
        [
            [L("src/requests/models.py", 548, 563)],
            [T("tests/test_requests.py", "test_params_are_added_before_fragment")],
        ],
        "query-fragments",
    ),
    (
        "ambiguous",
        "Why can Unicode request data have a content length larger than its character count?",
        [
            [L("src/requests/utils.py", 160, 170)],
            [T("tests/test_requests.py", "test_content_length_for_string_data_counts_bytes")],
        ],
        "byte-length",
    ),
    (
        "ambiguous",
        "How can user credentials and the fragment be removed from a URL while preserving its query?",
        [
            [F("src/requests/utils.py", "urldefragauth")],
            [T("tests/test_utils.py", "test_urldefragauth")],
        ],
        "url-sanitizing",
    ),
    (
        "ambiguous",
        "Can a cookie lifetime supplied as an integer produce an integer expiration timestamp?",
        [
            [L("src/requests/cookies.py", 534, 548)],
            [T("tests/test_requests.py", "test_max_age_valid_int")],
        ],
        "cookie-expiry",
    ),
    (
        "multi_block",
        "Trace the implementation that merges session and per-request headers and removes entries whose replacement value is null.",
        [[L("src/requests/sessions.py", 96, 105), L("src/requests/sessions.py", 547, 549)]],
        "header-merge",
    ),
    (
        "multi_block",
        "Trace the implementation that decides whether a redirected request keeps its authorization and then removes or rebuilds credentials.",
        [[L("src/requests/sessions.py", 156, 184), L("src/requests/sessions.py", 320, 332)]],
        "redirect-auth",
    ),
    (
        "multi_block",
        "Trace the implementation from uploaded file tuples to multipart bytes and the outgoing request content type.",
        [[L("src/requests/models.py", 221, 251), L("src/requests/models.py", 635, 650)]],
        "multipart-flow",
    ),
    (
        "multi_block",
        "Trace the implementation from certificate verification options to a direct or proxied connection pool.",
        [[L("src/requests/adapters.py", 97, 119), L("src/requests/adapters.py", 483, 508)]],
        "tls-pools",
    ),
    (
        "multi_block",
        "Trace the implementation from response headers to the encoding selected when bytes are decoded into text.",
        [
            [
                L("src/requests/adapters.py", 381, 385),
                L("src/requests/utils.py", 576, 591),
                L("src/requests/models.py", 1066, 1089),
            ]
        ],
        "response-decoding",
    ),
    (
        "multi_block",
        "Trace the implementation that decodes streaming byte chunks without losing a character split across chunks.",
        [[L("src/requests/models.py", 974, 977), L("src/requests/utils.py", 599, 610)]],
        "incremental-decoding",
    ),
    (
        "test",
        "Which automated test verifies that each response in a redirect sequence records the rewritten outgoing HTTP method?",
        [
            [
                T("tests/test_requests.py", "test_requests_are_updated_each_time"),
                L("tests/test_requests.py", 2605, 2638),
            ]
        ],
        "redirect-method-recording",
    ),
    (
        "test",
        "Which automated test verifies that query parameter merging distinguishes upper- and lowercase keys?",
        [[T("tests/test_requests.py", "test_params_are_merged_case_sensitive")]],
        "parameter-case",
    ),
    (
        "test",
        "Which automated test verifies that non-HTTP URL schemes bypass HTTP URL validation?",
        [[T("tests/test_requests.py", "test_nonhttp_schemes_dont_check_URLs")]],
        "non-http-schemes",
    ),
    (
        "test",
        "Which automated test checks that closing an unconsumed session response releases its connection?",
        [[T("tests/test_requests.py", "test_unconsumed_session_response_closes_connection")]],
        "response-cleanup",
    ),
    (
        "test",
        "Which automated test checks the error for a certificate-authority bundle path that does not exist?",
        [[T("tests/test_requests.py", "test_invalid_ca_certificate_path")]],
        "invalid-ca-path",
    ),
    (
        "test",
        "Which automated test demonstrates multiple query values for one parameter name?",
        [[T("tests/test_requests.py", "test_urlencoded_get_query_multivalued_param")]],
        "multi-value-query",
    ),
    (
        "implementation",
        "Locate the implementation that parses link relation parameters from an HTTP Link header.",
        [[L("src/requests/utils.py", 973, 999)]],
        "link-parsing",
    ),
    (
        "implementation",
        "Locate the implementation that returns false when a string cannot be parsed as an IPv4 address.",
        [[L("src/requests/utils.py", 756, 760)]],
        "ipv4-validation",
    ),
    (
        "implementation",
        "Locate the implementation that converts a prefix length into a dotted IPv4 netmask.",
        [[L("src/requests/utils.py", 748, 749)]],
        "netmask",
    ),
    (
        "implementation",
        "Locate the implementation that extracts a basename from a file-like object's name while rejecting angle-bracket placeholders.",
        [[L("src/requests/utils.py", 285, 287)]],
        "file-names",
    ),
    (
        "implementation",
        "Locate the implementation that removes a cookie by name with optional domain and path constraints.",
        [[L("src/requests/cookies.py", 171, 182)]],
        "cookie-removal",
    ),
    (
        "implementation",
        "Locate the implementation that initializes digest-authentication counters separately for each thread.",
        [[F("src/requests/auth.py", "HTTPDigestAuth.init_per_thread_state")]],
        "digest-thread-state",
    ),
]

EXPRESS: list[tuple] = [
    (
        "ambiguous",
        "How can JSON output escape characters that might otherwise be mistaken for HTML?",
        [
            [F("lib/response.js", "res.json"), L("lib/response.js", 1033, 1050)],
            [J("test/res.json.js", 112, 125)],
        ],
        "json-escaping",
    ),
    (
        "ambiguous",
        "How is the first acceptable response type chosen when several media types are offered?",
        [[F("lib/request.js", "req.accepts")], [J("test/req.accepts.js", 73, 84)]],
        "media-negotiation",
    ),
    (
        "ambiguous",
        "How is a successful retrieval considered unmodified when its entity tag matches?",
        [[F("lib/request.js", "req.fresh")], [J("test/req.fresh.js", 9, 22)]],
        "cache-freshness",
    ),
    (
        "ambiguous",
        "What prevents the same response variation field from being added twice?",
        [[F("lib/response.js", "res.vary")], [J("test/res.vary.js", 72, 88)]],
        "vary-deduplication",
    ),
    (
        "ambiguous",
        "How does a cookie lifetime in milliseconds become the lifetime sent in its header?",
        [[L("lib/response.js", 763, 781)], [J("test/res.cookie.js", 115, 126)]],
        "cookie-lifetime",
    ),
    (
        "ambiguous",
        "How are overlapping requested byte ranges combined when merging is enabled?",
        [[F("lib/request.js", "req.range")], [J("test/req.range.js", 88, 101)]],
        "combined-ranges",
    ),
    (
        "ambiguous",
        "How can repeated values be added to a response header while keeping its previous values?",
        [
            [F("lib/response.js", "res.append")],
            [J("test/res.append.js", 65, 83), F("test/res.append.js", "shouldHaveHeaderValues")],
        ],
        "header-append",
    ),
    (
        "multi_block",
        "Trace the implementation from a response's template rendering request to application-level rendering and invoking the selected view engine.",
        [
            [
                L("lib/response.js", 897, 921),
                L("lib/application.js", 548, 575),
                F("lib/application.js", "tryRender"),
                L("lib/view.js", 133, 159),
            ]
        ],
        "view-rendering",
    ),
    (
        "multi_block",
        "Trace the implementation that sets a download filename and passes the resolved file path into file transfer.",
        [[L("lib/response.js", 437, 486), L("lib/response.js", 375, 406)]],
        "download-transfer",
    ),
    (
        "multi_block",
        "Trace the implementation that mounts a child application, inherits parent settings and restores the parent request and response prototypes afterwards.",
        [[L("lib/application.js", 109, 122), L("lib/application.js", 218, 240)]],
        "subapp-mounting",
    ),
    (
        "multi_block",
        "Trace the implementation from a JSONP callback name to the wrapped JavaScript body and the final send operation.",
        [[L("lib/response.js", 264, 308), L("lib/response.js", 205, 222)]],
        "jsonp-send",
    ),
    (
        "multi_block",
        "Trace the implementation from a status-message response to its body preparation and omission for a HEAD request.",
        [
            [
                F("lib/response.js", "res.sendStatus"),
                L("lib/response.js", 154, 182),
                L("lib/response.js", 214, 222),
            ]
        ],
        "status-body",
    ),
    (
        "multi_block",
        "Trace the implementation that negotiates among response formats and adds the corresponding response variation header.",
        [
            [
                F("lib/response.js", "res.format"),
                F("lib/request.js", "req.accepts"),
                F("lib/response.js", "res.vary"),
            ]
        ],
        "format-selection",
    ),
    (
        "test",
        "Which automated test verifies that request parameters are restored after leaving a nested router?",
        [[J("test/app.router.js", 12, 37)]],
        "router-parameter-restoration",
    ),
    (
        "test",
        "Which automated test verifies that only error-handling callbacks run after an error is propagated through a route?",
        [[J("test/app.routes.error.js", 25, 60)]],
        "route-errors",
    ),
    (
        "test",
        "Which automated test checks that disabling query parsing leaves an empty query object?",
        [[J("test/req.query.js", 66, 72), F("test/req.query.js", "createApp")]],
        "disabled-query-parser",
    ),
    (
        "test",
        "Which automated test checks selecting the best character set from several supplied choices?",
        [[J("test/req.acceptsCharsets.js", 49, 60)]],
        "charset-negotiation",
    ),
    (
        "test",
        "Which automated test checks that Content-Type cannot be assigned an array?",
        [[J("test/res.set.js", 78, 89)]],
        "content-type-array",
    ),
    (
        "test",
        "Which automated test checks that an automatic OPTIONS response includes each method only once?",
        [[J("test/app.options.js", 20, 32)]],
        "options-deduplication",
    ),
    (
        "implementation",
        "Locate the implementation that builds an application's full mount path by walking its parent chain.",
        [[F("lib/application.js", "app.path")]],
        "mount-path",
    ),
    (
        "implementation",
        "Locate the implementation that reports whether the request protocol is secure.",
        [[F("lib/request.js", "req.secure")]],
        "secure-request",
    ),
    (
        "implementation",
        "Locate the implementation that turns named link relations into a response Link header.",
        [[F("lib/response.js", "res.links")]],
        "link-serialization",
    ),
    (
        "example",
        "In the runnable cookie-session demonstration, where is a counter persisted and incremented between visits?",
        [[L("examples/cookie-sessions/index.js", 12, 19)]],
        "example-cookie-session",
    ),
    (
        "example",
        "In the runnable route-map demonstration, where are nested paths traversed and registered as routes?",
        [[L("examples/route-map/index.js", 14, 29)]],
        "example-route-map",
    ),
    (
        "example",
        "In the runnable error-page demonstration, where are missing routes and application errors rendered with different statuses?",
        [[L("examples/error-pages/index.js", 63, 77), L("examples/error-pages/index.js", 91, 97)]],
        "example-error-pages",
    ),
]

FLASK: list[tuple] = [
    (
        "ambiguous",
        "What happens when an early request callback returns a response before the view runs?",
        [
            [L("src/flask/app.py", 1366, 1392)],
            [T("tests/test_basic.py", "test_request_preprocessing_early_return")],
        ],
        "early-response",
    ),
    (
        "ambiguous",
        "What determines whether markup in template values is automatically escaped?",
        [
            [L("src/flask/sansio/app.py", 546, 548)],
            [T("tests/test_templating.py", "test_escaping")],
        ],
        "template-escaping",
    ),
    (
        "ambiguous",
        "How can request details remain available while response content is generated later?",
        [
            [L("src/flask/helpers.py", 126, 148)],
            [T("tests/test_helpers.py", "test_streaming_with_context")],
        ],
        "stream-context",
    ),
    (
        "ambiguous",
        "Can an existing signed session remain readable after changing the application's signing key?",
        [
            [L("src/flask/sessions.py", 303, 321), L("src/flask/sessions.py", 323, 335)],
            [T("tests/test_basic.py", "test_session_secret_key_fallbacks")],
        ],
        "key-rotation",
    ),
    (
        "ambiguous",
        "How does a timestamp become an HTTP-style date in a JSON response?",
        [
            [L("src/flask/json/provider.py", 108, 110)],
            [T("tests/test_json.py", "test_jsonify_datetime")],
        ],
        "json-dates",
    ),
    (
        "ambiguous",
        "How is the same-site policy on the session cookie chosen?",
        [
            [
                L("src/flask/sessions.py", 208, 213),
                L("src/flask/sessions.py", 340, 346),
                L("src/flask/sessions.py", 374, 384),
            ],
            [T("tests/test_basic.py", "test_session_using_samesite_attribute")],
        ],
        "session-samesite",
    ),
    (
        "ambiguous",
        "Can configuration loading quietly ignore a missing environment-variable setting?",
        [[F("src/flask/config.py", "Config.from_envvar")], [L("tests/test_config.py", 144, 152)]],
        "missing-env-config",
    ),
    (
        "multi_block",
        "Trace the implementation from before-request callbacks through view dispatch to response processing and session saving.",
        [
            [
                L("src/flask/app.py", 1012, 1019),
                L("src/flask/app.py", 1039, 1051),
                L("src/flask/app.py", 1394, 1418),
            ]
        ],
        "request-lifecycle",
    ),
    (
        "multi_block",
        "Trace the implementation that runs request and application teardown and resets context bindings even if cleanup raises.",
        [
            [
                L("src/flask/ctx.py", 486, 504),
                F("src/flask/helpers.py", "_CollectErrors.__exit__"),
                F("src/flask/helpers.py", "_CollectErrors.raise_any"),
            ]
        ],
        "context-teardown",
    ),
    (
        "multi_block",
        "Trace the implementation that merges template context processors while preserving explicit values, then emits signals around rendering.",
        [[L("src/flask/app.py", 601, 616), F("src/flask/templating.py", "_render")]],
        "template-context-render",
    ),
    (
        "multi_block",
        "Trace the implementation that identifies an exception's status code and searches blueprint and application handlers through its inheritance hierarchy.",
        [
            [
                L("src/flask/sansio/scaffold.py", 686, 687),
                L("src/flask/sansio/scaffold.py", 703, 706),
                L("src/flask/sansio/app.py", 877, 892),
            ]
        ],
        "error-handler-resolution",
    ),
    (
        "multi_block",
        "Trace the implementation from an unavailable session serializer to the null session that rejects writes.",
        [
            [
                L("src/flask/sessions.py", 303, 305),
                L("src/flask/sessions.py", 323, 326),
                L("src/flask/ctx.py", 381, 393),
                L("src/flask/sessions.py", 150, 160),
                L("src/flask/sessions.py", 89, 94),
            ]
        ],
        "null-session",
    ),
    (
        "multi_block",
        "Trace the implementation that streams a template while keeping its request context alive until iteration completes.",
        [[F("src/flask/templating.py", "_stream"), L("src/flask/helpers.py", 126, 148)]],
        "template-streaming",
    ),
    (
        "test",
        "Which automated test verifies that a pre-render signal can change the template context?",
        [[T("tests/test_signals.py", "test_before_render_template")]],
        "pre-render-signal",
    ),
    (
        "test",
        "Which automated test ensures an already-handled earlier exception is not passed into a later request teardown?",
        [[T("tests/test_reqctx.py", "test_teardown_with_previous_exception")]],
        "teardown-exception",
    ),
    (
        "test",
        "Which automated test checks that JSON response construction rejects positional and keyword values together?",
        [[T("tests/test_basic.py", "test_jsonify_args_and_kwargs_check")]],
        "json-argument-conflict",
    ),
    (
        "test",
        "Which automated test verifies that package resources reject write-mode access?",
        [[T("tests/test_helpers.py", "test_open_resource_exceptions")]],
        "resource-modes",
    ),
    (
        "test",
        "Which automated test verifies that refreshing an unchanged permanent session still adds a cookie variation header?",
        [[T("tests/test_basic.py", "test_session_refresh_vary")]],
        "session-refresh",
    ),
    (
        "test",
        "Which automated test checks that two registrations cannot share the same blueprint name?",
        [[T("tests/test_blueprints.py", "test_unique_blueprint_names")]],
        "blueprint-name-collisions",
    ),
    (
        "implementation",
        "Locate the implementation that forwards URL anchor, scheme and method options to the application.",
        [[L("src/flask/helpers.py", 244, 251)]],
        "url-generation",
    ),
    (
        "implementation",
        "Locate the implementation that checks whether an application context is currently active.",
        [[L("src/flask/ctx.py", 257, 257)]],
        "app-context-guard",
    ),
    (
        "implementation",
        "Locate the implementation that finds a module's resource root using its file or loader and falls back to the working directory.",
        [[L("src/flask/helpers.py", 596, 617), L("src/flask/helpers.py", 619, 641)]],
        "resource-root",
    ),
    (
        "example",
        "In the runnable tutorial, where is the user identity saved in a session resolved before each request?",
        [[F("examples/tutorial/flaskr/auth.py", "load_logged_in_user")]],
        "tutorial-user-loading",
    ),
    (
        "example",
        "In the runnable tutorial, where is a missing post rejected and ownership checked before editing?",
        [[F("examples/tutorial/flaskr/blog.py", "get_post")]],
        "tutorial-post-ownership",
    ),
    (
        "example",
        "In the runnable tutorial sign-in flow, where is the password checked and the old session replaced with the authenticated identity?",
        [[F("examples/tutorial/flaskr/auth.py", "login")]],
        "tutorial-sign-in",
    ),
]


def resolve(root: Path, spec: tuple):
    path, anchor = spec
    if anchor.startswith("python-test:"):
        name = anchor.split(":", 1)[1]
        tree = ast.parse((root / path).read_text(encoding="utf-8"))
        found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(found) != 1:
            raise ValueError(f"ambiguous/missing test {path}:{name}")
        node = found[0]
        start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
        anchor = f"lines:{start}:{node.end_lineno}"
    return gold_range(root, path, anchor)


def assign_groups(cases: list[AmbiguousCase]) -> None:
    parent = list(range(len(cases)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, left in enumerate(cases):
        for j, right in enumerate(cases[:i]):
            if any(
                overlaps(a, b)
                for p in left.accepted_evidence
                for a in p
                for q in right.accepted_evidence
                for b in q
            ):
                parent[find(i)] = find(j)
    for i, case in enumerate(cases):
        case.question.group = f"{case.question.repo}:evidence-{find(i):02}"


def build() -> None:
    draft = "--draft" in sys.argv
    output = ROOT / ("data/comparison_draft" if draft else "eval/datasets/phase4_comparison_v3")
    if not draft and (output / "manifest.json").exists():
        raise FileExistsError("comparison labels frozen; create a new version to change them")
    output.mkdir(exist_ok=True)
    manifest: dict = {
        "version": 3,
        "authorship": "same assistant; independent label review pending",
        "selection": "stratified source-authored diagnostic sample, not random or independent holdout",
        "datasets": [],
    }
    preview = [
        "# Expanded comparison questions",
        "",
        "Source-authored before expanded retrieval scoring. Alternatives are sufficient proofs, not additive requirements across alternatives. Old evidence overlap is recorded in manifest.json; no independent review is claimed.",
        "",
    ]
    for name, seeds in (("requests", REQUESTS), ("express", EXPRESS), ("flask", FLASK)):
        repo, sha, relative = REPOS[name]
        root = ROOT / relative
        original = read_cases(ROOT / f"eval/datasets/phase4_ambiguous/{name}_qa.jsonl")
        cases: list[AmbiguousCase] = []
        for old in original:
            item = old.model_copy(deep=True)
            item.origin_case_id = old.question.id
            item.topic = f"retained-{old.question.id}"
            item.question.id = f"{name}-comparison-{len(cases) + 1:03}"
            cases.append(item)
        assert len(seeds) == 25
        for stratum, question, proofs, topic in seeds:
            accepted = [[resolve(root, spec) for spec in proof] for proof in proofs]
            q = Question(
                id=f"{name}-comparison-{len(cases) + 1:03}",
                repo=repo,
                commit_sha=sha,
                question=question,
                category=stratum,
                split="test",
                group="pending",
                provenance="v3 source-authored stratified comparison; no expanded scores inspected; independent review pending",
                gold=accepted[0],
                schema_version=2,
                cohort="legacy_regression",
                target_type=stratum if stratum in {"test", "example"} else "implementation",
            )
            cases.append(
                AmbiguousCase(
                    question=q,
                    accepted_evidence=accepted,
                    rationale="Ambiguous intent accepts labelled alternative sufficient proofs; explicit target intent and workflows require their labelled target evidence. Each proof includes all listed prerequisite blocks.",
                    related_v2_ids=[],
                    stratum=stratum,
                    topic=topic,
                )
            )
        assign_groups(cases)
        validate_cases(root, cases)
        corpus = production_corpus(root, repo, sha, Settings())
        for case in cases:
            if not any(
                coverage_scores(case.question.model_copy(update={"gold": proof}), corpus, 0.9)[1]
                == 1
                for proof in case.accepted_evidence
            ):
                raise ValueError(
                    f"no complete accepted proof in indexed corpus: {case.question.id}"
                )
        previous = read_questions(ROOT / f"eval/datasets/phase4_v2/{name}_qa.jsonl")
        overlaps_old = {}
        for c in cases:
            ids = sorted(
                q.id
                for q in previous
                if any(
                    overlaps(a, b)
                    for proof in c.accepted_evidence
                    for a in proof
                    for b in [*q.gold, *q.supporting_gold]
                )
            )
            c.related_v2_ids = ids
            overlaps_old[c.question.id] = ids
        path = output / f"{name}_qa.jsonl"
        path.write_text(
            "".join(json.dumps(c.model_dump(), ensure_ascii=False) + "\n" for c in cases),
            encoding="utf-8",
            newline="\n",
        )
        assert len(read_cases(path)) == 30
        record = {
            "repo": repo,
            "commit_sha": sha,
            "file": path.name,
            "questions": 30,
            "retained": 5,
            "added": 25,
            "strata": dict(Counter(c.stratum for c in cases)),
            "groups": len({c.question.group for c in cases}),
            "paths": len({g.path for c in cases for p in c.accepted_evidence for g in p}),
            "all_corpus_complete_proofs": len(cases),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "overlap_with_v2": overlaps_old,
        }
        manifest["datasets"].append(record)
        preview.extend([f"## {repo}", ""])
        for c in cases:
            preview.extend([f"### {c.question.id} — {c.stratum}", "", c.question.question, ""])
            for i, proof in enumerate(c.accepted_evidence, 1):
                preview.append(
                    f"- Proof {i}: "
                    + ", ".join(f"{g.path}:{g.start_line}-{g.end_line}" for g in proof)
                )
            preview.append("")
    write_json(output / "manifest.json", manifest)
    (output / "QUESTIONS.md").write_text("\n".join(preview), encoding="utf-8")
    print(
        json.dumps(
            [{k: v for k, v in d.items() if k != "overlap_with_v2"} for d in manifest["datasets"]],
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
