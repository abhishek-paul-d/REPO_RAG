"""Question text and labels for the Phase 7 bake-off v5, from the seeded draw in v5 targets.

Targets come from `eval/build_v5_targets.py` (seed fixed there). Each question below was written
from source inspection of the drawn chunk(s) without running retrieval or a model. Targets that
cannot make a defensible question are listed in DROPPED with the reason, in draw order; none was
replaced by hand-picking. Labels (gold chunks/files, probe audit) are derived from the capsule's
indexed code, and every probe term is checked to be absent from the indexed `src/` text.

    uv run python eval/build_v5_questions.py
"""

from __future__ import annotations

# ruff: noqa: E402
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.build_v5_targets import CAPSULE, REPOS, load_chunks
from eval.build_v5_targets import OUT as TARGETS

AUTHORED = ROOT / "eval/datasets/phase7_bakeoff_v5_authored.jsonl"
AUTHOR = "Claude (source inspection of the drawn chunk, no retrieval or model generation)"
REVIEW = "unreviewed by a person"
# Dev rows per (repo, suite); the remaining usable rows are test. Probes: 3 dev + 8 test per repo.
DEV_COUNTS = {
    ("psf/requests", "code"): 4,
    ("pallets/flask", "code"): 4,
    ("psf/requests", "architecture"): 3,
    ("pallets/flask", "architecture"): 4,
    ("psf/requests", "probes"): 3,
    ("pallets/flask", "probes"): 3,
}

# (target index in the draw, question). Order is the draw order restricted to usable targets.
CODE = {
    "psf/requests": [
        (1, "How is a cookie morsel converted into a Cookie, including how its expiry is derived?"),
        (
            2,
            "In what order does preparing a request apply the method, URL, headers, cookies, body, "
            "auth and hooks?",
        ),
        (
            3,
            "How are the headers for a request sent through a proxy built, including proxy "
            "credentials?",
        ),
        (4, "How does the no_proxy handling decide that a string is a valid CIDR network?"),
        (5, "How is a header value made of key=value pairs parsed into a dictionary?"),
        (8, "How does setting a cookie by name behave when the value is None or a Morsel?"),
        (
            11,
            "How does the cookie jar look up a single value by name, and what happens when "
            "several cookies match?",
        ),
        (12, "How is a Response object built from the low-level transport response?"),
        (13, "How are comma-separated header lists with quoted strings parsed?"),
        (14, "How is the Cookie header generated from the cookies passed to a request?"),
        (
            16,
            "How does reading the content of a response work, and what happens if the content "
            "was already consumed?",
        ),
        (
            17,
            "What diagnostic details does the bug-report helper collect about the platform and "
            "dependencies?",
        ),
        (18, "How are proxy settings re-evaluated when a redirect is followed?"),
        (19, "How is the path portion of the request line built, including the query string?"),
        (20, "How does the help module determine the Python implementation name and version?"),
        (
            22,
            "How is the proxy manager for a given proxy created and reused, including SOCKS "
            "proxies?",
        ),
    ],
    "pallets/flask": [
        (
            0,
            "How does flask.redirect choose which response class to build, and what status code "
            "does it use by default?",
        ),
        (
            2,
            "How can a view function be attached to an endpoint name when the URL rule was added "
            "without a view function?",
        ),
        (
            3,
            "How is the application config object created, and when is it rooted at the "
            "instance path?",
        ),
        (
            4,
            "How does the application-context globals object return an attribute or set a "
            "default for a missing one?",
        ),
        (5, "In what order are template loaders from the application and its blueprints searched?"),
        (
            6,
            "How can a function be registered to run after only the current request, and where "
            "may it be used?",
        ),
        (
            7,
            "How does Flask dispatch a request with before and after request processing and HTTP "
            "error handling?",
        ),
        (
            8,
            "How is the maximum request body size determined for a request, and what happens "
            "when it is exceeded?",
        ),
        (
            9,
            "How does Flask treat routing redirects that would drop the request body when debug "
            "mode is on?",
        ),
        (
            13,
            "How does the command line handle the environment file option when python-dotenv "
            "is not installed?",
        ),
        (
            14,
            "How does the default JSON provider convert dates, decimals, UUIDs, dataclasses and "
            "objects with an HTML representation?",
        ),
        (15, "Which template signals are sent when a template is rendered to a string?"),
        (
            17,
            "How does the config load values from a file with a caller-supplied loader "
            "function, and what does the silent option do?",
        ),
        (
            18,
            "How does flask.json.dump decide between the application's JSON provider and the "
            "standard library?",
        ),
        (19, "How does the run command validate the key option when the certificate is a file?"),
        (
            20,
            "How is the Flask application logger configured, including its level in debug mode "
            "and its default handler?",
        ),
    ],
}
ARCHITECTURE = {
    "psf/requests": [
        (
            0,
            "How do proxy credentials in a proxy URL end up as a Proxy-Authorization header on "
            "requests sent through that proxy?",
        ),
        (
            1,
            "How does a top-level call such as requests.get hand its work to a session, and how "
            "long does that session live?",
        ),
        (
            2,
            "Which errors do the request preparation steps raise for a malformed URL, a missing "
            "scheme or invalid JSON, and where are those errors defined?",
        ),
        (
            5,
            "How do cookies from a server response get from the low-level response into the "
            "Response cookie jar?",
        ),
        (
            8,
            "Which helper functions validate headers, unquote URLs or rewind request bodies, and "
            "which exceptions do they raise?",
        ),
        (
            9,
            "How are plain dictionaries of cookies converted into cookie jars and merged into an "
            "existing jar?",
        ),
        (
            10,
            "Why and how are response headers wrapped in a case-insensitive mapping when a "
            "Response is built from the transport response?",
        ),
        (11, "How does digest authentication parse the server challenge header into its parts?"),
        (
            12,
            "How does digest authentication keep cookies from the 401 challenge when it resends "
            "the request?",
        ),
        (13, "Where do sessions raise errors for an unsupported URL scheme or too many redirects?"),
        (18, "How are request and response headers stored so that name lookups ignore case?"),
    ],
    "pallets/flask": [
        (
            0,
            "How does the template loader find templates in blueprints as well as in the "
            "application?",
        ),
        (
            7,
            "How does popping an application context collect and re-raise errors from teardown "
            "functions?",
        ),
        (
            8,
            "How does the CLI test runner create the script information needed to invoke the "
            "application's commands?",
        ),
        (
            9,
            "How do the CLI app loading and the run command use the debug flag and dotenv "
            "loading helpers?",
        ),
        (
            10,
            "How are session values serialized into the signed cookie, including values such as "
            "tuples and dates?",
        ),
        (14, "How do class-based views call sync or async handlers through the application?"),
        (15, "How does a scaffold (application or blueprint) work out its root path?"),
        (
            16,
            "How does the request wrapper make form-encoding mistakes on file uploads easier to "
            "diagnose?",
        ),
        (18, "How do blueprints expose their own command-line command group?"),
        (
            19,
            "How does the request object work out the blueprints that apply to the current "
            "request?",
        ),
        (
            20,
            "How does the application build a test request context and a test client environment?",
        ),
        (
            21,
            "How are view return values turned into JSON responses through the application's "
            "JSON provider?",
        ),
        (23, "How does a blueprint serve its static files?"),
        (27, "How does the test client let tests read and modify the session between requests?"),
        (28, "How is the template context built and updated before a template renders?"),
        (
            35,
            "How does streaming a template keep the request context available while the "
            "response is generated?",
        ),
        (
            37,
            "How does the application explain which template loaders were tried when a "
            "template is not found?",
        ),
    ],
}
# (probe anchor index, question, terms that must not occur in the indexed src/ text)
PROBES = {
    "psf/requests": [
        (
            0,
            "Where does this Requests implementation define an exception for HTTP/2 stream resets?",
            ["http2", "goaway", "rst_stream"],
        ),
        (
            5,
            "Where does this Requests implementation run response hooks asynchronously on an "
            "event loop?",
            ["asyncio", "await ", "async def"],
        ),
        (
            7,
            "Where does this Requests implementation pin server certificate fingerprints for "
            "individual hosts?",
            ["fingerprint", "pinning", "pinned_cert"],
        ),
        (
            11,
            "Where does this Requests implementation upload the collected diagnostic "
            "information to a remote service?",
            ["telemetry", "upload_report", "sentry"],
        ),
        (
            4,
            "Where does this Requests implementation translate status code descriptions into "
            "other languages?",
            ["gettext", "locale", "translat"],
        ),
        (2, "Where does this Requests implementation switch to the uvloop event loop?", ["uvloop"]),
        (
            9,
            "Where does this Requests implementation encrypt cookie values before storing them?",
            ["fernet", "encrypt", "cipher"],
        ),
        (
            8,
            "Where does this Requests implementation refresh expired OAuth 2.0 access tokens?",
            ["refresh_token", "oauth2", "access_token"],
        ),
        (
            15,
            "Where does this Requests implementation check whether a newer release of the "
            "package is available?",
            ["pypi.org", "latest_version", "update_available"],
        ),
        (
            12,
            "Where does this Requests implementation provide a GraphQL query helper?",
            ["graphql"],
        ),
        (
            3,
            "Where does this Requests implementation compress request bodies with brotli "
            "before sending?",
            ["brotli", "zstd", "compress("],
        ),
    ],
    "pallets/flask": [
        (
            0,
            "Where does this Flask implementation load configuration values from a remote "
            "key-value store such as etcd?",
            ["etcd", "consul", "zookeeper"],
        ),
        (
            3,
            "Where does this Flask implementation deliver signals through a message broker to "
            "other processes?",
            ["rabbitmq", "kafka", "amqp"],
        ),
        (
            5,
            "Where does this Flask implementation define GraphQL resolvers for class-based views?",
            ["graphql", "resolver"],
        ),
        (
            15,
            "Where does this Flask implementation store sessions in a Redis backend?",
            ["redis", "memcache"],
        ),
        (
            14,
            "Where does this Flask implementation provide a built-in database migration command?",
            ["alembic", "migrate", "migration"],
        ),
        (
            12,
            "Where does this Flask implementation manage per-request database connection pooling?",
            ["connection pool", "create_engine", "pool_size"],
        ),
        (
            13,
            "Where does this Flask implementation serialize responses to MessagePack?",
            ["msgpack", "messagepack"],
        ),
        (
            4,
            "Where does this Flask implementation enforce per-user request rate limits?",
            ["rate limit", "ratelimit", "throttl"],
        ),
        (
            10,
            "Where does this Flask implementation ship application logs to a remote syslog or "
            "log aggregation service?",
            ["syslog", "logstash", "sockethandler"],
        ),
        (
            7,
            "Where does this Flask implementation support feature flags for individual blueprints?",
            ["feature flag", "feature_flag", "toggle"],
        ),
        (
            8,
            "Where does this Flask implementation record and replay HTTP traffic for tests?",
            ["vcr", "cassette", "replay"],
        ),
    ],
}
DROPPED = {
    "psf/requests": {
        "code": {
            0: "post: thin wrapper over request(), no behaviour to ask about",
            6: "RequestsCookieJar.get: chunk is the class header, symbol does not match content",
            7: "Response.iter_content.generate: topic overlaps the v4 streaming architecture "
            "question",
            9: "patch: thin wrapper over request()",
            10: "_init: fills the status code table, no defensible question",
            15: "RequestsCookieJar.items: one-line delegation",
            21: "get: thin wrapper over request()",
            23: "SessionRedirectMixin.send: type stub, no body",
        },
        "architecture": {
            3: "to_native_string edge: internal helper, name-matching artefact",
            4: "to_native_string edge: internal helper, name-matching artefact",
            6: "update/values resolved to cookies.py by name only",
            7: "to_native_string edge: internal helper",
            14: "read resolved to _types.py by name only",
            15: "to_native_string edge: internal helper",
            16: "read/has_read resolved to _types.py by name only",
            17: "read resolved to _types.py by name only",
            19: "module-level LookupDict instance, not a behaviour",
        },
    },
    "pallets/flask": {
        "code": {
            1: "get_debug_flag: chunk is a module header, symbol does not match content",
            10: "_stream: near-duplicate of _render (kept)",
            11: "setupmethod: chunk is a module header, symbol does not match content",
            12: "dumps: chunk is a module header, symbol does not match content",
            16: "get_signing_serializer: chunk is the class header, symbol does not match content",
            21: "_get_werkzeug_version: trivial cached lookup",
        },
        "architecture": {
            1: "pop/setdefault resolved to ctx.py by name only",
            2: "__main__ module entry point, trivial",
            3: "pop/setdefault resolved to ctx.py by name only",
            4: "pop resolved to ctx.py by name only",
            5: "setdefault resolved to ctx.py by name only",
            6: "setdefault resolved to ctx.py by name only",
            11: "_get_current_object proxy lookups, name-matching artefact",
            12: "_get_current_object proxy lookup, name-matching artefact",
            13: "same chunk as the code question on raise_routing_exception",
            17: "same chunk as the code question on create_logger",
            22: "extend resolved to blueprints.py by name only",
            24: "setdefault resolved to ctx.py by name only",
            25: "open resolved to testing.py by name only",
            26: "open/invoke resolved to testing.py by name only",
            29: "test_request_context edge duplicates the testing architecture question",
            30: "pop resolved to ctx.py by name only",
            31: "near-duplicate of the JSON provider question (kept 21)",
            32: "open resolved to testing.py by name only",
            33: "extend resolved to blueprints.py by name only",
            34: "extend resolved to blueprints.py by name only",
            36: "copy/from_environ resolved to ctx.py by name only",
            38: "near-duplicate of the template loader question (kept 0)",
        },
    },
}


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def group_id(repo: str, *parts: str) -> str:
    return hashlib.sha1(":".join([repo, *parts]).encode()).hexdigest()[:12]


T = TypeVar("T")


def assign(repo: str, suite: str, items: list[T]) -> list[tuple[str, T]]:
    """First DEV_COUNTS rows of the usable draw are dev, the rest test (draw order, not mine)."""
    dev = DEV_COUNTS[(repo, suite)]
    return [("dev" if i < dev else "test", item) for i, item in enumerate(items)]


def build() -> list[dict]:
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))
    rows: list[dict] = []
    with zipfile.ZipFile(CAPSULE) as archive:
        for repo, short in REPOS.items():
            chunks = load_chunks(archive, short)
            by_id = {c["chunk_id"]: c for c in chunks}
            src_text = "\n".join(c["text"].lower() for c in chunks if c["path"].startswith("src/"))
            commit = chunks[0]["commit_sha"]
            drawn = targets["repos"][repo]

            for counter, (split, (index, question)) in enumerate(
                assign(repo, "code", CODE[repo]), 1
            ):
                target = drawn["code"][index]
                chunk = by_id[target["chunk_id"]]
                row_id = f"v5-{short}-code-{counter:02d}"
                gold = {
                    "chunk_id": chunk["chunk_id"],
                    "end_line": chunk["end_line"],
                    "path": chunk["path"],
                    "start_line": chunk["start_line"],
                    "symbol": chunk["symbol"],
                    "text_sha256": digest_text(chunk["text"]),
                }
                rows.append(
                    {
                        "id": row_id,
                        "label": {
                            "author": AUTHOR,
                            "category": "nl",
                            "commit_sha": commit,
                            "gold": [gold],
                            "group": group_id(repo, chunk["path"], chunk["symbol"]),
                            "id": row_id,
                            "label_review": REVIEW,
                            "provenance": f"seeded draw {targets['seed']}, code target {index}",
                            "question": question,
                            "repo": repo,
                            "split": split,
                        },
                        "question": question,
                        "repo": repo,
                        "source": "eval/datasets/phase7_bakeoff_v5_targets.json",
                        "source_line": index,
                        "split": split,
                        "suite": "code",
                    }
                )

            for counter, (split, (index, question)) in enumerate(
                assign(repo, "architecture", ARCHITECTURE[repo]), 1
            ):
                target = drawn["architecture"][index]
                files = sorted({target["caller_path"], target["callee_path"]})
                row_id = f"v5-{short}-architecture-{counter:02d}"
                sources = []
                for path in files:
                    anchors = [c for c in chunks if c["path"] == path]
                    wanted = {call[0] for call in target["calls"]} | {
                        call[1] for call in target["calls"]
                    }
                    hit = next(
                        (
                            c
                            for c in anchors
                            if c["symbol"] in wanted or c["symbol"].split(".")[-1] in wanted
                        ),
                        None,
                    )
                    sources.append(
                        {"path": path, "anchor": hit["symbol"] if hit else "(file)"}
                        | (
                            {"start_line": hit["start_line"], "end_line": hit["end_line"]}
                            if hit
                            else {}
                        )
                    )
                rows.append(
                    {
                        "id": row_id,
                        "label": {
                            "author": AUTHOR,
                            "calls": target["calls"],
                            "commit_sha": commit,
                            "gold_directories": sorted(
                                {str(Path(p).parent.as_posix()) for p in files}
                            ),
                            "gold_files": files,
                            "group": group_id(repo, *files),
                            "id": row_id,
                            "label_review": REVIEW,
                            "label_sources": sources,
                            "provenance": f"seeded draw {targets['seed']}, architecture target {index}",
                            "question": question,
                            "repo": repo,
                            "split": split,
                            "type": "cross-file call",
                        },
                        "question": question,
                        "repo": repo,
                        "source": "eval/datasets/phase7_bakeoff_v5_targets.json",
                        "source_line": index,
                        "split": split,
                        "suite": "architecture",
                    }
                )

            for counter, (split, (index, question, terms)) in enumerate(
                assign(repo, "probes", PROBES[repo]), 1
            ):
                target = drawn["probes"][index]
                hits = [t for t in terms if t.lower() in src_text]
                if hits:
                    raise SystemExit(f"{repo} probe {question!r}: terms present in src: {hits}")
                row_id = f"v5-{short}-probe-{counter:02d}"
                rows.append(
                    {
                        "id": row_id,
                        "label": {
                            "author": AUTHOR,
                            "boundary_anchor": {
                                "chunk_id": target["chunk_id"],
                                "end_line": target["end_line"],
                                "path": target["path"],
                                "start_line": target["start_line"],
                                "symbol": target["symbol"],
                            },
                            "commit_sha": commit,
                            "expected": "abstain",
                            "id": row_id,
                            "label_review": REVIEW,
                            "provenance": f"seeded draw {targets['seed']}, probe anchor {index}",
                            "question": question,
                            "repo": repo,
                            "scope": (
                                "Built-in implementation in the pinned Python package only; "
                                "excludes dependencies, extensions and user applications."
                            ),
                            "source_audit": {
                                "indexed_src_chunks": sum(
                                    c["path"].startswith("src/") for c in chunks
                                ),
                                "terms": terms,
                                "hits": hits,
                            },
                            "split": split,
                        },
                        "question": question,
                        "repo": repo,
                        "source": "eval/datasets/phase7_bakeoff_v5_targets.json",
                        "source_line": index,
                        "split": split,
                        "suite": "probes",
                    }
                )
    return rows


def main() -> None:
    rows = build()
    AUTHORED.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    from collections import Counter

    print(Counter(f"{r['suite']}/{r['split']}" for r in rows))
    print(f"written {AUTHORED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
