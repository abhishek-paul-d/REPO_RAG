"""Author reproducible source-line gold from reviewed symbol anchors at pinned revisions.

Run from the project root; never fetches or changes either checkout.
The Requests seed is read as data, without importing the embedding benchmark.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from git import Repo  # noqa: E402

from eval.retrieval.anchors import definitions  # noqa: E402
from eval.retrieval.dataset import GoldRange, Question, evidence_hash, overlaps, validate_checkout  # noqa: E402
from repo_rag.config import Settings  # noqa: E402

REVISIONS = {
    "psf/requests": "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
    "expressjs/express": "0dee2c52c7c873498f8ed7d8ee4ddd0a04d3ada8",
}

# Questions were written against these implementations, not generated from retrieved chunks.
EXPRESS = [
    (
        "How is an Express application created with request and response prototypes?",
        "express.js",
        "createApplication",
    ),
    (
        "Where are default application settings and lazy router initialization configured?",
        "application.js",
        "app.defaultConfiguration",
    ),
    (
        "How does the application dispatch an incoming request and install a final error handler?",
        "application.js",
        "app.handle",
    ),
    (
        "How does mounting a sub-application restore the parent request and response prototypes?",
        "application.js",
        "app.use",
    ),
    ("Where is a chainable route created for a given URL path?", "application.js", "app.route"),
    ("How is a template engine registered for a file extension?", "application.js", "app.engine"),
    ("Where are callbacks for named route parameters registered?", "application.js", "app.param"),
    (
        "How do application settings compile query parser and trust proxy functions?",
        "application.js",
        "app.set",
    ),
    (
        "How does a mounted application compute its complete mount path?",
        "application.js",
        "app.path",
    ),
    ("How is one route handler registered for every HTTP method?", "application.js", "app.all"),
    (
        "How are view options merged and compiled views cached during rendering?",
        "application.js",
        "app.render",
    ),
    (
        "Where is the HTTP server created and its listen callback given server errors?",
        "application.js",
        "app.listen",
    ),
    (
        "How does setting the response status reject non-integers and out-of-range codes?",
        "response.js",
        "res.status",
    ),
    (
        "How are several link relations combined into the response Link header?",
        "response.js",
        "res.links",
    ),
    (
        "How does sending a response handle HEAD requests, content length and empty status codes?",
        "response.js",
        "res.send",
    ),
    (
        "How is an object serialized as JSON and assigned the JSON content type?",
        "response.js",
        "res.json",
    ),
    (
        "How are JSONP callback names sanitized before wrapping the JSON body?",
        "response.js",
        "res.jsonp",
    ),
    (
        "Where is an HTTP status code sent with its standard textual status message?",
        "response.js",
        "res.sendStatus",
    ),
    (
        "Why does sending a relative file path fail unless a root directory is supplied?",
        "response.js",
        "res.sendFile",
    ),
    (
        "How does a file download set Content-Disposition and resolve its file path?",
        "response.js",
        "res.download",
    ),
    (
        "How is a response content type inferred from a filename extension?",
        "response.js",
        "res.type",
    ),
    (
        "How does response format negotiation choose a handler or return status 406?",
        "response.js",
        "res.format",
    ),
    (
        "How does marking a response as an attachment set its filename and content type?",
        "response.js",
        "res.attachment",
    ),
    (
        "How can a response header append values without replacing its existing values?",
        "response.js",
        "res.append",
    ),
    (
        "How does setting Content-Type add a charset and reject an array value?",
        "response.js",
        "res.header",
    ),
    (
        "How is a cookie cleared with an expiry date while ignoring maxAge?",
        "response.js",
        "res.clearCookie",
    ),
    ("How are object cookies serialized and signed cookies created?", "response.js", "res.cookie"),
    ("How is the redirect Location header URL-encoded?", "response.js", "res.location"),
    (
        "How does a redirect select HTML or plain text and handle a HEAD request?",
        "response.js",
        "res.redirect",
    ),
    (
        "How does rendering a response forward template errors to the next middleware?",
        "response.js",
        "res.render",
    ),
    (
        "How does JSON serialization optionally escape less-than, greater-than and ampersand characters?",
        "response.js",
        "stringify",
    ),
    (
        "How does reading a request header treat Referer and Referrer as aliases?",
        "request.js",
        "req.header",
    ),
    (
        "How are acceptable response media types negotiated from the request?",
        "request.js",
        "req.accepts",
    ),
    ("Where are byte ranges from the Range request header parsed?", "request.js", "req.range"),
    (
        "How does the request protocol honor X-Forwarded-Proto only for trusted proxies?",
        "request.js",
        "req.protocol",
    ),
    (
        "How is the client IP address selected using the trust proxy setting?",
        "request.js",
        "req.ip",
    ),
    (
        "How are subdomains extracted using the configured subdomain offset?",
        "request.js",
        "req.subdomains",
    ),
    (
        "How does hostname extraction remove a port while preserving an IPv6 literal?",
        "request.js",
        "req.hostname",
    ),
    (
        "How does request freshness check the method, response status, ETag and Last-Modified?",
        "request.js",
        "req.fresh",
    ),
    (
        "Why does constructing a view fail if neither an extension nor a default engine is provided?",
        "view.js",
        "View",
    ),
]


def requests_seed() -> list[dict]:
    tree = ast.parse((ROOT / "eval/embedding_bench/bench.py").read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "DEV_QUESTIONS"
    )
    if node.value is None:
        raise ValueError("DEV_QUESTIONS has no value")
    rows = ast.literal_eval(node.value)
    rows.extend(
        [
            {
                "type": "nl",
                "q": "How does closing a session release its transport adapters?",
                "gold": ["src/requests/sessions.py::Session.close"],
            },
            {
                "type": "nl",
                "q": "How are outgoing request headers validated and stored case-insensitively?",
                "gold": ["src/requests/models.py::PreparedRequest.prepare_headers"],
            },
            {
                "type": "nl",
                "q": "How does closing a response release its underlying connection?",
                "gold": ["src/requests/models.py::Response.close"],
            },
            {
                "type": "nl",
                "q": "How is a dictionary converted to a cookie jar while respecting overwrite?",
                "gold": ["src/requests/cookies.py::cookiejar_from_dict"],
            },
        ]
    )
    return rows


def build(repo: str, checkout: Path) -> list[Question]:
    settings = Settings()
    if Repo(checkout).head.commit.hexsha != REVISIONS[repo]:
        raise ValueError(f"checkout {repo} at {REVISIONS[repo]} before building gold")
    seed = (
        requests_seed()
        if repo == "psf/requests"
        else [{"type": "nl", "q": q, "gold": [f"lib/{path}::{name}"]} for q, path, name in EXPRESS]
    )
    rows = []
    for i, item in enumerate(seed, 1):
        gold = []
        for target in item["gold"]:
            path, symbol = target.split("::")
            source = (checkout / path).read_text(encoding="utf-8")
            language = "python" if path.endswith(".py") else "javascript"
            found = [a for a in definitions(source, language) if a.symbol == symbol]
            if len(found) != 1:
                raise ValueError(
                    f"expected one independent source anchor: {target}, got {len(found)}"
                )
            anchor = found[0]
            gold.append(
                GoldRange(
                    path=path,
                    start_line=anchor.start,
                    end_line=anchor.end,
                    source_sha256=evidence_hash(source, anchor.start, anchor.end),
                    symbol=symbol,
                )
            )
        rows.append(
            Question(
                id=f"{repo.split('/')[-1]}-{i:03}",
                repo=repo,
                commit_sha=REVISIONS[repo],
                question=item["q"],
                category=item["type"],
                split="test",
                group="pending",
                provenance="embedding_bench DEV_QUESTIONS + 4 new"
                if repo == "psf/requests"
                else "manually authored against pinned Express source",
                gold=gold,
            )
        )
    # Connected components of overlapping gold: paraphrases and shared implementations
    # remain together, rather than leaking the same implementation into both splits.
    components: list[list[Question]] = []
    for row in rows:
        joined = [
            group
            for group in components
            if any(overlaps(a, b) for q in group for a in q.gold for b in row.gold)
        ]
        components = [group for group in components if group not in joined]
        components.append([row, *(q for group in joined for q in group)])
    components.sort(key=lambda g: min(q.id for q in g))
    random.Random(settings.eval_split_seed).shuffle(components)
    count = max(1, int(len(components) * settings.eval_dev_fraction))
    for i, group in enumerate(components):
        group_id = hashlib.sha256(",".join(sorted(q.id for q in group)).encode()).hexdigest()[:12]
        for row in group:
            row.split = "dev" if i < count else "test"
            row.group = group_id
    validate_checkout(checkout, rows)
    return rows


if __name__ == "__main__":
    roots = {
        "psf/requests": ROOT / "data/repos/psf__requests",
        "expressjs/express": ROOT / "data/eval_repos/expressjs__express",
    }
    for repo, checkout in roots.items():
        rows = build(repo, checkout)
        output = ROOT / "eval/datasets" / f"{repo.split('/')[-1]}_qa.jsonl"
        if output.exists():
            raise FileExistsError(
                "v1 gold is frozen; build follow-up datasets with eval/build_followup_datasets.py"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(q.model_dump(), ensure_ascii=False) + "\n" for q in rows),
            encoding="utf-8",
        )
        print(
            f"{repo}: {len(rows)} questions ({sum(q.split == 'dev' for q in rows)} dev, {sum(q.split == 'test' for q in rows)} test), {len({q.group for q in rows})} evidence groups"
        )
