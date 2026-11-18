"""Fresh questions to test a prompt change on questions it was not designed around.

    uv run python eval/build_fresh_questions.py

Writes eval/datasets/agent_eval_fresh_v1.jsonl: 10 "How do I ..." usage questions (docs, scored by
whether the answer names the doc file that covers the topic) and 10 "How does ..." code questions
(code, scored by whether a cited range overlaps the gold function). Run it with
`eval/agent_eval.py run --dataset eval/datasets/agent_eval_fresh_v1.jsonl`.

Why it exists: `docs_hint_narrow` fixed docs-045/046/047, but its rule ("how do I ...") was
written knowing those questions. These questions were written afterwards, from the docs and the
source at the indexed commit, and none repeats a question or a gold range of agent_eval_v1. The
usage topics were located in the docs by reading them; the code gold ranges are the `def` ranges
computed with `ast`, and the build refuses a range that overlaps an existing code question.

Nothing here was reviewed by a person. A usage question can also be answered from code, and a
code question can be answered from docs; the scorers only check the named doc path or gold range.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / "data/repos/psf__requests"
EXISTING = ROOT / "eval/datasets/agent_eval_v1.jsonl"
OUT = ROOT / "eval/datasets/agent_eval_fresh_v1.jsonl"

QS, AD, AU = (f"docs/user/{name}.rst" for name in ("quickstart", "advanced", "authentication"))

USAGE = [
    ("How do I send custom headers with a request?", [QS]),
    ("How do I send a JSON body in a POST request?", [QS]),
    ("How do I read the cookies a server sent back?", [QS]),
    ("How do I stream a large download to a file without loading it all into memory?", [QS, AD]),
    ("How do I turn off following redirects?", [QS]),
    ("How do I use a client side certificate?", [AD]),
    ("How do I use HTTP digest authentication?", [AU, AD]),
    ("How do I run my own function on every response?", [AD]),
    ("How do I use a custom transport adapter for one website only?", [AD]),
    ("How do I inspect and change a request before it is sent?", [AD]),
]

# (question, path, class or None, function)
CODE = [
    ("How does the library turn a cookie jar into a plain dictionary?",
     "src/requests/utils.py", None, "dict_from_cookiejar"),
    ("How does the library work out which Unicode encoding a JSON body uses?",
     "src/requests/utils.py", None, "guess_json_utf"),
    ("How does the library choose which proxy to use for a request URL?",
     "src/requests/utils.py", None, "select_proxy"),
    ("How does the library add a missing scheme to a URL?",
     "src/requests/utils.py", None, "prepend_scheme_if_needed"),
    ("How does the library extract a username and password from a URL?",
     "src/requests/utils.py", None, "get_auth_from_url"),
    ("How does the library recompute proxy settings when a redirect is followed?",
     "src/requests/sessions.py", "SessionRedirectMixin", "rebuild_proxies"),
    ("How does the transport adapter build the headers it sends to a proxy?",
     "src/requests/adapters.py", "HTTPAdapter", "proxy_headers"),
    ("How does the transport adapter decide between sending the full URL or only the path?",
     "src/requests/adapters.py", "HTTPAdapter", "request_url"),
    ("How does a response decide whether it is a redirect?",
     "src/requests/models.py", "Response", "is_redirect"),
    ("How does the library decode a streamed response into text chunks?",
     "src/requests/utils.py", None, "stream_decode_response_unicode"),
]  # fmt: skip


def function_range(path: str, cls: str | None, name: str) -> tuple[int, int]:
    tree = ast.parse((CHECKOUT / path).read_text(encoding="utf-8"))
    scope: list[ast.stmt] = tree.body
    if cls:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, node.end_lineno or node.lineno


def main() -> None:
    existing_questions: set[str] = set()
    existing_gold: list[tuple[str, int, int]] = []
    for line in EXISTING.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        existing_questions.add(row["question"].lower())
        for gold in row["expect"].get("gold", []):
            existing_gold.append((gold["path"], gold["start_line"], gold["end_line"]))

    records: list[dict] = []
    for n, (question, paths) in enumerate(USAGE, 1):
        assert question.lower() not in existing_questions, question
        for path in paths:
            assert (CHECKOUT / path).exists(), path
        records.append({"id": f"docs-f{n:02d}", "type": "docs", "question": question,
                        "expect": {"paths": paths}})  # fmt: skip
    for n, (question, path, cls, name) in enumerate(CODE, 1):
        assert question.lower() not in existing_questions, question
        start, end = function_range(path, cls, name)
        clash = [g for g in existing_gold if g[0] == path and not (end < g[1] or start > g[2])]
        assert not clash, f"{name} overlaps an existing gold range {clash}"
        records.append({"id": f"code-f{n:02d}", "type": "code", "question": question,
                        "expect": {"gold": [{"path": path, "start_line": start,
                                             "end_line": end, "symbol": name}]}})  # fmt: skip
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT.relative_to(ROOT)}: {len(records)} questions")


if __name__ == "__main__":
    sys.exit(main())
