"""Build eval/datasets/agent_eval_fresh_v2.jsonl: questions the system has not been tuned on.

Why: every number so far comes from questions the system was developed against (agent_eval_v1,
agent_eval_fresh_v1, requests_qa, the architecture sets). This set is new, written after those
were frozen, and checked against them:
  * no question text, and no gold code range, appears in an earlier set (the builder fails if so),
  * code gold ranges are whole definitions taken from the symbol table at the pinned commit,
  * docs gold files are checked to contain a keyword of the topic,
  * exact-fact answers are computed from git and metadata.db, with a different seed than agent_eval_v1.

Same record format as agent_eval_v1, so `eval/agent_eval.py run --dataset ...` scores it as it is.
There is no `commit` type: all 141 bug -> fix pairs of the repository were used for tuning or
selection, so no unseen gold exists for it.

    uv run python eval/build_fresh_v2.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPO = "psf/requests"
SHA = "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60"
CHECKOUT = ROOT / "data/repos/psf__requests"
OUT = ROOT / "eval/datasets/agent_eval_fresh_v2.jsonl"
EARLIER = [
    ROOT / "eval/datasets/agent_eval_v1.jsonl",
    ROOT / "eval/datasets/agent_eval_fresh_v1.jsonl",
    ROOT / "eval/datasets/requests_qa.jsonl",
]
SEED = 20261119

API = "src/requests/api.py"
SESSIONS = "src/requests/sessions.py"
ADAPTERS = "src/requests/adapters.py"
MODELS = "src/requests/models.py"
COOKIES = "src/requests/cookies.py"
UTILS = "src/requests/utils.py"

# (question, path, [qualified names]); every definition with that name in the file is gold.
CODE: list[tuple[str, str, list[str]]] = [
    (
        "When I call requests.get(url), which function does the real work, and what does it do with "
        "the Session it creates?",
        API,
        ["request", "get"],
    ),
    ("What default settings does a brand new Session start with?", SESSIONS, ["Session.__init__"]),
    (
        "After the adapter returns a response, what does Session.send do with it before handing it "
        "back?",
        SESSIONS,
        ["Session.send"],
    ),
    (
        "How does the redirect mixin find out where a redirect points to?",
        SESSIONS,
        ["SessionRedirectMixin.get_redirect_target"],
    ),
    ("How are request-level and session-level hook lists combined?", SESSIONS, ["merge_hooks"]),
    (
        "How does the transport adapter turn a urllib3 response into a requests Response object?",
        ADAPTERS,
        ["HTTPAdapter.build_response"],
    ),
    (
        "How does the transport adapter get a connection, and when does it go through a proxy "
        "manager?",
        ADAPTERS,
        ["HTTPAdapter.get_connection"],
    ),
    (
        "How are proxy connection managers created and reused, and how is a SOCKS proxy handled?",
        ADAPTERS,
        ["HTTPAdapter.proxy_manager_for"],
    ),
    (
        "Which pool settings does the HTTP adapter pass to urllib3's pool manager?",
        ADAPTERS,
        ["HTTPAdapter.init_poolmanager"],
    ),
    (
        "How does a Response decide whether it counts as true or false, and what does ok check?",
        MODELS,
        ["Response.__bool__", "Response.ok"],
    ),
    (
        "Where can a subclass of the transport adapter add extra headers to every request, and what "
        "does the default do?",
        ADAPTERS,
        ["HTTPAdapter.add_headers"],
    ),
    (
        "What decides that a response is a permanent redirect?",
        MODELS,
        ["Response.is_permanent_redirect"],
    ),
    (
        "How does a Request become a PreparedRequest?",
        MODELS,
        ["Request.prepare", "PreparedRequest.prepare"],
    ),
    (
        "How are cookies attached to an outgoing request?",
        MODELS,
        ["PreparedRequest.prepare_cookies"],
    ),
    ("How is a prepared request copied?", MODELS, ["PreparedRequest.copy"]),
    (
        "What does preparing authentication do when no auth argument is given but the URL itself "
        "carries credentials?",
        MODELS,
        ["PreparedRequest.prepare_auth"],
    ),
    (
        "How does setting a cookie to the value None behave in the cookie jar?",
        COOKIES,
        ["RequestsCookieJar.set"],
    ),
    (
        "What happens when two cookies with the same name exist for different domains and I read "
        "one by name?",
        COOKIES,
        ["RequestsCookieJar._find_no_duplicates"],
    ),
    (
        "How does the library build a Cookie object from a name and a value, and which arguments "
        "does it reject?",
        COOKIES,
        ["create_cookie"],
    ),
    ("How are two cookie sources merged into one jar?", COOKIES, ["merge_cookies"]),
    ("How is a cookie removed by name?", COOKIES, ["remove_cookie_by_name"]),
    (
        "How is a cookie Morsel from the standard library turned into a requests cookie?",
        COOKIES,
        ["morsel_to_cookie"],
    ),
    (
        "What does requests check about library versions when it is imported?",
        "src/requests/__init__.py",
        ["check_compatibility"],
    ),
    (
        "How does the library decide whether to use chardet or charset_normalizer?",
        "src/requests/compat.py",
        ["_resolve_char_detection"],
    ),
    (
        "What does the library collect for a bug report, such as platform and library versions?",
        "src/requests/help.py",
        ["info"],
    ),
    (
        "What does a RequestException keep, and how does it find its request?",
        "src/requests/exceptions.py",
        ["RequestException.__init__"],
    ),
    (
        "How does the library parse a header value such as Digest realm and qop into a dictionary?",
        UTILS,
        ["parse_dict_header", "unquote_header_value", "parse_list_header"],
    ),
    (
        "How does the library check whether an IP address is inside a CIDR range?",
        UTILS,
        ["address_in_network", "dotted_netmask", "is_valid_cidr"],
    ),
    (
        "How does the library decode a response body when the encoding is unknown or wrong?",
        UTILS,
        ["get_unicode_from_response"],
    ),
    (
        "How are names like requests.codes.ok created, and what happens for an unknown name?",
        "src/requests/status_codes.py",
        ["_init", "src/requests/structures.py|LookupDict.__getattr__"],
    ),
]

# (question, [(path, keyword the file must contain)])
DOCS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "How do I upload a file with a custom filename and content type?",
        [
            ("docs/user/quickstart.rst", "Multipart-Encoded File"),
            ("docs/user/advanced.rst", "files"),
        ],
    ),
    (
        "How do I upload several files in one request?",
        [("docs/user/advanced.rst", "POST Multiple Multipart-Encoded Files")],
    ),
    (
        "How do I send a request body that is generated piece by piece?",
        [("docs/user/advanced.rst", "Chunk-Encoded Requests")],
    ),
    (
        "How do I upload a large file without loading it all into memory?",
        [("docs/user/advanced.rst", "Streaming Uploads")],
    ),
    (
        "How do I delay downloading the response body until I want it?",
        [("docs/user/advanced.rst", "Body Content Workflow")],
    ),
    (
        "Does requests keep connections alive, and how do I turn that off?",
        [("docs/user/advanced.rst", "Keep-Alive")],
    ),
    (
        "How do I read the Link header of a response as links?",
        [("docs/user/advanced.rst", "Link Headers")],
    ),
    (
        "How do I save binary response content such as an image?",
        [("docs/user/quickstart.rst", "Binary Response Content")],
    ),
    (
        "How do I check a response's status code by name instead of by number?",
        [("docs/user/quickstart.rst", "Response Status Codes")],
    ),
    (
        "How do I write my own authentication scheme?",
        [("docs/user/advanced.rst", "Custom Authentication")],
    ),
]

PROBES = [
    ("How does requests encrypt files at rest with AES?", "aes"),
    ("Where does requests implement a Redis caching layer?", "redis"),
    ("How do I use requests with gRPC streaming?", "grpc"),
    ("Which function in requests trains a machine-learning model?", "machine-learning"),
    ("How does requests parse YAML configuration files?", "yaml"),
]


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(CHECKOUT), *args],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    )
    return done.stdout


def source(path: str) -> list[str]:
    return git("show", f"{SHA}:{path}").splitlines()


def gold_range(connection: sqlite3.Connection, path: str, qualname: str) -> list[dict]:
    rows = connection.execute(
        "SELECT start_line, end_line FROM code_symbols WHERE repo=? AND path=? AND qualname=? "
        "ORDER BY start_line",
        (REPO, path, qualname),
    ).fetchall()
    if not rows:
        raise SystemExit(f"no symbol {qualname} in {path}")
    lines = source(path)
    out = []
    for start, end in rows:
        text = "\n".join(lines[start - 1 : end]) + "\n"
        out.append(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "symbol": qualname,
            }
        )
    return out


def earlier_questions() -> tuple[set[str], list[tuple[str, int, int]], set[str]]:
    texts: set[str] = set()
    gold: list[tuple[str, int, int]] = []
    numbers: set[str] = set()
    for file in EARLIER:
        for line in file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            texts.add(re.sub(r"\W+", " ", row["question"]).strip().lower())
            numbers |= set(re.findall(r"#(\d+)", row["question"]))
            for g in (row.get("expect") or {}).get("gold", []) + row.get("gold", []):
                gold.append((g["path"], g["start_line"], g["end_line"]))
    return texts, gold, numbers


def first_tag_containing(sha: str) -> str | None:
    tags = git("tag", "--contains", sha, "--sort=creatordate").split()
    tags = [t for t in tags if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    return tags[0] if tags else None


def build() -> list[dict]:
    from repo_rag.config import Settings
    from repo_rag.history.metadata_db import MetadataDB

    texts, used_gold, used_numbers = earlier_questions()
    records: list[dict] = []

    def add(qtype: str, question: str, **expect: object) -> None:
        key = re.sub(r"\W+", " ", question).strip().lower()
        if key in texts:
            raise SystemExit(f"question already in an earlier set: {question}")
        texts.add(key)
        records.append(
            {
                "id": f"fresh2-{qtype}-{len(records) + 1:03d}",
                "type": qtype,
                "question": question,
                "expect": expect,
            }
        )

    connection = sqlite3.connect(f"file:{Settings().metadata_db_path}?mode=ro", uri=True)
    for question, path, names in CODE:
        gold = []
        for name in names:
            where, _, qualname = name.rpartition("|")  # "other/file.py|Qualified.name"
            gold += gold_range(connection, where or path, qualname)
        for g in gold:
            for p, a, b in used_gold:
                if p == g["path"] and g["start_line"] <= b and g["end_line"] >= a:
                    raise SystemExit(f"gold {g['symbol']} overlaps an earlier gold range")
        add("code", question, gold=gold)

    for question, files in DOCS:
        for path, keyword in files:
            if keyword.lower() not in "\n".join(source(path)).lower():
                raise SystemExit(f"{path} does not mention {keyword!r}")
        add("docs", question, paths=[p for p, _ in files])

    rng = random.Random(SEED)
    with MetadataDB(Settings().metadata_db_path) as db:
        prs = [p for p in db.find_prs(REPO, state="MERGED", limit=1_000_000) if p.author]
        prs = [p for p in prs if str(p.number) not in used_numbers]
        prs.sort(key=lambda p: p.number)
        picked = rng.sample(prs, 40)
        for pr in picked[:3]:
            add("history", f"Who merged PR #{pr.number} in {REPO}?", all=[pr.merged_by])
        for pr in picked[3:5]:
            add("history", f"Who authored PR #{pr.number} in {REPO}?", all=[pr.author])
        closing = [p for p in prs if p.closes_issues]
        for pr in rng.sample(closing, 2):
            add(
                "history",
                f"Which issues does PR #{pr.number} in {REPO} close?",
                numbers=[str(n) for n in pr.closes_issues],
            )
        # release questions: the first release tag that contains the PR's merge commit (git)
        released = 0
        for pr in picked[5:]:
            sha = getattr(pr, "merge_commit_sha", None)
            tag = first_tag_containing(sha) if sha else None
            if tag and released < 4:
                add(
                    "history",
                    f"Which release first contained the change merged in PR "
                    f"#{pr.number} in {REPO}?",
                    all=[tag],
                )
                released += 1
        if released < 4:
            raise SystemExit("not enough PRs with a release")
        # who made the most non-merge commits in a year: database vs git must agree
        for year in (2016, 2019, 2022):
            rows = connection.execute(
                "SELECT author_name, COUNT(*) n FROM history_commits WHERE repo=? "
                "AND substr(author_date,1,4)=? GROUP BY author_name ORDER BY n DESC LIMIT 2",
                (REPO, str(year)),
            ).fetchall()
            if rows[0][1] == rows[1][1]:
                raise SystemExit(f"tie for {year}")
            shortlog = git(
                "shortlog",
                "-sn",
                "--no-merges",
                f"--since={year}-01-01",
                f"--until={year + 1}-01-01",
                SHA,
            ).splitlines()[0]
            if rows[0][0] not in shortlog:
                raise SystemExit(f"git and the database disagree for {year}: {shortlog}")
            add(
                "history",
                f"Who made the most commits in {REPO} in {year}, not counting merge commits?",
                all=[rows[0][0]],
            )

        pr_numbers = {p.number for p in db.find_prs(REPO, limit=1_000_000)}
        issue_numbers = sorted(
            r[0] for r in connection.execute("SELECT number FROM issues WHERE repo=?", (REPO,))
        )
        every = sorted((pr_numbers | set(issue_numbers)) - {int(n) for n in used_numbers})
        outgoing, incoming = [], []
        for number in rng.sample(every, 400):
            refs = db.references(REPO, number)
            out = [m["number"] for m in refs["mentions"] if m["kind"] != "unknown"]
            inc = [m["number"] for m in refs["mentioned_by"]]
            if 1 <= len(out) <= 4 and not refs["truncated"]:
                outgoing.append((number, out))
            if 1 <= len(inc) <= 4 and not refs["truncated"]:
                incoming.append((number, inc))
        for number, found in outgoing[:2]:
            add(
                "refs",
                f"Which other PRs or issues does #{number} in {REPO} mention in its title or body?",
                numbers=[str(n) for n in found],
                exclude=str(number),
            )
        for number, found in incoming[:2]:
            add(
                "refs",
                f"Which PRs or issues in {REPO} mention #{number} in their title or body?",
                numbers=[str(n) for n in found],
                exclude=str(number),
            )
        only_issues = [
            n for n in issue_numbers if n not in pr_numbers and str(n) not in used_numbers
        ]
        for number in rng.sample(only_issues, 2):
            add("kind", f"Is #{number} in {REPO} a pull request or an issue?", kind="issue")
        for pr in rng.sample(prs, 2):
            add(
                "kind",
                f"Is #{pr.number} in {REPO} a pull request or an issue?",
                kind="pull request",
            )
        for label, items in (
            ("pull request", db.find_prs(REPO, limit=1_000_000)),
            ("issue", db.find_issues(REPO, limit=1_000_000)),
        ):
            titles: dict[str, list[int]] = {}
            for item in items:
                titles.setdefault(item.title.strip().lower(), []).append(item.number)
            usable = sorted(
                (nums[0], title)
                for title, nums in titles.items()
                if len(nums) == 1
                and 25 <= len(title) <= 120
                and "#" not in title
                and str(nums[0]) not in used_numbers
            )
            for number, title in rng.sample(usable, 3):
                add(
                    "prsearch",
                    f'In {REPO}, which {label} is about: "{title}"? Give its number.',
                    number=str(number),
                )
    for question, term in PROBES:
        add("probe", question, term=term)
    return records


def main() -> None:
    records = build()
    OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    kinds: dict[str, int] = {}
    for r in records:
        kinds[r["type"]] = kinds.get(r["type"], 0) + 1
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]
    print(f"wrote {OUT} ({len(records)} questions) sha256 {digest}\n{kinds}")


if __name__ == "__main__":
    main()
