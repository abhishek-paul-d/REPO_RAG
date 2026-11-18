"""Auto-generate the Phase 5b exact-answer check set. No hand labelling, no LLM.

Expected answers come from sources the tools do not use: the raw GraphQL nodes saved by the
sync (`<data_dir>/github_cache/sync/*.json`) and plain `git` subprocess calls. Nothing here
imports `MetadataDB` or `git_tools`. The tools are then asked the same questions by
`run_facts.py`; a pass means two independent readings of the same repository agree.

    uv run python eval/build_facts.py encode/httpx --checkout data/repos/encode__httpx \\
        --data-dir <dir holding github_cache/> [--per-type 6] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# How many facts of each type to sample (the plan asks for about 50 in total).
QUOTAS = {
    "pr_merged_by": 8,
    "pr_author": 5,
    "pr_title": 4,
    "pr_state": 4,
    "issue_author": 3,
    "issue_closed_by": 4,
    "release_first": 8,
    "blame_line": 6,
    "commits_in_month": 8,
    "top_committer": 3,
    "hot_file": 2,
    "tag_commit": 5,
}
HOT_PREFIX = "httpx/"
MAX_ZERO_COUNTS = 2


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "core.quotepath=false", "-C", str(root), *args],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    return done.stdout


def load_nodes(data_dir: Path, slug: str, kind: str) -> dict[int, dict[str, Any]]:
    path = data_dir / "github_cache" / "sync" / f"{slug.replace('/', '__')}.{kind}.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return {int(n): node for n, node in items.items()}


def login(node: dict[str, Any], key: str) -> str:
    return str((node.get(key) or {}).get("login") or "")


def fact(
    kind: str, ident: str, question: str, tool: str, args: dict[str, Any], expected: Any
) -> dict:
    compare = "set" if isinstance(expected, list) else "exact"
    return {
        "id": f"{kind}:{ident}",
        "type": kind,
        "question": question,
        "tool": tool,
        "args": args,
        "expected": expected,
        "compare": compare,
    }


# ------------------------------------------------------------------ GitHub facts


def github_facts(
    prs: dict[int, dict[str, Any]], issues: dict[int, dict[str, Any]], rng: random.Random
) -> list[dict]:
    out: list[dict] = []

    def pick(pool: list[int], kind: str) -> list[int]:
        return rng.sample(sorted(pool), min(QUOTAS[kind], len(pool)))

    merged = [n for n, p in prs.items() if p["state"] == "MERGED" and login(p, "mergedBy")]
    for n in pick(merged, "pr_merged_by"):
        out.append(
            fact(
                "pr_merged_by",
                str(n),
                f"Who merged #{n}?",
                "get_pr",
                {"number": n, "field": "merged_by"},
                login(prs[n], "mergedBy"),
            )
        )
    with_author = [n for n, p in prs.items() if login(p, "author")]
    for n in pick(with_author, "pr_author"):
        out.append(
            fact(
                "pr_author",
                str(n),
                f"Who opened PR #{n}?",
                "get_pr",
                {"number": n, "field": "author"},
                login(prs[n], "author"),
            )
        )
    for n in pick(list(prs), "pr_title"):
        out.append(
            fact(
                "pr_title",
                str(n),
                f"What is the title of PR #{n}?",
                "get_pr",
                {"number": n, "field": "title"},
                prs[n]["title"],
            )
        )
    by_state: dict[str, list[int]] = {}
    for n, p in prs.items():
        by_state.setdefault(p["state"], []).append(n)
    states = sorted(by_state)
    for i in range(QUOTAS["pr_state"]):  # round-robin so OPEN / CLOSED / MERGED all appear
        pool = by_state[states[i % len(states)]]
        n = rng.choice(sorted(pool))
        if any(f["id"] == f"pr_state:{n}" for f in out):
            continue
        out.append(
            fact(
                "pr_state",
                str(n),
                f"Is PR #{n} open, closed or merged?",
                "get_pr",
                {"number": n, "field": "state"},
                prs[n]["state"],
            )
        )
    with_issue_author = [n for n, i in issues.items() if login(i, "author")]
    for n in pick(with_issue_author, "issue_author"):
        out.append(
            fact(
                "issue_author",
                str(n),
                f"Who opened issue #{n}?",
                "get_issue_meta",
                {"number": n, "field": "author"},
                login(issues[n], "author"),
            )
        )
    # Closing PRs from the issue side; the tool reads the PR side. Only issues whose closing
    # PRs are all inside the synced PR set can be answered by the index.
    closed_by: dict[int, list[int]] = {}
    for n, i in issues.items():
        found = sorted(int(x["number"]) for x in i["closedByPullRequestsReferences"]["nodes"])
        if found and all(f in prs for f in found):
            closed_by[n] = found
    for n in pick(list(closed_by), "issue_closed_by"):
        out.append(
            fact(
                "issue_closed_by",
                str(n),
                f"Which PR closed issue #{n}?",
                "prs_closing_issue",
                {"number": n},
                [str(x) for x in closed_by[n]],
            )
        )
    return out


# ------------------------------------------------------------------ git facts


def tag_table(root: Path) -> list[tuple[str, int]]:
    lines = git(
        root, "for-each-ref", "refs/tags", "--format=%(refname:short)%09%(creatordate:unix)"
    )
    return sorted(
        ((n, int(t)) for n, t in (row.split("\t") for row in lines.splitlines() if row)),
        key=lambda x: (x[1], x[0]),
    )


def is_ancestor(root: Path, commit: str, tag: str) -> bool:
    done = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, f"{tag}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


def release_facts(root: Path, prs: dict[int, dict[str, Any]], rng: random.Random) -> list[dict]:
    tags = tag_table(root)
    rows: list[tuple[int, str, str]] = []  # (number, expected, merge sha)
    # One ancestor check per (PR, tag) is a subprocess each, so look at a random 30 only.
    merged = sorted(n for n, p in prs.items() if p["state"] == "MERGED")
    for n in sorted(rng.sample(merged, min(30, len(merged)))):
        sha = (prs[n].get("mergeCommit") or {}).get("oid")
        if prs[n]["state"] != "MERGED" or not sha:
            continue
        if subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", sha], capture_output=True
        ).returncode:
            continue
        containing = [(name, ts) for name, ts in tags if is_ancestor(root, sha, name)]
        if not containing:
            rows.append((n, "none", sha))
        elif len(containing) > 1 and containing[0][1] == containing[1][1]:
            continue  # two tags share the earliest date: the order is not defined
        else:
            rows.append((n, containing[0][0], sha))
    unreleased = [r for r in rows if r[1] == "none"]
    released = [r for r in rows if r[1] != "none"]
    quota = QUOTAS["release_first"]
    chosen = rng.sample(unreleased, min(2, len(unreleased)))
    chosen += rng.sample(released, min(quota - len(chosen), len(released)))
    return [
        fact(
            "release_first",
            str(n),
            f"Which release first included the change merged in #{n}?",
            "release_first",
            {"number": n},
            expected,
        )
        for n, expected, _ in sorted(chosen)
    ]


def blame_facts(root: Path, rng: random.Random) -> list[dict]:
    files = sorted(f for f in git(root, "ls-files", "httpx/*.py").splitlines() if f)
    out: list[dict] = []
    for _ in range(10 * QUOTAS["blame_line"]):
        if len(out) >= QUOTAS["blame_line"] or not files:
            break
        path = rng.choice(files)
        lines = git(root, "show", f"HEAD:{path}").splitlines()
        candidates = [i for i, text in enumerate(lines, start=1) if text.strip()]
        if not candidates:
            continue
        line = rng.choice(candidates)
        sha = git(root, "log", "-s", "-1", "--format=%H", f"-L{line},{line}:{path}", "HEAD").split()
        if not sha or any(f["id"] == f"blame_line:{path}:{line}" for f in out):
            continue
        out.append(
            fact(
                "blame_line",
                f"{path}:{line}",
                f"Which commit last changed line {line} of {path}?",
                "blame",
                {"path": path, "line": line},
                sha[0][:12],
            )
        )
    return out


def author_times(root: Path, *pathspec: str) -> list[tuple[str, datetime]]:
    text = git(root, "log", "--full-history", "--no-merges", "--format=%an%x09%at", "--", *pathspec)
    out = []
    for row in text.splitlines():
        name, stamp = row.rsplit("\t", 1)
        out.append((name, datetime.fromtimestamp(int(stamp), UTC)))
    return out


def month_bounds(year: int, month: int) -> tuple[str, str]:
    nxt = (year + (month == 12), month % 12 + 1)
    return f"{year}-{month:02d}-01", f"{nxt[0]}-{nxt[1]:02d}-01"


def count_facts(root: Path, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    paths = ["httpx", "tests", "docs", "httpx/_client.py", "httpx/_models.py", "pyproject.toml"]
    times = {p: [t for _, t in author_times(root, p)] for p in paths}
    everything = [t for _, t in author_times(root, ".")]
    for _ in range(200):
        if len(out) >= QUOTAS["commits_in_month"]:
            break
        path = rng.choice(paths)
        pivot = rng.choice(everything)  # a month that has history, so counts are mostly > 0
        since, until = month_bounds(pivot.year, pivot.month)
        key = f"{path}@{since}"
        if any(f["id"].endswith(key) for f in out):
            continue
        count = sum(1 for t in times[path] if since <= t.strftime("%Y-%m-%d") < until)
        if count == 0 and sum(f["expected"] == "0" for f in out) >= MAX_ZERO_COUNTS:
            continue  # a few zeros check "nothing", but most facts should need a real count
        out.append(
            fact(
                "commits_in_month",
                key,
                f"How many commits touched {path} in {pivot.strftime('%B %Y')}?",
                "count_commits",
                {"path": path, "since": since, "until": until},
                str(count),
            )
        )
    return out


def ranking_facts(root: Path, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    years = sorted({t.year for _, t in author_times(root, ".")})
    scopes = ["", "httpx", "tests"]
    attempts = 0
    while len(out) < QUOTAS["top_committer"] and attempts < 100:
        attempts += 1
        scope, year = rng.choice(scopes), rng.choice(years)
        rows = author_times(root, scope or ".")
        counts = Counter(n for n, t in rows if t.year == year).most_common(2)
        if not counts or (len(counts) > 1 and counts[0][1] == counts[1][1]):
            continue  # a tie has no single right answer
        key = f"{scope or 'repo'}@{year}"
        if any(f["id"].endswith(key) for f in out):
            continue
        where = f"touching {scope} " if scope else ""
        out.append(
            fact(
                "top_committer",
                key,
                f"Who made the most commits {where}in {year}?",
                "top_contributors",
                {"path": scope or None, "since": f"{year}-01-01", "until": f"{year + 1}-01-01"},
                counts[0][0],
            )
        )
    hot: list[dict] = []
    text = git(
        root,
        "log",
        "--full-history",
        "--no-merges",
        "--name-only",
        "--format=%x1e%at",
        "--",
        HOT_PREFIX.rstrip("/"),
    )
    commits: list[tuple[int, set[str]]] = []
    for block in text.split("\x1e"):
        if block.strip():
            stamp, *files = block.strip().split("\n")
            commits.append(
                (
                    datetime.fromtimestamp(int(stamp), UTC).year,
                    {f for f in files if f.startswith(HOT_PREFIX)},
                )
            )
    for year in rng.sample(years, len(years)):
        if len(hot) >= QUOTAS["hot_file"]:
            break
        counts: Counter[str] = Counter()
        for commit_year, files_in in commits:
            if commit_year == year:
                counts.update(files_in)
        top = counts.most_common(2)
        if not top or (len(top) > 1 and top[0][1] == top[1][1]):
            continue
        hot.append(
            fact(
                "hot_file",
                str(year),
                f"Which file under {HOT_PREFIX} was changed in the most commits in {year}?",
                "change_frequency",
                {
                    "path_prefix": HOT_PREFIX.rstrip("/"),
                    "since": f"{year}-01-01",
                    "until": f"{year + 1}-01-01",
                },
                top[0][0],
            )
        )
    return out + hot


def tag_facts(root: Path, rng: random.Random) -> list[dict]:
    names = [n for n, _ in tag_table(root)]
    kinds = dict(
        line.split("\t")
        for line in git(
            root, "for-each-ref", "refs/tags", "--format=%(refname:short)%09%(objecttype)"
        ).splitlines()
    )
    annotated = [n for n in names if kinds[n] == "tag"]
    plain = [n for n in names if kinds[n] != "tag"]
    chosen = rng.sample(annotated, min(3, len(annotated)))
    chosen += rng.sample(plain, min(QUOTAS["tag_commit"] - len(chosen), len(plain)))
    return [
        fact(
            "tag_commit",
            name,
            f"Which commit does the tag {name} point to?",
            "get_tag",
            {"name": name},
            git(root, "rev-list", "-n", "1", name).strip(),
        )
        for name in sorted(chosen)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("slug")
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    prs = load_nodes(args.data_dir, args.slug, "prs")
    issues = load_nodes(args.data_dir, args.slug, "issues")
    facts = github_facts(prs, issues, rng)
    facts += release_facts(args.checkout, prs, rng)
    facts += blame_facts(args.checkout, rng)
    facts += count_facts(args.checkout, rng)
    facts += ranking_facts(args.checkout, rng)
    facts += tag_facts(args.checkout, rng)
    head = git(args.checkout, "rev-parse", "HEAD").strip()
    out = args.out or Path("eval/datasets") / f"{args.slug.split('/')[1]}_facts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for item in facts:
            handle.write(
                json.dumps({**item, "repo": args.slug, "head": head}, ensure_ascii=False) + "\n"
            )
    print(f"{len(facts)} facts -> {out}")
    for kind, count in sorted(Counter(f["type"] for f in facts).items()):
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
