"""Build and hash a link-derived Phase 5 corpus BEFORE any retrieval or model loading.

GitHub calls are read-only and cached. The output directory is append-only during
preparation and refuses any existing freeze. No issue text enters commit documents.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from git import Repo  # noqa: E402

from repo_rag.config import Settings  # noqa: E402
from repo_rag.history.commit_indexer import (  # noqa: E402
    commit_document,
    is_fix_message,
    iter_commits,
)
from repo_rag.history.issue_linker import (  # noqa: E402
    GitHubClient,
    Issue,
    IssueLink,
    direct_references,
    link_commit,
)
from repo_rag.history.metadata_db import record_from_dict  # noqa: E402

PROTOCOL = ROOT / "eval/datasets/phase5_bugfix_v2/protocol.json"


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def read_lines(path: Path) -> list[dict]:
    """JSONL is separated by LF, not Python's extra Unicode splitlines separators."""
    with (
        path.open(encoding="utf-8")
        if path.exists()
        else gzip.open(Path(str(path) + ".gz"), "rt", encoding="utf-8")
    ) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    data = (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )
    return hashlib.sha256(data).hexdigest()


def grouped_pairs(
    repo: str, issues: list[Issue], links: list[IssueLink], protocol: dict
) -> list[dict]:
    gold: dict[tuple[str, int], set[str]] = defaultdict(set)
    paths: dict[tuple[str, int], set[str]] = defaultdict(set)
    for link in links:
        key = (link.issue_repo, link.issue_number)
        gold[key].add(link.commit_sha)
        paths[key].add(link.link_type)
    rows: list[dict] = [
        {
            "id": f"{repo}:{i.repo}#{i.number}",
            "repo": repo,
            "issue_repo": i.repo,
            "issue_number": i.number,
            "query": f"{i.title}\n\n{i.body}".strip(),
            "gold": sorted(gold[(i.repo, i.number)]),
            "link_paths": sorted(paths[(i.repo, i.number)]),
        }
        for i in issues
        if gold[(i.repo, i.number)]
    ]
    # Union connected issues through shared fix SHAs; do not split paraphrases or shared fixes.
    parents = list(range(len(rows)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        for sha in row["gold"]:
            if sha in owners:
                parents[find(index)] = find(owners[sha])
            owners[sha] = index
    groups: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    keys = sorted(groups)
    random.Random(protocol["split"]["seed"]).shuffle(keys)
    n_dev = max(1, round(len(keys) * protocol["split"]["dev_fraction"])) if len(keys) > 1 else 0
    dev = set(keys[:n_dev])
    for group_key, members in groups.items():
        group = hashlib.sha256("|".join(sorted(m["id"] for m in members)).encode()).hexdigest()
        for row in members:
            row.update(group=f"{repo}:{group}", split="dev" if group_key in dev else "test")
    return sorted(rows, key=lambda r: r["id"])


def prepare(
    repo: str, checkout: Path, output: Path, *, ref: str, protocol_path: Path = PROTOCOL
) -> dict:
    if (output / "freeze.json").exists():
        raise FileExistsError("Frozen dataset exists; preserve it and use a new output directory")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = protocol_path.with_name("protocol.freeze.json")
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8"))["protocol_sha256"] != digest(
        protocol_path
    ):
        raise ValueError("Protocol changed after its pre-run freeze")
    if repo not in protocol["repos"]:
        raise ValueError("repository is outside the frozen protocol")
    output.mkdir(parents=True, exist_ok=True)
    protocol_copy = output / "protocol.json"
    if protocol_copy.exists() and protocol_copy.read_bytes() != protocol_path.read_bytes():
        raise ValueError("preparation protocol changed; use a new output directory")
    protocol_copy.write_bytes(protocol_path.read_bytes())
    settings = Settings(
        max_commits=protocol["selection"]["max_commits"],
        commit_diff_chars=protocol["documents"]["commit_diff_chars"],
    )
    settings = settings.model_copy(update={"data_dir": output / "cache"})
    api = GitHubClient(settings)
    with Repo(checkout) as git:
        if git.git.rev_parse("--is-shallow-repository").strip() == "true":
            raise ValueError("History evaluation requires full history, not a shallow clone")
        pin = git.commit(ref).hexsha
    if pin != protocol["pins"][repo]:
        raise ValueError("ref differs from the prechosen protocol pin")
    records_file = output / "records.jsonl"
    preparation = output / "preparation.json"
    identity = {"repo": repo, "pin": pin, "protocol_sha256": digest(protocol_copy)}
    if preparation.exists() and json.loads(preparation.read_text(encoding="utf-8")) != identity:
        raise ValueError("repository/ref changed during preparation")
    write(preparation, identity)
    if records_file.exists():
        records = [record_from_dict(row) for row in read_lines(records_file)]
    else:
        records = []
        # Write to a temporary file, so an interruption cannot look like a complete corpus.
        pending = records_file.with_suffix(".tmp")
        with pending.open("w", encoding="utf-8") as stream:
            for record in iter_commits(checkout, repo, settings, ref=pin):
                records.append(record)
                stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                if len(records) % 100 == 0:
                    print(repo, "history records", len(records), flush=True)
        pending.replace(records_file)
    target = protocol["selection"]["target_issues_per_repo"]
    issues: dict[tuple[str, int], Issue] = {}
    links: list[IssueLink] = []
    skipped: list[dict] = []
    lookups = 0
    if repo in protocol["selection"]["direct_repos"]:
        candidates = [(record, direct_references(record.message, repo)) for record in records]
        # Resolve every direct edge for sampled issues, including multiple fixing commits.
        for record, refs in candidates:
            for issue_ref in refs:
                key = (issue_ref.repo, issue_ref.number)
                if key not in issues and len(issues) >= target:
                    continue
                issue = issues.get(key) or api.issue(issue_ref)
                if (
                    not issue
                    or issue.state != "CLOSED"
                    or not issue.title.strip()
                    or not issue.body.strip()
                ):
                    skipped.append(
                        {
                            "sha": record.sha,
                            "issue": asdict(issue_ref),
                            "reason": "not closed/nonempty GitHub Issue",
                        }
                    )
                    continue
                issues[key] = issue
                links.append(IssueLink(repo, record.sha, issue.repo, issue.number, "direct"))
    else:
        by_sha = {record.sha: record for record in records}
        pr_candidates = api.linked_pull_requests(
            repo, limit=protocol["selection"]["via_pr_search_max_prs"]
        )
        for pr_number, sha in pr_candidates:
            if sha not in by_sha:
                skipped.append(
                    {
                        "sha": sha,
                        "pr": pr_number,
                        "reason": "merge commit or outside capped pinned corpus",
                    }
                )
                continue
            record = by_sha[sha]
            found, fetched = link_commit(repo, record.sha, record.message, api)
            lookups += 1
            for issue in fetched:
                key = (issue.repo, issue.number)
                if (
                    issue.state == "CLOSED"
                    and issue.title.strip()
                    and issue.body.strip()
                    and (key in issues or len(issues) < target)
                ):
                    issues[key] = issue
            links.extend(
                link
                for link in found
                if (link.issue_repo, link.issue_number) in issues and link.link_type == "via_pr"
            )
            if lookups % 25 == 0:
                print(repo, "PR lookups", lookups, "issues", len(issues), flush=True)
            # Continue through the fixed PR search window to collect sampled multi-fix gold.
    links = list(dict.fromkeys(links))
    eligible = {r.sha for r in records if r.is_fix}
    if any(link.commit_sha not in eligible for link in links):
        raise ValueError("Gold candidate membership depends on issue links")
    docs = [commit_document(r, settings) for r in records]
    docs = [d for d in docs if d.metadata["is_fix"] == "true"]
    pairs = grouped_pairs(repo, list(issues.values()), links, protocol)
    if not pairs:
        raise ValueError("No valid issue/fix pairs; preparation remains unfrozen for diagnosis")
    write_lines(
        output / "corpus.jsonl", [{"text": d.page_content, "meta": d.metadata} for d in docs]
    )
    write_lines(output / "pairs.jsonl", pairs)
    write(output / "links.json", [asdict(link) for link in links])
    write(output / "issues.json", [asdict(issue) for issue in issues.values()])
    write(
        output / "selection.json",
        {
            **identity,
            "nonmerge_commits": len(records),
            "fix_candidates": len(docs),
            "questions": len(pairs),
            "groups": len({r["group"] for r in pairs}),
            "target": target,
            "shortfall": max(0, target - len(pairs)),
            "via_pr_lookups": lookups,
            "via_pr_gold_window": "only verified mergeCommit SHAs of the fixed PR search window in the capped nonmerge corpus; other fixes may be unlabelled",
            "skipped": skipped,
        },
    )
    names = [
        "protocol.json",
        "records.jsonl",
        "pairs.jsonl",
        "corpus.jsonl",
        "links.json",
        "issues.json",
        "selection.json",
        "preparation.json",
    ]
    freeze = {
        "repo": repo,
        "pin": pin,
        "sha256": {name: digest(output / name) for name in names},
        "stage": "frozen before any retrieval/model loading",
    }
    write(output / "freeze.json", freeze)
    return freeze


def verify_frozen(folder: Path) -> dict:
    freeze = json.loads((folder / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in freeze["sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return freeze


def verify_corpus(folder: Path) -> dict:
    """Rebuild every searchable document and connected-group label from frozen records.

    This catches appended issue text and changed split/gold semantics even when file hashes
    were recomputed. Historical AST extraction itself remains independently inspectable in
    records.jsonl; validating it again against git is a separate source check.
    """
    freeze = verify_frozen(folder)
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    if protocol["version"] < 2:
        raise ValueError("Historical v1 requires its archived source.zip; use v2 with current code")
    settings = Settings(commit_diff_chars=protocol["documents"]["commit_diff_chars"])
    records = [record_from_dict(row) for row in read_lines(folder / "records.jsonl")]
    if any(r.is_fix != is_fix_message(r.message) for r in records):
        raise ValueError("Stored candidate flags differ from message heuristic")
    links = [
        IssueLink(**row) for row in json.loads((folder / "links.json").read_text(encoding="utf-8"))
    ]
    issues = [
        Issue(**row) for row in json.loads((folder / "issues.json").read_text(encoding="utf-8"))
    ]
    pairs = grouped_pairs(freeze["repo"], issues, links, protocol)
    eligible = {r.sha for r in records if r.is_fix}
    if any(not set(pair["gold"]) <= eligible for pair in pairs):
        raise ValueError("Gold candidate membership depends on issue links")
    docs = [commit_document(record, settings) for record in records]
    expected = [
        {"text": doc.page_content, "meta": doc.metadata}
        for doc in docs
        if doc.metadata["is_fix"] == "true"
    ]
    if read_lines(folder / "corpus.jsonl") != expected:
        raise ValueError(
            "Corpus does not rebuild from commit-only records (possible issue-text leakage)"
        )
    if read_lines(folder / "pairs.jsonl") != pairs:
        raise ValueError("Gold/group/split does not rebuild from issue links")
    return {
        "repo": freeze["repo"],
        "records": len(records),
        "documents": len(expected),
        "questions": len(pairs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.checkout, args.output, ref=args.ref), indent=2))
