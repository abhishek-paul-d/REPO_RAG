"""Human review sheet for the saved agent answers, and judge-vs-person-vs-mechanical comparison.

    uv run python eval/judge_answers.py prepare ... --out DIR      # makes DIR/items.json (no GPU)
    uv run python eval/review_sheet.py sheet --out DIR [--sample N] # DIR/review_sheet.{md,csv}
    # a person fills the two verdict columns in review_sheet.csv, then, once a judge ran:
    uv run python eval/review_sheet.py compare --out DIR            # agreement table + disagreements.md

The sheet needs no model: it lays each answer next to the exact lines it cites and the reference
lines, so a person can mark `faithful` and `correct` quickly. After the judge has run (DIR/judgments.jsonl)
`compare` reports how often the judge, the person and the mechanical score agree, and lists the cases
to read. A person's verdicts are the only evidence here about whether the judge can be believed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SEED = 20261110
FAITH = ("faithful", "partial", "unfaithful")
CORRECT = ("correct", "partial", "incorrect")


def load_items(out: Path) -> list[dict[str, Any]]:
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    return [i for i in items if not i["skipped"]]


def pick(items: list[dict[str, Any]], sample: int | None, seed: int = SEED) -> list[dict[str, Any]]:
    """All items, or a seeded sample that keeps the code/docs mix (proportional, at least 1 each)."""
    if not sample or sample >= len(items):
        return sorted(items, key=lambda i: i["id"])
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_type.setdefault(item["type"], []).append(item)
    chosen: list[dict[str, Any]] = []
    for kind in sorted(by_type):
        group = sorted(by_type[kind], key=lambda i: i["id"])
        take = max(1, round(sample * len(group) / len(items)))
        chosen += rng.sample(group, min(take, len(group)))
    return sorted(chosen, key=lambda i: i["id"])


def render_markdown(items: list[dict[str, Any]], blind: bool = False) -> str:
    out = [
        "# Answer review sheet",
        "",
        "For each answer, mark in `review_sheet.csv`: **faithful** (every claim supported by the cited",
        "lines: faithful / partial / unfaithful) and **correct** (answers the question, agrees with the",
        "reference lines: correct / partial / incorrect). Use only the lines shown; no outside knowledge.",
        "",
    ]
    for n, item in enumerate(items, 1):
        mechanical = "pass" if item["mechanical_passed"] else "miss"
        heading = (
            f"{item['id']} ({item['type']})"
            if blind
            else (f"{item['id']} ({item['type']}, mechanical score: {mechanical})")
        )
        out += [
            f"## {n}. {heading}",
            "",
            f"**Question:** {item['question']}",
            "",
            f"**Answer:** {item['answer']}",
            "",
            "**Cited lines:**",
            "```",
            item["cited_evidence"] or "(the answer cites no resolvable lines)",
            "```",
            "",
            "**Reference lines:**",
            "```",
            item["reference_evidence"],
            "```",
            "",
        ]
    return "\n".join(out) + "\n"


def read_marks(path: Path) -> dict[str, dict[str, str]]:
    """Every row already in the CSV (verdicts and notes), keyed by id; {} if there is no file."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def write_csv(
    path: Path,
    items: list[dict[str, Any]],
    marks: dict[str, dict[str, str]],
    blind: bool = False,
) -> None:
    """Write the sheet, carrying over any verdicts and notes a person already entered."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "type", "mechanical_passed", "faithful", "correct", "notes"])
        for item in items:
            old = marks.get(item["id"], {})
            writer.writerow(
                [
                    item["id"],
                    item["type"],
                    "" if blind else item["mechanical_passed"],
                    old.get("faithful", ""),
                    old.get("correct", ""),
                    old.get("notes", ""),
                ]
            )


def read_person(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            faithful = row["faithful"].strip().lower()
            correct = row["correct"].strip().lower()
            if faithful in FAITH or correct in CORRECT:
                rows[row["id"]] = {
                    "faithful": faithful if faithful in FAITH else "",
                    "correct": correct if correct in CORRECT else "",
                }
    return rows


def read_judge(out: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    path = out / "judgments.jsonl"
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["role"] != "answer" or not row["ok"]:
            continue
        rows.setdefault(row["id"], {})[
            "faithful" if row["kind"] == "faithfulness" else "correct"
        ] = row["verdict"]
    return rows


def agreement(
    items: list[dict[str, Any]],
    person: dict[str, dict[str, str]],
    judge: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Exact-match agreement per dimension between person and judge, and the confusion counts."""
    result: dict[str, Any] = {}
    for dim, top in (("faithful", "faithful"), ("correct", "correct")):
        pairs = [
            (person[i["id"]][dim], judge[i["id"]][dim])
            for i in items
            if person.get(i["id"], {}).get(dim) and judge.get(i["id"], {}).get(dim)
        ]
        same = sum(p == j for p, j in pairs)
        # The judge's "top" verdict vs the person's, the number that matters most.
        top_pairs = [(p == top, j == top) for p, j in pairs]
        result[dim] = {
            "n": len(pairs),
            "exact_agreement": round(same / len(pairs), 4) if pairs else None,
            "top_vs_not_top": {
                f"person_{'top' if a else 'not'}_judge_{'top' if b else 'not'}": n
                for (a, b), n in sorted(Counter(top_pairs).items())
            },
            "disagreements": [
                {"id": i["id"], "person": person[i["id"]][dim], "judge": judge[i["id"]][dim]}
                for i in items
                if person.get(i["id"], {}).get(dim)
                and judge.get(i["id"], {}).get(dim)
                and person[i["id"]][dim] != judge[i["id"]][dim]
            ],
        }
    return result


def render_disagreements(
    items: list[dict[str, Any]],
    person: dict[str, dict[str, str]],
    judge: dict[str, dict[str, str]],
) -> str:
    """Cases worth reading: judge and mechanical score disagree, or judge and a person disagree."""
    out = ["# Disagreements to read", ""]
    count = 0
    for item in items:
        iid = item["id"]
        j = judge.get(iid, {})
        p = person.get(iid, {})
        notes: list[str] = []
        if item["mechanical_passed"] and j.get("correct") in ("incorrect", "partial"):
            notes.append(f"mechanical pass, judge says {j['correct']} (correct)")
        if not item["mechanical_passed"] and j.get("correct") == "correct":
            notes.append("mechanical miss, judge says correct")
        for dim in ("faithful", "correct"):
            if p.get(dim) and j.get(dim) and p[dim] != j[dim]:
                notes.append(f"{dim}: person {p[dim]}, judge {j[dim]}")
        if notes:
            count += 1
            out += [
                f"## {iid} ({item['type']})",
                "",
                *[f"- {note}" for note in notes],
                "",
                f"**Question:** {item['question']}",
                "",
                f"**Answer:** {item['answer']}",
                "",
            ]
    out.insert(2, f"{count} of {len(items)} judged answers.")
    out.insert(3, "")
    return "\n".join(out) + "\n"


def sheet(args: argparse.Namespace) -> None:
    out = Path(args.out)
    csv_path = out / "review_sheet.csv"
    marks = read_marks(csv_path)
    items = load_items(out)
    chosen = pick(items, args.sample)
    # Never drop an answer a person has already marked, even if a new sample would not pick it.
    marked = {
        i
        for i, row in marks.items()
        if row.get("faithful") or row.get("correct") or row.get("notes")
    }
    have = {i["id"] for i in chosen}
    chosen = sorted(chosen + [i for i in items if i["id"] in marked - have], key=lambda i: i["id"])
    (out / "review_sheet.md").write_text(render_markdown(chosen, args.blind), encoding="utf-8")
    write_csv(csv_path, chosen, marks, args.blind)
    print(
        f"{len(chosen)} answers -> {out / 'review_sheet.md'} and review_sheet.csv "
        f"({len(marked)} existing marks kept)"
    )


def mechanical_agreement(
    items: list[dict[str, Any]], verdicts: dict[str, dict[str, str]], source: str
) -> dict[str, Any]:
    """Mechanical pass/miss vs a source's `correct` verdict (correct = pass, else miss)."""
    counts: Counter[str] = Counter()
    for item in items:
        verdict = verdicts.get(item["id"], {}).get("correct")
        if verdict:
            counts[
                f"mechanical_{'pass' if item['mechanical_passed'] else 'miss'}_{source}_{verdict}"
            ] += 1
    n = sum(counts.values())
    same = sum(
        v
        for k, v in counts.items()
        if (k.startswith("mechanical_pass") and k.endswith("_correct"))
        or (k.startswith("mechanical_miss") and not k.endswith("_correct"))
    )
    return {
        "n": n,
        "agreement_pass_vs_correct": round(same / n, 4) if n else None,
        "counts": dict(sorted(counts.items())),
    }


def person_summary(person: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        dim: dict(Counter(row[dim] for row in person.values() if row.get(dim)))
        for dim in ("faithful", "correct")
    }


def compare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    items = load_items(out)
    judge = read_judge(out)
    csv_path = out / "review_sheet.csv"
    person = read_person(csv_path) if csv_path.is_file() else {}
    (out / "disagreements.md").write_text(
        render_disagreements(items, person, judge), encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "answers": len(items),
        "judged": len(judge),
        "reviewed_by_person": len(person),
    }
    if person:
        summary["person_verdicts"] = person_summary(person)
        summary["mechanical_vs_person"] = mechanical_agreement(items, person, "person")
    if judge:
        summary["mechanical_vs_judge"] = mechanical_agreement(items, judge, "judge")
    if person and judge:
        summary["person_vs_judge"] = agreement(items, person, judge)
    if not person and not judge:
        summary["note"] = "no judge output and no person verdicts yet; nothing to compare"
    (out / "agreement.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--out", required=True)
    s.add_argument("--sample", type=int, help="review this many answers (default: all)")
    s.add_argument("--blind", action="store_true", help="hide the mechanical score")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    args = parser.parse_args()
    {"sheet": sheet, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
