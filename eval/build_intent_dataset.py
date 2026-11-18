"""Build the intent/tie-break policy dataset: the frozen 90 questions plus 30 paraphrases.

The paraphrases reuse the gold evidence of the 12 implementation and 18 test questions but use
natural wording instead of the template ("Locate the implementation...", "Which automated
test..."). They test whether the intent detector, which was written from the template wording,
generalises. Paraphrases were authored by the same assistant that wrote the detector.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "eval/datasets/phase4_comparison_v3"
TARGET = ROOT / "eval/datasets/phase4_intent_v1"

PARAPHRASES = {
    "requests-comparison-019": "Is there a test that checks every response in a redirect chain gets the rewritten method?",
    "requests-comparison-020": "A test showing query parameter merging is case-sensitive about keys?",
    "requests-comparison-021": "Where is it tested that non-HTTP schemes skip URL validation?",
    "requests-comparison-022": "Which test shows closing a session releases the connection of an unread response?",
    "requests-comparison-023": "What covers the error raised for a missing CA bundle path?",
    "requests-comparison-024": "Where can I see a test with several values for one query parameter name?",
    "requests-comparison-025": "What function splits a Link header into dictionaries of URL and rel parameters?",
    "requests-comparison-026": "Which helper decides that a string is not an IPv4 address?",
    "requests-comparison-027": "How is a CIDR prefix length turned into a dotted netmask? I want the source.",
    "requests-comparison-028": "Where does the code get a filename from a file object's name, ignoring names like <stdin>?",
    "requests-comparison-029": "Where is the code that deletes a cookie by name, optionally limited to a domain and path?",
    "requests-comparison-030": "Where does digest auth keep its counters per thread?",
    "express-comparison-019": "What test makes sure params come back after exiting a nested router?",
    "express-comparison-020": "Where is it tested that after an error only error handlers run in a route?",
    "express-comparison-021": "Show me the test where turning off query parsing gives an empty query.",
    "express-comparison-022": "Which spec covers picking the best charset from a list of options?",
    "express-comparison-023": "Find a test that rejects an array for Content-Type.",
    "express-comparison-024": "What test checks that the automatic OPTIONS reply lists each method once?",
    "express-comparison-025": "Where in the source does an app compute its full mount path from its parents?",
    "express-comparison-026": "What defines req.secure?",
    "express-comparison-027": "Which code turns an object of relations into the Link response header?",
    "flask-comparison-019": "Which test checks that a before-render signal can modify the template context?",
    "flask-comparison-020": "Where is it tested that an exception handled earlier doesn't reach later request teardown?",
    "flask-comparison-021": "Which test covers jsonify rejecting args and kwargs together?",
    "flask-comparison-022": "What tests that package resources can't be opened for writing?",
    "flask-comparison-023": "Which test shows a refreshed but unchanged permanent session still adds Vary: Cookie?",
    "flask-comparison-024": "Where is the test that two registrations can't share a blueprint name?",
    "flask-comparison-025": "Which code passes _anchor, _scheme and _method on to the app's url_for?",
    "flask-comparison-026": "What checks whether an application context is active right now?",
    "flask-comparison-027": "Where does Flask work out a module's root path, falling back to the current directory?",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build() -> None:
    if TARGET.exists():
        raise FileExistsError("phase4_intent_v1 is frozen; create a new version instead")
    TARGET.mkdir()
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    used = set()
    datasets = []
    for entry in manifest["datasets"]:
        lines = []
        for raw in (SOURCE / entry["file"]).read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            lines.append(raw)
            case = json.loads(raw)
            qid = case["question"]["id"]
            if qid in PARAPHRASES:
                used.add(qid)
                para = json.loads(raw)
                para["question"]["id"] = qid + "-para"
                para["question"]["question"] = PARAPHRASES[qid]
                para["question"]["provenance"] = (
                    "natural-wording paraphrase of " + qid + "; same gold evidence"
                )
                para["origin_case_id"] = None
                para["topic"] = case.get("topic", "") + "-paraphrase"
                lines.append(json.dumps(para, ensure_ascii=False))
        path = TARGET / entry["file"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        datasets.append({**entry, "questions": len(lines), "sha256": digest(path)})
    if used != set(PARAPHRASES):
        raise ValueError(f"paraphrase ids missing from the dataset: {set(PARAPHRASES) - used}")
    shutil.copyfile(SOURCE / "protocol.json", TARGET / "protocol.json")
    (TARGET / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "parent": "phase4_comparison_v3",
                "authorship": "same assistant as the intent detector; independent review pending",
                "paraphrases": len(PARAPHRASES),
                "datasets": datasets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {sum(d['questions'] for d in datasets)} cases to {TARGET}")


if __name__ == "__main__":
    build()
