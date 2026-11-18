"""Check that each cited range shows what its sentence says; narrow or re-point ranges that do not.

`check_citations` in `citations.py` only asks whether a cited range overlaps a retrieved chunk. The
answer model also cites a whole chunk (a function plus the one after it), a range that misses the
function it names (it names `text` and cites `json`), or a range that runs past the file. This
module works from the identifiers a sentence puts in backticks:

- `citation_support` reports, per citation, which named identifiers appear in the cited lines
  (offline signal; no model).
- `snap_citations` rewrites a range using definition ranges from the symbol table, only when the
  sentence's names resolve unambiguously in that file and the new range overlaps a retrieved chunk.
  Every change is returned so the caller can show it. It never edits prose or adds a citation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from repo_rag.agent.citations import PATH_REF, Chunk

# A definition lookup: (name as written, path) -> [(qualified name, start line, end line)].
Lookup = Callable[[str, str], list[tuple[str, int, int]]]

_TICKED = re.compile(r"`([^`\n]+)`")
_NAME = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*")
MIN_NAME_CHARS = 3
# A citation is "wide" when it is this many times longer than the definitions the sentence names.
WIDE_FACTOR = 3
WIDE_SLACK = 10
COVERED = 0.8  # share of a named definition a range must contain to count as showing it


@dataclass(frozen=True)
class Support:
    ref: str
    names: tuple[str, ...]
    missing: tuple[str, ...]
    lines: int

    @property
    def checked(self) -> bool:
        return bool(self.names)

    @property
    def ok(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class Snap:
    old: str
    new: str
    why: str


@dataclass
class SnapResult:
    answer: str
    snaps: list[Snap] = field(default_factory=list)


def _names_in(sentence: str) -> list[str]:
    """Identifiers a sentence puts in backticks, minus the path:range references themselves."""
    names: list[str] = []
    for ticked in _TICKED.findall(sentence):
        if PATH_REF.search(ticked) or "/" in ticked or ticked.endswith((".py", ".rst", ".md")):
            continue
        for m in _NAME.finditer(ticked):
            name = m.group()
            if len(name.rpartition(".")[2]) >= MIN_NAME_CHARS and name not in names:
                names.append(name)
    return names


def _has(text: str, name: str) -> bool:
    last = name.rpartition(".")[2]
    return re.search(rf"(?<!\w){re.escape(last)}(?!\w)", text) is not None


def _ticked(answer: str, m: re.Match[str]) -> bool:
    """True when the reference sits alone inside a pair of backticks."""
    return m.start() > 0 and answer[m.start() - 1] == "`" and answer[m.end() : m.end() + 1] == "`"


def _segments(answer: str) -> list[tuple[re.Match[str], str]]:
    """Each citation with the text since the previous citation: a citation at the end of a
    paragraph answers for everything said before it. A reference's own backticks are left out so
    the ticks that remain pair up."""
    out: list[tuple[re.Match[str], str]] = []
    previous = 0
    previous_ticked = False
    for m in PATH_REF.finditer(answer):
        ticked = _ticked(answer, m)
        start = previous + (1 if previous_ticked else 0)
        end = m.start() - (1 if ticked else 0)
        out.append((m, answer[start:end]))
        previous, previous_ticked = m.end(), ticked
    return out


def citation_support(
    answer: str,
    read_lines: Callable[[str], list[str] | None],
    lookup: Lookup | None = None,
) -> list[Support]:
    """Per `path:start-end`: which backticked names said since the previous citation are shown by no
    citation to that file. A name counts as shown when any cited range in that file contains it, or
    (with `lookup`) lies inside the definition of that name, as a method does inside its class."""
    refs = _segments(answer)
    bodies: dict[str, list[str]] = {}
    for m, _ in refs:
        lines = read_lines(m["path"])
        start, end = int(m["start"]), int(m["end"])
        if lines and start <= end:
            bodies.setdefault(m["path"], []).append(chr(10).join(lines[start - 1 : end]))
    out: list[Support] = []
    for m, segment in refs:
        path, start, end = m["path"], int(m["start"]), int(m["end"])
        names = _names_in(segment)
        missing = []
        for name in names:
            if any(_has(body, name) for body in bodies.get(path, [])):
                continue
            if lookup and any(d[1] <= start and end <= d[2] for d in lookup(name, path)):
                continue
            missing.append(name)
        out.append(Support(m.group(), tuple(names), tuple(missing), max(0, end - start + 1)))
    return out


def _overlap(a: int, b: int, c: int, d: int) -> int:
    return max(0, min(b, d) - max(a, c) + 1)


def _resolve(
    names: list[str], path: str, lookup: Lookup, cited: tuple[int, int]
) -> list[tuple[int, int]]:
    """One definition range per name defined in `path` (the nearest to the cited range when a
    name is defined more than once). Names defined nowhere in the file are left out, and so is a
    definition that contains another named one (the class around a named method)."""
    ranges: list[tuple[int, int]] = []
    for name in names:
        found = lookup(name, path)
        if not found:
            continue
        found.sort(key=lambda r: (-_overlap(cited[0], cited[1], r[1], r[2]), r[1]))
        if (found[0][1], found[0][2]) not in ranges:
            ranges.append((found[0][1], found[0][2]))
    return [r for r in ranges if not any(o != r and r[0] <= o[0] and o[1] <= r[1] for o in ranges)]


def _status(cited: tuple[int, int], d: tuple[int, int]) -> str:
    """`inside` (the citation is part of the definition), `covers` (holds most of it), or `miss`."""
    if d[0] <= cited[0] and cited[1] <= d[1]:
        return "inside"
    if _overlap(*cited, *d) >= COVERED * (d[1] - d[0] + 1):
        return "covers"
    return "miss"


def snap_citations(answer: str, chunks: list[Chunk], lookup: Lookup) -> SnapResult:
    """Re-point a range that shows none of the definitions named before it, or narrow a wide one
    that contains all of them. A range that already shows (part of) a named definition is kept."""
    result = SnapResult(answer)
    edits: list[tuple[int, int, str]] = []
    everywhere = _names_in(answer)
    for m, segment in _segments(answer):
        names = _names_in(segment)
        path, start, end = m["path"], int(m["start"]), int(m["end"])
        defs = _resolve(names, path, lookup, (start, end)) if names else []
        if not defs:
            continue
        states = [_status((start, end), d) for d in defs]
        size = sum(d[1] - d[0] + 1 for d in defs)
        if all(x == "miss" for x in states):
            # A citation inside a definition the answer names elsewhere is a deliberate
            # sub-citation (the lines that do the work), not a stray range: keep it.
            named = _resolve(everywhere, path, lookup, (start, end))
            if any(_status((start, end), d) == "inside" for d in named):
                continue
            why = "re-pointed to the definitions named before it"
        elif (
            all(x == "covers" for x in states)
            and (end - start + 1) > WIDE_FACTOR * size + WIDE_SLACK
        ):
            why = "narrowed to the definitions named before it"
        else:
            continue
        ranges = sorted(defs)
        if not all(
            any(c.path == path and _overlap(*r, c.start_line, c.end_line) for c in chunks)
            for r in ranges
        ):
            continue  # the model never saw those lines: leave it alone
        new_ref = ", ".join(f"{path}:{r[0]}-{r[1]}" for r in ranges)
        if new_ref != m.group():
            edits.append((m.start(), m.end(), new_ref))
            result.snaps.append(Snap(m.group(), new_ref, why))
    text = answer
    for a, b, new in reversed(edits):
        text = text[:a] + new + text[b:]
    result.answer = text
    return result
