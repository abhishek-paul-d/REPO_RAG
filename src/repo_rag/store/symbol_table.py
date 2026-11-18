"""Symbol table in metadata.db: definitions and call sites, queried by `get_symbol`/`find_callers`.

Rows come from `chunking/symbols.py`. Call resolution is by name, so `find_callers` returns
candidates ranked by how well they match and says why (`how`); it never claims certainty it
doesn't have. Query handles open the database read-only and expose only typed methods.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from repo_rag.chunking.symbols import FileSymbols

_SCHEMA = """
CREATE TABLE IF NOT EXISTS code_symbols (
 repo TEXT NOT NULL, path TEXT NOT NULL, language TEXT NOT NULL, qualname TEXT NOT NULL,
 name TEXT NOT NULL, kind TEXT NOT NULL, parent TEXT NOT NULL, signature TEXT NOT NULL,
 start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
 PRIMARY KEY(repo, path, qualname, start_line));
CREATE INDEX IF NOT EXISTS code_symbols_name ON code_symbols(repo, name);
CREATE TABLE IF NOT EXISTS code_calls (
 repo TEXT NOT NULL, path TEXT NOT NULL, caller TEXT NOT NULL, callee TEXT NOT NULL,
 qualifier TEXT NOT NULL, line INTEGER NOT NULL,
 PRIMARY KEY(repo, path, caller, callee, qualifier, line));
CREATE INDEX IF NOT EXISTS code_calls_callee ON code_calls(repo, callee);
CREATE TABLE IF NOT EXISTS code_imports (
 repo TEXT NOT NULL, path TEXT NOT NULL, bound TEXT NOT NULL, module TEXT NOT NULL,
 original TEXT NOT NULL, PRIMARY KEY(repo, path, bound, module, original));
"""

_SELF_QUALIFIERS = frozenset({"self", "cls", "this", "super()", "super"})

# Best match first. A call may fit several candidate definitions; the best label wins.
HOW_SAME_CLASS = "same class"  # self/cls/this.f() inside the class that defines f
HOW_IMPORTED = "imported"  # f() where the file imports f from the module that defines it
HOW_QUALIFIED = "qualified"  # Class.f() or module.f() naming the definition's class/file
HOW_SAME_FILE = "same file"  # a bare f() in the file that defines f
HOW_RECEIVER = "receiver name"  # x.f() where x looks like the defining class (adapter, Adapter)
HOW_NAME_ONLY = "name only"  # same last name somewhere else; may be a different function
HOW_EXTERNAL = "external"  # no definition with this name in the repo (library, builtin)
_HOW_ORDER = (
    HOW_SAME_CLASS,
    HOW_IMPORTED,
    HOW_QUALIFIED,
    HOW_SAME_FILE,
    HOW_RECEIVER,
    HOW_NAME_ONLY,
)
# Languages where a bare `f()` cannot call a method (it needs `self.`/`this.`/a receiver).
_BARE_CALL_NEVER_METHOD = frozenset({"python", "javascript", "typescript", "tsx", "go"})
_RECEIVER_MIN_CHARS = 3

# (path, bound name) -> (module as written, imported name or "" for a module import)
Imports = dict[tuple[str, str], tuple[str, str]]


@dataclass(frozen=True)
class SymbolRow:
    qualname: str
    kind: str
    path: str
    start_line: int
    end_line: int
    signature: str
    parent: str
    language: str


@dataclass(frozen=True)
class CallerRow:
    caller: str  # qualified name of the calling definition, "<module>" for top-level code
    path: str
    line: int
    calls: str  # qualname of the definition it most plausibly calls
    expression: str  # the call as written: "self.client.send"
    how: str  # one of the HOW_* labels
    same_name_definitions: int  # how many definitions in the repo share the callee's name


@dataclass(frozen=True)
class CalleeRow:
    inside: str  # qualified name of the definition whose body contains the call
    line: int
    expression: str  # the call as written: "self.client.send"
    resolves_to: str  # qualname of the definition it most plausibly calls, "" if none in the repo
    resolves_path: str  # file of that definition, "" if none
    how: str  # one of the HOW_* labels, or HOW_EXTERNAL when no definition has that name
    same_name_definitions: int  # how many definitions in the repo share the callee's name


def _module_stem(path: str) -> str:
    """`src/requests/utils.py` and `.utils` and `requests.utils` all give `utils`."""
    path = path.replace(".", "/") if "/" not in path and not path.endswith(".py") else path
    stem = Path(path).stem
    return Path(path).parent.name if stem == "__init__" else stem


def _is_test_path(path: str) -> bool:
    parts = path.lower().split("/")
    return any(p in ("test", "tests", "testing") for p in parts[:-1]) or parts[-1].startswith(
        ("test_", "conftest")
    )


is_test_path = _is_test_path  # public name for other modules


def _how(call: sqlite3.Row, target: SymbolRow, imports: Imports) -> str:
    qualifier = call["qualifier"]
    target_module = _module_stem(target.path)
    if not qualifier:
        module, original = imports.get((call["path"], call["callee"]), ("", "?"))
        if original == target.qualname.rpartition(".")[2] and _module_stem(module) == target_module:
            return HOW_IMPORTED
    else:
        module, original = imports.get((call["path"], qualifier.split(".")[0]), ("", "?"))
        if original == "" and module and _module_stem(module) == target_module:
            return HOW_QUALIFIED
        if qualifier in _SELF_QUALIFIERS and call["path"] == target.path and target.parent:
            caller_parent = call["caller"].rpartition(".")[0]
            if caller_parent == target.parent or caller_parent.startswith(target.parent + "."):
                return HOW_SAME_CLASS
    last = qualifier.rpartition(".")[2]
    owner = target.parent.rpartition(".")[2]
    if qualifier and owner and last == owner:
        return HOW_QUALIFIED
    if not qualifier and call["path"] == target.path:
        return HOW_SAME_FILE
    if len(last) >= _RECEIVER_MIN_CHARS and owner and last.lower().strip("_") in owner.lower():
        return HOW_RECEIVER
    return HOW_NAME_ONLY


class SymbolTable:
    def __init__(self, path: Path, *, readonly: bool = True) -> None:
        self.readonly = readonly
        if readonly:
            self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
            self.connection.executescript(_SCHEMA)
        self.connection.row_factory = sqlite3.Row

    def __enter__(self) -> SymbolTable:
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def replace_repo(self, repo: str, files: Mapping[str, tuple[str, FileSymbols]]) -> None:
        """Atomically replace one repo's rows. `files` maps path -> (language, symbols)."""
        if self.readonly:
            raise PermissionError("SymbolTable is open read-only")
        with self.connection:
            for table in ("code_symbols", "code_calls", "code_imports"):
                self.connection.execute(f"DELETE FROM {table} WHERE repo=?", (repo,))
            for path, (language, symbols) in files.items():
                self.connection.executemany(
                    "INSERT OR REPLACE INTO code_symbols VALUES(?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            repo,
                            path,
                            language,
                            d.qualname,
                            d.name,
                            d.kind,
                            d.parent,
                            d.signature,
                            d.start_line,
                            d.end_line,
                        )
                        for d in symbols.definitions
                    ],
                )
                self.connection.executemany(
                    "INSERT OR REPLACE INTO code_calls VALUES(?,?,?,?,?,?)",
                    [(repo, path, c.caller, c.callee, c.qualifier, c.line) for c in symbols.calls],
                )
                self.connection.executemany(
                    "INSERT OR REPLACE INTO code_imports VALUES(?,?,?,?,?)",
                    [(repo, path, i.bound, i.module, i.original) for i in symbols.imports],
                )

    def counts(self, repo: str) -> tuple[int, int]:
        """(definitions, call sites) stored for this repo."""
        defs = self.connection.execute(
            "SELECT count(*) FROM code_symbols WHERE repo=?", (repo,)
        ).fetchone()[0]
        calls = self.connection.execute(
            "SELECT count(*) FROM code_calls WHERE repo=?", (repo,)
        ).fetchone()[0]
        return defs, calls

    @staticmethod
    def _row(r: sqlite3.Row) -> SymbolRow:
        return SymbolRow(
            r["qualname"],
            r["kind"],
            r["path"],
            r["start_line"],
            r["end_line"],
            r["signature"],
            r["parent"],
            r["language"],
        )

    def get_symbol(
        self, repo: str, symbol: str, *, path: str | None = None, limit: int = 20
    ) -> list[SymbolRow]:
        """Definitions named `symbol`: a qualified name (`Session.send`) or a bare name (`send`).

        A bare or partly qualified name matches every definition whose qualified name ends with
        it, so a name shared by several definitions returns all of them.
        """
        symbol = symbol.strip()
        if not symbol or limit < 1:
            raise ValueError("symbol must be non-empty and limit positive")
        name = symbol.rpartition(".")[2]
        clauses, params = ["repo=?", "name=?"], [repo, name]
        if path:
            clauses.append("path=?")
            params.append(path.replace("\\", "/").strip("/"))
        rows = self.connection.execute(
            f"SELECT * FROM code_symbols WHERE {' AND '.join(clauses)} "
            "ORDER BY qualname, path, start_line",
            params,
        ).fetchall()
        suffix = "." + symbol
        found = [r for r in rows if r["qualname"] == symbol or r["qualname"].endswith(suffix)]
        return [self._row(r) for r in found[:limit]]

    def find_callers(
        self, repo: str, symbol: str, *, path: str | None = None, limit: int = 50
    ) -> list[CallerRow]:
        """Call sites that plausibly call `symbol`, best match first (non-test code before tests).

        Matching is by name (see module docstring). `how` says why each row matched, and
        `same_name_definitions` > 1 warns that a name-only match could be another function.
        """
        targets = self.get_symbol(repo, symbol, path=path, limit=1000)
        if not targets or limit < 1:
            return []
        names = sorted({t.qualname.rpartition(".")[2] for t in targets})
        marks = ",".join("?" * len(names))
        # `from m import helper as h` makes `h(x)` a call of `helper`: find those call sites too.
        aliases = {
            (r["path"], r["bound"]): r["original"]
            for r in self.connection.execute(
                "SELECT path, bound, original FROM code_imports "
                f"WHERE repo=? AND original IN ({marks}) AND bound != original",
                [repo, *names],
            )
        }
        bound_names = sorted({bound for _, bound in aliases})
        all_names = [*names, *bound_names]
        calls = [
            c
            for c in self.connection.execute(
                "SELECT * FROM code_calls WHERE repo=? AND callee IN "
                f"({','.join('?' * len(all_names))}) "
                "ORDER BY path, line, caller",
                [repo, *all_names],
            ).fetchall()
            if c["callee"] in names or (not c["qualifier"] and (c["path"], c["callee"]) in aliases)
        ]
        shared = {
            name: self.connection.execute(
                "SELECT count(*) FROM code_symbols WHERE repo=? AND name=?", (repo, name)
            ).fetchone()[0]
            for name in names
        }
        imports: Imports = {
            (r["path"], r["bound"]): (r["module"], r["original"])
            for r in self.connection.execute(
                "SELECT path, bound, module, original FROM code_imports WHERE repo=?", (repo,)
            )
        }
        rows: list[tuple[int, CallerRow]] = []
        for call in calls:
            effective = (
                aliases.get((call["path"], call["callee"]), call["callee"])
                if not call["qualifier"] and call["callee"] not in names
                else call["callee"]
            )
            options = [t for t in targets if t.qualname.rpartition(".")[2] == effective]
            if not call["qualifier"]:  # a bare f() cannot be a method call in these languages
                options = [
                    t
                    for t in options
                    if not (t.kind == "method" and t.language in _BARE_CALL_NEVER_METHOD)
                ]
            if not options:
                continue
            rank, best = min(
                ((_HOW_ORDER.index(_how(call, t, imports)), t) for t in options),
                key=lambda pair: pair[0],
            )
            expression = (
                f"{call['qualifier']}.{call['callee']}" if call["qualifier"] else call["callee"]
            )
            rows.append(
                (
                    rank,
                    CallerRow(
                        call["caller"],
                        call["path"],
                        call["line"],
                        best.qualname,
                        expression,
                        _HOW_ORDER[rank],
                        shared[effective],
                    ),
                )
            )
        rows.sort(key=lambda r: (r[0], _is_test_path(r[1].path), r[1].path, r[1].line))
        return [row for _, row in rows[:limit]]

    def find_callees(
        self, repo: str, symbol: str, *, path: str | None = None, limit: int = 100
    ) -> list[CalleeRow]:
        """Calls made inside the definition(s) of `symbol`, in source order.

        Each call is matched to the most plausible definition by the same rules as
        `find_callers`. A call whose name matches no definition in the repo is `external`
        (a library or builtin). For a class this includes calls inside its methods, and for any
        definition it includes calls inside nested definitions.
        """
        targets = self.get_symbol(repo, symbol, path=path, limit=1000)
        if not targets or limit < 1:
            return []
        imports: Imports = {
            (r["path"], r["bound"]): (r["module"], r["original"])
            for r in self.connection.execute(
                "SELECT path, bound, module, original FROM code_imports WHERE repo=?", (repo,)
            )
        }
        definitions: dict[str, list[SymbolRow]] = {}
        rows: list[tuple[str, int, CalleeRow]] = []
        for target in targets:
            calls = self.connection.execute(
                "SELECT * FROM code_calls WHERE repo=? AND path=? AND (caller=? OR caller LIKE ?) "
                "ORDER BY line, callee",
                (repo, target.path, target.qualname, target.qualname.replace("%", "") + ".%"),
            ).fetchall()
            for call in calls:
                name = call["callee"]
                if not call["qualifier"]:
                    _, original = imports.get((call["path"], name), ("", ""))
                    name = original or name  # `import helper as h` then `h()` calls `helper`
                if name not in definitions:
                    definitions[name] = [
                        self._row(r)
                        for r in self.connection.execute(
                            "SELECT * FROM code_symbols WHERE repo=? AND name=?", (repo, name)
                        )
                    ]
                options = definitions[name]
                if not call["qualifier"]:
                    options = [
                        t
                        for t in options
                        if not (t.kind == "method" and t.language in _BARE_CALL_NEVER_METHOD)
                    ]
                written = call["callee"]
                expression = f"{call['qualifier']}.{written}" if call["qualifier"] else written
                if not options:
                    row = CalleeRow(
                        call["caller"],
                        call["line"],
                        expression,
                        "",
                        "",
                        HOW_EXTERNAL,
                        len(definitions[name]),
                    )
                else:
                    rank, best = min(
                        ((_HOW_ORDER.index(_how(call, t, imports)), t) for t in options),
                        key=lambda pair: pair[0],
                    )
                    row = CalleeRow(
                        call["caller"],
                        call["line"],
                        expression,
                        best.qualname,
                        best.path,
                        _HOW_ORDER[rank],
                        len(definitions[name]),
                    )
                rows.append((target.path, call["line"], row))
        rows.sort(key=lambda r: (r[0], r[1], r[2].expression))
        return [row for _, _, row in rows[:limit]]
