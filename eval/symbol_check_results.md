# Phase 6 symbol table: caller check against `git grep` (psf/requests)

Reproduce: `uv run repo-rag index-symbols psf/requests --checkout data/repos/psf__requests`, then
`uv run python eval/check_callers.py psf/requests data/repos/psf__requests <symbols>`.
2026-10-31, 807 definitions and 2,660 call sites from 35 files (3 s on CPU).

`git grep` lists every line with `name(` that is not a definition. The table must contain all of
them (recall) and nothing else (precision on call sites).

| Symbol | grep lines | table rows | grep-only | table-only | rows better than "name only" |
|---|---|---|---|---|---|
| `merge_setting` | 8 | 8 | 0 | 0 | 8 |
| `super_len` | 16 | 16 | 0 | 0 | 16 |
| `get_netrc_auth` | 5 | 5 | 0 | 0 | 5 |
| `HTTPAdapter.build_response` | 2 | 2 | 0 | 0 | 1 |
| `Session.prepare_request` | 13 | 13 | 0 | 0 | 7 |
| `dict_from_cookiejar` | 0 | 0 | 0 | 0 | 0 |
| `HTTPAdapter.send` | 45 | 43 | 2 | 0 | 1 |

The two `HTTPAdapter.send` grep-only lines are a docstring ("from the call to send()",
`adapters.py:609`) and a doctest (`>>> s.send(r)`, `models.py:395`): text, not calls, so excluding
them is right.

**What this does and does not show.** Every real call site of the name is found. Whether a
given row calls *that* function is a separate question: matching is by name, so for a method
name shared by several classes (`send`: 43 rows, 1 better than "name only") most rows are
`name only` and may call a different class's method. The label is returned with every row.
Unique module-level functions resolve well (`imported`/`same file`); `self.method()` calls
resolve to the defining class (`same class`); `adapter.send` ranks as `receiver name`.


## find_callees hand check (2026-11-02)

`repo-rag callees Session.send --repo psf/requests --path src/requests/sessions.py --external`
against the 22 call expressions in the function body (`sessions.py` lines 752-830, read by hand):
all 22 appear (9 resolved to a repo definition, 13 external: `kwargs.setdefault` x3, `isinstance`,
`ValueError`, `kwargs.pop`, `preferred_clock` x2, `timedelta`, `history.insert`, `history.pop`,
`next`). Resolved rows: `resolve_proxies`, `dispatch_hook`, `extract_cookies_to_jar` x2 (imported);
`self.get_adapter` (same class); `adapter.send` -> `BaseAdapter.send` (receiver name);
`self.resolve_redirects` x2 (name only: defined in the parent class `SessionRedirectMixin`, and
the name is unique, so it is correct, but the matcher cannot see inheritance).

Two defects found and what happened:

* `_is_prepared(request)` was reported as external. It is `from ._types import is_prepared as
  _is_prepared`. Fixed: import aliases are resolved in both `find_callees` and `find_callers`
  (callers of `is_prepared` previously missed every aliased call). Test:
  `test_aliased_imports_resolve_in_callers_and_callees`.
* `kwargs.get(...)` is matched by name to `get` in `api.py` (`name only`). This is wrong, and is the
  known limit of name-based matching: a receiver that is a plain variable cannot be typed. The
  `how` column marks it; not fixed.

Not handled: inheritance (parent-class methods called through `self`) and dynamic dispatch.
