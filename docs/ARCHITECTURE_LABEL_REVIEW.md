# Architecture label review (for a human)

> **Status: human reviewed (2026-11-18).** The verdicts below were drafted by Claude and accepted by the user as written.
> Sections A and B were checked against the psf/requests clone at `611c6162cb`. Section C was judged from knowledge of
> Flask's layout, without opening the pinned lines (there is no flask clone in `data/repos`).
> Result: 34 of 40 labels kept. Changes: A-11 (drop `adapters.py`), B-004 (drop `status_codes.py`), B-002 (add `models.py`),
> B-006 (add `sessions.py`), B-009 (add `adapters.py`). Need should go from 1 to 2 for A-04 and C-15.
> **The dataset files have not been changed yet,** and the saved runs have not been re-scored.

These are the gold labels behind the Phase 6 architecture-answer check. Before the review above, no person had reviewed any of them: they were written by an AI reading the source (the requests and flask sets say `Codex (source inspection, no retrieval)`; the fresh set was written and checked against the source by the same kind of reader). The pass/fail numbers in `docs/PROJECT_LOG.md` are only as good as these labels.

## What you are being asked to do

For each question, decide whether the listed **gold files** are the right ones. Open the link (it is pinned to the exact commit the label was written against), look at the highlighted lines, and tick one box. You are judging the **label**, not any model's answer.

## How a label turns into a pass or a miss

A model answer passes when its text names **at least half of the gold files** (rounded up; the `need` value below), by file path. It is a coverage proxy: it does not check that the explanation is correct or coherent. So two things make a label matter: a missing gold file can make a good answer miss, and a wrong gold file can make a bad answer pass.

## What is in this file

| Section | Questions | Repo and commit | Used by the Phase 6 gate? |
|---|---|---|---|
| A. Original set | 15 | psf/requests `611c6162cb` | Yes (gate 12 of 15) |
| B. Fresh set | 10 | psf/requests `611c6162cb` | Yes (gate 8 of 10) |
| C. Flask set | 15 | pallets/flask `d086db856b` | No (not indexed for the agent; lower priority) |

**Review A and B first.** They decide whether the Phase 6 result stands. C only matters if flask is indexed later.

## How to send it back

Edit this file (tick boxes, write notes) and tell me, or paste the questions you marked as wrong. I will then change the labels in the dataset files, say which question ids changed, and re-score the saved runs offline (the answers are saved, so no GPU is needed to see the effect on the 13/15 and 8/10 scores).

---

## A. Original set (psf/requests, 15 questions)

Agent-eval ids are `architecture-NNN` (same questions, in the order below).

### requests-architecture-01  ·  agent id `architecture-001`  ·  type `layout`  ·  split `dev`

**Question:** How is the public HTTP interface separated from persistent client state and network transport?

**To pass, an answer must name 2 of these 4 files:**

- `src/requests/__init__.py`  -  anchor `module imports / declarations`, lines 1-100  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/__init__.py#L1-L100)
- `src/requests/api.py`  -  anchor `request`, lines 24-71  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/api.py#L24-L71)
- `src/requests/sessions.py`  -  anchor `Session`, lines 395-905  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L395-L905)
- `src/requests/adapters.py`  -  anchor `HTTPAdapter`, lines 158-748  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L158-L748)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: `__init__`/`api` = public interface, `sessions` = state, `adapters` = transport.

### requests-architecture-02  ·  agent id `architecture-002`  ·  type `flow`  ·  split `dev`

**Question:** Trace a one-off HTTP call from a convenience entry point to the network connection pool.

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/api.py`  -  anchor `request`, lines 24-71  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/api.py#L24-L71)
- `src/requests/sessions.py`  -  anchor `Session.request`, lines 557-653  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L557-L653)
- `src/requests/adapters.py`  -  anchor `HTTPAdapter.send`, lines 634-748  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L634-L748)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: api.request -> Session.request -> HTTPAdapter.send -> urllib3 pool.

### requests-architecture-03  ·  agent id `architecture-003`  ·  type `dependency`  ·  split `dev`

**Question:** Which internal layers must a convenience HTTP entry point rely on to prepare and send a request?

**To pass, an answer must name 2 of these 4 files:**

- `src/requests/api.py`  -  anchor `request`, lines 24-71  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/api.py#L24-L71)
- `src/requests/sessions.py`  -  anchor `Session.request`, lines 557-653  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L557-L653)
- `src/requests/models.py`  -  anchor `PreparedRequest`, lines 378-729  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L378-L729)
- `src/requests/adapters.py`  -  anchor `HTTPAdapter`, lines 158-748  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L158-L748)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: api -> sessions -> models.PreparedRequest -> adapters is the real chain.

### requests-architecture-04  ·  agent id `architecture-004`  ·  type `responsibility`  ·  split `dev`

**Question:** Which part owns request preparation and response representation, and which part coordinates their use?

**To pass, an answer must name 1 of these 2 files:**

- `src/requests/models.py`  -  anchor `PreparedRequest`, lines 378-729  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L378-L729)
- `src/requests/sessions.py`  -  anchor `Session.prepare_request`, lines 511-555  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L511-L555)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Files are right, but need=1 lets an answer naming only models.py pass without saying who coordinates. Raise need to 2.

### requests-architecture-05  ·  agent id `architecture-005`  ·  type `flow`  ·  split `dev`

**Question:** How is a low-level transport response converted into the response object returned to the caller?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/adapters.py`  -  anchor `HTTPAdapter.build_response`, lines 365-401  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L365-L401)
- `src/requests/models.py`  -  anchor `Response`, lines 732-1184  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L732-L1184)
- `src/requests/sessions.py`  -  anchor `Session.send`, lines 752-829  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L752-L829)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: build_response creates the Response; Session.send runs hooks, cookies and redirects on it.

### requests-architecture-06  ·  agent id `architecture-006`  ·  type `responsibility`  ·  split `test`

**Question:** Which components jointly retain cookies between calls and attach them to outgoing requests?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/cookies.py`  -  anchor `RequestsCookieJar`, lines 191-476  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/cookies.py#L191-L476)
- `src/requests/models.py`  -  anchor `PreparedRequest.prepare_cookies`, lines 699-720  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L699-L720)
- `src/requests/sessions.py`  -  anchor `Session.prepare_request`, lines 511-555  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L511-L555)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Cookie jar, prepare_cookies, and the session merging cookies in prepare_request.

### requests-architecture-07  ·  agent id `architecture-007`  ·  type `flow`  ·  split `test`

**Question:** Trace the redirect cycle, including rebuilding credentials and preserving cookies across the next send.

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/sessions.py`  -  anchor `SessionRedirectMixin.resolve_redirects`, lines 186-307  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L186-L307)
- `src/requests/cookies.py`  -  anchor `extract_cookies_to_jar`, lines 135-150  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/cookies.py#L135-L150)
- `src/requests/models.py`  -  anchor `PreparedRequest`, lines 378-729  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L378-L729)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: resolve_redirects calls extract_cookies_to_jar, rebuild_auth and prepare_cookies.

### requests-architecture-08  ·  agent id `architecture-008`  ·  type `dependency`  ·  split `test`

**Question:** How does persistent client configuration depend on the case-insensitive header container and default settings helpers?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/sessions.py`  -  anchor `Session.__init__`, lines 442-503  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L442-L503)
- `src/requests/structures.py`  -  anchor `CaseInsensitiveDict`, lines 20-93  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/structures.py#L20-L93)
- `src/requests/utils.py`  -  anchor `default_headers`, lines 951-962  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py#L951-L962)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Session.__init__ uses default_headers(), which returns a CaseInsensitiveDict.

### requests-architecture-09  ·  agent id `architecture-009`  ·  type `responsibility`  ·  split `test`

**Question:** Where do authentication strategies live, and how do request preparation and redirect handling invoke them?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/auth.py`  -  anchor `AuthBase`, lines 78-82  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/auth.py#L78-L82)
- `src/requests/models.py`  -  anchor `PreparedRequest.prepare_auth`, lines 670-697  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L670-L697)
- `src/requests/sessions.py`  -  anchor `SessionRedirectMixin.rebuild_auth`, lines 309-332  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L309-L332)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: AuthBase, prepare_auth and rebuild_auth are the three places.

### requests-architecture-10  ·  agent id `architecture-010`  ·  type `dependency`  ·  split `test`

**Question:** Which pieces cooperate to choose certificate verification defaults and configure TLS for transport?

**To pass, an answer must name 2 of these 4 files:**

- `src/requests/adapters.py`  -  anchor `HTTPAdapter.cert_verify`, lines 307-363  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L307-L363)
- `src/requests/certs.py`  -  anchor `module imports / declarations`, lines 1-18  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/certs.py#L1-L18)
- `src/requests/utils.py`  -  anchor `DEFAULT_CA_BUNDLE_PATH`, lines 82-82  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py#L82-L82)
- `src/requests/sessions.py`  -  anchor `Session.merge_environment_settings`, lines 831-868  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L831-L868)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: cert_verify, certs.where() -> DEFAULT_CA_BUNDLE_PATH, merge_environment_settings reading REQUESTS_CA_BUNDLE.

### requests-architecture-11  ·  agent id `architecture-011`  ·  type `responsibility`  ·  split `test`

**Question:** Which modules cooperate to encode form and multipart upload bodies before network transmission?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/models.py`  -  anchor `RequestEncodingMixin`, lines 108-251  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L108-L251)
- `src/requests/utils.py`  -  anchor `to_key_val_list`, lines 376-404  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py#L376-L404)
- `src/requests/adapters.py`  -  anchor `HTTPAdapter.send`, lines 634-748  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L634-L748)

**Your review of this label**
- [ ] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [x] A listed file should not be required (name it): `src/requests/adapters.py`
- [ ] The question is unclear, or has no single right answer
- Notes: Encoding happens in models.py (_encode_params, _encode_files, urllib3 encode_multipart_formdata) and utils.py (to_key_val_list, guess_filename). HTTPAdapter.send only passes the finished body on (adapters.py:699). Today models + adapters passes with no encoding helper named.

### requests-architecture-12  ·  agent id `architecture-012`  ·  type `flow`  ·  split `test`

**Question:** Trace streamed response data from the raw connection through content decoding and iteration.

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/adapters.py`  -  anchor `HTTPAdapter.build_response`, lines 365-401  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L365-L401)
- `src/requests/models.py`  -  anchor `Response.iter_content`, lines 914-977  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L914-L977)
- `src/requests/utils.py`  -  anchor `stream_decode_response_unicode`, lines 594-610  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py#L594-L610)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: build_response stores raw; iter_content streams it; stream_decode_response_unicode decodes it.

### requests-architecture-13  ·  agent id `architecture-013`  ·  type `responsibility`  ·  split `test`

**Question:** How are user hooks registered and dispatched around the response lifecycle?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/hooks.py`  -  anchor `dispatch_hook`, lines 32-48  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/hooks.py#L32-L48)
- `src/requests/models.py`  -  anchor `RequestHooksMixin`, lines 254-281  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py#L254-L281)
- `src/requests/sessions.py`  -  anchor `Session.send`, lines 752-829  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py#L752-L829)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: register_hook registers, Session.send calls dispatch_hook (sessions.py:791), hooks.py holds dispatch_hook/default_hooks.

### requests-architecture-14  ·  agent id `architecture-014`  ·  type `layout`  ·  split `test`

**Question:** How are compatibility shims, dependency checks and the public exception hierarchy separated?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/compat.py`  -  anchor `module imports / declarations`, lines 1-100  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/compat.py#L1-L100)
- `src/requests/__init__.py`  -  anchor `check_compatibility`, lines 60-96  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/__init__.py#L60-L96)
- `src/requests/exceptions.py`  -  anchor `RequestException`, lines 20-35  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/exceptions.py#L20-L35)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: compat, check_compatibility and exceptions are the three areas named.

### requests-architecture-15  ·  agent id `architecture-015`  ·  type `dependency`  ·  split `test`

**Question:** Which layers translate connection and timeout failures from the HTTP backend into library-specific errors?

**To pass, an answer must name 1 of these 2 files:**

- `src/requests/adapters.py`  -  anchor `HTTPAdapter.send`, lines 634-748  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py#L634-L748)
- `src/requests/exceptions.py`  -  anchor `ConnectionError`, lines 70-71  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/exceptions.py#L70-L71)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: need=1 is lenient (exceptions.py alone passes) but acceptable: it is the error hierarchy asked about.

---

## B. Fresh set (psf/requests, 10 questions)

Written on 2026-11-10 after the first 15 scored 5/15, on topics the first set does not cover. These labels have **no line anchors**, only file names; the links open the whole file at the pinned commit.

### archfresh-001  ·  type `architecture`

**Question:** Which modules decide which proxy a request goes through, and how is it applied to the connection?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/sessions.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py)
- `src/requests/utils.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py)
- `src/requests/adapters.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: utils decides (resolve_proxies, select_proxy, should_bypass_proxies), sessions merges (rebuild_proxies, merge_environment_settings), adapters applies (proxy_manager_for).

### archfresh-002  ·  type `architecture`

**Question:** How do credentials from a netrc file end up on a request when the caller passes no auth?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/utils.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py)
- `src/requests/sessions.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py)
- `src/requests/auth.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/auth.py)

**Your review of this label**
- [ ] Right: these files are what a good answer must name
- [x] A needed file is missing (name it): `src/requests/models.py`
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: get_netrc_auth -> Session.prepare_request (sessions.py:538) gives a tuple; PreparedRequest.prepare_auth turns it into HTTPBasicAuth (models.py:685). Add as a 4th file, keep need=2.

### archfresh-003  ·  type `architecture`

**Question:** Which parts normalise and validate a URL before a request is sent, and what do they raise?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/models.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py)
- `src/requests/utils.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py)
- `src/requests/exceptions.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/exceptions.py)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: prepare_url in models, requote_uri/unquote_unreserved in utils, MissingSchema/InvalidURL in exceptions.

### archfresh-004  ·  type `architecture`

**Question:** How does an HTTP error status code become an exception for the caller?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/models.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py)
- `src/requests/exceptions.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/exceptions.py)
- `src/requests/status_codes.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/status_codes.py)

**Your review of this label**
- [ ] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [x] A listed file should not be required (name it): `src/requests/status_codes.py`
- [ ] The question is unclear, or has no single right answer
- Notes: raise_for_status compares numeric ranges and never uses `codes` (models only uses codes for redirect sets, lines 96-100, 887). models + status_codes passes with no HTTPError. Use models + exceptions, need=2.

### archfresh-005  ·  type `architecture`

**Question:** How is a response body decoded as JSON, including guessing its encoding, and which errors surface?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/models.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py)
- `src/requests/utils.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py)
- `src/requests/exceptions.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/exceptions.py)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Response.json uses utils.guess_json_utf and raises exceptions.JSONDecodeError. Optionally also accept compat.py (complexjson).

### archfresh-006  ·  type `architecture`

**Question:** How does digest authentication answer a 401 challenge, and how is it attached to the response?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/auth.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/auth.py)
- `src/requests/models.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py)
- `src/requests/hooks.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/hooks.py)

**Your review of this label**
- [ ] Right: these files are what a good answer must name
- [x] A needed file is missing (name it): `src/requests/sessions.py`
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: HTTPDigestAuth registers handle_401 via register_hook (auth.py:339); it runs when Session.send calls dispatch_hook (sessions.py:791). Add as a 4th file, keep need=2.

### archfresh-007  ·  type `architecture`

**Question:** Which parts decide whether a request body is sent with a content length or chunked?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/models.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py)
- `src/requests/utils.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py)
- `src/requests/adapters.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: prepare_body/prepare_content_length, super_len, adapters.py:679 (chunked when no Content-Length).

### archfresh-008  ·  type `architecture`

**Question:** Which components release network connections when a session or a response is closed?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/sessions.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py)
- `src/requests/adapters.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/adapters.py)
- `src/requests/models.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Session.close closes adapters, HTTPAdapter.close clears pools, Response.close calls release_conn.

### archfresh-009  ·  type `architecture`

**Question:** How is the character encoding of a response chosen when its text is read?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/models.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/models.py)
- `src/requests/utils.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py)
- `src/requests/compat.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/compat.py)

**Your review of this label**
- [ ] Right: these files are what a good answer must name
- [x] A needed file is missing (name it): `src/requests/adapters.py`
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: build_response sets response.encoding = get_encoding_from_headers(...) (adapters.py:385), where the encoding is first chosen. compat only resolves the chardet/charset_normalizer import. Add as a 4th file, keep need=2.

### archfresh-010  ·  type `architecture`

**Question:** Where do the default headers such as the user agent come from, and how do they reach a request?

**To pass, an answer must name 2 of these 3 files:**

- `src/requests/utils.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/utils.py)
- `src/requests/sessions.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/sessions.py)
- `src/requests/__version__.py`  -  [open](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/src/requests/__version__.py)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: default_headers/default_user_agent in utils, __version__ for the version string, Session.__init__ + merge_setting in sessions.

---

## C. Flask set (pallets/flask, 15 questions) - lower priority

Not used by the Phase 6 gate: the agent has only been run on psf/requests.

### flask-architecture-01  ·  type `layout`  ·  split `dev`

**Question:** How is the framework divided between reusable setup machinery and the concrete WSGI application?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/app.py`  -  anchor `Flask`, lines 109-1625  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L109-L1625)
- `src/flask/sansio/app.py`  -  anchor `App`, lines 60-1014  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L60-L1014)
- `src/flask/sansio/scaffold.py`  -  anchor `Scaffold`, lines 52-706  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/scaffold.py#L52-L706)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-02  ·  type `dependency`  ·  split `dev`

**Question:** Which internal layers supply the public application object, HTTP wrappers and setup base classes?

**To pass, an answer must name 2 of these 4 files:**

- `src/flask/__init__.py`  -  anchor `module imports / declarations`, lines 1-39  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/__init__.py#L1-L39)
- `src/flask/app.py`  -  anchor `Flask`, lines 109-1625  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L109-L1625)
- `src/flask/wrappers.py`  -  anchor `Request`, lines 18-219  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/wrappers.py#L18-L219)
- `src/flask/sansio/app.py`  -  anchor `App`, lines 60-1014  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L60-L1014)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-03  ·  type `responsibility`  ·  split `dev`

**Question:** Where are method-based views adapted into callable endpoints and registered as routing rules?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/views.py`  -  anchor `View.as_view`, lines 86-135  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/views.py#L86-L135)
- `src/flask/sansio/app.py`  -  anchor `App.add_url_rule`, lines 606-662  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L606-L662)
- `src/flask/sansio/scaffold.py`  -  anchor `Scaffold.route`, lines 344-373  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/scaffold.py#L344-L373)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-04  ·  type `responsibility`  ·  split `dev`

**Question:** Which components connect application template configuration, loader dispatch and rendering helpers?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/templating.py`  -  anchor `DispatchingJinjaLoader`, lines 49-120  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/templating.py#L49-L120)
- `src/flask/app.py`  -  anchor `Flask.create_jinja_environment`, lines 467-505  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L467-L505)
- `src/flask/sansio/scaffold.py`  -  anchor `Scaffold.jinja_loader`, lines 272-282  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/scaffold.py#L272-L282)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-05  ·  type `flow`  ·  split `dev`

**Question:** How do a template rendering call and its default context reach the template environment?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/templating.py`  -  anchor `render_template`, lines 136-148  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/templating.py#L136-L148)
- `src/flask/app.py`  -  anchor `Flask.update_template_context`, lines 588-616  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L588-L616)
- `src/flask/globals.py`  -  anchor `module imports / declarations`, lines 1-77  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/globals.py#L1-L77)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-06  ·  type `layout`  ·  split `test`

**Question:** How is reusable route grouping divided between its public object, shared setup logic and application registration?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/blueprints.py`  -  anchor `Blueprint`, lines 18-128  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/blueprints.py#L18-L128)
- `src/flask/sansio/blueprints.py`  -  anchor `Blueprint`, lines 119-692  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/blueprints.py#L119-L692)
- `src/flask/sansio/app.py`  -  anchor `App.register_blueprint`, lines 571-596  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L571-L596)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-07  ·  type `flow`  ·  split `test`

**Question:** Trace deferred route registration from a reusable route group into the application's URL map.

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/sansio/blueprints.py`  -  anchor `Blueprint.register`, lines 273-377  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/blueprints.py#L273-L377)
- `src/flask/sansio/app.py`  -  anchor `App.register_blueprint`, lines 571-596  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L571-L596)
- `src/flask/sansio/scaffold.py`  -  anchor `Scaffold.route`, lines 344-373  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/scaffold.py#L344-L373)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-08  ·  type `dependency`  ·  split `test`

**Question:** Which common setup layer is shared by the application and its reusable route groups?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/sansio/scaffold.py`  -  anchor `Scaffold`, lines 52-706  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/scaffold.py#L52-L706)
- `src/flask/sansio/app.py`  -  anchor `App`, lines 60-1014  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L60-L1014)
- `src/flask/sansio/blueprints.py`  -  anchor `Blueprint`, lines 119-692  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/blueprints.py#L119-L692)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-09  ·  type `responsibility`  ·  split `test`

**Question:** Which components implement context-local application, request and session access and context cleanup?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/globals.py`  -  anchor `module imports / declarations`, lines 1-77  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/globals.py#L1-L77)
- `src/flask/ctx.py`  -  anchor `AppContext`, lines 260-525  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/ctx.py#L260-L525)
- `src/flask/app.py`  -  anchor `Flask.do_teardown_request`, lines 1420-1451  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L1420-L1451)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened. Assumes the merged AppContext model (no separate RequestContext); confirm at d086db856b.

### flask-architecture-10  ·  type `flow`  ·  split `test`

**Question:** How does the request lifecycle open a signed cookie session and save it back on the response?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/ctx.py`  -  anchor `AppContext._get_session`, lines 381-393  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/ctx.py#L381-L393)
- `src/flask/sessions.py`  -  anchor `SecureCookieSessionInterface`, lines 284-385  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sessions.py#L284-L385)
- `src/flask/app.py`  -  anchor `Flask.process_response`, lines 1394-1418  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L1394-L1418)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened. Assumes the merged AppContext model (no separate RequestContext); confirm at d086db856b.

### flask-architecture-11  ·  type `flow`  ·  split `test`

**Question:** Trace an incoming WSGI request through context creation, view dispatch and response conversion.

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/app.py`  -  anchor `Flask.wsgi_app`, lines 1566-1616  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L1566-L1616)
- `src/flask/ctx.py`  -  anchor `AppContext.from_environ`, lines 340-348  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/ctx.py#L340-L348)
- `src/flask/wrappers.py`  -  anchor `Response`, lines 222-257  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/wrappers.py#L222-L257)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened. Assumes the merged AppContext model (no separate RequestContext); confirm at d086db856b.

### flask-architecture-12  ·  type `responsibility`  ·  split `test`

**Question:** Where do error-handler registration, exception dispatch and failure signals fit together?

**To pass, an answer must name 2 of these 4 files:**

- `src/flask/sansio/scaffold.py`  -  anchor `Scaffold.register_error_handler`, lines 650-662  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/scaffold.py#L650-L662)
- `src/flask/sansio/app.py`  -  anchor `App._find_error_handler`, lines 869-892  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L869-L892)
- `src/flask/app.py`  -  anchor `Flask.handle_exception`, lines 897-948  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L897-L948)
- `src/flask/signals.py`  -  anchor `module imports / declarations`, lines 1-17  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/signals.py#L1-L17)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-13  ·  type `dependency`  ·  split `test`

**Question:** Which components connect URL-building helpers to the active application and its routing map?

**To pass, an answer must name 2 of these 4 files:**

- `src/flask/helpers.py`  -  anchor `url_for`, lines 200-251  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/helpers.py#L200-L251)
- `src/flask/app.py`  -  anchor `Flask.url_for`, lines 1102-1222  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L1102-L1222)
- `src/flask/globals.py`  -  anchor `module imports / declarations`, lines 1-77  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/globals.py#L1-L77)
- `src/flask/sansio/app.py`  -  anchor `App.add_url_rule`, lines 606-662  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/sansio/app.py#L606-L662)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-14  ·  type `responsibility`  ·  split `test`

**Question:** How are application defaults, configuration loading and request-specific limits divided?

**To pass, an answer must name 2 of these 3 files:**

- `src/flask/app.py`  -  anchor `Flask`, lines 109-1625  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L109-L1625)
- `src/flask/config.py`  -  anchor `Config`, lines 50-367  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/config.py#L50-L367)
- `src/flask/wrappers.py`  -  anchor `Request.max_content_length`, lines 89-90  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/wrappers.py#L89-L90)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened.

### flask-architecture-15  ·  type `flow`  ·  split `test`

**Question:** Trace command-line application discovery into running the development server.

**To pass, an answer must name 1 of these 2 files:**

- `src/flask/cli.py`  -  anchor `ScriptInfo.load_app`, lines 333-372  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/cli.py#L333-L372)
- `src/flask/app.py`  -  anchor `Flask.run`, lines 630-753  -  [open](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/src/flask/app.py#L630-L753)

**Your review of this label**
- [x] Right: these files are what a good answer must name
- [ ] A needed file is missing (name it):
- [ ] A listed file should not be required (name it):
- [ ] The question is unclear, or has no single right answer
- Notes: Judged from knowledge of Flask's layout; no flask clone, pinned lines not opened. Files are right, but need=1 lets an answer naming only app.py pass. Raise need to 2, or swap Flask.run for cli.py run_command.

