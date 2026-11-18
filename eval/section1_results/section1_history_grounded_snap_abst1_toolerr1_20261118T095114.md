# Section 1 sample questions: history_grounded_snap_abst1_toolerr1

32 questions, 3 flagged by the automatic checks. Read the flagged ones first; an unflagged answer has only passed the citation checks, not been verified.

| id | category | flags |
|---|---|---|
| arch-1 | architecture | none |
| arch-2 | architecture | none |
| arch-3 | architecture | none |
| arch-4 | architecture | none |
| explain-1 | explain | none |
| explain-2 | explain | none |
| explain-3 | explain | none |
| explain-4 | explain | none |
| bug-1 | bug report | none |
| bug-2 | bug report | none |
| bug-3 | bug report | none |
| nav-1 | navigational | 1 |
| nav-2 | navigational | none |
| impact-1 | impact | none |
| impact-2 | impact | none |
| hist-1 | change history | none |
| hist-2 | change history | none |
| people-1 | people | none |
| people-2 | people | none |
| people-3 | people | none |
| people-4 | people | none |
| pr-1 | pull requests | none |
| pr-2 | pull requests | none |
| pr-3 | pull requests | 1 |
| issue-1 | issues | none |
| rel-1 | releases | 1 |
| rel-2 | releases | none |
| rel-3 | releases | none |
| stats-1 | stats | none |
| stats-2 | stats | none |
| onboard-1 | onboarding | none |
| oos-1 | out of scope | none |

## arch-1 (architecture, code)

**Q:** Give me a high-level overview of this repo's architecture. What are the main modules and how do they talk to each other?

**A:** This is the psf/requests library, a Python HTTP client. Its architecture is organized around a few core modules under src/requests/. The public API lives in src/requests/api.py:1-21, which implements the top-level request functions (get, post, etc.) by delegating to a Session. The Session class in src/requests/sessions.py:1-73 manages and persists settings across requests (cookies, auth, proxies) and is the central orchestrator: it builds a PreparedRequest, dispatches hooks, and hands the request to a transport adapter. The transport layer is in src/requests/adapters.py:158-199, where HTTPAdapter (a BaseAdapter) provides the general-case interface for contacting HTTP/HTTPS URLs by wrapping urllib3 connection pools. Supporting modules include src/requests/models.py (Request/Response/PreparedRequest, re-exported in sessions.py), src/requests/cookies.py (cookie handling), src/requests/auth.py (authentication), src/requests/exceptions.py (error hierarchy), src/requests/status_codes.py (HTTP status codes), src/requests/structures.py (CaseInsensitiveDict), and src/requests/utils.py (helpers like proxy resolution and header defaults). Compatibility and packaging concerns are handled in src/requests/compat.py:1-82 (urllib3 version detection, character-detection library resolution, json/simplejson) and src/requests/packages.py:1-23 (backwards-compatible re-exposure of urllib3/idna/chardet). The package entry point src/requests/__init__.py:188-219 re-exports the public names (Session, Response, the verb functions, codes, etc.). In short, the flow is: api.py -> Session (sessions.py) -> HTTPAdapter (adapters.py) -> urllib3, with models, cookies, auth, exceptions, status_codes, structures and utils providing the data types and helpers that these layers share.

**Sources shown to the model:**
- src/requests/sessions.py:1-73 <module>
- src/requests/__init__.py:188-219 <module>
- docs/conf.py:108-158 <module>
- src/requests/packages.py:1-23 <module>
- src/requests/compat.py:1-82 _resolve_char_detection
- src/requests/api.py:1-21 <module>
- src/requests/help.py:67-132 info
- src/requests/adapters.py:158-199 HTTPAdapter

Tools: search_code; 27.5 s

## arch-2 (architecture, code)

**Q:** How does a request flow from Session.request to the network connection?

**A:** A request flows from Session.request to the network connection as follows: Session.request builds a PreparedRequest via prepare_request, which merges the request's settings with the session's settings (cookies, auth, headers, params, hooks) and calls PreparedRequest.prepare (src/requests/sessions.py:505-555). It then calls Session.send with that PreparedRequest (src/requests/sessions.py:752-799). In send, defaults for stream/verify/cert are set, proxies are resolved, the appropriate adapter is chosen via self.get_adapter(url=request.url), and the actual network transmission happens through adapter.send(request, **kwargs) (src/requests/sessions.py:752-799). The HTTPAdapter.send method performs the actual send over the connection pool (src/requests/adapters.py:634-679). After the response returns, send dispatches response hooks, persists cookies, and if allow_redirects is set, resolves redirects via resolve_redirects, which may call send again for each redirect (src/requests/sessions.py:801-829 and src/requests/sessions.py:237-307).

**Sources shown to the model:**
- src/requests/sessions.py:505-555 Session.__enter__
- src/requests/sessions.py:370-392 SessionRedirectMixin.rebuild_method
- src/requests/api.py:48-71 request
- src/requests/sessions.py:908-920 session
- src/requests/sessions.py:752-799 Session.send
- src/requests/models.py:378-422 PreparedRequest.__init__
- src/requests/adapters.py:634-679 HTTPAdapter.send
- src/requests/cookies.py:114-161 MockResponse

Tools: search_code, search_code, search_code; 12.1 s

## arch-3 (architecture, code)

**Q:** Where is authentication handled, and what happens to the request before it is sent?

**A:** Authentication is handled in src/requests/auth.py, which contains the authentication handlers (src/requests/auth.py:1-31). Before a request is sent, the auth data is prepared by PreparedRequest.prepare_auth, which sets the Authorization header (src/requests/models.py:654-697). For example, HTTPBasicAuth.__call__ attaches the Authorization header to the PreparedRequest (src/requests/auth.py:85-121), and HTTPProxyAuth.__call__ attaches the Proxy-Authorization header (src/requests/auth.py:85-121). On redirects, SessionRedirectMixin.rebuild_auth strips or reapplies authentication to avoid leaking credentials (src/requests/sessions.py:309-332).

**Sources shown to the model:**
- src/requests/auth.py:85-121 HTTPBasicAuth
- src/requests/sessions.py:309-332 SessionRedirectMixin.rebuild_auth
- src/requests/auth.py:124-155 HTTPDigestAuth.__init__
- src/requests/auth.py:1-31 <module>
- src/requests/sessions.py:442-488 Session.__init__
- src/requests/models.py:654-697 PreparedRequest.prepare_content_length
- src/requests/adapters.py:565-611 HTTPAdapter.request_url
- src/requests/cookies.py:31-87 MockRequest.__init__

Tools: search_code; 4.8 s

## arch-4 (architecture, code)

**Q:** What design pattern do the transport adapters use?

**A:** The transport adapters use the Strategy pattern: a common BaseAdapter interface defines send and close, and concrete adapters like HTTPAdapter implement it so a Session can swap the transport strategy per URL prefix (src/requests/adapters.py:122-155, src/requests/adapters.py:158-199, src/requests/sessions.py:888-905).

**Sources shown to the model:**
- src/requests/adapters.py:122-155 BaseAdapter
- src/requests/adapters.py:158-199 HTTPAdapter
- src/requests/sessions.py:888-905 Session.mount
- src/requests/adapters.py:1-67 <module>
- src/requests/sessions.py:831-886 Session.merge_environment_settings
- src/requests/auth.py:85-121 HTTPBasicAuth
- src/requests/models.py:72-105 <module>
- src/requests/utils.py:85-93 <module>

Tools: search_code; 4.0 s

## explain-1 (explain, code)

**Q:** Explain what `rebuild_auth` does, step by step, and why it might be written this way.

**A:** rebuild_auth is a method on SessionRedirectMixin that handles authentication when a request is redirected, to avoid leaking credentials while preserving them where safe (src/requests/sessions.py:309-332). Step by step: (1) It takes the prepared redirect request and the response that caused the redirect, and asserts both are prepared, then reads the original request from response.request (src/requests/sessions.py:309-332). (2) It extracts the prepared request's headers and both the original and new URLs (src/requests/sessions.py:309-332). (3) If the request has an Authorization header and should_strip_auth(original_url, url) is true — meaning the redirect crosses to a different host — it deletes the Authorization header so credentials are not sent to a new host (src/requests/sessions.py:309-332). (4) It then checks .netrc for credentials for the new host via get_netrc_auth(url), but only if self.trust_env is set; if found, it reapplies them with prepared_request.prepare_auth(new_auth) (src/requests/sessions.py:309-332). It is written this way because a redirect to a different origin could otherwise leak the original credentials to an untrusted host, so the method strips auth on cross-host redirects while still allowing .netrc-based auth to be re-applied for the new host when the environment is trusted (src/requests/sessions.py:309-332).

**Sources shown to the model:**
- src/requests/sessions.py:309-332 SessionRedirectMixin.rebuild_auth
- src/requests/auth.py:321-354 HTTPDigestAuth.__call__
- src/requests/auth.py:124-155 HTTPDigestAuth.__init__
- src/requests/auth.py:268-319 HTTPDigestAuth.handle_redirect
- src/requests/auth.py:85-121 HTTPBasicAuth
- src/requests/models.py:654-697 PreparedRequest.prepare_content_length
- src/requests/adapters.py:403-440 HTTPAdapter.build_connection_pool_key_attributes
- src/requests/utils.py:1040-1095 prepend_scheme_if_needed

Tools: search_code; 6.9 s

## explain-2 (explain, code)

**Q:** Who calls `get_environ_proxies`, and what happens if it raises?

**A:** get_environ_proxies is called by Session.merge_environment_settings (src/requests/sessions.py:831-886), by resolve_proxies (src/requests/utils.py:911-962), and by the tests TestGetEnvironProxies.test_bypass, test_not_bypass, test_bypass_no_proxy_keyword, and test_not_bypass_no_proxy_keyword (tests/test_utils.py:242, 253, 265, 282). If it raises, the exception is not caught by any of these callers: in merge_environment_settings the call is made directly inside the trust_env branch (src/requests/sessions.py:831-886) and in resolve_proxies it is called directly when trust_env is true and the URL is not bypassed (src/requests/utils.py:911-962), so the exception propagates up to the caller of those functions. The function itself (src/requests/utils.py:873-908) only returns {} when should_bypass_proxies is true, otherwise returns getproxies(); it has no internal try/except, so any error from getproxies() or should_bypass_proxies propagates unchanged.

**Sources shown to the model:**
- src/requests/utils.py:873-908 get_environ_proxies
- src/requests/utils.py:911-962 resolve_proxies
- src/requests/sessions.py:831-886 Session.merge_environment_settings
- src/requests/utils.py:810-870 should_bypass_proxies
- src/requests/sessions.py:334-368 SessionRedirectMixin.rebuild_proxies
- src/requests/adapters.py:512-563 HTTPAdapter.get_connection
- src/requests/compat.py:83-115 <module>
- src/requests/__init__.py:188-219 <module>

Tools: find_callers, search_code; 6.8 s

## explain-3 (explain, code)

**Q:** What's the difference between `Session.send` and `Session.request`?

**A:** Session.request is the high-level entry point: it takes a method, URL and optional parameters (params, data, json, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert), constructs a Request, prepares it with the session's settings, and then sends it, returning a Response (src/requests/sessions.py:557-600). Session.send is the lower-level method: it accepts an already-constructed PreparedRequest (raising ValueError if given a plain Request), applies session defaults (stream, verify, cert, proxies), dispatches the request through the appropriate adapter, runs response hooks, persists cookies, and resolves redirects, returning a Response (src/requests/sessions.py:752-829). In short, request builds and prepares the request for you, while send transmits a PreparedRequest you have already prepared (src/requests/models.py:378-422).

**Sources shown to the model:**
- src/requests/sessions.py:752-799 Session.send
- src/requests/sessions.py:801-829 Session.send
- src/requests/sessions.py:695-740 Session.post
- src/requests/sessions.py:505-555 Session.__enter__
- src/requests/sessions.py:557-600 Session.request
- src/requests/api.py:48-71 request
- src/requests/models.py:378-422 PreparedRequest.__init__
- src/requests/adapters.py:634-679 HTTPAdapter.send

Tools: search_code; 6.0 s

## explain-4 (explain, code)

**Q:** `super_len` has no docstring. Write one based on how it's used.

**A:** """Return the length of a request body, supporting strings, bytes, file-like objects, and other objects with a known size. It determines the total length via __len__, a .len attribute, or os.fstat on a fileno, then subtracts the current position (from .tell(), or by seeking to the end for seekable objects) and returns max(0, total_length - current_position), falling back to 0 when the length cannot be determined (src/requests/utils.py:160-228). It is used to compute the Content-Length header for request bodies (src/requests/models.py:654-697, src/requests/models.py:603-652)."""

**Sources shown to the model:**
- src/requests/utils.py:160-199 super_len
- src/requests/utils.py:201-228 super_len
- src/requests/models.py:654-697 PreparedRequest.prepare_content_length
- src/requests/models.py:72-105 <module>
- src/requests/models.py:603-652 PreparedRequest.prepare_body
- src/requests/_types.py:1-64 SupportsRead
- src/requests/auth.py:34-82 _basic_auth_str
- src/requests/compat.py:1-82 _resolve_char_detection

Tools: get_symbol, search_code; 5.6 s

## bug-1 (bug report, exact)

**Q:** Users report that response.text shows garbled characters for a page with no charset in its headers. Has something like this been fixed before?

**A:** Yes. When no charset is present in the headers, Response.text falls back to an auto-detected encoding: if self.encoding is None it uses self.apparent_encoding (guessed via charset_normalizer or chardet), then decodes with errors="replace" (src/requests/models.py:1034-1089). This behavior was added in a fix noted in the history: 'If no Response.encoding is specified and chardet is available, Response.text will guess an encoding' (commit 497695864b48).

**Sources shown to the model:**
- src/requests/models.py:1034-1089 Response.content
- src/requests/utils.py:1008-1037 guess_json_utf
- src/requests/utils.py:569-630 get_encoding_from_headers
- src/requests/utils.py:522-566 get_encodings_from_content
- src/requests/help.py:1-32 <module>
- src/requests/compat.py:1-82 _resolve_char_detection
- src/requests/__init__.py:1-57 <module>
- src/requests/cookies.py:114-161 MockResponse

Tools: search_commits, search_code; 6.7 s

## bug-2 (bug report, exact)

**Q:** Timeouts are ignored when proxies are set. Find commits that touched proxy and timeout handling.

**A:** c93419f5125b, c64c0ab12151, c2aeaa3959b5, dfa41afd43a1, d2d576b6b110

**Sources shown to the model:**
- commit c93419f5125b TimeoutSauce!
- commit 6d0315f78fb2 Clarify timeout behavior in quickstart doc
- commit 58e3dd9de85c fix urllib3 bug
- commit d2d576b6b110 Update urllib3 to a27758625e4169330fcf965652b1093faf5aaaa2
- commit c64c0ab12151 prototype for new urllib3 timeouts
- commit c2aeaa3959b5 Add support for connect timeouts
- commit c64c0ab12151 prototype for new urllib3 timeouts
- commit dfa41afd43a1 Document Timeout behaviour

Tools: search_commits, search_commits; 4.4 s

## bug-3 (bug report, exact)

**Q:** Which commit introduced the redirect behaviour change in v2.3?

**A:** 470af31f4a5e

**Sources shown to the model:**
- commit 969195ad55f9 Clarify our 1.2.1 behaviour change.
- commit 9968a10fcfad Typo in changelog.
- commit 470af31f4a5e Handle 301s 'properly'.
- commit 679ab5bc20d0 Initial 2.3.0 changelog.
- commit f24d56a1ce65 Consolidate logic for changing method during redirects

Tools: search_commits; 2.1 s

## nav-1 (navigational, code)

**Q:** Where is `DEFAULT_REDIRECT_LIMIT` defined and where is it used?

**A:** DEFAULT_REDIRECT_LIMIT is defined in src/requests/models.py:72-105 as `DEFAULT_REDIRECT_LIMIT: int = 30`. It is used as the default value for the `max_redirects` attribute of the Session class (src/requests/sessions.py:395-440), which is then checked in `SessionRedirectMixin.resolve_redirects` to raise `TooManyRedirects` when the redirect count is exceeded (src/requests/sessions.py:186-235).

**Flags:**
- src/requests/sessions.py:395-440: no cited range shows DEFAULT_REDIRECT_LIMIT

**Sources shown to the model:**
- src/requests/sessions.py:127-152 SessionRedirectMixin.send
- src/requests/sessions.py:186-235 SessionRedirectMixin.resolve_redirects
- src/requests/models.py:72-105 <module>
- src/requests/auth.py:268-319 HTTPDigestAuth.handle_redirect
- src/requests/sessions.py:395-440 Session
- src/requests/utils.py:85-93 <module>
- src/requests/adapters.py:239-267 HTTPAdapter.init_poolmanager
- src/requests/status_codes.py:101-128 _init

Tools: get_symbol, search_code, find_callers, search_code; 8.0 s

## nav-2 (navigational, code)

**Q:** Show me all the places that read environment variables.

**A:** Places that read environment variables: set_environ reads os.environ.get(env_name) (src/requests/utils.py:763-807); get_netrc_auth reads os.environ.get("NETRC") (src/requests/utils.py:231-287); Session.merge_environment_settings reads os.environ for verify/cert settings (src/requests/sessions.py:831-886).

**Sources shown to the model:**
- src/requests/utils.py:763-807 is_valid_cidr
- src/requests/utils.py:873-908 get_environ_proxies
- src/requests/utils.py:911-962 resolve_proxies
- src/requests/utils.py:231-287 get_netrc_auth
- src/requests/utils.py:96-146 <module>
- src/requests/sessions.py:831-886 Session.merge_environment_settings
- docs/conf.py:108-158 <module>
- src/requests/_types.py:1-64 SupportsRead

Tools: search_code, search_code; 7.7 s

## impact-1 (impact, code)

**Q:** What breaks if I change the signature of `Session.send`?

**A:** Changing the signature of Session.send (def send(self, request: PreparedRequest, **kwargs: Any) -> Response, src/requests/sessions.py:752-799) would break its internal callers and the public API: Session.request calls self.send (src/requests/sessions.py:557-600), and SessionRedirectMixin.resolve_redirects calls self.send (src/requests/sessions.py:292). It also breaks external callers that pass a PreparedRequest plus kwargs such as stream/verify/cert/proxies/allow_redirects, e.g. the tests that call s.send / session.send (tests/test_requests.py:325, tests/test_requests.py:622, tests/test_requests.py:630, tests/test_requests.py:639, tests/test_requests.py:2652). The method relies on **kwargs to set defaults (stream, verify, cert, proxies) and to pop allow_redirects (src/requests/sessions.py:752-799), so removing or renaming those parameters would break request dispatch, redirect resolution and hook dispatch.

**Sources shown to the model:**
- src/requests/sessions.py:752-799 Session.send
- src/requests/sessions.py:695-740 Session.post
- src/requests/sessions.py:801-829 Session.send
- src/requests/sessions.py:908-920 session
- src/requests/sessions.py:557-600 Session.request
- src/requests/adapters.py:681-693 HTTPAdapter.send
- src/requests/models.py:378-422 PreparedRequest.__init__
- src/requests/auth.py:321-354 HTTPDigestAuth.__call__

Tools: search_code, find_callers; 7.8 s

## impact-2 (impact, docs)

**Q:** Which third-party libraries does requests depend on, and which versions are pinned?

**A:** requests depends on four third-party libraries, all with version ranges (not exact pins): charset_normalizer>=2,<4, idna>=2.5,<4, urllib3>=1.26,<3, and certifi>=2023.5.7 (pyproject.toml:1-24). None are pinned to an exact version; they use minimum/maximum bounds. The docs build pins Sphinx==7.2.6 (docs/requirements.txt:1-3), and dev/test dependencies include pytest-httpbin==2.1.0 and httpbin~=0.10.0 (requirements-dev.txt:1-7, pyproject.toml:58-115).

**Sources shown to the model:**
- docs/requirements.txt:1-3 requirements.txt
- requirements-dev.txt:1-7 requirements-dev.txt
- README.md:30-38 Requests > Installing Requests and Supported Versions
- pyproject.toml:1-24 pyproject.toml
- pyproject.toml:58-115 pyproject.toml

Tools: search_docs; 3.1 s

## hist-1 (change history, exact)

**Q:** Show me the history of `prepare_url`: who changed it and what each change did.

**A:** History of prepare_url in src/requests/models.py (newest first): 561e4b6889f5 by Nate Prewitt (2026-05-03, PR #7272) 'Add inline types to Requests'; bda7f0171f8b by Sergey Lyapustin (2022-06-29, PR #6188) 'Fixed warnings message to suggest using secure protocol'; 2a6f290bc093 by Nate Prewitt (2022-04-29, PR #6095) 'Add automatic code formatting to Requests'; 8bce583b9547 by Nate Prewitt (2022-03-25) 'Farewell, sweet Concorde!'; d09659997cd1 by Ian Stapleton Cordasco (2021-12-28) 'Merge pull request #5414 from mondeja/5367'; 658c10e84a46 by Álvaro Mondéjar (2020-04-05) 'Raise InvalidURL if hosts starts with "." as was reported at #5367'; 4246792b818c by Nate Prewitt (2021-12-28) 'Fix erroneous use of "schema" instead of "scheme"'; 9cdf29410778 by Nihaal (2019-08-13) 'Updated references to previous requests/requests GitHub path'.

Tools: file_history, file_history; 4.5 s

## hist-2 (change history, exact)

**Q:** What were the three most recent changes to src/requests/adapters.py?

**A:** 1) 661970d171d9 by Joren Hammudoglu on 2026-06-08: Disable commonly ignored Pyright linting rules (#7497). 2) fd628095d7b9 by typhon8 on 2026-05-10: Preserve leading slashes in request path_url (#7315). 3) 561e4b6889f5 by Nate Prewitt on 2026-05-03: Add inline types to Requests (#7272).

Tools: file_history; 2.0 s

## people-1 (people, exact)

**Q:** Who wrote most of src/requests/auth.py?

**A:** Nate Prewitt

Tools: top_contributors; 1.0 s

## people-2 (people, exact)

**Q:** Who last touched lines 40-80 of src/requests/sessions.py?

**A:** Joren Hammudoglu

Tools: blame; 1.7 s

## people-3 (people, exact)

**Q:** Who knows the sessions code best?

**A:** Nate Prewitt

**Sources shown to the model:**
- src/requests/sessions.py:908-920 session
- src/requests/sessions.py:1-73 <module>
- src/requests/sessions.py:395-440 Session
- src/requests/sessions.py:490-503 Session.__init__
- src/requests/sessions.py:888-905 Session.mount
- src/requests/cookies.py:191-227 RequestsCookieJar.get
- src/requests/api.py:1-21 <module>
- src/requests/auth.py:124-155 HTTPDigestAuth.__init__

Tools: top_contributors, search_code, top_contributors; 4.5 s

## people-4 (people, exact)

**Q:** Who opened, who reviewed and who merged PR #5856?

**A:** Opened by steveberdy, reviewed by sigmavirus24 and nateprewitt, merged by nateprewitt.

Tools: get_pr; 1.9 s

## pr-1 (pull requests, exact)

**Q:** What's PR #7272 about? Give me its description.

**A:** PR #7272, titled "Add inline types to Requests", adds inline type annotations throughout the Requests library to codify the type contract alongside the code. It is a path for what inline typing in Requests could look like, aiming to establish which APIs are intended to support, make more reasoned decisions about code changes, and make using Requests in a modern development environment easier. The package passes strict type checking with pyright (0 errors), with a few internal decisions deferred for maintainer/community discussion. Key design decisions include using cast() over assert/isinstance for type narrowing to avoid breaking downstream code at runtime, a new _types.py module centralizing Protocols, type aliases, and TypeVars, SupportsRead/SupportsItems as @runtime_checkable Protocols replacing hasattr checks, is_prepared() as a TypeIs guard that is a no-op at runtime (always returns True), and about 100 documented type suppressions. It is related to issue #7271 (RFC: Adding inline type annotations to Requests).

Tools: get_pr; 3.5 s

## pr-2 (pull requests, exact)

**Q:** Which PR switched the chardet dependency to charset_normalizer?

**A:** - 5797

Tools: search_prs; 2.8 s

## pr-3 (pull requests, exact)

**Q:** What did reviewers push back on in #5856?

**A:** No relevant implementation was found in the retrieved context.

**Flags:**
- abstained

Tools: get_pr, get_references, search_prs; 4.9 s

## issue-1 (issues, exact)

**Q:** Are there open bugs about timeouts?

**A:** No open bugs about timeouts were found; all matching timeout issues are closed.

Tools: search_prs; 3.7 s

## rel-1 (releases, exact)

**Q:** What changed between v2.30.0 and v2.31.0?

**A:** No relevant implementation was found in the retrieved context.

**Flags:**
- abstained

**Sources shown to the model:**
- src/requests/__init__.py:60-108 check_compatibility
- src/requests/__version__.py:1-14 <module>
- src/requests/auth.py:34-82 _basic_auth_str
- src/requests/compat.py:1-82 _resolve_char_detection
- src/requests/help.py:1-32 <module>
- src/requests/_types.py:1-64 SupportsRead
- src/requests/sessions.py:76-124 merge_setting
- src/requests/packages.py:1-23 <module>

Tools: get_tag, get_tag, search_code; 4.7 s

## rel-2 (releases, exact)

**Q:** Which release first contained the fix in PR #5856?

**A:** v2.27.0

Tools: get_pr, releases_containing; 2.9 s

## rel-3 (releases, exact)

**Q:** When did the `json=` parameter ship?

**A:** 2.4.2

**Sources shown to the model:**
- src/requests/models.py:1091-1142 Response.json
- src/requests/_types.py:177-188 GetKwargs
- src/requests/models.py:576-602 PreparedRequest.prepare_body
- src/requests/api.py:117-151 post
- src/requests/exceptions.py:42-99 JSONDecodeError
- src/requests/utils.py:1008-1037 guess_json_utf
- src/requests/cookies.py:531-576 morsel_to_cookie
- src/requests/compat.py:1-82 _resolve_char_detection

Tools: search_code, search_commits; 4.8 s

## stats-1 (stats, exact)

**Q:** Who are the top 5 contributors?

**A:** Kenneth Reitz, Cory Benfield, Nate Prewitt, Ian Cordasco, dependabot[bot]

Tools: top_contributors; 1.3 s

## stats-2 (stats, exact)

**Q:** Which files change most often?

**A:** requests/models.py, test_requests.py, requests/sessions.py

Tools: change_frequency; 1.1 s

## onboard-1 (onboarding, docs)

**Q:** How do I install requests and run its tests?

**A:** To install Requests, run `python -m pip install requests` (docs/user/install.rst:10-15, README.md:30-38). To run its tests, use the Makefile targets: `make init` installs dev dependencies via `python -m pip install -r requirements-dev.txt`, and `make test` runs `python -m pytest tests` (Makefile:1-27).

**Sources shown to the model:**
- docs/user/install.rst:10-15 Installation of Requests > $ python -m pip install requests
- docs/user/install.rst:1-7 Installation of Requests
- README.md:30-38 Requests > Installing Requests and Supported Versions
- docs/user/quickstart.rst:1-17 Quickstart
- Makefile:1-27 Makefile

Tools: search_docs; 2.6 s

## oos-1 (out of scope, out_of_scope)

**Q:** Show me all the places we write to the cache.

**A:** No relevant implementation was found in the retrieved context.

**Sources shown to the model:**
- src/requests/structures.py:20-74 CaseInsensitiveDict.__init__
- src/requests/cookies.py:229-291 RequestsCookieJar.set
- src/requests/cookies.py:579-625 cookiejar_from_dict
- src/requests/cookies.py:348-399 RequestsCookieJar.__iter__
- src/requests/cookies.py:479-528 _copy_cookie_jar
- src/requests/status_codes.py:1-59 <module>
- src/requests/utils.py:290-338 extract_zipped_paths
- src/requests/sessions.py:888-905 Session.mount

Tools: search_code, search_code; 6.8 s
