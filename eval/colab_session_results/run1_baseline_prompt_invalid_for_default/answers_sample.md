# Answer sample

Read these against the evidence; a pass here is only mechanical.

## fresh_default

- `fresh2-code-028` PASS | 'How does the library check whether an IP address is inside a CIDR range?'
  - The library checks membership with address_in_network, which converts the IP and the network address to 32-bit integers via socket.inet_aton, derives the netmask from the /bits prefix using dotted_netmask, and returns True when (ipaddr & netmask) == (network & netmask) (src/requests/utils.py:722-733). The /bits prefix is turned into a dotted netmask by dotted_netmask, which computes 0xFFFFFFFF ^ (1 << 32 - mask) - 1 and formats it with socket.inet_ntoa (src/requests/utils.py:736-743). This is used by should_bypass_proxies, which first validates the CIDR with is_valid_cidr (checking for exactly one '/', a mask between 1 and 32, and a valid IPv4 address) and then calls address_in_network for e
- `fresh2-code-013` PASS | 'How does a Request become a PreparedRequest?'
  - A Request becomes a PreparedRequest by calling Request.prepare(), which creates a new PreparedRequest and passes the Request's method, url, headers, files, data, json, params, auth, cookies, and hooks into PreparedRequest.prepare() (src/requests/models.py:353-375). PreparedRequest.prepare() then runs a fixed sequence of sub-preparers: prepare_method, prepare_url, prepare_headers, prepare_cookies, prepare_body, prepare_auth, and finally prepare_hooks, with prepare_auth required to be last so auth schemes can operate on a fully prepared request (src/requests/models.py:424-456). Alternatively, Session.prepare_request() builds a PreparedRequest by merging the Request's settings with the Session'
- `fresh2-code-025` PASS | 'What does the library collect for a bug report, such as platform and library versions?'
  - The library's bug report helper, info(), collects: the platform system and release (src/requests/help.py:67-132); the Python implementation name and version via _implementation() (src/requests/help.py:35-64); the system SSL/OpenSSL version (src/requests/help.py:67-132); whether pyOpenSSL and charset_normalizer are in use (src/requests/help.py:67-132); and the versions of the dependency libraries pyOpenSSL (including its OpenSSL version number), urllib3, chardet, charset_normalizer, cryptography, and idna, plus the requests library's own version (src/requests/help.py:67-132).
- `fresh2-code-029` MISS | 'How does the library decode a response body when the encoding is unknown or wrong?'
  - When the encoding is unknown, Response.text falls back to self.apparent_encoding, which guesses the encoding using charset_normalizer or chardet (src/requests/models.py:1034-1089). It then decodes with str(self.content, encoding or "utf-8", errors="replace"); if that raises LookupError or TypeError (e.g. a misspelled or None encoding), it decodes blindly with str(self.content, errors="replace") (src/requests/models.py:1034-1089). For JSON, Response.json first tries guess_json_utf on the raw bytes when no encoding is set, and if that UTF codec fails it falls back to self.text, which uses the apparent-encoding guess (src/requests/models.py:1091-1142). The header-based encoding is obtained via 
- `fresh2-docs-037` MISS | 'How do I read the Link header of a response as links?'
  - Use the Response.links property, which reads the 'link' header and returns a dict of parsed links keyed by their rel (or url) (src/requests/models.py:1091-1142). It calls parse_header_links, which splits the header into a list of dicts each containing a 'url' plus any parameters like 'rel' and 'type' (src/requests/utils.py:965-1005). Example: r = requests.get(url); r.links['next']['url'].
- `fresh2-docs-031` MISS | 'How do I upload a file with a custom filename and content type?'
  - Pass the file as a tuple in the files dict: a 3-tuple ('filename', fileobj, 'content_type') sets a custom filename and content type, or a 4-tuple ('filename', fileobj, 'content_type', custom_headers) also adds per-file headers (src/requests/api.py:24-47). The encoding logic unpacks these tuples and builds the multipart body, using the provided filename and content type (src/requests/models.py:182-251). Example: requests.post(url, files={'file': ('myname.txt', open('f','rb'), 'text/plain')}).
- `fresh2-docs-035` MISS | 'How do I delay downloading the response body until I want it?'
  - Pass stream=True to the request so the response body is not downloaded into memory immediately; then read it later with Response.iter_content (src/requests/sessions.py:557-600, src/requests/models.py:914-965).
- `fresh2-docs-038` MISS | 'How do I save binary response content such as an image?'
  - Access the response's binary data via the Response.content property, which returns the content as bytes (src/requests/models.py:1034-1055), then write those bytes to a file in binary mode, e.g. open('image.png','wb').write(r.content). For large responses, use Response.iter_content to stream chunks to disk instead of loading everything into memory (src/requests/models.py:914-977).
- `fresh2-history-048` PASS | 'Which release first contained the change merged in PR #7196 in psf/requests?'
  - v2.33.0
- `fresh2-history-047` PASS | 'Which issues does PR #1030 in psf/requests close?'
  - - 1027
- `fresh2-history-045` PASS | 'Who authored PR #6500 in psf/requests?'
  - dependabot
- `fresh2-history-054` PASS | 'Who made the most commits in psf/requests in 2022, not counting merge commits?'
  - Nate Prewitt  Note: commit counts cover non-merge commits only (merge commits are not indexed).  Activity from the commit history: - Nate Prewitt: 21 commits in this history, first 2022-01-03, most recent 2022-06-29
- `fresh2-kind-061` PASS | 'Is #2672 in psf/requests a pull request or an issue?'
  - pull request
- `fresh2-kind-062` PASS | 'Is #1125 in psf/requests a pull request or an issue?'
  - pull request
- `fresh2-kind-059` PASS | 'Is #5166 in psf/requests a pull request or an issue?'
  - issue
- `fresh2-kind-060` PASS | 'Is #6344 in psf/requests a pull request or an issue?'
  - issue
- `fresh2-probe-071` PASS | 'How do I use requests with gRPC streaming?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-probe-070` PASS | 'Where does requests implement a Redis caching layer?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-probe-069` PASS | 'How does requests encrypt files at rest with AES?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-probe-072` PASS | 'Which function in requests trains a machine-learning model?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-prsearch-067` PASS | 'In psf/requests, which issue is about: "how can i pass gmail\'s cookie to requests session?"? Give its number.'
  - 3624  Details from the search results: - #3624 (issue, closed): Issue #3624: how can I pass gmail's cookie to requests session?
- `fresh2-prsearch-068` PASS | 'In psf/requests, which issue is about: "rfe: please provide support for latest `urllib3`"? Give its number.'
  - 6278  Details from the search results: - #6278 (issue, closed): Issue #6278: RFE: please provide support for latest `urllib3`
- `fresh2-prsearch-064` PASS | 'In psf/requests, which pull request is about: "docs - clarify json() error exception for python 3"? Give its number.'
  - - 5259  Details from the search results: - #5259 (pull request, closed): PR #5259: docs - clarify json() error exception for Python 3
- `fresh2-prsearch-066` PASS | 'In psf/requests, which issue is about: "timeout exception: session.get() vs requests.get() behavior discrepancy"? Give i'
  - 5381  Details from the search results: - #5381 (issue, closed): Issue #5381: timeout exception: session.get() vs requests.get() behavior discrepancy
- `fresh2-refs-055` PASS | 'Which other PRs or issues does #7339 in psf/requests mention in its title or body?'
  - - 6990
- `fresh2-refs-057` PASS | 'Which PRs or issues in psf/requests mention #1750 in their title or body?'
  - - 5671
- `fresh2-refs-058` PASS | 'Which PRs or issues in psf/requests mention #1847 in their title or body?'
  - - 1850
- `fresh2-refs-056` PASS | 'Which other PRs or issues does #7251 in psf/requests mention in its title or body?'
  - - 7228

## fresh_guards

- `fresh2-code-016` PASS | 'What does preparing authentication do when no auth argument is given but the URL itself carries credentials?'
  - When no auth argument is given, prepare_auth extracts credentials from the URL itself: it calls get_auth_from_url on self.url and, if any part of the resulting (username, password) tuple is non-empty, uses that tuple as the auth (src/requests/models.py:672-697). get_auth_from_url parses the URL and returns the unquoted username and password, or an empty tuple if the URL has no credentials (src/requests/utils.py:1070-1084). If the extracted auth is a two-element tuple, it is special-cased into an HTTPBasicAuth handler, which then sets the Authorization header to a Basic auth string (src/requests/models.py:680-697, src/requests/auth.py:85-105).
- `fresh2-code-018` PASS | 'What happens when two cookies with the same name exist for different domains and I read one by name?'
  - When you read a cookie by name without specifying a domain, the behavior depends on which method you use. The dict-style __getitem__ and get() both call _find_no_duplicates, which iterates over all cookies and raises a CookieConflictError if more than one cookie matches the given name (and optional domain/path), so two cookies with the same name for different domains will raise CookieConflictError (src/requests/cookies.py:423-452, src/requests/cookies.py:191-227). Internally, Requests uses _find, which does not raise on conflicts but arbitrarily returns the value of the first matching cookie it encounters (src/requests/cookies.py:401-421). To avoid the conflict, pass the domain (and optional
- `fresh2-code-004` PASS | 'How does the redirect mixin find out where a redirect points to?'
  - The redirect mixin determines the redirect target in SessionRedirectMixin.get_redirect_target, which checks resp.is_redirect and, if true, reads the 'location' header from the response, re-encodes it from latin1 to utf8, and returns it as a native string; otherwise it returns None (src/requests/sessions.py:127-152). The is_redirect property is True when the response has a 'location' header and its status code is in REDIRECT_STATI (src/requests/models.py:861-913). resolve_redirects then uses that target, joining relative locations against resp.url when there is no netloc (src/requests/sessions.py:237-280).
- `fresh2-code-012` PASS | 'What decides that a response is a permanent redirect?'
  - A response is a permanent redirect when it has a 'location' header and its status code is one of the permanent redirect codes, 301 (moved_permanently) or 308 (permanent_redirect), as checked by Response.is_permanent_redirect (src/requests/models.py:861-913). The status code names map to 301 and 308 in src/requests/status_codes.py:1-59.
- `fresh2-docs-037` PASS | 'How do I read the Link header of a response as links?'
  - Requests automatically parses Link headers into the response's .links dictionary, so you access them by rel name, e.g. r.links["next"] returns {'url': '...', 'rel': 'next'} (docs/user/advanced.rst:944-966).
- `fresh2-docs-036` PASS | 'Does requests keep connections alive, and how do I turn that off?'
  - Yes — thanks to urllib3, keep-alive is 100% automatic within a session, so any requests made within a session automatically reuse the appropriate connection (docs/user/advanced.rst:336-347). The docs do not describe a setting to turn keep-alive off; they only note that connections are released back to the pool for reuse once all body data has been read, so you should set stream to False or read the content property (docs/user/advanced.rst:336-347).
- `fresh2-docs-034` PASS | 'How do I upload a large file without loading it all into memory?'
  - To upload a large file without loading it all into memory, pass a file-like object as the request body so Requests streams it: `with open('massive-body', 'rb') as f: requests.post('http://some.url/streamed', data=f)` (docs/user/advanced.rst:349-366). Open the file in binary mode so the Content-Length header is set to the byte count (docs/user/advanced.rst:349-366). For a very large file sent as multipart/form-data, Requests does not stream by default; use the separate `requests-toolbelt` package (docs/user/quickstart.rst:349-366).
- `fresh2-docs-035` PASS | 'How do I delay downloading the response body until I want it?'
  - Pass stream=True to the request so only the headers are downloaded and the body is deferred until you access Response.content (docs/user/advanced.rst:300-329). You can then retrieve the body conditionally, e.g. if int(r.headers['content-length']) < TOO_LONG: content = r.content (docs/user/advanced.rst:300-329). For finer control use Response.iter_content() or Response.iter_lines(), or read the undecoded body from Response.raw (docs/user/advanced.rst:300-329). Note that with stream=True the connection is not returned to the pool until you consume all the data or call Response.close(), so wrap the request in a with statement to ensure it is closed (docs/user/advanced.rst:300-334).
- `fresh2-history-048` PASS | 'Which release first contained the change merged in PR #7196 in psf/requests?'
  - v2.33.0
- `fresh2-history-054` PASS | 'Who made the most commits in psf/requests in 2022, not counting merge commits?'
  - Nate Prewitt  Note: commit counts cover non-merge commits only (merge commits are not indexed).  Activity from the commit history: - Nate Prewitt: 21 commits in this history, first 2022-01-03, most recent 2022-06-29
- `fresh2-history-049` PASS | 'Which release first contained the change merged in PR #4996 in psf/requests?'
  - v2.23.0
- `fresh2-history-045` PASS | 'Who authored PR #6500 in psf/requests?'
  - dependabot
- `fresh2-kind-059` PASS | 'Is #5166 in psf/requests a pull request or an issue?'
  - issue
- `fresh2-kind-061` PASS | 'Is #2672 in psf/requests a pull request or an issue?'
  - pull request
- `fresh2-kind-062` PASS | 'Is #1125 in psf/requests a pull request or an issue?'
  - pull request
- `fresh2-kind-060` PASS | 'Is #6344 in psf/requests a pull request or an issue?'
  - issue
- `fresh2-probe-072` PASS | 'Which function in requests trains a machine-learning model?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-probe-069` PASS | 'How does requests encrypt files at rest with AES?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-probe-071` PASS | 'How do I use requests with gRPC streaming?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-probe-070` PASS | 'Where does requests implement a Redis caching layer?'
  - No relevant implementation was found in the retrieved context.
- `fresh2-prsearch-065` PASS | 'In psf/requests, which pull request is about: "only the path should be unquoted/quoted."? Give its number.'
  - - 149  Details from the search results: - #149 (pull request, merged): PR #149: Only the path should be unquoted/quoted.
- `fresh2-prsearch-064` PASS | 'In psf/requests, which pull request is about: "docs - clarify json() error exception for python 3"? Give its number.'
  - - 5259  Details from the search results: - #5259 (pull request, closed): PR #5259: docs - clarify json() error exception for Python 3
- `fresh2-prsearch-068` PASS | 'In psf/requests, which issue is about: "rfe: please provide support for latest `urllib3`"? Give its number.'
  - 6278  Details from the search results: - #6278 (issue, closed): Issue #6278: RFE: please provide support for latest `urllib3`
- `fresh2-prsearch-066` PASS | 'In psf/requests, which issue is about: "timeout exception: session.get() vs requests.get() behavior discrepancy"? Give i'
  - 5381  Details from the search results: - #5381 (issue, closed): Issue #5381: timeout exception: session.get() vs requests.get() behavior discrepancy
- `fresh2-refs-055` PASS | 'Which other PRs or issues does #7339 in psf/requests mention in its title or body?'
  - - 6990
- `fresh2-refs-058` PASS | 'Which PRs or issues in psf/requests mention #1847 in their title or body?'
  - - 1850
- `fresh2-refs-057` PASS | 'Which PRs or issues in psf/requests mention #1750 in their title or body?'
  - - 5671
- `fresh2-refs-056` PASS | 'Which other PRs or issues does #7251 in psf/requests mention in its title or body?'
  - - 7228

## commit_default

- `commitfull-103` PASS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\nSSLV3_ALERT'
  - a62a2d35d918  Matched commit details: - a62a2d35d918 (2024-05-22, Ian Stapleton Cordasco): Allow for overriding of specific pool key params; ranked 4 of 15 search results; a text match on the report, not confirmed against the code.
- `commitfull-019` MISS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\ntest_unicod'
  - 003c795afed5  Matched commit details: - 003c795afed5 (2013-05-21, Ib Lundgren): Only pass unicode fieldnames to urllib3.; ranked 1 of 15 search results; a text match on the report, not confirmed against the code.
- `commitfull-070` PASS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\nHTTPDigestA'
  - 5c19d3e07b68  Matched commit details: - 5c19d3e07b68 (2015-12-05, Matt Jordan): requests/auth: Handle an empty 'qop' attribute in a Authenticate challenge; ranked 1 of 15 search results; a text match on the report, not confirmed against the code.
- `commitfull-058` PASS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\n__build__ o'
  - dacf69a16df7  Matched commit details: - dacf69a16df7 (2015-04-05, Ian Cordasco): Fix build version number in metadata; ranked 1 of 15 search results; a text match on the report, not confirmed against the code.

## commit_explain

- `commitfull-012` MISS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\nCookies won'
  - Yes. The most probable fix is commit 87abd9c6094f (2014-01-12, "Use calendar.timegm when calculating cookie expiration"), which changed requests/cookies.py::morsel_to_cookie to compute the cookie expiration with calendar.timegm instead of time.mktime - time.timezone, so that a Set-Cookie with expires=Thu, 01-Jan-1970 00:00:01 GMT is correctly treated as expired and the cookie is deleted. Related earlier work includes 88f13598f32b (2013-04-04), which added the unit test test_cookie_removed_on_expire reproducing exactly this scenario, and ea4570da5714 (2013-12-05), which fixed the timezone handling in morsel_to_cookie.  Matched commit details: - 87abd9c6094f (2014-01-12, Ian Cordasco): Use cal
- `commitfull-011` PASS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\nSession.req'
  - Yes. Commit ca0aea640d0c (2013-04-02, Kamil Szot) added the return_response parameter to Session.request, but it was immediately reverted by commit 54ed5ed469ff (2013-04-02, Kenneth Reitz), so the parameter was not actually kept; the docstring was separately corrected in 1abd13700b32 to remove the stale return_response documentation.  Matched commit details: - ca0aea640d0c (2013-04-02, Kamil Szot): Fix for the issue; ranked 1 of 15 search results; a text match on the report, not confirmed against the code. - 54ed5ed469ff (2013-04-02, Kenneth Reitz): Revert "Fix for the issue ; ranked 5 of 15 search results; a text match on the report, not confirmed against the code. - 1abd13700b32 (2013-04-0
- `commitfull-041` MISS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\nLink header'
  - Yes. Commit 4594667f07ca (2014-10-08, mikecool1000) changed parse_header_links in requests/utils.py from value.split(",") to re.split(",\ *<", value), which is a probable fix for link headers whose attribute values contain commas (e.g. Memento datetime values).  Matched commit details: - 4594667f07ca (2014-10-08, mikecool1000): Update utils.py; ranked 3 of 15 search results; a text match on the report, not confirmed against the code.
- `commitfull-113` MISS | 'A user reports the bug below. Was something like this fixed before in this repository? If so, which commit?\n\nReal loggin'
  - No answer could be produced: the answer model failed (RuntimeError: Model server returned 530: ).

