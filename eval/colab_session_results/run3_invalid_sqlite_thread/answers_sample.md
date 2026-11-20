# Answer sample

Read these against the evidence; a pass here is only mechanical.

## fresh_default

- `fresh2-code-028` PASS | 'How does the library check whether an IP address is inside a CIDR range?'
  - The library checks membership with address_in_network, which converts the IP and the network address to 32-bit integers via socket.inet_aton, derives the netmask from the /bits prefix using dotted_netmask, and returns True when (ipaddr & netmask) == (network & netmask) (src/requests/utils.py:722-733). The helper dotted_netmask converts a /xx prefix into a dotted netmask by computing 0xFFFFFFFF ^ (1 << 32 - mask) - 1 and formatting it with socket.inet_ntoa (src/requests/utils.py:736-744). This is used by should_bypass_proxies, which first validates the CIDR with is_valid_cidr and then calls address_in_network for IPv4 hostnames (src/requests/utils.py:810-870).
- `fresh2-code-013` PASS | 'How does a Request become a PreparedRequest?'
  - A Request becomes a PreparedRequest by calling Request.prepare(), which constructs a new PreparedRequest and passes the Request's method, url, headers, files, data, json, params, auth, cookies, and hooks into PreparedRequest.prepare() (src/requests/models.py:358-375). PreparedRequest.prepare() then runs a fixed sequence of sub-preparers: prepare_method, prepare_url, prepare_headers, prepare_cookies, prepare_body, prepare_auth, and finally prepare_hooks, with prepare_auth last so authentication schemes can operate on a fully prepared request (src/requests/models.py:424-452). Alternatively, Session.prepare_request() builds a PreparedRequest by merging the Request's settings with the Session's 
- `fresh2-code-025` PASS | 'What does the library collect for a bug report, such as platform and library versions?'
  - The library's bug report helper, info(), collects: the platform system and release (src/requests/help.py:67-132); the Python implementation name and version via _implementation() (src/requests/help.py:35-64); the system SSL/OpenSSL version (src/requests/help.py:67-132); whether pyOpenSSL and charset_normalizer are in use (src/requests/help.py:67-132); and the versions of the dependency libraries pyOpenSSL (including its OpenSSL version number), urllib3, chardet, charset_normalizer, cryptography, and idna, plus the requests library's own version (src/requests/help.py:67-132).
- `fresh2-code-029` MISS | 'How does the library decode a response body when the encoding is unknown or wrong?'
  - When the encoding is unknown, Response.text falls back to self.apparent_encoding, which guesses the encoding using charset_normalizer or chardet (src/requests/models.py:1034-1089). It then decodes with str(self.content, encoding or "utf-8", errors="replace"); if that raises LookupError or TypeError (e.g. a misspelled or None encoding), it decodes blindly with str(self.content, errors="replace") (src/requests/models.py:1034-1089). For JSON, Response.json first tries guess_json_utf on the raw bytes when no encoding is set, and if that UTF codec fails it falls back to self.text, which uses the apparent-encoding guess (src/requests/models.py:1091-1142). The header-based encoding is obtained via 
- `fresh2-docs-037` PASS | 'How do I read the Link header of a response as links?'
  - Requests automatically parses Link headers into a dictionary you can access via the response's .links attribute, keyed by the rel value. For example, after r = requests.head(url), r.links["next"] returns {'url': '...', 'rel': 'next'} and r.links["last"] returns {'url': '...', 'rel': 'last'} (docs/user/advanced.rst:944-966).
- `fresh2-docs-031` PASS | 'How do I upload a file with a custom filename and content type?'
  - To upload a file with a custom filename and content type, pass a tuple to the files dict: the filename, the file object opened in binary mode, and the content type (optionally followed by a headers dict). Example: files = {'file': ('report.xls', open('report.xls', 'rb'), 'application/vnd.ms-excel', {'Expires': '0'})} then requests.post(url, files=files) (docs/user/quickstart.rst:319-329).
- `fresh2-docs-035` PASS | 'How do I delay downloading the response body until I want it?'
  - Use the stream parameter set to True when making the request; this defers downloading the response body until you access the Response.content attribute (docs/user/advanced.rst:300-329). For example: r = requests.get(url, stream=True). You can then conditionally retrieve the content, or use Response.iter_content()/iter_lines() for finer control (docs/user/advanced.rst:300-329). Note that with stream=True the connection stays open until you consume all the data or call Response.close(), so prefer a with statement (docs/user/advanced.rst:329-334).
- `fresh2-docs-038` PASS | 'How do I save binary response content such as an image?'
  - Access the response body as bytes via r.content, then write it to a file in binary mode; for an image you can also open it with PIL using BytesIO(r.content) (docs/user/quickstart.rst:121-140).
- `fresh2-history-048` MISS | 'Which release first contained the change merged in PR #7196 in psf/requests?'
  - No answer could be produced: a tool failed (get_pr: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 4916.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-history-047` MISS | 'Which issues does PR #1030 in psf/requests close?'
  - No answer could be produced: a tool failed (get_pr: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 17444.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-history-045` MISS | 'Who authored PR #6500 in psf/requests?'
  - No answer could be produced: a tool failed (get_pr: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 4916.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-history-054` MISS | 'Who made the most commits in psf/requests in 2022, not counting merge commits?'
  - No answer could be produced: a tool failed (top_contributors: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 17444.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-kind-061` MISS | 'Is #2672 in psf/requests a pull request or an issue?'
  - No answer could be produced: a tool failed (get_pr: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 17444.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-kind-062` MISS | 'Is #1125 in psf/requests a pull request or an issue?'
  - No answer could be produced: a tool failed (get_pr: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 27436.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-kind-059` MISS | 'Is #5166 in psf/requests a pull request or an issue?'
  - No answer could be produced: a tool failed (get_pr: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 4916.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-kind-060` MISS | 'Is #6344 in psf/requests a pull request or an issue?'
  - No answer could be produced: a tool failed (get_pr: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 27436.). Commit, PR and issue data may not be indexed for this repository.
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
- `fresh2-refs-055` MISS | 'Which other PRs or issues does #7339 in psf/requests mention in its title or body?'
  - No answer could be produced: a tool failed (get_references: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 12932.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-refs-057` MISS | 'Which PRs or issues in psf/requests mention #1750 in their title or body?'
  - No answer could be produced: a tool failed (get_references: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 4916.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-refs-058` MISS | 'Which PRs or issues in psf/requests mention #1847 in their title or body?'
  - No answer could be produced: a tool failed (get_references: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 12932.). Commit, PR and issue data may not be indexed for this repository.
- `fresh2-refs-056` MISS | 'Which other PRs or issues does #7251 in psf/requests mention in its title or body?'
  - No answer could be produced: a tool failed (get_references: ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6220 and this is thread id 27436.). Commit, PR and issue data may not be indexed for this repository.

