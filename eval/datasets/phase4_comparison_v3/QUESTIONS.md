# Expanded comparison questions

Source-authored before expanded retrieval scoring. Alternatives are sufficient proofs, not additive requirements across alternatives. Old evidence overlap is recorded in manifest.json; no independent review is claimed.

## psf/requests

### requests-comparison-001 — ambiguous

Why does putting a line break inside an HTTP header value cause a failure?

- Proof 1: src/requests/utils.py:1093-1095, src/requests/utils.py:1103-1119, src/requests/_internal_utils.py:15-19
- Proof 2: tests/test_requests.py:1817-1840

### requests-comparison-002 — ambiguous

How can an already partially consumed upload be read again from its saved starting position?

- Proof 1: src/requests/utils.py:1139-1155
- Proof 2: tests/test_requests.py:2025-2037

### requests-comparison-003 — ambiguous

Why can removing an HTTP header use different capitalization from when it was stored?

- Proof 1: src/requests/structures.py:59-68
- Proof 2: tests/test_structures.py:6-10, tests/test_structures.py:15-27

### requests-comparison-004 — ambiguous

What happens when several response callbacks run in sequence and one returns no replacement value?

- Proof 1: src/requests/hooks.py:39-48
- Proof 2: tests/test_hooks.py:6-18

### requests-comparison-005 — ambiguous

Can a copied HTTP header mapping contain the same values while being a separate object?

- Proof 1: src/requests/structures.py:49-57, src/requests/structures.py:89-90
- Proof 2: tests/test_structures.py:6-10, tests/test_structures.py:37-40

### requests-comparison-006 — ambiguous

How are byte-string credentials encoded for HTTP basic authentication?

- Proof 1: src/requests/auth.py:65-75
- Proof 2: tests/test_requests.py:572-580

### requests-comparison-007 — ambiguous

What failure occurs when JSON is requested from a response with no content?

- Proof 1: src/requests/models.py:1119-1124, src/requests/models.py:1070-1071
- Proof 2: tests/test_requests.py:2230-2239

### requests-comparison-008 — ambiguous

Why does looking up a cookie by name fail when several stored cookies match it?

- Proof 1: src/requests/cookies.py:437-452
- Proof 2: tests/test_requests.py:1424-1433

### requests-comparison-009 — ambiguous

How do extra query parameters stay ahead of a URL fragment?

- Proof 1: src/requests/models.py:548-563
- Proof 2: tests/test_requests.py:150-165

### requests-comparison-010 — ambiguous

Why can Unicode request data have a content length larger than its character count?

- Proof 1: src/requests/utils.py:160-170
- Proof 2: tests/test_requests.py:3074-3084

### requests-comparison-011 — ambiguous

How can user credentials and the fragment be removed from a URL while preserving its query?

- Proof 1: src/requests/utils.py:1122-1136
- Proof 2: tests/test_utils.py:729-741

### requests-comparison-012 — ambiguous

Can a cookie lifetime supplied as an integer produce an integer expiration timestamp?

- Proof 1: src/requests/cookies.py:534-548
- Proof 2: tests/test_requests.py:2519-2525

### requests-comparison-013 — multi_block

Trace the implementation that merges session and per-request headers and removes entries whose replacement value is null.

- Proof 1: src/requests/sessions.py:96-105, src/requests/sessions.py:547-549

### requests-comparison-014 — multi_block

Trace the implementation that decides whether a redirected request keeps its authorization and then removes or rebuilds credentials.

- Proof 1: src/requests/sessions.py:156-184, src/requests/sessions.py:320-332

### requests-comparison-015 — multi_block

Trace the implementation from uploaded file tuples to multipart bytes and the outgoing request content type.

- Proof 1: src/requests/models.py:221-251, src/requests/models.py:635-650

### requests-comparison-016 — multi_block

Trace the implementation from certificate verification options to a direct or proxied connection pool.

- Proof 1: src/requests/adapters.py:97-119, src/requests/adapters.py:483-508

### requests-comparison-017 — multi_block

Trace the implementation from response headers to the encoding selected when bytes are decoded into text.

- Proof 1: src/requests/adapters.py:381-385, src/requests/utils.py:576-591, src/requests/models.py:1066-1089

### requests-comparison-018 — multi_block

Trace the implementation that decodes streaming byte chunks without losing a character split across chunks.

- Proof 1: src/requests/models.py:974-977, src/requests/utils.py:599-610

### requests-comparison-019 — test

Which automated test verifies that each response in a redirect sequence records the rewritten outgoing HTTP method?

- Proof 1: tests/test_requests.py:2649-2667, tests/test_requests.py:2605-2638

### requests-comparison-020 — test

Which automated test verifies that query parameter merging distinguishes upper- and lowercase keys?

- Proof 1: tests/test_requests.py:1764-1768

### requests-comparison-021 — test

Which automated test verifies that non-HTTP URL schemes bypass HTTP URL validation?

- Proof 1: tests/test_requests.py:1910-1919

### requests-comparison-022 — test

Which automated test checks that closing an unconsumed session response releases its connection?

- Proof 1: tests/test_requests.py:2191-2198

### requests-comparison-023 — test

Which automated test checks the error for a certificate-authority bundle path that does not exist?

- Proof 1: tests/test_requests.py:969-976

### requests-comparison-024 — test

Which automated test demonstrates multiple query values for one parameter name?

- Proof 1: tests/test_requests.py:1068-1071

### requests-comparison-025 — implementation

Locate the implementation that parses link relation parameters from an HTTP Link header.

- Proof 1: src/requests/utils.py:973-999

### requests-comparison-026 — implementation

Locate the implementation that returns false when a string cannot be parsed as an IPv4 address.

- Proof 1: src/requests/utils.py:756-760

### requests-comparison-027 — implementation

Locate the implementation that converts a prefix length into a dotted IPv4 netmask.

- Proof 1: src/requests/utils.py:748-749

### requests-comparison-028 — implementation

Locate the implementation that extracts a basename from a file-like object's name while rejecting angle-bracket placeholders.

- Proof 1: src/requests/utils.py:285-287

### requests-comparison-029 — implementation

Locate the implementation that removes a cookie by name with optional domain and path constraints.

- Proof 1: src/requests/cookies.py:171-182

### requests-comparison-030 — implementation

Locate the implementation that initializes digest-authentication counters separately for each thread.

- Proof 1: src/requests/auth.py:147-155

## expressjs/express

### express-comparison-001 — ambiguous

Why do spaces in a redirect destination become percent escapes?

- Proof 1: lib/response.js:797-799
- Proof 2: test/res.location.js:35-47

### express-comparison-002 — ambiguous

Where can I follow how the client address is chosen behind a trusted proxy?

- Proof 1: lib/request.js:340-343
- Proof 2: test/req.ip.js:40-53

### express-comparison-003 — ambiguous

Why does clearing a cookie expire it immediately even when a positive lifetime was supplied?

- Proof 1: lib/response.js:715-722
- Proof 2: test/res.clearCookie.js:36-47

### express-comparison-004 — ambiguous

Why is a fractional HTTP response status rejected?

- Proof 1: lib/response.js:65-69
- Proof 2: test/res.status.js:144-154

### express-comparison-005 — ambiguous

What happens if I try to read a request header without supplying its name?

- Proof 1: lib/request.js:63-67
- Proof 2: test/req.get.js:36-46

### express-comparison-006 — ambiguous

How can JSON output escape characters that might otherwise be mistaken for HTML?

- Proof 1: lib/response.js:236-250, lib/response.js:1033-1050
- Proof 2: test/res.json.js:112-125

### express-comparison-007 — ambiguous

How is the first acceptable response type chosen when several media types are offered?

- Proof 1: lib/request.js:127-130
- Proof 2: test/req.accepts.js:73-84

### express-comparison-008 — ambiguous

How is a successful retrieval considered unmodified when its entity tag matches?

- Proof 1: lib/request.js:469-486
- Proof 2: test/req.fresh.js:9-22

### express-comparison-009 — ambiguous

What prevents the same response variation field from being added twice?

- Proof 1: lib/response.js:878-882
- Proof 2: test/res.vary.js:72-88

### express-comparison-010 — ambiguous

How does a cookie lifetime in milliseconds become the lifetime sent in its header?

- Proof 1: lib/response.js:763-781
- Proof 2: test/res.cookie.js:115-126

### express-comparison-011 — ambiguous

How are overlapping requested byte ranges combined when merging is enabled?

- Proof 1: lib/request.js:214-218
- Proof 2: test/req.range.js:88-101

### express-comparison-012 — ambiguous

How can repeated values be added to a response header while keeping its previous values?

- Proof 1: lib/response.js:635-647
- Proof 2: test/res.append.js:65-83, test/res.append.js:107-116

### express-comparison-013 — multi_block

Trace the implementation from a response's template rendering request to application-level rendering and invoking the selected view engine.

- Proof 1: lib/response.js:897-921, lib/application.js:548-575, lib/application.js:625-631, lib/view.js:133-159

### express-comparison-014 — multi_block

Trace the implementation that sets a download filename and passes the resolved file path into file transfer.

- Proof 1: lib/response.js:437-486, lib/response.js:375-406

### express-comparison-015 — multi_block

Trace the implementation that mounts a child application, inherits parent settings and restores the parent request and response prototypes afterwards.

- Proof 1: lib/application.js:109-122, lib/application.js:218-240

### express-comparison-016 — multi_block

Trace the implementation from a JSONP callback name to the wrapped JavaScript body and the final send operation.

- Proof 1: lib/response.js:264-308, lib/response.js:205-222

### express-comparison-017 — multi_block

Trace the implementation from a status-message response to its body preparation and omission for a HEAD request.

- Proof 1: lib/response.js:325-332, lib/response.js:154-182, lib/response.js:214-222

### express-comparison-018 — multi_block

Trace the implementation that negotiates among response formats and adds the corresponding response variation header.

- Proof 1: lib/response.js:574-599, lib/request.js:127-130, lib/response.js:878-882

### express-comparison-019 — test

Which automated test verifies that request parameters are restored after leaving a nested router?

- Proof 1: test/app.router.js:12-37

### express-comparison-020 — test

Which automated test verifies that only error-handling callbacks run after an error is propagated through a route?

- Proof 1: test/app.routes.error.js:25-60

### express-comparison-021 — test

Which automated test checks that disabling query parsing leaves an empty query object?

- Proof 1: test/req.query.js:66-72, test/req.query.js:94-106

### express-comparison-022 — test

Which automated test checks selecting the best character set from several supplied choices?

- Proof 1: test/req.acceptsCharsets.js:49-60

### express-comparison-023 — test

Which automated test checks that Content-Type cannot be assigned an array?

- Proof 1: test/res.set.js:78-89

### express-comparison-024 — test

Which automated test checks that an automatic OPTIONS response includes each method only once?

- Proof 1: test/app.options.js:20-32

### express-comparison-025 — implementation

Locate the implementation that builds an application's full mount path by walking its parent chain.

- Proof 1: lib/application.js:399-403

### express-comparison-026 — implementation

Locate the implementation that reports whether the request protocol is secure.

- Proof 1: lib/request.js:326-328

### express-comparison-027 — implementation

Locate the implementation that turns named link relations into a response Link header.

- Proof 1: lib/response.js:98-111

### express-comparison-028 — example

In the runnable cookie-session demonstration, where is a counter persisted and incremented between visits?

- Proof 1: examples/cookie-sessions/index.js:12-19

### express-comparison-029 — example

In the runnable route-map demonstration, where are nested paths traversed and registered as routes?

- Proof 1: examples/route-map/index.js:14-29

### express-comparison-030 — example

In the runnable error-page demonstration, where are missing routes and application errors rendered with different statuses?

- Proof 1: examples/error-pages/index.js:63-77, examples/error-pages/index.js:91-97

## pallets/flask

### flask-comparison-001 — ambiguous

How can a prefix be removed from configuration keys while their letter case is normalized?

- Proof 1: src/flask/config.py:353-364
- Proof 2: tests/test_config.py:217-226

### flask-comparison-002 — ambiguous

Why does supplying a list of HTTP methods to a method-specific route decorator fail?

- Proof 1: src/flask/sansio/scaffold.py:284-293, src/flask/sansio/scaffold.py:295-301
- Proof 2: tests/test_basic.py:70-72

### flask-comparison-003 — ambiguous

Why is a dot forbidden in a blueprint name?

- Proof 1: src/flask/sansio/blueprints.py:198-199
- Proof 2: tests/test_blueprints.py:242-244

### flask-comparison-004 — ambiguous

Why does a precise decimal number appear as a quoted string in JSON?

- Proof 1: src/flask/json/provider.py:112-113
- Proof 2: tests/test_json.py:191-193

### flask-comparison-005 — ambiguous

How can an environment variable populate a deeply nested configuration dictionary?

- Proof 1: src/flask/config.py:152-183
- Proof 2: tests/test_config.py:79-107

### flask-comparison-006 — ambiguous

What happens when an early request callback returns a response before the view runs?

- Proof 1: src/flask/app.py:1366-1392
- Proof 2: tests/test_basic.py:735-759

### flask-comparison-007 — ambiguous

What determines whether markup in template values is automatically escaped?

- Proof 1: src/flask/sansio/app.py:546-548
- Proof 2: tests/test_templating.py:71-88

### flask-comparison-008 — ambiguous

How can request details remain available while response content is generated later?

- Proof 1: src/flask/helpers.py:126-148
- Proof 2: tests/test_helpers.py:229-240

### flask-comparison-009 — ambiguous

Can an existing signed session remain readable after changing the application's signing key?

- Proof 1: src/flask/sessions.py:303-321, src/flask/sessions.py:323-335
- Proof 2: tests/test_basic.py:396-421

### flask-comparison-010 — ambiguous

How does a timestamp become an HTTP-style date in a JSON response?

- Proof 1: src/flask/json/provider.py:108-110
- Proof 2: tests/test_json.py:133-142

### flask-comparison-011 — ambiguous

How is the same-site policy on the session cookie chosen?

- Proof 1: src/flask/sessions.py:208-213, src/flask/sessions.py:340-346, src/flask/sessions.py:374-384
- Proof 2: tests/test_basic.py:356-380

### flask-comparison-012 — ambiguous

Can configuration loading quietly ignore a missing environment-variable setting?

- Proof 1: src/flask/config.py:102-124
- Proof 2: tests/test_config.py:144-152

### flask-comparison-013 — multi_block

Trace the implementation from before-request callbacks through view dispatch to response processing and session saving.

- Proof 1: src/flask/app.py:1012-1019, src/flask/app.py:1039-1051, src/flask/app.py:1394-1418

### flask-comparison-014 — multi_block

Trace the implementation that runs request and application teardown and resets context bindings even if cleanup raises.

- Proof 1: src/flask/ctx.py:486-504, src/flask/helpers.py:665-674, src/flask/helpers.py:676-679

### flask-comparison-015 — multi_block

Trace the implementation that merges template context processors while preserving explicit values, then emits signals around rendering.

- Proof 1: src/flask/app.py:601-616, src/flask/templating.py:123-133

### flask-comparison-016 — multi_block

Trace the implementation that identifies an exception's status code and searches blueprint and application handlers through its inheritance hierarchy.

- Proof 1: src/flask/sansio/scaffold.py:686-687, src/flask/sansio/scaffold.py:703-706, src/flask/sansio/app.py:877-892

### flask-comparison-017 — multi_block

Trace the implementation from an unavailable session serializer to the null session that rejects writes.

- Proof 1: src/flask/sessions.py:303-305, src/flask/sessions.py:323-326, src/flask/ctx.py:381-393, src/flask/sessions.py:150-160, src/flask/sessions.py:89-94

### flask-comparison-018 — multi_block

Trace the implementation that streams a template while keeping its request context alive until iteration completes.

- Proof 1: src/flask/templating.py:163-178, src/flask/helpers.py:126-148

### flask-comparison-019 — test

Which automated test verifies that a pre-render signal can change the template context?

- Proof 1: tests/test_signals.py:25-47

### flask-comparison-020 — test

Which automated test ensures an already-handled earlier exception is not passed into a later request teardown?

- Proof 1: tests/test_reqctx.py:29-43

### flask-comparison-021 — test

Which automated test checks that JSON response construction rejects positional and keyword values together?

- Proof 1: tests/test_basic.py:1355-1358

### flask-comparison-022 — test

Which automated test verifies that package resources reject write-mode access?

- Proof 1: tests/test_helpers.py:363-368

### flask-comparison-023 — test

Which automated test verifies that refreshing an unchanged permanent session still adds a cookie variation header?

- Proof 1: tests/test_basic.py:603-617

### flask-comparison-024 — test

Which automated test checks that two registrations cannot share the same blueprint name?

- Proof 1: tests/test_blueprints.py:1063-1077

### flask-comparison-025 — implementation

Locate the implementation that forwards URL anchor, scheme and method options to the application.

- Proof 1: src/flask/helpers.py:244-251

### flask-comparison-026 — implementation

Locate the implementation that checks whether an application context is currently active.

- Proof 1: src/flask/ctx.py:257-257

### flask-comparison-027 — implementation

Locate the implementation that finds a module's resource root using its file or loader and falls back to the working directory.

- Proof 1: src/flask/helpers.py:596-617, src/flask/helpers.py:619-641

### flask-comparison-028 — example

In the runnable tutorial, where is the user identity saved in a session resolved before each request?

- Proof 1: examples/tutorial/flaskr/auth.py:33-43

### flask-comparison-029 — example

In the runnable tutorial, where is a missing post rejected and ownership checked before editing?

- Proof 1: examples/tutorial/flaskr/blog.py:28-57

### flask-comparison-030 — example

In the runnable tutorial sign-in flow, where is the password checked and the old session replaced with the authenticated identity?

- Proof 1: examples/tutorial/flaskr/auth.py:85-109
