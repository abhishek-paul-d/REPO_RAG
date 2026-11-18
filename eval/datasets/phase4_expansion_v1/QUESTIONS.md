# Frozen expansion questions

Codex source-authored and author-checked; independent review pending

## requests-expansion-001

Trace how the top-level GET helper delegates a request and ensures its temporary session's adapters are closed afterwards.

Required together: src/requests/api.py:74-87, src/requests/api.py:67-71, src/requests/sessions.py:505-509, src/requests/sessions.py:883-886

## requests-expansion-002

Trace how copying a prepared request copies its cookies, including the fallback for a generic cookie jar without its own copy method.

Required together: src/requests/models.py:456-465, src/requests/cookies.py:479-491

## requests-expansion-003

Trace how environment proxy values fill missing request proxy entries and how the resulting map is combined with session settings and null entries removed.

Required together: src/requests/sessions.py:845-851, src/requests/sessions.py:862-868, src/requests/sessions.py:84-105

## requests-expansion-004

Trace how a digest-authentication challenge is saved and retried, and how the nonce counter becomes the eight-digit field in the new Authorization header.

Required together: src/requests/auth.py:291-312, src/requests/auth.py:162-166, src/requests/auth.py:227-237, src/requests/auth.py:250-266

## requests-expansion-005

Trace how line-by-line response iteration obtains byte chunks and preserves a final unfinished line when a line crosses a chunk boundary; leave Unicode decoding disabled.

Required together: src/requests/models.py:1010-1032, src/requests/models.py:935-956, src/requests/models.py:967-977

## requests-expansion-006

Trace how an HTTP error status produces an exception that retains both the response and, when not supplied separately, the response's originating request.

Required together: src/requests/models.py:1160-1171, src/requests/exceptions.py:21-35, src/requests/exceptions.py:66-67

## express-expansion-001

Trace how assigning the application's ETag setting selects and stores the weak or strong generator, up to its delegation to the external hashing library.

Required together: lib/application.js:359-366, lib/utils.js:130-152, lib/utils.js:40-40, lib/utils.js:51-51, lib/utils.js:249-257

## express-expansion-002

Trace how setting a numeric proxy trust count stores a compiled trust function and how that function decides whether a hop is trusted.

Required together: lib/application.js:359-360, lib/application.js:370-379, lib/utils.js:194-205

## express-expansion-003

Trace the extended-query-parser setting from application configuration through selection of its parser to the options passed to the external query-string parser.

Required together: lib/application.js:359-369, lib/utils.js:162-184, lib/utils.js:267-271

## express-expansion-004

Trace how setting an attachment filename derives its content type and writes the attachment disposition, including how the derived type is assigned to the response.

Required together: lib/response.js:609-618, lib/response.js:507-514, lib/response.js:670-692

## express-expansion-005

Trace how a redirect destination becomes the encoded Location header, is reused in the negotiated body, and yields no body for a HEAD request.

Required together: lib/response.js:797-799, lib/response.js:838-866

## express-expansion-006

Trace how application initialization installs a lazy router getter and how registering a route accesses that router rather than building a new one for each registration.

Required together: lib/application.js:59-82, lib/application.js:256-258

## flask-expansion-001

Trace a JSON response request with several positional values through argument normalization and the default provider's debug-mode formatting and response construction.

Required together: src/flask/json/__init__.py:170-170, src/flask/json/provider.py:75-87, src/flask/json/provider.py:205-215

## flask-expansion-002

Trace how loading configuration from a file hands the parsed mapping to the configuration object and accepts only uppercase keys.

Required together: src/flask/config.py:290-302, src/flask/config.py:314-321

## flask-expansion-003

Trace how a categorized flash message is written to the session and later retrieved only once into request-local storage, filtered, and returned with or without categories.

Required together: src/flask/helpers.py:348-350, src/flask/helpers.py:391-399

## flask-expansion-004

Trace how the file-response wrapper combines caller options with application and request defaults before handing them to Werkzeug; stop at that delegation boundary.

Required together: src/flask/helpers.py:528-540, src/flask/helpers.py:402-414

## flask-expansion-005

Trace how a blueprint URL rule is deferred, invoked during registration, and given its URL prefix and namespaced endpoint before being added to the application.

Required together: src/flask/sansio/blueprints.py:433-441, src/flask/sansio/blueprints.py:334-335, src/flask/sansio/blueprints.py:98-116

## flask-expansion-006

Trace how a newly created Jinja environment obtains its default autoescape callback and how that callback handles an unnamed template or an HTML filename regardless of case.

Required together: src/flask/app.py:479-494, src/flask/sansio/app.py:546-548

## click-expansion-001

Trace how option decorators attach parameters to a callback and how command construction consumes that stored list in declaration order.

Required together: src/click/decorators.py:370-377, src/click/decorators.py:314-321, src/click/decorators.py:221-230, src/click/decorators.py:248-250

## click-expansion-002

Trace command forwarding: how the current context fills missing explicit arguments and how the invoked command fills any remaining defaults before calling its callback.

Required together: src/click/core.py:952-959, src/click/core.py:907-940

## click-expansion-003

Trace how a typed object-passing decorator finds the nearest matching parent context object and, with creation enabled, constructs and stores one when no match exists.

Required together: src/click/decorators.py:76-95, src/click/core.py:770-789

## click-expansion-004

Trace registration of a context cleanup callback through explicit context closure, including execution of the exit stack and replacement of that stack for reuse.

Required together: src/click/core.py:707-724, src/click/core.py:738-742

## click-expansion-005

Trace how creating a command context parses arguments using a parser populated with the command's parameters, up to parameter-result handling.

Required together: src/click/core.py:1386-1394, src/click/core.py:1396-1404, src/click/core.py:1256-1261

## click-expansion-006

Trace how a parameter falls back from command-line and environment input to a context default map, including invoking a callable stored default and recording the source of the value.

Required together: src/click/core.py:2693-2726, src/click/core.py:825-835, src/click/core.py:798-803

## click-expansion-007

Trace integer-range conversion when clamping is enabled and a value lies outside an open bound, showing how the numeric conversion and adjusted bound are computed.

Required together: src/click/types.py:598-610, src/click/types.py:669-686, src/click/types.py:761-765

## click-expansion-008

Trace case-insensitive choice conversion through normalization of the input and available choices, returning the original choice object rather than the normalized spelling.

Required together: src/click/types.py:462-470, src/click/types.py:398-404, src/click/types.py:416-424

## click-expansion-009

Trace how converting an eagerly opened file parameter schedules closing or flushing at context shutdown, and how the scheduled wrapper suppresses cleanup exceptions.

Required together: src/click/types.py:1006-1021, src/click/core.py:707-717, src/click/utils.py:36-49, src/click/core.py:719-724, src/click/core.py:738-742

## click-expansion-010

Trace how a group's command decorator constructs a command, registers its chosen name, and rejects registration without a command name.

Required together: src/click/core.py:1881-1892, src/click/decorators.py:239-250, src/click/core.py:1835-1843

## click-expansion-011

Trace how two-column help entries are measured and padded before their descriptions are wrapped to the available width.

Required together: src/click/formatting.py:14-28, src/click/formatting.py:243-269, src/click/formatting.py:62-107

## click-expansion-012

Trace how a missing filesystem path is accepted when existence is optional and then converted to the requested string, bytes, or path-object result type.

Required together: src/click/types.py:1167-1180, src/click/types.py:1147-1159
