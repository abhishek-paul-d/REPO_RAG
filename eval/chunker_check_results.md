# Chunker check (budget 1500 non-ws chars)

Run 2026-10-22 with `eval/chunker_check.py`. Repos at: psf/requests `611c6162`, pallets/flask
`d086db85`, encode/httpx `b5addb64`, expressjs/express `0dee2c52`, trpc/trpc `1c9bf51e`,
google/gson `e2292f9c`, spf13/cobra `adbc8813`. Summary and history: docs/PROJECT_LOG.md.

## psf/requests  (0.3 s, duplicate chunk ids: 0)

| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| python | 37 | 274 | 274 / 0 | 775 | **0** | 0 | 0 | 0 | 1220 / 1469 / 2579 | 1 | 0 |

- **python over budget (largest)** (1): docs/_themes/flask_theme_support.py:7-86 [class] class FlaskyStyle(Style):

## pallets/flask  (0.4 s, duplicate chunk ids: 0)

| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| python | 83 | 429 | 429 / 0 | 1572 | **0** | 0 | 0 | 0 | 1163 / 1468 / 1500 | 0 | 0 |


## encode/httpx  (0.4 s, duplicate chunk ids: 0)

| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| python | 60 | 391 | 391 / 0 | 1206 | **0** | 0 | 0 | 0 | 1217 / 1467 / 1503 | 1 | 0 |

- **python over budget (largest)** (1): httpx/_client.py:595-637 [block] """

## expressjs/express  (0.5 s, duplicate chunk ids: 0)

| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| javascript | 141 | 384 | 384 / 0 | 291 | **0** | 0 | 0 | 0 | 1178 / 1454 / 1515 | 1 | 0 |

- **javascript over budget (largest)** (1): test/express.json.js:226-295 [module] describe('with strict option', function () {

## trpc/trpc  (2.2 s, duplicate chunk ids: 0)

| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| javascript | 25 | 38 | 38 / 0 | 15 | **0** | 0 | 0 | 0 | 588 / 1444 / 1485 | 0 | 0 |
| tsx | 236 | 563 | 563 / 0 | 980 | **0** | 0 | 0 | 0 | 1032 / 1453 / 4584 | 9 | 0 |
| typescript | 598 | 1510 | 1510 / 0 | 2880 | **0** | 0 | 0 | 0 | 1043 / 1449 / 2331 | 7 | 0 |

- **tsx over budget (largest)** (9): examples/next-prisma-todomvc/src/components/footer.tsx:55-55 [block] d="M60.4375 15.2266V26.5H61.8438V22.4766H64.6797C66.7969 22.4766 68.30; packages/react-query/test/invalidateRouters.test.tsx:266-361 [module] const utils = render(; packages/react-query/test/withTRPC.test.tsx:116-210 [module] test('useQuery: conditional ssr false', async () => {; packages/react-query/test/polymorphism.test.tsx:268-344 [module] test('can use a sub-typed factory router with the interfaces from the ; packages/tanstack-react-query/test/infiniteQueryOptions.test.tsx:328-389 [module] test('regression #6701: infinite suspense has non-optional getNextPage; packages/react-query/test/regression/issue-3455-56-invalidate-queries.test.tsx:40-110 [module] test('invalidate with filter', async () => {
- **typescript over budget (largest)** (7): examples/next-sse-chat/src/server/routers/post.ts:43-141 [module] infinite: publicProcedure; packages/server/src/unstable-core-do-not-import/procedureBuilder.ts:541-632 [function] export function createBuilder<TContext, TMeta, TErrorShape = unknown>(; packages/tests/server/regression/issue-5037-context-inference.test.ts:66-165 [module] test('using generically constructed tRPC instance should have correctl; packages/openapi/src/schemaExtraction.ts:137-201 [function] function walkZodShape(; packages/tests/server/caller.test.ts:4-85 [module] test('experimental caller', async () => {; packages/server/src/adapters/ws.ts:1-57 [class] import type { IncomingMessage } from 'http';

## google/gson  (1.2 s, duplicate chunk ids: 0)

| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| java | 264 | 1478 | 1478 / 0 | 4021 | **0** | 0 | 0 | 0 | 1179 / 1464 / 2889 | 8 | 0 |

- **java over budget (largest)** (8): gson/src/main/java/com/google/gson/internal/GsonTypes.java:348-445 [block] private static Type resolve(; gson/src/main/java/com/google/gson/FieldNamingPolicy.java:148-212 [block] /**; gson/src/test/java/com/google/gson/ParameterizedTypeFixtures.java:109-161 [block] implements JsonSerializer<MyParameterizedType<T>>, JsonDeserializer<My; gson/src/test/java/com/google/gson/GsonTest.java:100-153 [method] @Test; gson/src/test/java/com/google/gson/common/TestTypes.java:112-177 [class] public static class BagOfPrimitives {; gson/src/test/java/com/google/gson/JsonObjectAsMapTest.java:210-249 [method] @Test

## spf13/cobra  (0.4 s, duplicate chunk ids: 0)

| Language | Files | Chunks | AST / fallback | Defs checked | **Split** | Text mismatch | Lost-line files | Overlaps | Median / p90 / max size | Over budget | Anonymous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| go | 36 | 358 | 358 / 0 | 569 | **0** | 0 | 0 | 0 | 1280 / 1479 / 4933 | 4 | 0 |

- **go over budget (largest)** (4): flag_groups_test.go:22-161 [block] func TestValidateFlagGroups(t *testing.T) {; powershell_completions.go:77-152 [block] $ShellCompDirectiveNoFileComp=%[6]d; doc/rest_docs.go:62-130 [function] func GenReSTCustom(cmd *cobra.Command, w io.Writer, linkHandler func(s; completions_test.go:4019-4074 [function] func TestCustomDefaultShellCompDirective(t *testing.T) {
