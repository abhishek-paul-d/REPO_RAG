# Plan to finish the project (written 2026-11-08)

State: Phases 0 to 5b ticked. Open: 6, 7, 8, 9. Nothing committed. Estimates are mine, not
measured; "session" means one sitting of mine, "GPU session" means one Colab A100 serve or
indexing run that you start. Wall-clock depends mostly on your availability for the GPU
sessions and the label review.

## Decisions only you can make

| # | Decision | Needed by | Default if you do not decide |
|---|---|---|---|
| D1 | Phase 8 "done when" (paste a URL, wait, chat): keep as written (build a CPU background indexer, slow) or change to "build the index in Colab, upload the zip" | Stage 3 | Change the wording; plan.md already says it is unmet |
| D2 | Who reviews the 30 architecture labels (about 2 to 4 hours of a person) | Stage 4 | Phase 6 stays unticked |
| D3 | A judge model for the Phase 7 answer-quality check (local open-source only) | Stage 2 | **Decided 2026-11-10:** quantized Qwen3.8-27B (user wrote "Qwen3.7-28B"). The Phase 4 judge is a reranker, so it does not fit |
| D4 | Which Phase 9 items to do (see Stage 6) | Stage 6 | README + incremental re-index + MCP |
| D5 | Adding the `mcp` dependency, and `lancedb` if chosen | Stage 6 | Ask when we get there |
| D6 | When to commit | any time | Nothing is committed until you say |

## Stage 0: housekeeping (0.5 session)

- Confirm the full suite is green; replace the Drive zip with the current hash.
- Check the Streamlit UI in the built-in browser against the real API (never done).
- Commit when you say (a lot is uncommitted).

## Stage 1: GPU session A, A/B repeats (1 session + 1 GPU session of about 2 h)

- Baseline, `docs_hint_narrow`, `prs_narrow`: code, docs, prsearch, 3 repeats each (9 passes,
  about 10 min each).
- Read it as: how often does a question flip by chance, and does a variant beat baseline by more
  than that. Adopt a variant only if it does, and say so with the repeat data.
- Done when: the default prompt is decided and logged, whichever way.

## Stage 2: close Phase 7 (3 to 5 sessions, 2 GPU sessions)

1. ~~Fix wrong-range citations (code-016, code-037)~~ Diagnosed 2026-11-09: not range bugs. The gold
   ranges are correct and the model's cited range equals a real chunk. 016 is a selection error among the
   top 5; 037 is a retrieval miss (gold chunk at rank 7, tool shows 5). See PROJECT_LOG 2026-11-09.
2. Commit questions at 3/10: diagnose before changing anything. Compare the model's own queries with
   and without the reranker, per question (1 to 2 sessions, GPU). It may not be fixable; if the
   cause is the model, record that and stop.
3. LLM-judge answer-quality check on the golden set (faithfulness and correctness), small and cheap
   (1 to 2 sessions to build, 1 GPU session to run). D3 decided. **Built 2026-11-10, awaiting the GPU run:**
   `notebooks/answer_judge_on_colab.ipynb` + `notebooks/answer_judge_bundle.zip`.
4. Show all §1 sample questions give cited, sensible answers on the demo repo; the architecture
   ones depend on Stage 4.
5. Tick Phase 7 when: fact gate (met, 59/60), judge check run and recorded, §1 questions pass.

## Stage 3: close Phase 8 (1 to 2 sessions)

- D1 decided 2026-11-09 (user): **change the wording**. plan.md Phase 8 is reworded (done). Tick Phase 8
  after the remaining check below.
- **Stage 3 done 2026-11-10:** the import-while-asking check was run offline (409 for repos being
  replaced), and the user ticked Phase 8 in plan.md.
- ~~If D1 is "build it"~~ Not taken (D1 = change the wording).
- ~~Check asking while an import runs.~~ Done.

## Stage 4: close Phase 6 (1 to 2 sessions + your review time + 1 GPU session)

1. GPU session: run the E3 notebook (Qwen3-4B-Instruct-2507, approved) (1 to 2 h).
2. D2: a person reviews the 30 labels.
3. Run the coherent-architecture check; tick Phase 6 only if it passes. **Correction 2026-11-10:** this check
   was not frozen anywhere. It is now built (`agent_eval.py build-architecture`, 15 requests questions,
   coverage of gold files). **Gate decided 2026-11-10 (user): at least 12 of 15, plus you read the answers.**
   **Run 2026-11-10: 5/15, gate not met** (one `search_code` call per question; answers name 1 of the needed
   files). Phase 6 stays unticked; see PROJECT_LOG 2026-11-10 for what was and was not tried.
   If the router still lowers recall, the honest result is "built, not adopted", and the criterion
   may need your decision.

## Stage 5: second repository (2 to 3 sessions, 1 to 2 GPU sessions)

- Index a second repo on Colab, sync its PRs and issues, build a small question set, rerun the
  agent eval. Without it every number here is from one repo (psf/requests, plus httpx facts).

## Stage 6: Phase 9 (pick from this list; D4)

| Item | Effort | Note |
|---|---|---|
| README with architecture diagram and eval table | 1 session | **Done 2026-11-10** (no demo GIF; refresh the eval table after the judge run and the second repo) |
| Incremental re-index (diff since last SHA) | 1 to 2 sessions | **Built 2026-11-10** as `repo-rag update` (vector reuse + plan + refresh); not yet run on a real embedding job |
| MCP server (`search_code`, `search_commits`) | 1 session | Needs D5 |
| GitHub webhook / Action to re-index on push | 1 session | Depends on the incremental re-index |
| LanceDB backend + comparison with Chroma | 2 to 3 sessions | Needs D5 and a comparison eval |
| Live GitHub tools (CI status, open PR checks) | 1 to 2 sessions | Needs network at query time |

## Totals

| Scope | Sessions (mine) | GPU sessions (yours) | Elapsed |
|---|---|---|---|
| Tick Phases 6, 7, 8 (Stages 0 to 4) | about 8 to 13 | about 4 to 5 | about 2 to 3 weeks |
| Plus a second repo (Stage 5) | +2 to 3 | +1 to 2 | +3 to 5 days |
| Plus Phase 9 minimal (README, incremental re-index, MCP) | +3 to 4 | 0 | +4 to 6 days |
| Plus all of Phase 9 | +7 to 11 | 0 | +2 to 3 weeks |

Biggest uncertainties: whether commit questions are fixable at all (Stage 2.2), whether the
architecture router ever passes its frozen gate (Stage 4), the time a person needs for D2, and how
fast you can run GPU sessions. Any of these can add a week or turn a tick into "built, not adopted".
