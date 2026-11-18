# Embedding model benchmark

Picks the embedding model for `code_chunks`, `docs` and `commits`. It runs on a Colab A100.

## Candidates

| Key | Model | Size | License | Why it's in |
|---|---|---|---|---|
| `coderankembed` | [nomic-ai/CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) | 137M | MIT | Small code specialist. Can embed queries on a laptop CPU |
| `qwen3-0.6b` | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.6B | Apache-2.0 | General-purpose, strong on code. Task instructions, 32k context, Matryoshka dims |
| `nomic-embed-code` | [nomic-ai/nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code) | 7B | Apache-2.0 | Quality ceiling. Reported on par with voyage-code-3 on CodeSearchNet. Needs a GPU at query time |
| `voyage-code-3` | Voyage API | n/a | Proprietary | Optional paid reference (our current config default). Runs only if `VOYAGE_API_KEY` is set |
| `bm25` | rank-bm25 | n/a | n/a | Lexical baseline. Every model is also scored as a **BM25 hybrid** (RRF), which is what production will use |

Left out: **jina-code-embeddings** (0.5B/1.5B), whose license is **CC-BY-NC-4.0**, so it can't be
used commercially. Our `config.py` currently names `jina-code-embeddings-0.5b` as the local
fallback, so that should change whatever this benchmark shows.

## What is measured

**Quality** (Hit@1/5/10, MRR@10, nDCG@10), over one shared corpus of about 2.9k Python
function/class chunks from requests, flask and httpx, all pinned to fixed SHAs. httpx's
look-alike code makes hard negatives for requests questions.

| Task | Queries | Gold |
|---|---|---|
| T1 docstring → code | ~500, generated automatically | The function the docstring came from. Docstrings are stripped from the corpus |
| T2 developer questions | 36 hand-written (25 natural-language, 5 naming identifiers, 6 bug-symptom style) | Symbols checked against the pinned source |
| T3 bug report → fix | Up to 200 GitHub issues (title + body) | Commits linked by "fixes #N", directly or through the PR's merge. Issue refs are scrubbed from the commit text |

**Speed and cost**: model load time, corpus throughput (docs/s, tokens/s, bf16 on A100),
single-query latency on GPU and on CPU (CPU only for laptop-viable models), peak GPU memory,
and the share of chunks over the 2,048-token cap.

**Other factors**: license and commercial use, embedding dim and index size per 100k chunks,
Matryoshka support (Qwen3 is also scored at 256 and 512 dims, with no re-encoding), max context,
and whether queries can be embedded on a laptop CPU.

The chunker is a fixed stdlib-`ast` chunker, the same for every model. It is **not** the Phase 2
tree-sitter chunker; the chunker comparison is a separate eval (Phase 4).

## Run it

1. Rebuild the notebook if `bench.py` changed: `uv run python eval/embedding_bench/build_notebook.py`
2. Open `embedding_bench.ipynb` in Colab (File → Upload notebook) and pick an **A100** runtime.
3. Optional secrets: `GITHUB_TOKEN` (enables T3), `VOYAGE_API_KEY`, `HF_TOKEN`.
4. Runtime → Run all. The last cell downloads `results.zip`. Unzip it into
   `eval/embedding_bench/results/`.

Outputs: `summary.md` (pivot tables), `metrics.csv`, `speed.csv`, `factors.csv`,
`t2_per_question.csv` (which questions each model misses), `quality_vs_speed.png` and
`run_meta.json`.

## How to decide

Production retrieval is **hybrid** (dense + BM25), so compare the `hybrid_bm25` rows first.

1. **Hard constraints first.**
   - The license must allow commercial use.
   - Query embedding must fit the deployment: on a laptop CPU, single-query p50 should be under ~300 ms. A 7B model rules out laptop-only deployment and needs a GPU or a hosted endpoint at query time.
2. **Quality on T2 and T3 counts most.** They are this project's real questions. T1 is a
   CodeSearchNet-style sanity check, and docstrings are often near-paraphrases of the code.
3. **A big model must earn its cost.** If the 7B model beats the next one by less than about
   0.03 mean nDCG@10 (hybrid), take the smaller model. With 36 T2 questions, one question
   moves Hit@5 by about 2.8 points, so treat T2 gaps under ~0.05 as noise.
4. **Tie-breakers:** index size (dim), Matryoshka support (smaller indexes later), indexing
   throughput (re-index time for big repos), and max context (long functions).

Record the decision and its numbers in `eval/RESULTS.md` and update `embedding_*` in
`src/repo_rag/config.py`.
