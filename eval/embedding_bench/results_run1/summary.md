# Embedding benchmark results

## nDCG@10 by task

|                                     |   T1_docstring_to_code |   T2_dev_questions |   mean |
|:------------------------------------|-----------------------:|-------------------:|-------:|
| ('bm25', 'lexical')                 |                  0.305 |              0.418 |  0.361 |
| ('coderankembed', 'dense')          |                  0.842 |              0.29  |  0.566 |
| ('coderankembed', 'hybrid_bm25')    |                  0.586 |              0.392 |  0.489 |
| ('nomic-embed-code', 'dense')       |                  0.892 |              0.277 |  0.585 |
| ('nomic-embed-code', 'hybrid_bm25') |                  0.596 |              0.413 |  0.505 |
| ('qwen3-0.6b', 'dense')             |                  0.813 |              0.558 |  0.686 |
| ('qwen3-0.6b', 'dense@256d')        |                  0.802 |              0.544 |  0.673 |
| ('qwen3-0.6b', 'dense@512d')        |                  0.818 |              0.572 |  0.695 |
| ('qwen3-0.6b', 'hybrid_bm25')       |                  0.583 |              0.534 |  0.558 |

## Hit@5 by task

|                                     |   T1_docstring_to_code |   T2_dev_questions |
|:------------------------------------|-----------------------:|-------------------:|
| ('bm25', 'lexical')                 |                  0.367 |              0.611 |
| ('coderankembed', 'dense')          |                  0.942 |              0.361 |
| ('coderankembed', 'hybrid_bm25')    |                  0.693 |              0.528 |
| ('nomic-embed-code', 'dense')       |                  0.97  |              0.361 |
| ('nomic-embed-code', 'hybrid_bm25') |                  0.693 |              0.556 |
| ('qwen3-0.6b', 'dense')             |                  0.924 |              0.75  |
| ('qwen3-0.6b', 'dense@256d')        |                  0.904 |              0.694 |
| ('qwen3-0.6b', 'dense@512d')        |                  0.916 |              0.722 |
| ('qwen3-0.6b', 'hybrid_bm25')       |                  0.665 |              0.694 |

## Speed and memory

| model            |   load_s |   dim |   corpus_docs |   corpus_encode_s |   docs_per_s |   tokens_per_s |   truncated_pct |   query_ms_p50_cuda |   query_ms_p95_cuda |   peak_gpu_gb |   query_ms_p50_cpu |   query_ms_p95_cpu |
|:-----------------|---------:|------:|--------------:|------------------:|-------------:|---------------:|----------------:|--------------------:|--------------------:|--------------:|-------------------:|-------------------:|
| coderankembed    |     38.9 |   768 |          5760 |              29.7 |        194   |          33398 |               0 |                16.8 |                19.3 |         26.82 |               67.1 |               83.2 |
| qwen3-0.6b       |     18   |  1024 |          5760 |              18.5 |        311.2 |          41534 |               0 |                44.4 |                49.4 |          9.4  |              268.6 |              293.3 |
| nomic-embed-code |     97.3 |  3584 |          5760 |              74.5 |         77.3 |          10242 |               0 |                36.8 |                39.3 |         17.08 |              nan   |              nan   |

## Other factors

| model            | hf_id                     | params   | license    | commercial_use   |   dim | mrl_dims   |   max_context | laptop_cpu_query_ok   |   index_mb_per_100k_chunks_fp32 |
|:-----------------|:--------------------------|:---------|:-----------|:-----------------|------:|:-----------|--------------:|:----------------------|--------------------------------:|
| coderankembed    | nomic-ai/CodeRankEmbed    | 137M     | MIT        | True             |   768 | -          |          8192 | True                  |                             307 |
| qwen3-0.6b       | Qwen/Qwen3-Embedding-0.6B | 0.6B     | Apache-2.0 | True             |  1024 | 256,512    |         32768 | True                  |                             410 |
| nomic-embed-code | nomic-ai/nomic-embed-code | 7B       | Apache-2.0 | True             |  3584 | -          |         32768 | False                 |                            1434 |

## T2 Hit@5 by question type

|                                     |   ident |   nl |   symptom |
|:------------------------------------|--------:|-----:|----------:|
| ('coderankembed', 'dense')          |     0.8 | 0.36 |     0     |
| ('coderankembed', 'hybrid_bm25')    |     1   | 0.48 |     0.333 |
| ('nomic-embed-code', 'dense')       |     0.8 | 0.36 |     0     |
| ('nomic-embed-code', 'hybrid_bm25') |     1   | 0.52 |     0.333 |
| ('qwen3-0.6b', 'dense')             |     1   | 0.72 |     0.667 |
| ('qwen3-0.6b', 'hybrid_bm25')       |     1   | 0.68 |     0.5   |
