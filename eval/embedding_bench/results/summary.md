# Embedding benchmark results

## nDCG@10 by task

|                                     |   T1_docstring_to_code |   T2_dev_questions |   T2r_dev_questions_in_repo |   T3_bug_to_fix |   mean |
|:------------------------------------|-----------------------:|-------------------:|----------------------------:|----------------:|-------:|
| ('bm25', 'lexical')                 |                  0.305 |              0.418 |                       0.522 |           0.347 |  0.398 |
| ('coderankembed', 'dense')          |                  0.842 |              0.29  |                       0.416 |           0.436 |  0.496 |
| ('coderankembed', 'hybrid_bm25')    |                  0.586 |              0.392 |                       0.549 |           0.443 |  0.493 |
| ('nomic-embed-code', 'dense')       |                  0.892 |              0.277 |                       0.431 |           0.596 |  0.549 |
| ('nomic-embed-code', 'hybrid_bm25') |                  0.596 |              0.413 |                       0.576 |           0.524 |  0.527 |
| ('qwen3-0.6b', 'dense')             |                  0.813 |              0.558 |                       0.736 |           0.488 |  0.649 |
| ('qwen3-0.6b', 'dense@256d')        |                  0.802 |              0.544 |                       0.726 |           0.463 |  0.634 |
| ('qwen3-0.6b', 'dense@512d')        |                  0.818 |              0.572 |                       0.738 |           0.482 |  0.653 |
| ('qwen3-0.6b', 'hybrid_bm25')       |                  0.583 |              0.534 |                       0.704 |           0.466 |  0.572 |

## Hit@5 by task

|                                     |   T1_docstring_to_code |   T2_dev_questions |   T2r_dev_questions_in_repo |   T3_bug_to_fix |
|:------------------------------------|-----------------------:|-------------------:|----------------------------:|----------------:|
| ('bm25', 'lexical')                 |                  0.367 |              0.611 |                       0.806 |           0.438 |
| ('coderankembed', 'dense')          |                  0.942 |              0.361 |                       0.556 |           0.521 |
| ('coderankembed', 'hybrid_bm25')    |                  0.693 |              0.528 |                       0.778 |           0.535 |
| ('nomic-embed-code', 'dense')       |                  0.97  |              0.361 |                       0.722 |           0.688 |
| ('nomic-embed-code', 'hybrid_bm25') |                  0.693 |              0.556 |                       0.778 |           0.604 |
| ('qwen3-0.6b', 'dense')             |                  0.924 |              0.75  |                       0.889 |           0.535 |
| ('qwen3-0.6b', 'dense@256d')        |                  0.904 |              0.694 |                       0.917 |           0.562 |
| ('qwen3-0.6b', 'dense@512d')        |                  0.916 |              0.722 |                       0.861 |           0.549 |
| ('qwen3-0.6b', 'hybrid_bm25')       |                  0.665 |              0.694 |                       0.917 |           0.569 |

## Speed and memory

| model            |   load_s |   dim |   corpus_docs |   corpus_encode_s |   docs_per_s |   tokens_per_s |   truncated_pct |   query_ms_p50_cuda |   query_ms_p95_cuda |   peak_gpu_gb |   query_ms_p50_cpu |   query_ms_p95_cpu |
|:-----------------|---------:|------:|--------------:|------------------:|-------------:|---------------:|----------------:|--------------------:|--------------------:|--------------:|-------------------:|-------------------:|
| coderankembed    |     31.5 |   768 |         10642 |              65.2 |        163.2 |          32724 |               0 |                15.8 |                16.5 |         26.82 |               64.4 |               70.7 |
| qwen3-0.6b       |     13.3 |  1024 |         10642 |              44.5 |        239   |          38986 |               0 |                42.5 |                43.7 |          9.4  |              257.9 |              272   |
| nomic-embed-code |     89.7 |  3584 |         10642 |             173.8 |         61.2 |           9924 |               0 |                36.1 |                36.7 |         17.08 |              nan   |              nan   |

## Other factors

| model            | hf_id                     | params   | license    | commercial_use   |   dim | mrl_dims   |   max_context | laptop_cpu_query_ok   |   index_mb_per_100k_chunks_fp32 |
|:-----------------|:--------------------------|:---------|:-----------|:-----------------|------:|:-----------|--------------:|:----------------------|--------------------------------:|
| coderankembed    | nomic-ai/CodeRankEmbed    | 137M     | MIT        | True             |   768 | -          |          8192 | True                  |                             307 |
| qwen3-0.6b       | Qwen/Qwen3-Embedding-0.6B | 0.6B     | Apache-2.0 | True             |  1024 | 256,512    |         32768 | True                  |                             410 |
| nomic-embed-code | nomic-ai/nomic-embed-code | 7B       | Apache-2.0 | True             |  3584 | -          |         32768 | False                 |                            1434 |

## T2 Hit@5 by question type

|                                                                  |   ident |   nl |   symptom |
|:-----------------------------------------------------------------|--------:|-----:|----------:|
| ('T2_dev_questions', 'coderankembed', 'dense')                   |     0.8 | 0.36 |     0     |
| ('T2_dev_questions', 'coderankembed', 'hybrid_bm25')             |     1   | 0.48 |     0.333 |
| ('T2_dev_questions', 'nomic-embed-code', 'dense')                |     0.8 | 0.36 |     0     |
| ('T2_dev_questions', 'nomic-embed-code', 'hybrid_bm25')          |     1   | 0.52 |     0.333 |
| ('T2_dev_questions', 'qwen3-0.6b', 'dense')                      |     1   | 0.72 |     0.667 |
| ('T2_dev_questions', 'qwen3-0.6b', 'hybrid_bm25')                |     1   | 0.68 |     0.5   |
| ('T2r_dev_questions_in_repo', 'coderankembed', 'dense')          |     0.8 | 0.6  |     0.167 |
| ('T2r_dev_questions_in_repo', 'coderankembed', 'hybrid_bm25')    |     1   | 0.76 |     0.667 |
| ('T2r_dev_questions_in_repo', 'nomic-embed-code', 'dense')       |     1   | 0.72 |     0.5   |
| ('T2r_dev_questions_in_repo', 'nomic-embed-code', 'hybrid_bm25') |     1   | 0.76 |     0.667 |
| ('T2r_dev_questions_in_repo', 'qwen3-0.6b', 'dense')             |     1   | 0.88 |     0.833 |
| ('T2r_dev_questions_in_repo', 'qwen3-0.6b', 'hybrid_bm25')       |     1   | 0.88 |     1     |
