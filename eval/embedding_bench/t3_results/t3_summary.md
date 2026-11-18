# T3 tuning (qwen3-0.6b)

## Dev nDCG@10 at 512 dims (selection grid)

| instruction       |   diff2000 |   message |   symbols |   symbols_diff800 |
|:------------------|-----------:|----------:|----------:|------------------:|
| baseline          |      0.5   |     0.285 |     0.419 |             0.482 |
| code_instruction  |      0.524 |     0.293 |     0.436 |             0.507 |
| issue_to_commit   |      0.45  |     0.277 |     0.396 |             0.46  |
| none              |      0.501 |     0.295 |     0.429 |             0.507 |
| symptom_to_change |      0.511 |     0.293 |     0.407 |             0.504 |

## Test nDCG@10 at 512 dims (reported, not used for selection)

| instruction       |   diff2000 |   message |   symbols |   symbols_diff800 |
|:------------------|-----------:|----------:|----------:|------------------:|
| baseline          |      0.464 |     0.391 |     0.469 |             0.464 |
| code_instruction  |      0.482 |     0.394 |     0.449 |             0.482 |
| issue_to_commit   |      0.437 |     0.369 |     0.443 |             0.45  |
| none              |      0.455 |     0.391 |     0.463 |             0.447 |
| symptom_to_change |      0.464 |     0.402 |     0.458 |             0.476 |

## Doc formats

| doc_format      |   median_tokens |   encode_s |   docs_per_s |
|:----------------|----------------:|-----------:|-------------:|
| diff2000        |             121 |       27.8 |        175.6 |
| symbols_diff800 |             129 |       19.4 |        252.1 |
| symbols         |              24 |        8.4 |        581.8 |
| message         |               7 |        8   |        612   |

## Verdict

```json
{
  "model": "qwen3-0.6b",
  "selected_on": "dev nDCG@10 at 512 dims",
  "best": {
    "instruction": "code_instruction",
    "doc_format": "diff2000"
  },
  "baseline": {
    "instruction": "baseline",
    "doc_format": "diff2000"
  },
  "test_ndcg@10": {
    "baseline": 0.4643,
    "best": 0.4815
  },
  "test_hit@5": {
    "baseline": 0.5278,
    "best": 0.5417
  },
  "all_ndcg@10": {
    "baseline": 0.4822,
    "best": 0.5029
  },
  "paired_bootstrap_test": {
    "mean_gain": 0.0172,
    "ci95_low": -0.0166,
    "ci95_high": 0.0503,
    "p_gain_le_0": 0.1616
  },
  "n_dev": 72,
  "n_test": 72,
  "n_commits": 4882
}
```
