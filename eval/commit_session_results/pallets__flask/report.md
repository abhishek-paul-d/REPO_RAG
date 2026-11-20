# Commit session report

| step | strict | lenient | abstained | mean search calls | get_commit calls | won | lost | sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control_a | 105/136 | 105/136 | 19 | 1.07 | 0 |  |  |  |
| control_b | 105/136 | 105/136 | 19 | 1.06 | 0 | 0 | 0 | 1.000 |
| fused | 105/136 | 105/136 | 17 | 1.05 | 0 | 7 | 7 | 1.000 |
| summary_fused_hidden | 110/136 | 110/136 | 14 | 1.05 | 0 | 14 | 9 | 0.405 |
| summary_fused_hidden_b | 110/136 | 110/136 | 14 | 1.05 | 0 | 14 | 9 | 0.405 |
| code_fallback | 107/136 | 107/136 | 17 | 1.09 | 0 | 6 | 4 | 0.754 |
| fused_code_fallback | 0/136 | 0/136 | 136 | 0.06 | 0 | 0 | 105 | 0.000 |

won/lost are questions passed/failed compared with control_a.

Noise: the two identical control runs differ on 0 of 136 questions.
