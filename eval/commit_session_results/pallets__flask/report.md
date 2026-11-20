# Commit session report

| step | strict | lenient | abstained | mean search calls | get_commit calls | won | lost | sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control_a | 101/136 | 101/136 | 23 | 1.02 | 0 |  |  |  |
| control_b | 100/136 | 100/136 | 24 | 1.01 | 0 | 2 | 3 | 1.000 |
| fused | 103/136 | 103/136 | 22 | 1.01 | 0 | 7 | 5 | 0.774 |
| control_c | 101/136 | 101/136 | 24 | 1.02 | 0 | 2 | 2 | 1.000 |
| summary_fused_hidden | 106/136 | 106/136 | 18 | 1.01 | 0 | 12 | 7 | 0.359 |
| summary_fused_hidden_b | 107/136 | 107/136 | 17 | 1.01 | 0 | 12 | 6 | 0.238 |

won/lost are questions passed/failed compared with control_a.

Noise: the two identical control runs differ on 5 of 136 questions.
