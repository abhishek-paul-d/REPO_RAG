# Commit session report

| step | strict | lenient | abstained | mean search calls | get_commit calls | won | lost | sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control_a | 76/113 | 76/113 | 12 | 1.12 | 0 |  |  |  |
| control_b | 76/113 | 76/113 | 11 | 1.11 | 0 | 2 | 2 | 1.000 |
| fused | 79/113 | 80/113 | 10 | 1.06 | 0 | 6 | 3 | 0.508 |
| verify | 69/113 | 70/113 | 13 | 1.07 | 105 | 2 | 9 | 0.065 |
| fused_verify | 69/113 | 69/113 | 12 | 1.09 | 110 | 2 | 9 | 0.065 |
| fused_fix_only | 85/113 | 85/113 | 9 | 1.07 | 0 | 13 | 4 | 0.049 |
| summary_hidden | 78/113 | 78/113 | 10 | 1.07 | 0 | 10 | 8 | 0.815 |
| fused_w05 | 76/113 | 76/113 | 11 | 1.09 | 0 | 3 | 3 | 1.000 |
| summary_fused_hidden | 82/113 | 83/113 | 8 | 1.04 | 0 | 12 | 6 | 0.238 |
| control_c | 78/113 | 79/113 | 10 | 1.12 | 0 | 4 | 2 | 0.688 |
| summary_fused_hidden_b | 84/113 | 85/113 | 7 | 1.07 | 0 | 13 | 5 | 0.096 |

won/lost are questions passed/failed compared with control_a.

Caveat, fused_fix_only: every gold commit of this set passes the is_fix test by construction (the data protocol requires it), so this number is an upper bound for real bug reports
Noise: the two identical control runs differ on 4 of 113 questions.
