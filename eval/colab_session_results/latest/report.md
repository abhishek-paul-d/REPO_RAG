# Colab session report

| step | architecture | code | commit | docs | history | kind | probe | prsearch | refs | seconds/question |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh_default |  | 29/30 (abst 0, fail 0) |  | 10/10 (abst 0, fail 0) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 5/5 (abst 5, fail 0) | 6/6 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 6.1 |
| fresh_guards |  | 29/30 (abst 0, fail 0) |  | 10/10 (abst 0, fail 0) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 5/5 (abst 5, fail 0) | 6/6 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 6.7 |
| commit_default |  |  | 74/113 (abst 14, fail 2) |  |  |  |  |  |  | 9.8 |
| commit_explain |  |  | 78/113 (abst 10, fail 1) |  |  |  |  |  |  | 9.9 |
| regression_default |  | 40/40 (abst 0, fail 0) | 7/10 (abst 3, fail 1) | 11/12 (abst 0, fail 0) | 11/11 (abst 0, fail 0) | 5/5 (abst 0, fail 0) | 5/5 (abst 5, fail 0) | 10/10 (abst 0, fail 0) | 10/10 (abst 0, fail 0) | 6.6 |
| architecture_v1 | 12/15 (abst 1, fail 1) |  |  |  |  |  |  |  |  | 12.5 |
| architecture_fresh | 9/10 (abst 0, fail 0) |  |  |  |  |  |  |  |  | 12.3 |
| fresh_repeat_2 |  | 29/30 (abst 0, fail 0) |  | 10/10 (abst 0, fail 0) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 5/5 (abst 5, fail 0) | 6/6 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 5.9 |
| fresh_repeat_3 |  | 29/30 (abst 0, fail 0) |  | 10/10 (abst 0, fail 0) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 5/5 (abst 5, fail 0) | 6/6 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 5.9 |

- **fresh_default -> fresh_guards:** 0 questions gained [], 0 lost []
- **fresh_default -> fresh_repeat_2:** 0 questions gained [], 0 lost []
- **fresh_repeat_2 -> fresh_repeat_3:** 0 questions gained [], 0 lost []
- **commit_default -> commit_explain:** 6 questions gained ['commitfull-006', 'commitfull-011', 'commitfull-019', 'commitfull-031', 'commitfull-083', 'commitfull-106'], 2 lost ['commitfull-036', 'commitfull-061']
- **Noise on the fresh set** (fresh_default, fresh_repeat_2, fresh_repeat_3): passes [72, 72, 72], spread 0 questions
