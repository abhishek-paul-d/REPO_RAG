# Colab session report

| step | architecture | code | commit | docs | history | kind | probe | prsearch | refs | seconds/question |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh_default |  | 0/30 (abst 30, fail 30) |  | 0/10 (abst 10, fail 10) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 0/5 (abst 5, fail 5) | 0/6 (abst 6, fail 6) | 4/4 (abst 0, fail 0) | 7.5 |
| fresh_guards |  | 0/30 (abst 30, fail 30) |  | 0/10 (abst 10, fail 10) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 0/5 (abst 5, fail 5) | 0/6 (abst 6, fail 6) | 4/4 (abst 0, fail 0) | 12.3 |
| commit_default |  |  | 0/113 (abst 113, fail 113) |  |  |  |  |  |  | 10.1 |
| commit_explain |  |  | 25/113 (abst 81, fail 76) |  |  |  |  |  |  | 9.0 |
| regression_default |  | 0/40 (abst 40, fail 40) | 0/10 (abst 10, fail 10) | 0/12 (abst 12, fail 12) | 11/11 (abst 0, fail 0) | 5/5 (abst 0, fail 0) | 0/5 (abst 5, fail 5) | 0/10 (abst 10, fail 10) | 10/10 (abst 0, fail 0) | 7.1 |
| architecture_v1 | 0/15 (abst 15, fail 15) |  |  |  |  |  |  |  |  | 12.9 |
| architecture_fresh | 0/10 (abst 10, fail 10) |  |  |  |  |  |  |  |  | 14.3 |
| fresh_repeat_2 |  | 10/30 (abst 20, fail 20) |  | 5/10 (abst 5, fail 5) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 2/5 (abst 5, fail 3) | 3/6 (abst 3, fail 3) | 4/4 (abst 0, fail 0) | 6.0 |
| fresh_repeat_3 |  | 28/30 (abst 1, fail 1) |  | 10/10 (abst 0, fail 0) | 14/14 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 5/5 (abst 5, fail 0) | 6/6 (abst 0, fail 0) | 4/4 (abst 0, fail 0) | 6.8 |

- **fresh_default -> fresh_guards:** 0 questions gained [], 0 lost []
- **fresh_default -> fresh_repeat_2:** 20 questions gained ['fresh2-code-001', 'fresh2-code-005', 'fresh2-code-009', 'fresh2-code-013', 'fresh2-code-014', 'fresh2-code-018', 'fresh2-code-019', 'fresh2-code-023', 'fresh2-code-025', 'fresh2-code-028', 'fresh2-docs-032', 'fresh2-docs-034', 'fresh2-docs-035', 'fresh2-docs-038', 'fresh2-docs-040', 'fresh2-probe-072', 'fresh2-probe-073', 'fresh2-prsearch-063', 'fresh2-prsearch-067', 'fresh2-prsearch-068'], 0 lost []
- **fresh_repeat_2 -> fresh_repeat_3:** 30 questions gained ['fresh2-code-002', 'fresh2-code-003', 'fresh2-code-004', 'fresh2-code-006', 'fresh2-code-007', 'fresh2-code-008', 'fresh2-code-010', 'fresh2-code-011', 'fresh2-code-012', 'fresh2-code-015', 'fresh2-code-016', 'fresh2-code-017', 'fresh2-code-020', 'fresh2-code-021', 'fresh2-code-022', 'fresh2-code-024', 'fresh2-code-026', 'fresh2-code-027', 'fresh2-code-030', 'fresh2-docs-031', 'fresh2-docs-033', 'fresh2-docs-036', 'fresh2-docs-037', 'fresh2-docs-039', 'fresh2-probe-069', 'fresh2-probe-070', 'fresh2-probe-071', 'fresh2-prsearch-064', 'fresh2-prsearch-065', 'fresh2-prsearch-066'], 1 lost ['fresh2-code-001']
- **commit_default -> commit_explain:** 25 questions gained ['commitfull-002', 'commitfull-008', 'commitfull-009', 'commitfull-014', 'commitfull-022', 'commitfull-030', 'commitfull-042', 'commitfull-044', 'commitfull-045', 'commitfull-049', 'commitfull-053', 'commitfull-056', 'commitfull-062', 'commitfull-063', 'commitfull-075', 'commitfull-076', 'commitfull-079', 'commitfull-088', 'commitfull-091', 'commitfull-093', 'commitfull-101', 'commitfull-103', 'commitfull-104', 'commitfull-108', 'commitfull-109'], 0 lost []
- **Noise on the fresh set** (fresh_default, fresh_repeat_2, fresh_repeat_3): passes [22, 42, 71], spread 49 questions
