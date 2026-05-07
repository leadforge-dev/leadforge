# v2 Decision Log — `leadforge-lead-scoring-v2`

This log tracks every external finding against
`leadforge-lead-scoring-v1` and the disposition the maintainer
took on each one. It exists so a contributor in 2027 can see
*why* a v2 design call was made (or why a v1 quirk was kept).

The log starts empty. The first real entry will be added when
the first issue lands; the schema below is what that entry
will fill in.

## Schema

Each row is one disposition. Add new rows at the bottom; never
edit historical entries.

| Field | Required | Format | Notes |
|---|---|---|---|
| `received_at` | yes | `YYYY-MM-DD` | Date the finding was received (issue opened / reviewer comment / direct message). Use the wall-clock date in the maintainer's timezone. |
| `source` | yes | one of `issue:#NNN`, `pr:#NNN`, `email`, `direct` | Where the finding came in. `issue` and `pr` link via the GitHub number. |
| `topic` | yes | one short phrase | What the finding is about — e.g. "expected_acv realism", "industry conversion rates", "cohort-by-segment drift". |
| `severity` | yes | `low` / `medium` / `high` | Reporter's claim, sanity-checked by the maintainer. `high` is the equivalent of the breakage-report `high` severity tier. |
| `verdict` | yes | one of `accepted-for-v2`, `deferred`, `wont-fix`, `needs-investigation` | See vocabulary below. |
| `next_step` | yes | one sentence | What concretely happens next (or has happened). Free-form but specific — "tracked in v2 milestone as #NNN", "documented as v1 simplification in dataset card", etc. |
| `link` | optional | URL or path | Pointer to the resulting commit, doc change, or v2 work item. Empty for `wont-fix` and `needs-investigation`. |

### Verdict vocabulary

| Verdict | When |
|---|---|
| `accepted-for-v2` | The finding is real and the fix lands in v2. There should be a linked v2 milestone work item. |
| `deferred` | The finding is real but the fix is post-v2 (or unsized). Counts as a backlog entry, not a v2 commitment. |
| `wont-fix` | The finding is correct but the design call is intentional. The dataset card or roadmap should already document it; if not, the entry should result in a doc update. |
| `needs-investigation` | The finding is plausible but not yet reproduced or scoped. Stays in this state for at most one cycle; the maintainer must promote it to one of the other three verdicts before declaring v2 ready. |

## Log

(no entries yet — first entry lands when the first external finding is received)
