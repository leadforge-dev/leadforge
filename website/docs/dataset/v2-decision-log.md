---
sidebar_position: 5
title: v2 decision log
---

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

| received_at | source | topic | severity | verdict | next_step | link |
|---|---|---|---|---|---|---|
| 2026-05-08 | pr:#76 | F002 — Gaussian noise on float features produces non-physical values (negative ACV, negative day-deltas, day-deltas > snapshot_day=30) without disclosure in `dataset_card.md` Caveats | medium | accepted-for-v2 | Add a "Noise artefacts" bullet to the per-tier `dataset_card.md` Caveats section in v2. Requires touching `leadforge/narrative/dataset_card.py` (auto-rendered file), so out of scope for PR 7.1's no-bundle-regen rule | release/validation/llm_critique_raw_20260508T204359.124834Z.json#F002 |
| 2026-05-08 | pr:#76 | F003 — `release/README.md` `](../foo)` relative links would 404 on Kaggle / Hugging Face if shipped as-is | medium | wont-fix | Already treated by `scripts/_release_common.py::rewrite_release_links()` — both platform packagers (PR 5.1, 5.2) rewrite `](../foo)` → GitHub blob URL at packaging time before the README is inlined onto Kaggle / HF; the as-committed `release/README.md` keeps the relative paths so it renders correctly on github.com. The LLM critique didn't have visibility into the platform packagers (intentional — they're not in the input bundle) and made a wrong inference | scripts/_release_common.py |
| 2026-05-08 | pr:#76 | F005 — `calibration_max_bin_error = 0.5234` on advanced tier is driven by an n=2 high-probability bin; `validation_report.md` headline table reports the value with no minimum-bin-count footnote | medium | accepted-for-v2 | Either compute `calibration_max_bin_error` only over bins with `n >= 20`, OR expose both raw and n-weighted variants and add a footnote. Not a 1-line change — touches `leadforge/validation/release_quality.py`'s metric definition and would require regenerating `validation_report.{json,md}`, which PR 7.1's brief explicitly forbids ("`validation_report.{json,md}` should not need regeneration for this PR") | release/validation/llm_critique_raw_20260508T204359.124834Z.json#F005 |
| 2026-05-08 | pr:#76 | Missing — Datasheets §Biases enumeration in `release/README.md` (industry/region/persona uniformity, channel-conditional independence) | medium | accepted-for-v2 | The README's "Known limitations" lists individual symptoms (weak channel signal, flat AUC across tiers); a dedicated §Biases section listing the *generative* bias axes is a v2 polish item | release/validation/llm_critique_raw_20260508T204359.124834Z.json#missing-biases |
| 2026-05-08 | pr:#76 | Missing — Datasheets §Privacy in `release/README.md` (no real CRM seed, no PII-shaped strings, public-artefacts-only reproducibility) | medium | accepted-for-v2 | The README treats "fictional" as sufficient privacy disclosure; an explicit Privacy section will land in v2 alongside §Biases | release/validation/llm_critique_raw_20260508T204359.124834Z.json#missing-privacy |
| 2026-05-08 | pr:#76 | Missing — per-bundle `dataset_card.md` Group-split warning section disclosing `account_id` / `contact_id` overlap | high | accepted-for-v2 | The README-side warning is added in PR 7.1 (resolves F001's load-bearing path); replicating it into the auto-rendered per-tier `dataset_card.md` requires the same `leadforge/narrative/dataset_card.py` change as F002 and lands in v2 | release/README.md ("Group-leakage warning"), release/validation/llm_critique_raw_20260508T204359.124834Z.json#missing-group-split |
| 2026-05-08 | pr:#76 | Q1 — does the simulator window event tables before or after Gaussian-noise injection on float features (the 43.46-day `days_since_first_touch` finding) | low | wont-fix | Intended noise artefact, not a windowing bug. Float features pass through `_apply_difficulty_distortions()` *after* snapshot-window aggregation, so additive Gaussian noise on `days_since_first_touch` can push the value past the 30-day snapshot. F002 captures the disclosure side; the mechanism itself is correct | leadforge/mechanisms/measurement.py |
| 2026-05-08 | pr:#76 | Q2 — `top_decile_rate` naming clarity (precision-at-top-10 vs recall-at-top-10) | low | accepted-for-v2 | Rename to `top_decile_precision` (current implementation is precision at top 10 %) in v2 alongside any other release-quality field renames; touches `leadforge/validation/release_quality.py` public API | release/validation/llm_critique_raw_20260508T204359.124834Z.json#Q2 |
| 2026-05-08 | pr:#76 | Q3 — does Kaggle / Hugging Face upload include `docs/release/` and `docs/external_review/` subtrees | low | wont-fix | No — only `release/` ships per the platform packagers (`scripts/package_kaggle_release.py`, `scripts/package_hf_release.py`). Cross-tree links are rewritten to GitHub blob URLs by `_release_common.py::rewrite_release_links()`. F003's verdict above carries the answer | scripts/_release_common.py |
