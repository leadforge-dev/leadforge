# Break Me — adversarial playbook for `leadforge-lead-scoring-v1`

We *want* this dataset to be broken on purpose. The notebooks
ship the headline walkthroughs (notebook 03 dissects the
documented `total_touches_all` trap; notebook 04 covers
calibration, value-aware ranking, and cohort shift). This guide
is the **meta-recipe**: the patterns to look for on any
synthetic teaching dataset, with worked-example pointers back
into the v1 bundle so each pattern is grounded in a number
you can reproduce.

If you find one of these on `leadforge-lead-scoring-v1`,
file an issue using one of the templates in
[`.github/ISSUE_TEMPLATE/`](../../.github/ISSUE_TEMPLATE).
Accepted findings are logged in
[`v2_decision_log.md`](v2_decision_log.md).

## Triage labels

When you file an issue, suggest one of these labels in the
title or body. The maintainer applies the final label.

| Label | When |
|---|---|
| `critical-leakage` | The dataset reconstructs the label via a path that wasn't documented. Highest priority — blocks v1 if reproducible on the as-shipped bundle. |
| `realism` | A modelled distribution disagrees with what a domain expert expects (industry mix, persona behaviour, funnel timing, channel attribution, pricing). Belongs in the realism issue template. |
| `difficulty` | A tier sits outside its declared band on a metric documented in `release/validation/validation_report.md`. Likely a band recalibration in v2. |
| `documentation` | A claim in the dataset card or notebooks doesn't match the artefact. Cheap to fix; please file. |
| `platform` | Kaggle / HF artefact issue (broken link, malformed YAML, schema mismatch). Phase 5 territory. |
| `notebook` | A notebook fails to execute, or its tolerance gate fires on a fresh checkout. |
| `pedagogy` | The teaching framing is misleading even though the artefact is technically correct. |
| `v2-idea` | A capability worth adding (cohort drift, channel-conditional probabilities, non-linear motifs). |
| `out-of-scope-v1` | True observation, but explicitly deferred — the dataset card already documents it as a v1 simplification. |

## The meta-recipe

Notebook 03 §7 introduces a three-step recipe (read the feature
dictionary → ablate, don't just probe → check the time window).
This guide extends it with one more step that the notebook
doesn't cover, then organises the patterns to apply each step
to.

1. **Read the feature dictionary first.** Every public bundle
   ships `feature_dictionary.csv` with a `leakage_risk` column.
   Treat that as the primary leakage audit before any modelling.
2. **Ablate, don't just probe.** A standalone-AUC probe on a
   single feature can rate a column as ~0.5 AUC while a tree
   model extracts non-trivial lift from the same column once
   it can combine it with the rest of the panel. Notebook 03
   §4–§5 demonstrate the gap on `total_touches_all`
   (standalone 0.531 → GBM lift +0.032 vs LR lift +0.009).
3. **Check the time window.** If you have any event table
   with timestamps, cross-check every aggregate feature against
   `lead_created_at + snapshot_day`. The validation report's
   `post_snapshot_aggregates` baseline (`$.tiers.intermediate.per_seed[*].baselines.post_snapshot_aggregates`)
   bench-tests this same idea at scale.
4. **Treat the train/test split as untrusted.** The split file
   says one thing; what the model sees during fitting is what
   matters. Sections 5 and 6 below cover the most common ways
   the two diverge.

The pattern catalogue below maps each pattern to the recipe
step it operationalises.

---

## Leakage patterns

### 1. Naming smells the dictionary should already flag

A column whose name mentions `total`, `all`, `lifetime`,
`final`, `outcome`, or any superlative that crosses the
prediction horizon is suspicious by default on a snapshot-
anchored task. `leadforge-lead-scoring-v1` ships exactly one
such column — `total_touches_all` — and the
`feature_dictionary.csv` row for it sets `leakage_risk=True`
and explains *why* in the description.

**How to detect on any dataset.** Grep the column list for
`*_total`, `*_all`, `*_lifetime`, `*_final`, `*_outcome`,
`current_*`, `is_*` (especially `is_won`, `is_closed`).
Cross-check each hit against the dataset's stated prediction
horizon and snapshot anchor. If the column name implies a
window the snapshot can't have observed, the dictionary should
either flag it or rename it; if neither, that's a `documentation`
issue at minimum and probably `critical-leakage`.

**Worked example.** Notebook 03 §2 shows the dictionary read
in three lines of pandas; the column it surfaces is
`total_touches_all`.

### 2. The standalone-AUC undersell (tree-friendly leakage)

A feature can score ~0.5 AUC as a single-column ranker and
still hand a tree model material lift once interactions with
other columns are available. The validation report's
`post_snapshot_aggregates` baseline (HistGBM on the trap
column alone, see
[`leadforge/validation/release_quality.py`](../../leadforge/validation/release_quality.py))
gives ~0.55 AUC on intermediate (median across seeds 42–46;
0.52–0.61 across all tier × seed pairs) — the trap "looks"
innocuous even when scored by a tree model on its own.
Notebook 03 §5 then runs a full panel ablation and HistGBM
extracts +0.032 AUC; LR with the same preprocessing only
extracts +0.009 because it can't represent the relevant
interaction.

**How to detect on any dataset.** Don't audit leakage with
single-feature AUC. For every column you flagged in pattern 1,
fit two tree models on the same train/test split — one with
the column, one without — and read the AUC delta. A delta
larger than your sampling noise is a flag, regardless of the
standalone number.

**Worked example.** Notebook 03 §4 (standalone) and §5
(ablation), with the side-by-side bar chart in §5.1. The
sign-aware tolerance gate in §6 (`MIN_GBM_LIFT = 0.015`)
formalises the asymmetry as a CI assertion.

### 3. Time-window violations on engineered features

The non-negotiable rule: no feature on a snapshot-anchored
task may use events later than `lead_created_at + snapshot_day`.
The public bundle's event tables (`touches`, `sessions`,
`sales_activities`, `opportunities`) are pre-filtered to
satisfy this rule (notebook 02 §3 verifies the contract on
the bundle as shipped, including a *minimum headroom under
cutoff* readout). The hazard you can still create yourself is
to engineer a feature that joins back to a non-event table
without filtering — for instance, joining `customers` (which
exists only for *converted* leads) into a feature panel.

**How to detect on any dataset.** For every per-lead
aggregate you build, write the query as `SELECT … WHERE
event.timestamp <= lead.created_at + INTERVAL '<snapshot_day>'`
explicitly, even when the underlying table is already filtered.
If the same SQL works against the instructor companion (full-
horizon tables) AND the public bundle, you'll catch
yourself if you accidentally rely on rows that exist only in
the unfiltered view.

**Worked example.** Notebook 02 §3 implements the per-table
inline assertion. The validation report's
`$.tiers.<tier>.per_seed[*].baselines.post_snapshot_aggregates`
HistGBM AUC documents what a model can recover when the rule
is intentionally violated.

### 4. Target-encoding leakage on test

Mean-target encoding of a categorical feature is a textbook
hazard: fit the encoding on the *full* train+test population
and you've leaked test labels into the feature. Notebook 02
§4.4 demonstrates the train-only-fit posture on `industry`
(four industries — logistics, healthcare_non_clinical,
manufacturing, professional_services — encoded by their
training-split conversion rate, with a global-mean fallback
for industries not seen in train). The leakage variant is a
one-liner — `pd.concat([train, test]).groupby('industry')['target'].mean()`
— and the notebook deliberately doesn't show it, because the
lesson there is the discipline. This guide shows the leakage
form (above) so you recognise it during code review.

**How to detect on any dataset.** When mean-target encoding
shows up in a notebook or pipeline, check three things in
order: (a) the encoding's `.fit()` call sees only training
labels; (b) the same encoding is applied to test via merge
or join, never re-fitted; (c) categories present in test but
not train fall back to a deterministic value (global mean is
fine; computing a fallback from test is not). If the encoding
is fit on test labels even partially — including via a
"smoothed" encoder that uses pooled train+test counts — you
have target leakage.

**Worked example.** Notebook 02 §4.4 (train-only fit) and
§4.5 (the merge that applies the encoding to test). The
fallback-to-train-mean handling is in `attach_engineered`.

---

## Split discipline

### 5. Train-test contamination

The bundle ships a deterministic 70/15/15 split on `lead_id`
(see `tasks/<task>/task_manifest.json`). That guarantees
`lead_id` uniqueness across splits — but `account_id` and
`contact_id` are *not* split on. On the as-shipped intermediate
bundle, **518 of 557 test accounts (93 %) also appear in train**,
and the contact-level overlap is similar in magnitude (the
split is `lead_id`-keyed and `account_id` / `contact_id` are
shared foreign keys); the same proportions hold on intro and
advanced because the splitter is tier-invariant. Models can
ride account- or contact-level signal across the split boundary
in ways that don't generalise to a fresh account or fresh
contact.

**How to detect on any dataset.** Repeat the snippet below per
group key — every reusable foreign-key column the dataset
exposes (`account_id`, `contact_id`, and any derived strata
like `industry × region` you bake into engineered features) is
a separate group-leakage axis.

```python
import pandas as pd
train = pd.read_parquet("intermediate/tasks/converted_within_90_days/train.parquet")
test  = pd.read_parquet("intermediate/tasks/converted_within_90_days/test.parquet")
for key in ("account_id", "contact_id"):
    overlap = set(train[key]) & set(test[key])
    print(f"shared {key}: {len(overlap)} / {test[key].nunique()}")
```

If any overlap is non-empty *and* you've engineered any
group-level features, retrain with group-aware splitting
(e.g. `GroupKFold` on the relevant key) and re-read the AUC
delta. The delta is the amount of "free" lift the random-split
was buying you. The right framing isn't "remove the leak"; it's
*report both numbers so the reader knows which is which.*

**Worked example.** Notebook 02 §4.2 builds an account-level
density feature using *only* train leads' touches — a
defensive posture against this hazard. The
`tasks/converted_within_90_days/task_manifest.json` records
the split policy and is the right artefact to cite when filing
an issue under this label. A bundle-level group-overlap audit
isn't included in v1 — the validation report's split-leakage
probe (`probe_split_id_overlap`) checks `lead_id` only;
extending it to enumerate `account_id` and `contact_id`
overlap is a `v2-idea` candidate.

### 6. Cohort-by-segment evaluation

Notebook 04 §7 demonstrates **tier-wide** cohort shift —
sort leads chronologically, train on the first 85 %, score
the last 15 % — and finds intermediate cohort-split AUC
sits *higher* than random-split AUC by ~0.0155 (the v1
simulator has no time drift baked in over the 90-day horizon).
The richer stress test is **per-segment** cohort shift:
chronological resplit *within* each industry, region, or
revenue tier, and read the same delta per segment. Segment-
conditional drift can hide inside a stable tier-wide number
— industry A drifting up by 0.04 cancels industry B drifting
down by 0.04 in the average.

**How to detect on any dataset.** For each segment column
(`industry`, `region`, `employee_band`,
`estimated_revenue_band`), repeat the cohort-split protocol
from notebook 04 §7 conditioned on that segment. Report the
per-segment AUC degradation and the spread across segments.
A spread larger than the tier's cross-seed GBM-AUC band
(`$.tiers.<tier>.spreads.gbm_auc` — same model the cohort-shift
block uses) is a realism flag: the simulator is producing a
homogeneous world that real production cohorts wouldn't be.

**Worked example.** Notebook 04 §7 (tier-wide, validator-
mirrored). The validation report's `cohort_shift.<tier>.auc_degradation`
field gives the v1 baseline you're trying to refine. v1
intentionally runs only the tier-wide check; the per-segment
audit is a `v2-idea` candidate.

---

## Metric and ranking traps

### 7. Value-aware ranking surprises

P(convert) ranking and `P(convert) × expected_acv` ranking
are both reasonable depending on the operational question.
Notebook 04 §5 shows the gap on this bundle — at top-50, ACV
capture jumps from 0.16 (P-only) to 0.40 (P × ACV). The trap
is reaching for one metric when the operational question
demands the other and not noticing the inversion. AUC ranks
*everything* by P(convert); a salesperson with capacity for
50 leads cares about revenue-weighted top-50 capture.

**How to detect on any dataset.** Compute both `precision_at_k`
and `expected_acv_capture_at_k` for the same top-K. If their
ranking of model variants disagrees, that's a finding — at
minimum a `pedagogy` issue, possibly `realism` if the gap is
so large it suggests the simulator's ACV column has unrealistic
correlation with P(convert).

**Worked example.** Notebook 04 §5 produces both curves
side-by-side; the validation report's per-seed scalars live
under
`$.tiers.<tier>.per_seed[*].expected_acv_capture_at_k.50`
(and `.100` for top-100), keyed by string K.

### 8. Threshold-vs-rank semantics

A `precision >= threshold` operating point and a `top-K by
rank` operating point are not the same thing when probabilities
have ties. Notebook 04 §6 picks a threshold that "should"
admit 50 leads and reads back `actually_above` as a defensive
instrument — on the as-shipped intermediate bundle the realised
count matches capacity, but the readout exists so a seed where
ties cluster at the operating probability fails loud rather
than silently inflating the slate.

**How to detect on any dataset.** When you set a probability
threshold for a fixed-capacity decision, always log the
*realised* count above threshold, not just the threshold value.
If realised > capacity by more than a few percent, ties are
inflating the slate and you need either a finer probability
grid (less likely to help on a calibrated model) or a
secondary rank score to break ties.

**Worked example.** Notebook 04 §6 prints
`capacity / threshold / actually_above / precision / recall`
and walks through the threshold sweep for context. The
calibration-bin output in §3 is the related receipt — a model
with poor bin-error is more likely to have ties at common
probabilities.

---

## Robustness and realism

### 9. Calibration drift across cohorts and segments

The validation report tracks `calibration_max_bin_error`
per tier (`$.tiers.<tier>.medians.calibration_max_bin_error`)
— intermediate ~0.25, intro ~0.25, advanced ~0.52. That's a
single number per tier on a single split; in principle it can
mask segment-conditional miscalibration. Whether v1 actually
exhibits such drift is an open question — the per-segment
audit is the way to find out. Notebook 04 §3 shows the
tier-level reliability diagram on the public bundle; the
analogous per-segment diagram is the next stress test.

**How to detect on any dataset.** Reproduce notebook 04 §3's
binning protocol *within* each segment column you care about
(`industry`, `region`, `employee_band`,
`estimated_revenue_band`). Report `max_bin_error` per segment
and the spread across segments. A segment whose max-bin-error
is materially worse than the tier-level number is a `realism`
finding — the world isn't producing the correlation structure
between segment and outcome that real production data would.

**Worked example.** Notebook 04 §3 covers the tier-level
case end-to-end. The cohort-shift block in §7 is the
chronological analogue (calibration over time, in
expectation, via AUC degradation as a coarse summary). v1
doesn't ship a per-segment calibration audit; it's a
`v2-idea`.

---

## What to do when you find one

1. Reproduce the finding from a clean checkout against the
   as-shipped bundle. Note the seed, tier, and the test-split
   sha256 from `manifest.json` — under
   `tasks.converted_within_90_days.test_sha256`. That single
   hash uniquely identifies the bundle the finding was
   reproduced on; the manifest also carries per-table hashes
   under `tables.<name>.sha256` if a table-specific hash is
   the right anchor for the finding.
2. Pick the issue template that fits — leakage / contamination
   / metric findings go in
   [`dataset_breakage_report.yml`](../../.github/ISSUE_TEMPLATE/dataset_breakage_report.yml);
   distributional / realism critiques go in
   [`realism_feedback.yml`](../../.github/ISSUE_TEMPLATE/realism_feedback.yml).
3. Suggest a triage label from the table at the top of this
   guide. The maintainer applies the final label.
4. Watch [`v2_decision_log.md`](v2_decision_log.md) for the
   disposition. Accepted findings get an entry with a verdict
   (`accepted-for-v2`, `deferred`, `wont-fix`,
   `needs-investigation`) and a pointer to the resulting v2
   work item.
