# leadforge-lead-scoring-v1 — release quality report

**Package version:** `1.0.0`
**Generated:** `2026-05-06T06:12:04+00:00`
**Seeds:** [42, 43, 44, 45, 46]
Every value below cites the JSON field that backs it; see `validation_report.json` for the machine-readable form.

## Per-tier headline metrics

| Tier | Conv. rate (test) | LR AUC | GBM AUC | GBM−LR | LR AP | Brier | Cal. max-bin err | Top-decile rate |
|---|---|---|---|---|---|---|---|---|
| advanced | 0.0840 (`$.tiers.advanced.medians.conversion_rate_test`) | 0.8861 (`$.tiers.advanced.medians.lr_auc`) | 0.8726 (`$.tiers.advanced.medians.gbm_auc`) | -0.0133 (`$.tiers.advanced.medians.gbm_minus_lr_auc`) | 0.3514 (`$.tiers.advanced.medians.lr_average_precision`) | 0.0611 (`$.tiers.advanced.medians.brier_score`) | 0.5234 (`$.tiers.advanced.medians.calibration_max_bin_error`) | 0.3333 (`$.tiers.advanced.medians.top_decile_rate`) |
| intermediate | 0.2160 (`$.tiers.intermediate.medians.conversion_rate_test`) | 0.8859 (`$.tiers.intermediate.medians.lr_auc`) | 0.8755 (`$.tiers.intermediate.medians.gbm_auc`) | -0.0072 (`$.tiers.intermediate.medians.gbm_minus_lr_auc`) | 0.5752 (`$.tiers.intermediate.medians.lr_average_precision`) | 0.1096 (`$.tiers.intermediate.medians.brier_score`) | 0.2490 (`$.tiers.intermediate.medians.calibration_max_bin_error`) | 0.5867 (`$.tiers.intermediate.medians.top_decile_rate`) |
| intro | 0.4267 (`$.tiers.intro.medians.conversion_rate_test`) | 0.8788 (`$.tiers.intro.medians.lr_auc`) | 0.8729 (`$.tiers.intro.medians.gbm_auc`) | -0.0045 (`$.tiers.intro.medians.gbm_minus_lr_auc`) | 0.7608 (`$.tiers.intro.medians.lr_average_precision`) | 0.1301 (`$.tiers.intro.medians.brier_score`) | 0.2497 (`$.tiers.intro.medians.calibration_max_bin_error`) | 0.7733 (`$.tiers.intro.medians.top_decile_rate`) |

## Cross-seed stability (G8.1)

| Tier | Seeds | LR AUC spread | GBM AUC spread | AP spread | Brier spread |
|---|---|---|---|---|---|
| advanced | [42, 43, 44, 45, 46] | 0.0401 (`$.tiers.advanced.spreads.lr_auc`) | 0.0171 (`$.tiers.advanced.spreads.gbm_auc`) | 0.0814 (`$.tiers.advanced.spreads.lr_average_precision`) | 0.0152 (`$.tiers.advanced.spreads.brier_score`) |
| intermediate | [42, 43, 44, 45, 46] | 0.0230 (`$.tiers.intermediate.spreads.lr_auc`) | 0.0270 (`$.tiers.intermediate.spreads.gbm_auc`) | 0.0863 (`$.tiers.intermediate.spreads.lr_average_precision`) | 0.0161 (`$.tiers.intermediate.spreads.brier_score`) |
| intro | [42, 43, 44, 45, 46] | 0.0272 (`$.tiers.intro.spreads.lr_auc`) | 0.0232 (`$.tiers.intro.spreads.gbm_auc`) | 0.0670 (`$.tiers.intro.spreads.lr_average_precision`) | 0.0184 (`$.tiers.intro.spreads.brier_score`) |

## Cross-tier ordering (G7.4)

- AP ranking (descending): ['intro', 'intermediate', 'advanced'] (`$.cross_tier_ordering.by_average_precision`)
- P@100 ranking (descending): ['intro', 'intermediate', 'advanced'] (`$.cross_tier_ordering.by_precision_at_100`)
- GBM−LR ranking (descending): ['intro', 'intermediate', 'advanced'] (`$.cross_tier_ordering.by_gbm_minus_lr`)
- Conversion-rate ranking (descending): ['intro', 'intermediate', 'advanced'] (`$.cross_tier_ordering.by_conversion_rate`)
- AP intro > intermediate: **True** (`$.cross_tier_ordering.average_precision_intro_gt_intermediate`)
- AP intermediate > advanced: **True** (`$.cross_tier_ordering.average_precision_intermediate_gt_advanced`)
- GBM−LR positive in every tier: **False** (`$.cross_tier_ordering.gbm_minus_lr_positive_in_every_tier`)

## Cohort-shift evaluation (G6.4)

| Tier | Random-split AUC | Cohort-split AUC | Degradation (random − cohort) |
|---|---|---|---|
| advanced | 0.8726 (`$.cohort_shift.advanced.random_split_auc`) | 0.8628 (`$.cohort_shift.advanced.cohort_split_auc`) | 0.0098 (`$.cohort_shift.advanced.auc_degradation`) |
| intermediate | 0.8754 (`$.cohort_shift.intermediate.random_split_auc`) | 0.8908 (`$.cohort_shift.intermediate.cohort_split_auc`) | -0.0155 (`$.cohort_shift.intermediate.auc_degradation`) |
| intro | 0.8729 (`$.cohort_shift.intro.random_split_auc`) | 0.8573 (`$.cohort_shift.intro.cohort_split_auc`) | 0.0156 (`$.cohort_shift.intro.auc_degradation`) |

## Baseline AUCs (G5.* / leakage probes)

Each cell is HistGBM AUC trained on the named feature subset only.

| Tier | seed | engagement_only | id_only | post_snapshot_aggregates | source_only |
|---|---|---|---|---|---|
| advanced | 42 | 0.5884 (`$.tiers.advanced.per_seed[0].baselines.engagement_only`) | 0.5062 (`$.tiers.advanced.per_seed[0].baselines.id_only`) | 0.5317 (`$.tiers.advanced.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5226 (`$.tiers.advanced.per_seed[0].baselines.source_only`) |
| advanced | 43 | 0.5039 (`$.tiers.advanced.per_seed[1].baselines.engagement_only`) | 0.4003 (`$.tiers.advanced.per_seed[1].baselines.id_only`) | 0.5447 (`$.tiers.advanced.per_seed[1].baselines.post_snapshot_aggregates`) | 0.4245 (`$.tiers.advanced.per_seed[1].baselines.source_only`) |
| advanced | 44 | 0.5850 (`$.tiers.advanced.per_seed[2].baselines.engagement_only`) | 0.4507 (`$.tiers.advanced.per_seed[2].baselines.id_only`) | 0.5218 (`$.tiers.advanced.per_seed[2].baselines.post_snapshot_aggregates`) | 0.5396 (`$.tiers.advanced.per_seed[2].baselines.source_only`) |
| advanced | 45 | 0.5703 (`$.tiers.advanced.per_seed[3].baselines.engagement_only`) | 0.5116 (`$.tiers.advanced.per_seed[3].baselines.id_only`) | 0.5441 (`$.tiers.advanced.per_seed[3].baselines.post_snapshot_aggregates`) | 0.4748 (`$.tiers.advanced.per_seed[3].baselines.source_only`) |
| advanced | 46 | 0.6362 (`$.tiers.advanced.per_seed[4].baselines.engagement_only`) | 0.5249 (`$.tiers.advanced.per_seed[4].baselines.id_only`) | 0.5620 (`$.tiers.advanced.per_seed[4].baselines.post_snapshot_aggregates`) | 0.4604 (`$.tiers.advanced.per_seed[4].baselines.source_only`) |
| intermediate | 42 | 0.6196 (`$.tiers.intermediate.per_seed[0].baselines.engagement_only`) | 0.4949 (`$.tiers.intermediate.per_seed[0].baselines.id_only`) | 0.5461 (`$.tiers.intermediate.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5139 (`$.tiers.intermediate.per_seed[0].baselines.source_only`) |
| intermediate | 43 | 0.5525 (`$.tiers.intermediate.per_seed[1].baselines.engagement_only`) | 0.5341 (`$.tiers.intermediate.per_seed[1].baselines.id_only`) | 0.5994 (`$.tiers.intermediate.per_seed[1].baselines.post_snapshot_aggregates`) | 0.5109 (`$.tiers.intermediate.per_seed[1].baselines.source_only`) |
| intermediate | 44 | 0.5708 (`$.tiers.intermediate.per_seed[2].baselines.engagement_only`) | 0.5608 (`$.tiers.intermediate.per_seed[2].baselines.id_only`) | 0.5253 (`$.tiers.intermediate.per_seed[2].baselines.post_snapshot_aggregates`) | 0.4392 (`$.tiers.intermediate.per_seed[2].baselines.source_only`) |
| intermediate | 45 | 0.5931 (`$.tiers.intermediate.per_seed[3].baselines.engagement_only`) | 0.5015 (`$.tiers.intermediate.per_seed[3].baselines.id_only`) | 0.5754 (`$.tiers.intermediate.per_seed[3].baselines.post_snapshot_aggregates`) | 0.4778 (`$.tiers.intermediate.per_seed[3].baselines.source_only`) |
| intermediate | 46 | 0.5788 (`$.tiers.intermediate.per_seed[4].baselines.engagement_only`) | 0.4333 (`$.tiers.intermediate.per_seed[4].baselines.id_only`) | 0.5388 (`$.tiers.intermediate.per_seed[4].baselines.post_snapshot_aggregates`) | 0.5156 (`$.tiers.intermediate.per_seed[4].baselines.source_only`) |
| intro | 42 | 0.5885 (`$.tiers.intro.per_seed[0].baselines.engagement_only`) | 0.4884 (`$.tiers.intro.per_seed[0].baselines.id_only`) | 0.5617 (`$.tiers.intro.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5014 (`$.tiers.intro.per_seed[0].baselines.source_only`) |
| intro | 43 | 0.5877 (`$.tiers.intro.per_seed[1].baselines.engagement_only`) | 0.5189 (`$.tiers.intro.per_seed[1].baselines.id_only`) | 0.5343 (`$.tiers.intro.per_seed[1].baselines.post_snapshot_aggregates`) | 0.5254 (`$.tiers.intro.per_seed[1].baselines.source_only`) |
| intro | 44 | 0.5818 (`$.tiers.intro.per_seed[2].baselines.engagement_only`) | 0.4840 (`$.tiers.intro.per_seed[2].baselines.id_only`) | 0.5344 (`$.tiers.intro.per_seed[2].baselines.post_snapshot_aggregates`) | 0.4839 (`$.tiers.intro.per_seed[2].baselines.source_only`) |
| intro | 45 | 0.6436 (`$.tiers.intro.per_seed[3].baselines.engagement_only`) | 0.4748 (`$.tiers.intro.per_seed[3].baselines.id_only`) | 0.6144 (`$.tiers.intro.per_seed[3].baselines.post_snapshot_aggregates`) | 0.4864 (`$.tiers.intro.per_seed[3].baselines.source_only`) |
| intro | 46 | 0.5785 (`$.tiers.intro.per_seed[4].baselines.engagement_only`) | 0.5261 (`$.tiers.intro.per_seed[4].baselines.id_only`) | 0.5220 (`$.tiers.intro.per_seed[4].baselines.post_snapshot_aggregates`) | 0.4824 (`$.tiers.intro.per_seed[4].baselines.source_only`) |

## Figures

- Lift curves: `figures/lift_curve_intro.png`, `figures/lift_curve_intermediate.png`, `figures/lift_curve_advanced.png`
- Calibration (intermediate): `figures/calibration_intermediate.png`
- Leakage / baseline deltas: `figures/leakage_delta.png`
- Value capture: `figures/value_capture.png`
- Cohort shift: `figures/cohort_shift.png`

---

**Gate references** (see `docs/release/v1_acceptance_gates.md`):

- **G6.4** — Cohort/time-shift AUC degradation band.
- **G7.\*** — Per-tier ROC-AUC, AP, P@K, lift, calibration bands.
- **G7.4** — Cross-tier ordering (AP / P@K / GBM−LR / conversion-rate).
- **G8.1** — Cross-seed stability (per-metric spread within tolerance).

_Renderer: `leadforge.validation.reporting`. JSON sibling: `validation_report.json`._
