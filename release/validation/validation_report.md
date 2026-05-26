# leadforge-lead-scoring-v1 — release quality report

**Package version:** `1.0.0`
**Generated:** `2026-05-26T04:39:42+00:00`
**Seeds:** [42]
Every value below cites the JSON field that backs it; see `validation_report.json` for the machine-readable form.

## Per-tier headline metrics

| Tier | Conv. rate (test) | LR AUC | GBM AUC | GBM−LR | LR AP | Brier | Cal. max-bin err | Top-decile rate |
|---|---|---|---|---|---|---|---|---|
| advanced | 0.0787 (`$.tiers.advanced.medians.conversion_rate_test`) | 0.5629 (`$.tiers.advanced.medians.lr_auc`) | 0.5331 (`$.tiers.advanced.medians.gbm_auc`) | -0.0297 (`$.tiers.advanced.medians.gbm_minus_lr_auc`) | 0.0926 (`$.tiers.advanced.medians.lr_average_precision`) | 0.0732 (`$.tiers.advanced.medians.brier_score`) | 0.0726 (`$.tiers.advanced.medians.calibration_max_bin_error`) | 0.0800 (`$.tiers.advanced.medians.top_decile_rate`) |
| intermediate | 0.2227 (`$.tiers.intermediate.medians.conversion_rate_test`) | 0.6704 (`$.tiers.intermediate.medians.lr_auc`) | 0.6524 (`$.tiers.intermediate.medians.gbm_auc`) | -0.0179 (`$.tiers.intermediate.medians.gbm_minus_lr_auc`) | 0.3584 (`$.tiers.intermediate.medians.lr_average_precision`) | 0.1629 (`$.tiers.intermediate.medians.brier_score`) | 0.2789 (`$.tiers.intermediate.medians.calibration_max_bin_error`) | 0.3867 (`$.tiers.intermediate.medians.top_decile_rate`) |
| intro | 0.4267 (`$.tiers.intro.medians.conversion_rate_test`) | 0.6708 (`$.tiers.intro.medians.lr_auc`) | 0.6485 (`$.tiers.intro.medians.gbm_auc`) | -0.0223 (`$.tiers.intro.medians.gbm_minus_lr_auc`) | 0.5683 (`$.tiers.intro.medians.lr_average_precision`) | 0.2221 (`$.tiers.intro.medians.brier_score`) | 0.1761 (`$.tiers.intro.medians.calibration_max_bin_error`) | 0.6267 (`$.tiers.intro.medians.top_decile_rate`) |

## Cross-seed stability (G8.1)

| Tier | Seeds | LR AUC spread | GBM AUC spread | AP spread | Brier spread |
|---|---|---|---|---|---|
| advanced | [42] | 0.0000 (`$.tiers.advanced.spreads.lr_auc`) | 0.0000 (`$.tiers.advanced.spreads.gbm_auc`) | 0.0000 (`$.tiers.advanced.spreads.lr_average_precision`) | 0.0000 (`$.tiers.advanced.spreads.brier_score`) |
| intermediate | [42] | 0.0000 (`$.tiers.intermediate.spreads.lr_auc`) | 0.0000 (`$.tiers.intermediate.spreads.gbm_auc`) | 0.0000 (`$.tiers.intermediate.spreads.lr_average_precision`) | 0.0000 (`$.tiers.intermediate.spreads.brier_score`) |
| intro | [42] | 0.0000 (`$.tiers.intro.spreads.lr_auc`) | 0.0000 (`$.tiers.intro.spreads.gbm_auc`) | 0.0000 (`$.tiers.intro.spreads.lr_average_precision`) | 0.0000 (`$.tiers.intro.spreads.brier_score`) |

## Cross-tier ordering (G7.4)

- AP ranking (descending): ['intro', 'intermediate', 'advanced'] (`$.cross_tier_ordering.by_average_precision`)
- P@100 ranking (descending): ['intro', 'intermediate', 'advanced'] (`$.cross_tier_ordering.by_precision_at_100`)
- GBM−LR ranking (descending): ['intermediate', 'intro', 'advanced'] (`$.cross_tier_ordering.by_gbm_minus_lr`)
- Conversion-rate ranking (descending): ['intro', 'intermediate', 'advanced'] (`$.cross_tier_ordering.by_conversion_rate`)
- AP intro > intermediate: **True** (`$.cross_tier_ordering.average_precision_intro_gt_intermediate`)
- AP intermediate > advanced: **True** (`$.cross_tier_ordering.average_precision_intermediate_gt_advanced`)
- GBM−LR positive in every tier: **False** (`$.cross_tier_ordering.gbm_minus_lr_positive_in_every_tier`)

## Cohort-shift evaluation (G6.4)

| Tier | Random-split AUC | Cohort-split AUC | Degradation (random − cohort) |
|---|---|---|---|
| advanced | 0.5331 (`$.cohort_shift.advanced.random_split_auc`) | 0.5780 (`$.cohort_shift.advanced.cohort_split_auc`) | -0.0448 (`$.cohort_shift.advanced.auc_degradation`) |
| intermediate | 0.6524 (`$.cohort_shift.intermediate.random_split_auc`) | 0.5933 (`$.cohort_shift.intermediate.cohort_split_auc`) | 0.0592 (`$.cohort_shift.intermediate.auc_degradation`) |
| intro | 0.6485 (`$.cohort_shift.intro.random_split_auc`) | 0.6560 (`$.cohort_shift.intro.cohort_split_auc`) | -0.0076 (`$.cohort_shift.intro.auc_degradation`) |

## Baseline AUCs (G5.* / leakage probes)

Each cell is HistGBM AUC trained on the named feature subset only.

| Tier | seed | engagement_only | id_only | post_snapshot_aggregates | source_only |
|---|---|---|---|---|---|
| advanced | 42 | 0.5121 (`$.tiers.advanced.per_seed[0].baselines.engagement_only`) | 0.5062 (`$.tiers.advanced.per_seed[0].baselines.id_only`) | 0.5640 (`$.tiers.advanced.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5226 (`$.tiers.advanced.per_seed[0].baselines.source_only`) |
| intermediate | 42 | 0.6246 (`$.tiers.intermediate.per_seed[0].baselines.engagement_only`) | 0.4949 (`$.tiers.intermediate.per_seed[0].baselines.id_only`) | 0.5541 (`$.tiers.intermediate.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5139 (`$.tiers.intermediate.per_seed[0].baselines.source_only`) |
| intro | 42 | 0.6040 (`$.tiers.intro.per_seed[0].baselines.engagement_only`) | 0.4884 (`$.tiers.intro.per_seed[0].baselines.id_only`) | 0.5589 (`$.tiers.intro.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5014 (`$.tiers.intro.per_seed[0].baselines.source_only`) |

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
