# leadforge-lead-scoring-v1 — release quality report

**Package version:** `1.0.0`
**Generated:** `2026-05-26T21:23:32+00:00`
**Seeds:** [42, 43, 44, 45, 46]
Every value below cites the JSON field that backs it; see `validation_report.json` for the machine-readable form.

## Per-tier headline metrics

| Tier | Conv. rate (test) | LR AUC | GBM AUC | GBM−LR | LR AP | Brier | Cal. max-bin err | Top-decile rate |
|---|---|---|---|---|---|---|---|---|
| advanced | 0.0840 (`$.tiers.advanced.medians.conversion_rate_test`) | 0.6236 (`$.tiers.advanced.medians.lr_auc`) | 0.6003 (`$.tiers.advanced.medians.gbm_auc`) | -0.0242 (`$.tiers.advanced.medians.gbm_minus_lr_auc`) | 0.1218 (`$.tiers.advanced.medians.lr_average_precision`) | 0.0758 (`$.tiers.advanced.medians.brier_score`) | 0.2210 (`$.tiers.advanced.medians.calibration_max_bin_error`) | 0.1067 (`$.tiers.advanced.medians.top_decile_rate`) |
| intermediate | 0.2160 (`$.tiers.intermediate.medians.conversion_rate_test`) | 0.6625 (`$.tiers.intermediate.medians.lr_auc`) | 0.6339 (`$.tiers.intermediate.medians.gbm_auc`) | -0.0179 (`$.tiers.intermediate.medians.gbm_minus_lr_auc`) | 0.3318 (`$.tiers.intermediate.medians.lr_average_precision`) | 0.1604 (`$.tiers.intermediate.medians.brier_score`) | 0.2785 (`$.tiers.intermediate.medians.calibration_max_bin_error`) | 0.3200 (`$.tiers.intermediate.medians.top_decile_rate`) |
| intro | 0.4267 (`$.tiers.intro.medians.conversion_rate_test`) | 0.6708 (`$.tiers.intro.medians.lr_auc`) | 0.6838 (`$.tiers.intro.medians.gbm_auc`) | -0.0105 (`$.tiers.intro.medians.gbm_minus_lr_auc`) | 0.5547 (`$.tiers.intro.medians.lr_average_precision`) | 0.2197 (`$.tiers.intro.medians.brier_score`) | 0.1761 (`$.tiers.intro.medians.calibration_max_bin_error`) | 0.6133 (`$.tiers.intro.medians.top_decile_rate`) |

## Cross-seed stability (G8.1)

| Tier | Seeds | LR AUC spread | GBM AUC spread | AP spread | Brier spread |
|---|---|---|---|---|---|
| advanced | [42, 43, 44, 45, 46] | 0.1000 (`$.tiers.advanced.spreads.lr_auc`) | 0.1056 (`$.tiers.advanced.spreads.gbm_auc`) | 0.0560 (`$.tiers.advanced.spreads.lr_average_precision`) | 0.0156 (`$.tiers.advanced.spreads.brier_score`) |
| intermediate | [42, 43, 44, 45, 46] | 0.0594 (`$.tiers.intermediate.spreads.lr_auc`) | 0.0517 (`$.tiers.intermediate.spreads.gbm_auc`) | 0.1237 (`$.tiers.intermediate.spreads.lr_average_precision`) | 0.0202 (`$.tiers.intermediate.spreads.brier_score`) |
| intro | [42, 43, 44, 45, 46] | 0.0871 (`$.tiers.intro.spreads.lr_auc`) | 0.1214 (`$.tiers.intro.spreads.gbm_auc`) | 0.1041 (`$.tiers.intro.spreads.lr_average_precision`) | 0.0293 (`$.tiers.intro.spreads.brier_score`) |

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
| advanced | 0.5331 (`$.cohort_shift.advanced.random_split_auc`) | 0.5780 (`$.cohort_shift.advanced.cohort_split_auc`) | -0.0448 (`$.cohort_shift.advanced.auc_degradation`) |
| intermediate | 0.6524 (`$.cohort_shift.intermediate.random_split_auc`) | 0.5933 (`$.cohort_shift.intermediate.cohort_split_auc`) | 0.0592 (`$.cohort_shift.intermediate.auc_degradation`) |
| intro | 0.6485 (`$.cohort_shift.intro.random_split_auc`) | 0.6560 (`$.cohort_shift.intro.cohort_split_auc`) | -0.0076 (`$.cohort_shift.intro.auc_degradation`) |

## Baseline AUCs (G5.* / leakage probes)

Each cell is HistGBM AUC trained on the named feature subset only.

| Tier | seed | engagement_only | id_only | post_snapshot_aggregates | source_only |
|---|---|---|---|---|---|
| advanced | 42 | 0.5121 (`$.tiers.advanced.per_seed[0].baselines.engagement_only`) | 0.5062 (`$.tiers.advanced.per_seed[0].baselines.id_only`) | 0.5640 (`$.tiers.advanced.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5226 (`$.tiers.advanced.per_seed[0].baselines.source_only`) |
| advanced | 43 | 0.5593 (`$.tiers.advanced.per_seed[1].baselines.engagement_only`) | 0.4003 (`$.tiers.advanced.per_seed[1].baselines.id_only`) | 0.5825 (`$.tiers.advanced.per_seed[1].baselines.post_snapshot_aggregates`) | 0.4245 (`$.tiers.advanced.per_seed[1].baselines.source_only`) |
| advanced | 44 | 0.5831 (`$.tiers.advanced.per_seed[2].baselines.engagement_only`) | 0.4507 (`$.tiers.advanced.per_seed[2].baselines.id_only`) | 0.5162 (`$.tiers.advanced.per_seed[2].baselines.post_snapshot_aggregates`) | 0.5396 (`$.tiers.advanced.per_seed[2].baselines.source_only`) |
| advanced | 45 | 0.5906 (`$.tiers.advanced.per_seed[3].baselines.engagement_only`) | 0.5116 (`$.tiers.advanced.per_seed[3].baselines.id_only`) | 0.5589 (`$.tiers.advanced.per_seed[3].baselines.post_snapshot_aggregates`) | 0.4748 (`$.tiers.advanced.per_seed[3].baselines.source_only`) |
| advanced | 46 | 0.5738 (`$.tiers.advanced.per_seed[4].baselines.engagement_only`) | 0.5249 (`$.tiers.advanced.per_seed[4].baselines.id_only`) | 0.5302 (`$.tiers.advanced.per_seed[4].baselines.post_snapshot_aggregates`) | 0.4604 (`$.tiers.advanced.per_seed[4].baselines.source_only`) |
| intermediate | 42 | 0.6246 (`$.tiers.intermediate.per_seed[0].baselines.engagement_only`) | 0.4949 (`$.tiers.intermediate.per_seed[0].baselines.id_only`) | 0.5541 (`$.tiers.intermediate.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5139 (`$.tiers.intermediate.per_seed[0].baselines.source_only`) |
| intermediate | 43 | 0.5989 (`$.tiers.intermediate.per_seed[1].baselines.engagement_only`) | 0.5341 (`$.tiers.intermediate.per_seed[1].baselines.id_only`) | 0.5847 (`$.tiers.intermediate.per_seed[1].baselines.post_snapshot_aggregates`) | 0.5109 (`$.tiers.intermediate.per_seed[1].baselines.source_only`) |
| intermediate | 44 | 0.5507 (`$.tiers.intermediate.per_seed[2].baselines.engagement_only`) | 0.5608 (`$.tiers.intermediate.per_seed[2].baselines.id_only`) | 0.5221 (`$.tiers.intermediate.per_seed[2].baselines.post_snapshot_aggregates`) | 0.4392 (`$.tiers.intermediate.per_seed[2].baselines.source_only`) |
| intermediate | 45 | 0.5518 (`$.tiers.intermediate.per_seed[3].baselines.engagement_only`) | 0.5015 (`$.tiers.intermediate.per_seed[3].baselines.id_only`) | 0.5786 (`$.tiers.intermediate.per_seed[3].baselines.post_snapshot_aggregates`) | 0.4778 (`$.tiers.intermediate.per_seed[3].baselines.source_only`) |
| intermediate | 46 | 0.5633 (`$.tiers.intermediate.per_seed[4].baselines.engagement_only`) | 0.4333 (`$.tiers.intermediate.per_seed[4].baselines.id_only`) | 0.5438 (`$.tiers.intermediate.per_seed[4].baselines.post_snapshot_aggregates`) | 0.5156 (`$.tiers.intermediate.per_seed[4].baselines.source_only`) |
| intro | 42 | 0.6040 (`$.tiers.intro.per_seed[0].baselines.engagement_only`) | 0.4884 (`$.tiers.intro.per_seed[0].baselines.id_only`) | 0.5589 (`$.tiers.intro.per_seed[0].baselines.post_snapshot_aggregates`) | 0.5014 (`$.tiers.intro.per_seed[0].baselines.source_only`) |
| intro | 43 | 0.6115 (`$.tiers.intro.per_seed[1].baselines.engagement_only`) | 0.5189 (`$.tiers.intro.per_seed[1].baselines.id_only`) | 0.5483 (`$.tiers.intro.per_seed[1].baselines.post_snapshot_aggregates`) | 0.5254 (`$.tiers.intro.per_seed[1].baselines.source_only`) |
| intro | 44 | 0.5770 (`$.tiers.intro.per_seed[2].baselines.engagement_only`) | 0.4840 (`$.tiers.intro.per_seed[2].baselines.id_only`) | 0.5360 (`$.tiers.intro.per_seed[2].baselines.post_snapshot_aggregates`) | 0.4839 (`$.tiers.intro.per_seed[2].baselines.source_only`) |
| intro | 45 | 0.6437 (`$.tiers.intro.per_seed[3].baselines.engagement_only`) | 0.4748 (`$.tiers.intro.per_seed[3].baselines.id_only`) | 0.6181 (`$.tiers.intro.per_seed[3].baselines.post_snapshot_aggregates`) | 0.4864 (`$.tiers.intro.per_seed[3].baselines.source_only`) |
| intro | 46 | 0.5635 (`$.tiers.intro.per_seed[4].baselines.engagement_only`) | 0.5261 (`$.tiers.intro.per_seed[4].baselines.id_only`) | 0.5145 (`$.tiers.intro.per_seed[4].baselines.post_snapshot_aggregates`) | 0.4824 (`$.tiers.intro.per_seed[4].baselines.source_only`) |

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
