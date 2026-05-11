# Claims register — `leadforge-lead-scoring-v1`

Every numerical / structural claim made in `release/README.md` (and
copied onto the Kaggle / HuggingFace dataset pages), paired with the
artifact and path that backs it.  This file is auto-rendered from
[`release/claims_register_source.yaml`](claims_register_source.yaml)
by `scripts/build_claims_register.py`.  Edit the YAML, not this file.

Tip for AI reviewers: `claims_register.json` is the machine-readable
twin of this document with the same data plus a schema block.

## calibration

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c05` | Conversion rate (cross-seed median, seeds 42-46): intro 42.67%, intermediate 21.60%, advanced 8.40%. | `release/metrics.json` | `$.tiers.<tier>.medians.conversion_rate_test` | `scripts/validate_release_candidate.py` |
| `c06` | Cross-seed median LR AUC: intro 0.879, intermediate 0.886, advanced 0.886. | `release/metrics.json` | `$.tiers.<tier>.medians.lr_auc` | `scripts/validate_release_candidate.py` |
| `c07` | Cross-seed median LR Average Precision: intro 0.761, intermediate 0.575, advanced 0.351. | `release/metrics.json` | `$.tiers.<tier>.medians.lr_average_precision` | `scripts/validate_release_candidate.py` |
| `c08` | Cross-seed median P@100: intro 0.80, intermediate 0.59, advanced 0.34. | `release/metrics.json` | `$.tiers.<tier>.medians.precision_at_100` | `scripts/validate_release_candidate.py` |
| `c09` | Cross-seed median Brier score: intro 0.130, intermediate 0.110, advanced 0.061. | `release/metrics.json` | `$.tiers.<tier>.medians.brier_score` | `scripts/validate_release_candidate.py` |

## composition

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c01` | Three difficulty tiers (intro / intermediate / advanced), 5,000 leads each. | `release/<tier>/manifest.json` | `$.n_leads` | `leadforge validate` |
| `c02` | Each tier has 1,500 accounts and 4,200 contacts. | `release/<tier>/manifest.json` | `$.n_accounts, $.n_contacts` | `leadforge validate` |
| `c03` | Public bundles ship 7 snapshot-safe relational tables (accounts, contacts, leads, touches, sessions, sales_activities, opportunities). | `release/<tier>/manifest.json` | `$.tables (keys)` | `leadforge validate` |
| `c04` | Instructor companion ships 9 tables (the 7 public ones plus customers and subscriptions). | `release/intermediate_instructor/manifest.json` | `$.tables (keys)` | `leadforge validate` |

## difficulty

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c10` | Conversion-rate, AP, and P@100 orderings hold intro > intermediate > advanced. | `release/metrics.json` | `$.cross_tier_ordering.{by_conversion_rate, by_average_precision, by_precision_at_100}` | `scripts/validate_release_candidate.py` |
| `c11` | Difficulty knobs by tier: signal strength 0.90/0.70/0.50, noise scale 0.10/0.30/0.55, missing rate 2%/8%/18%. | `release/<tier>/metrics.json` | `$.difficulty_knobs` | `leadforge inspect` |

## intended_use

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c25` | Intended uses: teaching baseline lead scoring, relational feature engineering, leakage detection, calibration / lift / P@K / value-aware ranking, model-family comparison under a controlled DGP. | `release/README.md` | `section 'Intended uses'` | `n/a (prose contract)` |

## limitations

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c12` | GBM-LR AUC delta is slightly negative in every tier (-0.0045 / -0.0072 / -0.0133); v1's snapshot is dominated by linear features. | `release/metrics.json` | `$.tiers.<tier>.medians.gbm_minus_lr_auc` | `scripts/validate_release_candidate.py` |
| `c13` | lead_source is weakly informative — out-of-sample univariate AUC ~0.50-0.52 across tiers, per-channel rate spread <=0.05. | `release/docs/channel_signal_audit.md` | `n/a (prose)` | `scripts/audit_channel_signal.py` |
| `c14` | Cohort-shift AUC degradation is small (v1 has no time-of-year drift baked in). | `release/metrics.json` | `$.cohort_shift.<tier>.auc_degradation` | `scripts/validate_release_candidate.py` |

## out_of_scope

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c26` | Out of scope: production lead scoring, vendor benchmarking, causal-inference research requiring DGP recovery, demographic / fairness research. | `release/README.md` | `section 'Out-of-scope uses'` | `n/a (prose contract)` |

## provenance

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c22` | Recipe b2b_saas_procurement_v1, canonical seed 42, cross-seed sweep 42-46, bundle schema version 5, package leadforge 1.0.0+. | `release/<tier>/manifest.json` | `$.recipe_id, $.seed, $.bundle_schema_version, $.package_version` | `leadforge validate` |
| `c23` | Every file in the bundle is SHA-256 hashed in manifest.json; the bundle is verifiable end-to-end with `leadforge validate`. | `release/<tier>/manifest.json` | `$.tables.*.sha256, $.tasks.*.{train,valid,test}_sha256` | `leadforge validate` |
| `c24` | Acceptance bands for every gate live as YAML at release/docs/v1_acceptance_gates_bands.yaml; bands are recipe gates, not achievable ranges. | `release/docs/v1_acceptance_gates_bands.yaml` | `per_tier` | `scripts/validate_release_candidate.py` |

## redaction

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c15` | Public leads.parquet drops conversion_timestamp and converted_within_90_days. | `release/<tier>/manifest.json` | `$.structural_redactions.columns.leads` | `scripts/probe_relational_leakage.py` |
| `c16` | Public opportunities.parquet drops close_outcome and closed_at. | `release/<tier>/manifest.json` | `$.structural_redactions.columns.opportunities` | `scripts/probe_relational_leakage.py` |
| `c17` | Public bundles omit customers and subscriptions tables entirely. | `release/<tier>/manifest.json` | `$.structural_redactions.omitted_tables` | `scripts/probe_relational_leakage.py` |
| `c18` | Snapshot-filtered event tables (touches, sessions, sales_activities, opportunities) keep only rows with <ts> <= lead_created_at + snapshot_day. | `release/<tier>/manifest.json` | `$.relational_snapshot_safe, $.snapshot_day` | `scripts/probe_relational_leakage.py` |
| `c19` | total_touches_all is the deliberate leakage trap: it counts touches over the full 90-day window and is flagged leakage_risk=True. | `release/<tier>/feature_dictionary.csv` | `row[name=='total_touches_all'].leakage_risk` | `grep on feature_dictionary.csv` |

## splits

| ID | Claim | Backing artifact | Path | Verifier |
|---|---|---|---|---|
| `c20` | Splits are 70/15/15 train/valid/test, deterministic given seed; recorded in tasks/converted_within_90_days/task_manifest.json. | `release/<tier>/tasks/converted_within_90_days/task_manifest.json` | `n/a (whole file)` | `leadforge validate` |
| `c21` | Splitter keyed on lead_id only — 518/557 (~93%) of test accounts also appear in train on the intermediate bundle. Use GroupKFold(account_id) for a generalisation-faithful number. | `release/docs/break_me_guide.md` | `section 5` | `scripts/probe_relational_leakage.py --max-accuracy` |
