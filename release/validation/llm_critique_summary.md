# LLM critique summary — `leadforge-lead-scoring-v1`

- **Release:** `leadforge-lead-scoring-v1`
- **Model:** `claude-opus-4-7` (effort: `high`, thinking: `adaptive`)
- **Run timestamp:** 2026-05-08T20:43:59.124834Z
- **Input-bundle SHA256:** `ce1e4c204f6f3747dc050f3323accd56dabb669d679db7c0eb6272aa76fb7540`
- **Overall score:** 7/10

## Overall assessment

The bundle ships cleanly on the structural axes — manifest fields are complete, redaction contract is single-sourced, validation report reconciles against the README headline table, and the documented `total_touches_all` trap is consistently flagged across card, dictionary, and break-me guide. No high-severity leakage path beyond the documented trap surfaces in the inputs. The one high-severity issue is pedagogical: the 93% account_id overlap between train and test is fully described in `break_me_guide.md` §5 but absent from the dataset card and README, so a notebook-01 student will silently train an account-leaky baseline. Remaining findings are noise-injection realism gaps, relative-path hygiene for Kaggle/HF, and adversarial-framing completeness around contact-level contamination.

## Findings

### Severity: high (1)

#### F001 — `documentation` / `D6`

**Claim.** The 93% test-account overlap with train is documented only in the adversarial guide, not in the dataset card or README, so a baseline-notebook student will not know their AUC is account-leaky.

**Evidence.** `break_me_guide.md` §5 quotes '518 of 557 test accounts (93%) also appear in train' and notes 'A bundle-level account_id overlap audit isn't included in v1'; release/README.md 'Composition' section says only 'Splits. 70/15/15 train/valid/test, deterministic given seed' with no group-leakage warning; release/intermediate/dataset_card.md has no mention of account-level overlap.

**Reproducer.** python -c "import pandas as pd; tr=pd.read_parquet('release/intermediate/tasks/converted_within_90_days/train.parquet'); te=pd.read_parquet('release/intermediate/tasks/converted_within_90_days/test.parquet'); print(len(set(tr.account_id)&set(te.account_id)), '/', te.account_id.nunique())"

**Suggested fix.** Add a one-paragraph 'Group-leakage warning' to release/README.md 'Splits' subsection and to dataset_card.md 'Caveats', citing the 518/557 figure and pointing at break_me_guide §5 plus a GroupKFold(account_id) recipe.

### Severity: medium (4)

#### F002 — `documentation` / `D1`

**Claim.** Noise injection produces physically impossible values (negative ACV, negative `days_since_last_touch`, `days_since_first_touch` > snapshot_day) that the dataset card's 'Caveats' does not disclose.

**Evidence.** Test-split describe(): `opportunity_estimated_acv` min = -140151.06, `expected_acv` min = -125614.81, `days_since_last_touch` min = -29.73, `days_since_first_touch` max = 43.46 (snapshot_day = 30 per manifest). Dataset_card.md caveat states 'event-aggregate features ... observe only the first 30 days' with no mention that Gaussian noise can push float features outside their physical range.

**Reproducer.** python -c "import pandas as pd; df=pd.read_parquet('release/intermediate/tasks/converted_within_90_days/test.parquet'); print(df[['expected_acv','days_since_last_touch','days_since_first_touch']].describe())"

**Suggested fix.** Add a 'Noise artefacts' bullet to dataset_card.md Caveats: 'Gaussian noise on float features can produce non-physical values (negative ACV, negative day-deltas, day-deltas > snapshot_day=30). Models should treat these as noise rather than clip; clipping silently shifts the conditional distribution.'

#### F003 — `platform` / `D8`

**Claim.** release/README.md links to files outside the release/ tree using `](../foo)` paths that will 404 once the README is inlined onto Kaggle and Hugging Face.

**Evidence.** README references `[gemini_v2_summary.md](../docs/external_review/summaries/gemini_v2_summary.md)`, `[generation_method.md](../docs/release/generation_method.md)`, `[leakage_probes.py](../leadforge/validation/leakage_probes.py)`, `[v1_acceptance_gates_bands.yaml](../docs/release/v1_acceptance_gates_bands.yaml)`, `[channel_signal_audit.md](../docs/release/channel_signal_audit.md)`, `[break_me_guide.md](../docs/release/break_me_guide.md)`, `[feature_dictionary.md](../docs/release/feature_dictionary.md)`, plus two `.github/ISSUE_TEMPLATE/*.yml` references — none of which ship in the release bundle.

**Reproducer.** grep -nE '\]\(\.\./' release/README.md

**Suggested fix.** Replace each `../<path>` link with an absolute URL of the form `https://github.com/leadforge-dev/leadforge/blob/v1.0.0/<path>` so off-platform links resolve from Kaggle / HF; ship a thin `docs/release/` redirect inside the bundle for the two files external readers actually need (generation_method.md and break_me_guide.md).

#### F004 — `pedagogy` / `D9`

**Claim.** `break_me_guide.md` pattern 5 covers train/test contamination on `account_id` but ignores the parallel hazard on `contact_id`, despite contacts being shared at a similar magnitude given the lead-keyed split.

**Evidence.** Test-split sample shows `contact_id` unique=684/750; with 4,200 contacts split across 3,500/750/750 task rows and the splitter keyed only on `lead_id` (per task_manifest.json policy referenced in break_me_guide §5), contact-level overlap is structurally guaranteed. Pattern 5 names only `account_id` and lists no contact-keyed analogue.

**Reproducer.** python -c "import pandas as pd; tr=pd.read_parquet('release/intermediate/tasks/converted_within_90_days/train.parquet'); te=pd.read_parquet('release/intermediate/tasks/converted_within_90_days/test.parquet'); print('contact overlap:', len(set(tr.contact_id)&set(te.contact_id)), '/', te.contact_id.nunique())"

**Suggested fix.** Extend break_me_guide §5 to enumerate `account_id`, `contact_id`, and any other reusable foreign-key column (e.g. derived `industry × region` strata) as group-leakage axes; reuse the same overlap-snippet template per key.

#### F005 — `pedagogy` / `D5`

**Claim.** The advanced-tier headline `calibration_max_bin_error = 0.5234` is driven by 2- and 3-sample high-probability bins, and the validation report surfaces the headline without the n-count caveat.

**Evidence.** `$.tiers.advanced.per_seed[1].calibration_bins[5]` records `{bin_lower: 0.5, mean_actual: 0.0, mean_predicted: 0.5234, n: 2}` — the bin that drives the 0.5234 headline; `validation_report.md` 'Per-tier headline metrics' table reports 0.5234 with no minimum-bin-count footnote.

**Reproducer.** python -c "import json; r=json.load(open('release/validation/validation_report.json')); [print(b['n'], b['mean_predicted']-b['mean_actual']) for b in r['tiers']['advanced']['per_seed'][1]['calibration_bins']]"

**Suggested fix.** Compute `calibration_max_bin_error` only over bins with `n >= 20` (or expose both raw and n-weighted variants) and add a footnote to the headline table noting that low-positive-rate tiers can show large bin-errors driven by small-n high-probability bins.

### Severity: low (1)

#### F006 — `documentation` / `D1`

**Claim.** release/README.md 'Dataset summary' table claims '24–61%' / '12–31%' / '4–12%' as the conversion-rate recipe bands, but the validation report shows observed test conversion-rate spreads only 8–10% / 18–22% / 34–43% across seeds 42–46, so the bands are documented as recipe-acceptance windows without saying so.

**Evidence.** release/README.md 'Conversion rate (recipe band)' row vs `$.tiers.{intro,intermediate,advanced}.per_seed[*].conversion_rate_test` actual values (intro 0.3427–0.4347, intermediate 0.176–0.2227, advanced 0.0787–0.0987).

**Reproducer.** python -c "import json; r=json.load(open('release/validation/validation_report.json'))['tiers']; [print(t, sorted(s['conversion_rate_test'] for s in r[t]['per_seed'])) for t in r]"

**Suggested fix.** Rename the column header to 'Conversion rate (acceptance band, gate G7.*)' and add a one-sentence note that observed five-seed spreads sit comfortably inside the gate band — otherwise readers infer that the simulator can produce 4% or 61% on the same tier, which it can't.

## Missing sections

- missing: Datasheets §Biases — the README out-of-scope mentions fairness research is unsupported but does not enumerate which biases the synthetic generator does encode (industry/region/persona uniformity, channel-conditional independence per known-limitations).
- missing: Datasheets §Privacy — the README treats 'fictional' as sufficient privacy disclosure but does not state that no real CRM was used as seed data, that no PII-shaped strings (job titles, emails, names) appear, and that the recipe is reproducible from public artefacts only.
- missing: dataset_card.md §Group-split warning — no per-bundle disclosure of account_id / contact_id overlap across train/valid/test (see F001, F004).

## Questions for the maintainer

- Does the simulator window event tables before or after Gaussian-noise injection on float features — i.e. is the 43.46-day `days_since_first_touch` a windowing bug or an intended noise artefact?
- Is `top_decile_rate` defined as precision at top 10% or recall at top 10%, and should the validation_report.md headline rename it accordingly so it isn't read as a synonym for P@100?
- Will Kaggle / Hugging Face uploads include the `docs/release/` and `docs/external_review/` subtrees, or only the `release/` subtree — the answer determines whether F003 is medium or high?

## Bundle hashes (audit)

| File / block | SHA256 |
|---|---|
| `docs/release/break_me_guide.md` | `87694a4cc397…` |
| `docs/release/generation_method.md` | `60c663cf1edc…` |
| `public_instructor_diff` | `2c626ea25480…` |
| `public_safe_mechanism_summary` | `05e6d5bb12ec…` |
| `release/README.md` | `7a27b000f7fc…` |
| `release/intermediate/dataset_card.md` | `5d4a68b59ad2…` |
| `release/intermediate/feature_dictionary.csv` | `4fe5724049e6…` |
| `release/intermediate/manifest.json` | `da802eedf92f…` |
| `release/intermediate/tasks/test.parquet[head]` | `6f33b2f2235e…` |
| `release/validation/validation_report.json` | `2f165370fdc8…` |
| `release/validation/validation_report.md` | `04250633a39d…` |
