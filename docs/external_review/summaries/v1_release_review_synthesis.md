# LeadForge v1 — Release Preview Review Synthesis
_Review round: v1 release preview (Shmaggle + ShmuggingFace mock site), 2026-05-25._
_Sources: Claude (standard + adaptive thinking), ChatGPT (standard + extended thinking), Gemini (standard + extended thinking)._
_Distinct from the earlier design-phase reviews in `docs/external_review/{gemini,chatgpt}/`: those covered architecture and DGP design; this round reviewed the near-final release bundle, preview site, and dataset quality._
_Goal: de-duplicated, cross-pollinated, ordered HIGH → MEDIUM → LOW. All detail preserved._

---

## 1. LeadForge Codebase Feedback

### HIGH

**[HIGH-C1] `has_open_opportunity` and `opportunity_estimated_acv` carry post-snapshot, label-correlated signal — a direct violation of the project's hardest invariant.**

All three models identified this as the single most important code bug. The mechanics, established most precisely by Claude (adaptive), are:

In `leadforge/render/snapshots.py`, `open_opps = od[od["close_outcome"].isna()]` is used to derive `has_open_opportunity=True` and `opportunity_estimated_acv`. The filter `od` is correctly restricted to opportunities *created* on/before the snapshot day, but `close_outcome` is **not** a snapshot-day field: `leadforge/simulation/engine.py` sets `close_outcome`/`closed_at` from the lead's *eventual* full-horizon trajectory after the full 90-day simulation. So `close_outcome.isna()` encodes "this deal never reached a terminal state during the entire 90-day run," which the day-30 snapshot cannot know. Consequences: a lead whose in-window opportunity closes_won by day 90 receives `has_open_opportunity=False` and `opportunity_estimated_acv=NaN`; a lead whose opportunity is still open at day 90 (stalled non-converter) receives `True`. Both features therefore carry post-snapshot, label-correlated signal, and `expected_acv` inherits it (it backfills the band midpoint precisely for the converter rows whose `opportunity_estimated_acv` was nulled out).

The impact is large: the public snapshot-filtered `opportunities.parquet` carries 4,426/4,255/4,004 rows against 5,000 leads per tier, so a large fraction of leads have an in-window opportunity whose open/closed status is read from the future.

Fix: gate on `closed_at` vs the snapshot cutoff, i.e., `open ⟺ closed_at is null OR closed_at > lead_created_at + snapshot_day`. After fixing, measure AUC contribution before/after to confirm scope. Also flag `has_open_opportunity` and `opportunity_estimated_acv` with `leakage_risk=True` in the feature dictionary until the fix ships.

ChatGPT (standard) additionally notes that `total_touches_all` is documented as *the* deliberate trap, and that the dataset card explicitly claims engagement features are "computed strictly over events on days [0, 30]" — so the opportunity-feature leak is an undocumented, unintentional second trap that contradicts the published promise.

**[HIGH-C2] No CI configuration is present in the reviewable artifact, but the project's entire trust model rests on CI-enforced guarantees.**

Flagged at HIGH by Claude (standard and adaptive). No `.github/workflows/*.yml` and no `.github/` directory of any kind appears in the 300-file foldermix dump. Meanwhile `pyproject.toml` contains prose describing specific CI jobs: "CI's 'test' job (which installs only `[dev]`)", "the CI `notebooks` job nbclient-executes `release/notebooks/*.ipynb`", "CI's type-check job doesn't install `anthropic`." `CLAUDE.md` asserts branch protection on `main`. An internal review note (`docs/external_review/summaries/key_findings.md`) states CI runs lint/mypy/pytest plus dataset jobs but is "Missing: release-candidate workflow."

Since every headline number on the dataset card is produced by `scripts/validate_release_candidate.py`, the most important confirmation is that this driver runs in CI on the release artifacts — otherwise the published metrics are only ever validated by hand. The README links to `.github/ISSUE_TEMPLATE/dataset_breakage_report.yml` and `realism_feedback.yml`, which are also absent from the pack.

Either (a) the workflows exist but were excluded from the pack — in which case the pack is not the complete authoritative artifact and CI should be added to the review bundle — or (b) they do not exist, in which case every "enforced in CI" guarantee is unbacked. This must be resolved before publication.

Gemini separately notes that the `[publish]` optional dependency group (`datasets`, `kaggle`, `markdown-it-py`) is not included in the main automated test matrix, so `build_shmuggingface_site.py` and related release-packaging scripts risk silent regressions.

**[HIGH-C3] The hidden DAG is validated and exported, but it may not be the executable DGP — the graph may function more as narrative metadata than as a causal engine.**

Flagged at HIGH by ChatGPT (standard and extended). The architecture promises a variable hidden world graph with five motif families, stochastic rewiring, mechanism assignment, and a DAG invariant, but the simulation appears to use the graph mainly through `world_graph.motif_family`; the actual rewired topology and edge weights may not drive node-by-node structural equations in the daily simulator. That makes the graph closer to narrative metadata than a causal data-generating graph, which is materially weaker than the stated design. If the graph shape does not causally determine conditional distributions, the "five motif families + stochastic rewiring" story overstates what actually varies across seeds.

ChatGPT (extended) adds: the documented graph invariant is stronger than what the code enforces. `generation_method.md` claims the sampled graph is acyclic, every node is reachable from a root, and the outcome node is reachable from every non-root subgraph. The actual `WorldGraph._validate()` checks acyclicity, type legality, nondegeneracy, and only that each outcome is reachable from *at least one* root. Either implement the stronger all-node / all-relevant-subgraph reachability checks, or weaken the public DGP claim to match what is enforced.

Claude (standard, as a HIGH-positive) confirms the DAG-enforcement machinery is real: acyclicity (`nx.is_directed_acyclic_graph` + `find_cycle` error message), node-type legality, non-degeneracy, and outcome-reachability are all enforced at construction; the label is event-derived in `simulation/engine.py` (true only when a `closed_won` event lands inside the window). The gap is specifically between what the graph structure represents vs what it causally drives.

**[HIGH-C4] Core leakage and exposure invariants are genuinely enforced in code — this is the strongest part of the package. (Positive finding.)**

Claude (standard) flagged this HIGH-positive so it is not lost among the problems. `leadforge/render/relational_snapshot_safe.py` drops `BANNED_LEAD_COLUMNS`/`BANNED_OPP_COLUMNS`, filters `SNAPSHOT_FILTERED_TABLES` per-lead to `lead_created_at + snapshot_day`, and omits `BANNED_TABLES`. These constants are single-sourced in `leadforge/validation/leakage_probes.py` and re-imported by the writer, manifest builder, and validator — they cannot drift. The writer is correctly wired into the public path in `leadforge/api/bundle.py` (gated on `bundle_filter.relational_snapshot_safe`, with a hard `ValueError` if `snapshot_day` is `None`). The label is event-derived, never sampled. This is an unusually rigorous synthetic-data release compared to most.

---

### MEDIUM

**[MEDIUM-C1] The leakage-probe taxonomy has no flat-feature snapshot-consistency probe — exactly the gap that let the `has_open_opportunity` bug through.**

Claude (adaptive). `leadforge/validation/leakage_probes.py` is thorough on the relational surface: `probe_banned_columns`, `probe_banned_tables`, `probe_snapshot_window` (audits event-table timestamps ≤ cutoff), `probe_deterministic_reconstruction`, split-overlap, near-duplicate, and label-drift probes. But `probe_snapshot_window` validates relational event timestamps, not *derived* flat features; and `post_snapshot_aggregates` only targets `total_touches_all`. There is no probe asserting that each flat feature is reconstructable from events in `[0, snapshot_day]`. Recommended fix: add a probe that recomputes the opportunity-derived features under a strict `closed_at > cutoff` rule and asserts equality with the shipped column.

**[MEDIUM-C2] `to_dataframes_snapshot_safe` silently no-ops the opportunities filter if `opportunities` lacks a `lead_id` column.**

Claude (standard). `_build_anchor` correctly asserts `leads.lead_id` is unique (preventing row inflation), but `_filter_to_snapshot_window` left-merges each event table on `lead_id` without asserting the event table actually carries `lead_id`. The schema defines `opportunities.lead_id` as an FK, so in practice this holds; but a schema change that renamed or dropped that key would cause the merge to produce all-`NaT` cutoffs → `(ts <= NaT)` → `fillna(False)` → **every opportunity row dropped silently** rather than raising. Since this is the public-bundle leakage boundary, the function should assert `lead_id in events.columns` for each `SNAPSHOT_FILTERED_TABLES` entry and fail loud.

**[MEDIUM-C3] `total_touches_all` receives MCAR missingness distortion in higher tiers, partially defeating its purpose as a clean leakage trap.**

Claude (standard). In `render/snapshots.py`, `_NUMERIC_DISTORTION_COLS` includes every non-target `Int64`/`Float64` column, so `total_touches_all` is eligible for the missingness pass (2% intro → 18% advanced). The trap is pedagogically strongest when a student can cleanly show that adding it spikes AUC; injecting up to 18% NaN muddies that demonstration and is not disclosed in the feature dictionary's trap description. Either exempt `total_touches_all` from distortion (cleanest) or document that the trap is itself degraded in advanced.

**[MEDIUM-C4] The label window comparison is strict `<`, contradicting the documented inclusive "within 90 days."**

Claude (standard). `engine.py` sets the label via `state.conversion_day < config.label_window_days`. A conversion on exactly day 90 is labelled negative, whereas "converted within 90 days" reads as inclusive (`<= 90`). With `snapshot_day=30` features and a 90-day label this affects very few rows, but it is a spec/code mismatch a careful student auditing the boundary (exactly what the break-me guide invites) will find. Make the comparison inclusive or document the half-open convention in `generation_method.md`.

**[MEDIUM-C5] The canonical package layout in CLAUDE.md is substantially out of sync with the real tree.**

Both Claude versions. Listed modules that don't exist: `core/time.py`, `narrative/company.py`/`product.py`/`market.py`/`funnel.py`, `structure/templates.py`/`constraints.py`, `simulation/world.py`/`scheduler.py`/`interventions.py`, `render/graph_export.py`/`metadata.py`/`notebooks.py`, `exposure/redaction.py`, `validation/artifact_checks.py`. Modules that exist but are omitted: `render/relational_snapshot_safe.py`, `render/tasks.py`, `exposure/metadata.py`, `validation/{bundle_checks,lead_scoring,leakage_probes,llm_critique,release_quality,reporting}.py`, `mechanisms/influence.py`. Relatedly, `tests/exposure/test_redaction.py` exists with no `exposure/redaction.py` module. None of this is a runtime bug, but it makes the canonical doc untrustworthy as an onboarding map; either regenerate it from the tree or delete it.

**[MEDIUM-C6] `leadforge validate` is much weaker than the release-quality story implies.**

ChatGPT (extended). The release documentation says `scripts/validate_release_candidate.py` runs the full panel (AUC, PR-AUC, calibration, lift, P@K, expected-ACV capture, leakage probes, cross-seed bands) and exits non-zero outside declared bands. By contrast, package-level `check_difficulty()` only verifies that the manifest names a known difficulty profile, and `realism._check_conversion_rate()` accepts anything between 1% and 95%. That is fine for lightweight bundle validation, but it is not release-grade invariant enforcement. Add an explicit `leadforge validate --release-gates` path, or make the docs consistently distinguish "bundle sanity validation" from "v1 release acceptance validation."

**[MEDIUM-C7] The `is_sql=False` direct-conversion bypass is a structural patch rather than a causal fix.**

Gemini (standard and extended). The CHANGELOG and `engine.py` describe a workaround for the deterministic invariant where pre-SQL leads never convert: a "rare direct conversion path" with a heavily discounted daily probability patches the 0% conversion bug but functions as operational duct-tape. A proper fix should modify the causal graph mechanisms to allow legitimate, fast-tracked conversions rather than forcing a random noise discount.

**[MEDIUM-C8] The `[publish]` extra dependencies are not in the main CI test matrix.**

Gemini (standard). `pyproject.toml` defines a `publish` optional dependency group containing `datasets`, `kaggle`, and `markdown-it-py`. The CI pipeline primarily relies on `[dev]` dependencies. Without explicitly including `[publish]` extras in the automated testing matrix, release-packaging scripts risk silent regressions.

---

### LOW

**[LOW-C1] `_apply_difficulty_distortions` reconstructs `RNGRoot(seed)` from scratch rather than deriving a named substream from the run's root RNG.**

Claude (standard). Determinism is preserved (it's a pure function of `seed`), but it bypasses the "single seeded root, deterministically-derived substreams" architectural invariant. If two callers ever pass the same `seed` but different upstream substream consumption, the distortion noise will be identical — a latent correlation surprise. Prefer threading the existing root.

**[LOW-C2] `package-lock.json` pins `git+ssh://` while `package.json` declares `github:` (https).**

Claude (standard and adaptive), ChatGPT (extended). A clean `npm install` in an environment without an SSH key/agent — fresh CI runner, contributor with only HTTPS credentials — will fail to resolve `@shmuggingface/core`. Re-generate the lockfile so it resolves via HTTPS. (Also noted as a ShmuggingFace framework issue in §4.)

**[LOW-C3] `generation_method.md` overstates the outcome-reachability invariant.**

Claude (adaptive). The doc says the outcome node is "reachable from every non-root subgraph," but `_check_outcome_reachable` in `structure/graph.py` only verifies reachability from at least one root. Either implement the stronger check or weaken the public claim. Separately, the `graph` property returns the live internal `nx.DiGraph` with only a "read-only intent" docstring — callers can mutate it past validation; returning a copy or frozen view would enforce the invariant.

**[LOW-C4] Determinism rests on legacy numpy `RandomState`.**

Claude (adaptive). `core/rng.py` derives substreams from `SHA-256(seed:name)` — clean and reproducible. The docstring itself flags `RandomState` (vs `Generator`) as tracked tech debt. Fine for v1, but pin the numpy major version in `pyproject.toml`/CI so byte-determinism (asserted by `verify_hash_determinism.py`) can't silently break on a numpy upgrade.

**[LOW-C5] `api/recipes.py` config precedence uses many discrete kwargs; a typed `GenerationOverride` object would clean this up.**

Gemini. Passing up to a dozen discrete kwargs (`snapshot_day`, `label_window_days`, `n_leads`, etc.) through the API and CLI layers will become brittle as the simulation engine expands.

**[LOW-C6] A dedicated `leadforge release` CLI command group would prevent script rot.**

Gemini. Release and packaging pipelines are scattered across ad-hoc scripts in `scripts/`. Integrating a release command group would unify the developer experience for subsequent versions.

---

## 2. LeadForge Dataset v1 — Data Feedback

### HIGH

**[HIGH-D1] The three difficulty tiers are almost entirely a prevalence/noise axis, not a learnable signal-complexity axis — and the headline framing does not match this reality.**

All three models, at HIGH, with strongest detail from Claude (adaptive). LR AUC is flat across tiers (0.879/0.886/0.886) while conversion rates move 42.7%→21.6%→8.4%. AUC is base-rate invariant, so its flatness is the tell: the *ranking* problem barely changes across tiers. Reading `mechanisms/policies.py` confirms the construction: `signal_strength` attenuates only the secondary latent-score weights while the conversion hazard is recalibrated per tier to hit the target conversion-rate band — so the dominant lever is the target conversion rate, not signal degradation.

Claude (adaptive) adds a particularly sharp observation: prevalence-normalized, "advanced" is actually the *most* separable tier — lift over random-AP baseline is ≈1.78×/2.66×/4.18× (intro→advanced) and top-decile lift is ≈1.81×/2.72×/3.97×. A student told "advanced is harder" is mostly learning that rarer positives depress precision/AP/Brier — a base-rate lesson mislabelled as a signal-complexity lesson. The README discloses flat AUC, but then frames AP/P@K/Brier as the difficulty signal, which is the conflation.

What changes meaningfully across tiers: AP (0.761/0.575/0.351), P@100 (0.80/0.59/0.34), top-decile rate (0.773/0.587/0.333), Brier (0.130/0.110/0.061), missingness rate, and noise scale.

ChatGPT (standard) adds: GBM−LR is −0.0045/−0.0072/−0.0133 across all tiers, so "Advanced" does not reward non-linear models. Students comparing model families will find linear wins slightly, everywhere — a thin and potentially misleading lesson.

Gemini: "The 'Advanced' tier is merely an imbalanced dataset, not a structurally harder modeling problem."

Recommended fixes: (a) re-tune `signal_strength` so it meaningfully separates AUC/AP beyond what base rate explains, or (b) relabel the axis honestly as a prevalence/precision/noise regime, not a discrimination-difficulty regime. If option (b), update the README, dataset card, Kaggle/HF metadata, and tier-picker guidance accordingly.

**[HIGH-D2] `has_open_opportunity` and `opportunity_estimated_acv` constitute an undocumented, unintentional second leakage trap beyond `total_touches_all`.**

Claude (adaptive, marked HIGH). The dataset card claims `total_touches_all` is *the* deliberate trap and that engagement features are "computed strictly over events on days [0, 30]." The opportunity-feature post-snapshot leak (see §1 finding C1 for full mechanism) quietly breaks that promise. Until the code is fixed, these features must be flagged `leakage_risk=True` in the feature dictionary and added to Known Limitations.

ChatGPT (standard) adds: `opportunity_created`, `has_open_opportunity`, and `opportunity_estimated_acv` can dominate the classification task and represent business state rather than ordinary lead features; they all deserve equally careful treatment to the `total_touches_all` trap documentation.

**[HIGH-D3] Advanced-tier calibration is severely poor (max-bin error 0.5234) and the README's Calibration section omits the metric that shows it.**

Claude (standard and adaptive, both HIGH). The headline Calibration table lists LR AUC/AP/P@100/Brier — and Brier *improves* as difficulty rises (0.130→0.110→0.061) purely because the base rate falls, making calibration appear to get *better* with difficulty. But `calibration_max_bin_error` is 0.2497/0.2490/**0.5234** — the advanced tier's worst calibration bin is off by more than 50 percentage points. Cross-seed spread is 0.4828 for advanced (vs 0.249 for intermediate), meaning it is also highly unstable. Since "teaching calibration" is a stated intended use and Notebook 04 is built on it, shipping a tier with that calibration gap without surfacing it in the dataset card is misleading. Add `calibration_max_bin_error` to the README Calibration table and add a known-limitation bullet. Flag advanced calibration as unreliable, or confine the calibration teaching use to intro/intermediate.

**[HIGH-D4] The acceptance bands are fitted to the generator's own output rather than to external realism — passing the gates proves reproducibility, not fidelity.**

Claude (standard, HIGH). `release/docs/v1_acceptance_gates_bands.yaml` carries inline comments that say so directly: the intro GBM−LR band comment reads "band fits the data," and `gbm_minus_lr_auc` is set to `{min: -0.05}` specifically to accommodate the observed negative deltas, whereas the example band in `validation/difficulty.py` docstring shows the *intended* posture `{min: 0.005}`. The `calibration_max_bin_error` band for advanced is `{max: 0.90}`, accommodating an observed 0.5234 — a model can be more than half-miscalibrated in its worst bin and still "pass."

This is fine as a regression gate (it prevents drift), but the README's "every realism claim is backed by validation_report.md" risks being read as "the data is realistic because it passes the gates." State plainly in `v1_acceptance_gates.md` that bands are descriptive regression fences, not realism thresholds (the YAML comments say this; the README does not).

**[HIGH-D5] The 93% train/test account overlap is a structural evaluation problem for a B2B dataset — shipping only the random split as the default is insufficient.**

Gemini called this HIGH (most strongly); Claude and ChatGPT called it MEDIUM. Gemini: "In real-world mid-market procurement, the primary challenge is generalizing to *unseen* accounts. With this overlap, tree models will simply memorize account-level latent traits rather than learning generalizable firmographic signals." Claude (adaptive): the README and `break_me_guide.md` §5 are admirably explicit (518/557 ≈ 93% of test accounts also in train), but the *as-shipped* `tasks/.../{train,valid,test}.parquet` are the random split, and every headline metric is computed on it.

Recommended: ship a second `GroupKFold(account_id)` split (or a chronological split) as a first-class task variant, rather than leaving it as a "retrain yourself" exercise. As-is, the easy path produces the inflated number and the honest path requires extra work most students won't do.

---

### MEDIUM

**[MEDIUM-D1] Intro conversion rate (42.7%) is implausible for top-of-funnel B2B lead scoring and should be explicitly labelled as a pedagogical setting, not a realistic B2B funnel.**

Claude (adaptive), ChatGPT (standard and extended). A 42.7% lead→closed-won rate is more like a warm MQL or qualified-pipeline population. Claude (adaptive) adds: the snapshot-filtered `opportunities.parquet` shows ~80–88% of leads have an in-window opportunity by day 30, versus a realistic MQL→opportunity rate of ~10–20%. The data behaves like a late-stage opportunity dataset wearing a lead-scoring label. The Known-Limitations section is candid about many things but omits this. For a teaching dataset whose pitch is "the confusions students hit on real CRM data," the unrealistic funnel geometry deserves an explicit caveat. Suggested framing: Intro = high-prevalence classroom warm-up (not a realistic unqualified B2B lead funnel).

**[MEDIUM-D2] `lead_source` and `first_touch_channel` are byte-identical in v1, producing a redundant column and a near-zero-signal channel axis that contradicts the domain narrative.**

All three models. The feature dictionary admits both columns "encode the same origination channel under different field names." The channel audit shows per-channel rate spread ≤ 0.043 and out-of-sample univariate AUC ≈ 0.50–0.52. ChatGPT (extended) notes the scenario prose implies channel should matter ("inbound 45%, SDR 35%, partner 20%"), but conversion is driven by motif hazards keyed off latent traits rather than channel-conditional probabilities. Consider dropping the duplicate column for v1 (keep `lead_source`) and labelling the channel-signal weakness more prominently — currently buried as limitation #3.

**[MEDIUM-D3] GBM never beats LR in any tier, which undercuts "comparing model families" as an intended use.**

Claude (standard), ChatGPT (standard and extended). GBM−LR AUC is −0.0045/−0.0072/−0.0133. The README attributes this to a linear-dominated snapshot and defers non-linear interactions to v2 — a reasonable disclosure — but "comparing model families" is listed as an intended use, and in v1 that comparison has exactly one outcome (linear wins, slightly, everywhere). Either remove model-family comparison from intended uses for v1 or inject at least one genuine interaction so the comparison is pedagogically live.

**[MEDIUM-D4] The "relational feature engineering" use case is barely supported — the signal lives in static columns, not in cross-table structure.**

Claude (adaptive). The signal lives in static firmographic/personographic columns (full LR AUC ~0.88 vs `engagement_only` ~0.58 and `source_only` ~0.50). Notebook 02's own honest takeaway puts the GBM(eng)−GBM(flat) AUC lift at +0.0147 on seed 42 — smaller than the documented cross-seed `gbm_auc` spread of ~0.027 — i.e., the relational lift is within noise. That undercuts a headline intended use ("Teaching relational feature engineering against snapshot-safe tables"). v2 should inject signal that genuinely lives in cross-table structure; for v1, the README should temper the claim.

**[MEDIUM-D5] Advanced-tier calibration is too unstable to be a reliable teaching surface.**

Claude (adaptive). `calibration_max_bin_error` median 0.5234 with cross-seed spread 0.4828. This is a base-rate artifact (few positives in high-score bins at 8.4% prevalence) and interacts with the prevalence finding (#D1) — Brier *improves* as difficulty rises (advanced 0.061), which is itself misleading as a "difficulty" signal. Recommend flagging advanced calibration as unreliable, or teaching calibration only on intro/intermediate.

**[MEDIUM-D6] The feature set is useful for a teaching dataset but misses several important B2B lead-scoring signals.**

ChatGPT (standard and extended). Missing or underdeveloped signals: technographics, competitor/complement stack, campaign/UTM hierarchy, email opens/clicks/replies, SDR/rep capacity and territory, account history, buying-committee size, budget/authority/need/timeline proxies (BANT), trial/product-activation telemetry, deduplication/merge history, CRM enrichment confidence, negative behaviors (unsubscribe, career-page visits), sequence step and SLA/follow-up latency. The design documents identify many of these as canonical or candidate variables. The generation method already admits v1 is not temporally rich and uses clean demographic strings, which is honest but should be more visible on the public card.

**[MEDIUM-D7] Non-physical noise artifacts are disclosed but the Known-Limitations section does not call them out prominently enough.**

ChatGPT (extended), Claude (adaptive, LOW). Gaussian noise on float features (including `days_since_first_touch`, `days_since_last_touch`) can produce negative values; outlier injection sets values to `median ± 5σ`. The README's "Known limitations" section emphasizes flat AUC, GBM underperformance, weak channel signal, and small cohort-shift degradation, but does not clearly call out that Gaussian distortion can produce non-physical feature values. Recommended addition: "advanced-tier noise can make some bounded/time/count-like proxies non-physical; treat these as synthetic distortion artifacts." Note: Claude (standard) also observes a simple post-noise clamp to physical ranges would remove this wart at virtually no cost.

**[MEDIUM-D8] Account/contact overlap should be elevated from caveat to primary evaluation warning, and the headline metrics should clearly state the split type.**

ChatGPT (extended). 518 of 557 test accounts in the intermediate bundle also appear in train (≈93%). The validation report's cohort split does not show consistent degradation, but the conceptual leakage risk remains. For students, the right framing is: random split metrics are a baseline exercise; group-split metrics are the generalization check. The current structure makes the inflated number easy and the honest number extra work.

---

### LOW

**[LOW-D1] Gaussian-noise distortion can produce non-physical values and nothing clamps them.**

All three models (various severities). `_apply_difficulty_distortions` adds `N(0, noise_scale·σ)` to float columns without clipping. The README lists this as a known limitation. Claude (standard): a post-noise clamp to physical ranges (e.g., `days_since_x >= 0`) would cost nothing and remove a known wart. Claude (adaptive): negative days/ACV is a giveaway that the data is synthetic. ChatGPT (extended): the HF preview showed a negative numeric value in the row preview, consistent with noise leaking into a duration/recency-like field.

**[LOW-D2] `touches_week_1` spans 8 days (days 0–7 inclusive) while the name implies 7 days.**

Claude (adaptive) and ChatGPT (extended). The builder uses `_day <= 7`, which is 8 day values (0, 1, 2, 3, 4, 5, 6, 7). The feature dictionary footnotes this, but the name is a footgun. Rename to `touches_days_0_7` or change the implementation to days 0–6.

**[LOW-D3] Feature count claims are inconsistent across the corpus.**

Claude (standard). The brief says "40+ features"; the README/feature dictionary correctly say 32 public columns. Align the brief and marketing copy to the real number.

**[LOW-D4] `lead_created_at` ships as an ISO-8601 string in the flat CSV.**

Claude (standard). The feature dictionary advises binning/dropping it for production and using it as a cohort key, but shipping it as a raw string invites the "feed raw timestamp to a linear model" mistake. Consider also shipping a pre-binned `cohort_month` to make the intended cohort-split workflow the path of least resistance.

**[LOW-D5] The flagship leakage and value-ranking lifts are single-seed and near the noise floor.**

Claude (adaptive). Notebook 03's headline trap lift (~+0.03 GBM AUC) and Notebook 04's value-ranking gain are measured only on seed 42, and the trap lift sits close to the documented cross-seed `gbm_auc` spread (~0.027); the CI gates assert the *sign* on a single seed. A small seed sweep would make the lessons robust rather than anecdotal. Notebook 02 already concedes this for relational lift. Separately, nb02 teaches plain whole-train target encoding without out-of-fold cross-fitting — a notable omission in a series whose theme is leakage discipline.

**[LOW-D6] Known-limitations honesty is strong overall but has three disclosed gaps.**

Claude (adaptive). The project's candid disclosure of flat AUC, GBM≤LR, weak channel signal, and 93% account overlap is genuinely unusual and improves reviewer trust. Three gaps remain: (a) the opportunity-feature post-snapshot leak (#D2 above), (b) the unrealistic conversion/funnel rates (#D1 MEDIUM), and (c) the framing that AP/P@K/Brier measure "difficulty" when they primarily measure prevalence (#D1 HIGH).

---

## 3. LeadForge Dataset v1 — Presentation Feedback

### HIGH

**[HIGH-P1] `release/kaggle/dataset-metadata.json` ships with `"isPrivate": true` — a one-line publish blocker.**

Claude (adaptive). If uploaded as-is, the dataset publishes as a *private* Kaggle dataset — invisible to the students it targets. This requires an immediate fix before any Kaggle release. (It also illustrates the ShmuggingFace preview gap in §4 finding SF1: the preview is built from other files and would never surface this error.)

**[HIGH-P2] The Kaggle-style (Shmaggle) page contains leftover sock/laundry demo copy.**

ChatGPT (extended). The Shmaggle "Objective" section says: "Do an EDA and try to predict which socks and laundry conditions achieve suspiciously stable pair success." This is a publication blocker — it makes the page look templated and unreviewed, and it directly contradicts the LeadForge lead-scoring narrative. Root cause: hardcoded at `node_modules/@shmuggingface/core/src/generate.mjs:1161`.

**[HIGH-P3] The Kaggle preview advertises fabricated Usability scores and dataset medals.**

Claude (standard, HIGH). `scripts/build_shmuggingface_site.py` hardcodes `TIER_USABILITY = {"intro":"9.4","intermediate":"9.1","advanced":"8.9"}` and `TIER_MEDAL = {"intro":"Gold","intermediate":"Silver","advanced":"Bronze"}` and emits them into the per-tier config as `kaggleUsability`/`kaggleMedals`. Kaggle computes Usability itself from completeness/license/etc., and Kaggle does not award Bronze/Silver/Gold "medals" to datasets in the way implied. These values should be removed — they are not only wrong but could mislead reviewers into evaluating non-existent platform behavior. Claude (adaptive) notes the committed `kaggle.html` does not actually render these fields (see finding P7 on preview drift), so they are currently dead config — but they remain latent misinformation.

**[HIGH-P4] The HF data viewer shows incorrect split row counts — train shows 5,000 rows despite the 70/15/15 split.**

ChatGPT (standard and extended). The preview says "train · 5,000 rows" even though the full dataset is 5,000 rows total. A first-time user would reasonably think the train split alone has 5,000 rows. Claude (standard) identifies the underlying cause as a ShmuggingFace config issue (framework may render dataset-level row count as split-level row count rather than reading per-split counts from the manifest).

**[HIGH-P5] Difficulty tier communication is misleading — "Advanced" implies harder nonlinear modeling, not higher class imbalance.**

All three models. Gemini (extended): "Because the AUC is flat, students downloading the 'Advanced' bundle will waste time tuning complex neural networks for non-existent non-linearities, assuming the boundary is highly complex." Claude (adaptive): the README discloses flat AUC but frames AP/P@K/Brier as the difficulty signal, which is the conflation. ChatGPT (extended): the tiers need to be reframed as "business-prioritization difficulty" rather than "classification difficulty."

Recommended copy (ChatGPT extended): "Intro = high-prevalence classroom warm-up; Intermediate = default benchmark for most notebooks; Advanced = low-prevalence prioritization/calibration exercise, not harder nonlinear classification."

---

### MEDIUM

**[MEDIUM-P1] The instructor companion (`intermediate_instructor/`) is listed in the public README tree — verify it is never co-published.**

Claude (adaptive, MEDIUM; elevated risk if HF instructor dataset is public). `release/README.md`'s "What's inside" tree lists `intermediate_instructor/` (full-horizon tables + hidden DAG + latent registry + `customers`/`subscriptions`) as a sibling of the public tiers inside `release/`. The full-horizon instructor bundle reconstructs the label by construction, so if it lands in the public Kaggle/HF upload this is a total redaction bypass. The Kaggle `resources` list correctly omits it, but the README copy that is *also* the Kaggle/HF description still advertises it. Confirm the instructor dataset is gated/separate and remove `intermediate_instructor/` from the public-facing "what's inside" tree.

**[MEDIUM-P2] Notebooks 01 and 02 contain stale internal forward-references that should not appear in published teaching material.**

Claude (adaptive). Notebooks 01 and 02 repeatedly cite "Notebook 03 *(coming in PR 6.2)*," "Notebook 04 *(coming in PR 6.2)*," "the break_me_guide template lands in PR 6.3," and `feature_dictionary.md` says "see notebook 4 once Phase 6 ships" — yet all four notebooks ship in this release. Notebooks 03/04 were updated to drop these; 01/02 were not. Internal PR/phase numbers should not appear in published teaching material. ChatGPT (extended) additionally notes that Notebook 01's markdown still says Notebook 03 is "coming in PR 6.2."

**[MEDIUM-P3] Notebook 01's baseline keeps `total_touches_all` and contradicts the README's stated default, risking teaching the trap by example.**

Claude (adaptive) and ChatGPT (extended). The README says "Drop `total_touches_all` unless you're demonstrating leakage detection," but Notebook 01 deliberately *keeps* it (to reproduce the validation panel). A beginner who lifts Notebook 01's "drop IDs + label only" feature selection as a template inherits the leakage trap. ChatGPT (extended): for public pedagogy, Notebook 01 should start with a clean baseline that drops all `leakage_risk=True` columns, then optionally show an "as-shipped validation reproduction" appendix.

**[MEDIUM-P4] The notebook arc uses only the intermediate tier, so the advanced tier's worst pathologies are never shown.**

Claude (adaptive). All four notebooks set `BUNDLE = Path("../intermediate")`. Notebook 04 demonstrates calibration where it looks good (max-bin error ~0.13) and concludes "the LR baseline is well-calibrated," while the advanced tier's 0.52 calibration error (the case a student is invited to "graduate" to) is never shown or addressed. A short "switch to advanced and watch calibration break" cell would close the gap between what's taught and what's shipped.

**[MEDIUM-P5] Two copies of the README can (and do) drift; the Kaggle copy omits content the GitHub/HF copy advertises.**

Claude (adaptive). `release/README.md` and the README embedded in `dataset-metadata.json.description` have divergent "What's inside" trees — the Kaggle copy lists `dataset-metadata.json`/`LICENSE`/cover image and omits the `notebooks/`, `validation/`, and `intermediate_instructor/` lines; the canonical README shows the opposite set. The Kaggle `resources` manifest ships no notebooks at all, yet the shared README/HF copy promotes "notebooks/ 01 baseline · 02 relational · 03 leakage · 04 calibration." Single-source the README and decide whether notebooks ship on Kaggle.

**[MEDIUM-P6] Column descriptions are empty for the 2nd and 3rd tier in the Kaggle preview.**

Claude (standard). In the committed `kaggle.html`, the first tier's column table shows full descriptions (e.g., `expected_acv` → "opportunity ACV if available by snapshot..."), but repeated per-tier tables show `<td class="col__desc"></td>` for the same columns. Since the feature set is identical across tiers, every tier's schema preview should carry the same descriptions. Either a build-script bug (descriptions loaded once) or a ShmuggingFaceCore rendering limitation — trace which before publish.

**[MEDIUM-P7] The committed preview HTML almost certainly drifted from the current build script.**

Claude (standard). The build script emits `kaggleUsability`, `kaggleMedals`, `downloads`, `likes`, `discussions`, and per-tier `subtitle` strings, yet the committed `kaggle.html`/`huggingface_*.html` do not surface the medal/usability values — meaning the committed HTML was produced by a different (older) script or core version. Re-generate the committed previews from the current script and confirm what actually renders before treating them as the review surface.

**[MEDIUM-P8] `touch_type` appears in the Kaggle preview's column list but is a relational table column, not a flat task column; and the description uses unrendered RST literal backticks.**

Claude (standard). The relational `touches` table has `touch_type`, but the dataset-page column preview should reflect the flat task schema (the 32 columns ending in `converted_within_90_days`). Seeing `touch_type` with prose like `` ``email``, ``call``... `` (literal double-backticks, an RST artifact) suggests the preview is concatenating relational-table schema rows into the flat-dataset column view. Confirm the preview separates "task columns" from "relational table columns," and fix the `` `` `` literal-backtick rendering.

**[MEDIUM-P9] The README is accurate but too dense above the fold — the most important operational guidance is buried.**

ChatGPT (standard and extended). A student needs a clearer first-screen path: "Start with Intermediate," "drop `total_touches_all`," "use parquet splits," "use GroupKFold for account-level generalization," "run notebook 01 first." Currently buried below tables and narrative. Gemini also recommends a prominent "Start Here" quick-link to the first baseline notebook directly in the Kaggle description and HF Dataset Card.

**[MEDIUM-P10] The notebook sequence lacks a group-aware evaluation checkpoint.**

ChatGPT (standard and extended), Gemini. The four notebooks form a good teaching arc, but the missing lesson is a required account-level or contact-level split comparison — especially since the release discloses 93% account overlap. Recommended: add a `GroupKFold(account_id)` section in Notebook 02 or 04 and show how metrics change.

**[MEDIUM-P11] HF default config is set to `intermediate`, but `intro` is the correct pedagogical entry point.**

Gemini (standard). In the dataset card YAML (`release/huggingface/README.md`), `intermediate` is explicitly set to `default: true`. A student executing `load_dataset("leadforge/leadforge-lead-scoring-v1")` without specifying a subset bypasses the introductory tier entirely. The default configuration should map to the easiest pedagogical entry point (`intro`).

**[MEDIUM-P12] Mock social/activity metrics are distracting and occasionally contradictory.**

ChatGPT (standard). The Shmaggle activity panel shows nonzero "views" and "downloads" language that is not real platform telemetry, and the "Downloads 0 / 213 in last 30 days" display is confusing. These elements should either be removed, labeled as mock placeholders, or replaced with neutral zeros before external review/publication.

**[MEDIUM-P13] The data explorer preview is not reliable enough for a first-time visitor.**

ChatGPT (extended). The HF viewer reports "Subset … 5,000 rows" and "Split train · 5,000 rows" despite the release describing 70/15/15 splits. The Shmaggle explorer says "10 of 32 columns" but renders the entire column list inline, then reports obviously broken distribution summaries such as `converted_within_90_days` with "136 unique values." The row preview also concatenates adjacent fields without delimiters, e.g., employee and revenue bands run together. This weakens trust before users ever download the data.

---

### LOW

**[LOW-P1] HF metadata still contains mock-only fields.**

ChatGPT (extended). The HF-style page shows `Homepage: shmuggingface.local` and `Paper: Mock release review`. Appropriate for the preview but should be stripped or replaced before a real Hugging Face publish.

**[LOW-P2] Metadata formats label says "CSV" only; release also includes Parquet.**

ChatGPT (standard). Should say "CSV + Parquet," and the metadata/schema display should distinguish the flat CSV from task-split Parquet files.

**[LOW-P3] Tag sets are inconsistent across surfaces.**

Claude (adaptive). HF: 7 tags including non-topical `datasets`/`pandas`; Kaggle: 8 including `education`/`saas`; preview hard-codes its own 8. Also: the parquet task-split schemas in `dataset-metadata.json` type integer count columns as `"number"` (the feature dictionary declares `Int64`). The `split` column ships in `lead_scoring.csv` but isn't in the authoritative `feature_dictionary.csv`.

**[LOW-P4] Flagship lifts are single-seed and near the noise floor.**

Claude (adaptive). See §2 LOW-D5 — repeated here because it affects the notebook's teaching credibility as a presentation artifact.

**[LOW-P5] Flat CSV naming convention could confuse Kaggle-native users.**

Gemini. Presenting a unified `lead_scoring.csv` with a `split` column instead of separate `train.csv`/`test.csv` files is non-standard for the Kaggle audience. Clarify in the dataset card why this was chosen.

**[LOW-P6] Notebook arc is pedagogically coherent and is a genuine strength.**

Claude (standard), ChatGPT (standard). The arc 01-baseline → 02-relational-FE → 03-leakage/time-windows → 04-lift/calibration is logical with no obvious gap. Notebook 03 explicitly dissects the `total_touches_all` trap. The one identified gap: no notebook walks the random-vs-grouped split delta despite the README calling it the single most impactful split issue and the break-me guide documenting the recipe.

---

## 4. ShmuggingFace Integration Script (`scripts/build_shmuggingface_site.py`)

_These issues were extracted by the ShmuggingFace dev from the upstream `shmuggingface_review_synthesis.md` as downstream (LeadForge-side) integration responsibilities, confirmed 2026-05-25. Three items are partially disputed and covered in the rebuttal (`replies/shmuggingface_dev_rebuttal.md`). Cross-references to §3 (Presentation) are noted where the finding also appears there._

---

### HIGH

**[HIGH-I1] The integration script reads different source files than the real Kaggle/HF metadata — the preview structurally cannot surface the most important publish bugs.**

`scripts/build_shmuggingface_site.py` builds each tier's config from `manifest.json`, `metrics.json`, `feature_dictionary.csv`, `lead_scoring.csv`, and the rendered `release/README.md`, and hard-codes the task (`tabular-classification`), license (`MIT`), splits, and tags list. It **never reads** `release/kaggle/dataset-metadata.json` or `release/huggingface/README.md` — the files that actually drive the published pages. Specific consequences:

- The `isPrivate: true` Kaggle publish blocker (§3 HIGH-P1) is invisible in the preview by construction.
- Tag, task, license, split-config, and schema mismatches can pass the review round unnoticed.
- Per-tier `metrics.json`, `manifest.json`, and `lead_scoring.csv` are missing from the review bundle, so offline reviewers and AI agents cannot verify the generated config against real inputs — which contradicts the "self-contained for AI review" claim in the README.

Fix: drive the preview config from the same two canonical metadata files the platforms consume, or add a diff/lint step that exits non-zero when preview fields disagree with `dataset-metadata.json` / HF frontmatter. Also ship at least one tier's per-tier bundle files in the reviewable artifact set.

**[HIGH-I2] Only six files are mapped into the preview file listing — the claimed self-contained release bundle is vastly underrepresented.**

The HF-style preview lists only `lead_scoring.csv`, `feature_dictionary.csv`, three task-split Parquet files, and `dataset_card.md`. The release README claims a richer bundle: manifests, metrics, relational tables, docs, claims register, notebooks, validation artifacts. ShmuggingFaceCore supports real backing for every listed file via `sourcePath` or `downloadUrl`, so this is an integration-script omission. The Kaggle quick-start command also references `data/train.csv`, which does not match any displayed file path (see §3 MEDIUM-P13 for the UX consequence).

Fix: include the full artifact tree in the generated config, grouped by type (relational tables, docs, notebooks, manifests/metrics, validation). Ensure quick-start code examples reference actual displayed file names.

**[HIGH-I3] `package-lock.json` resolves `@shmuggingface/core` over SSH — breaks CI and contributors without GitHub SSH keys.**

`package.json` declares `github:ShmuggingFace/ShmuggingFaceCore#v1.0.0` (which resolves as HTTPS), but `npm install` generated a lockfile that pins `git+ssh://git@github.com/ShmuggingFace/ShmuggingFaceCore.git`. Any CI runner or contributor without SSH keys configured for GitHub will fail at `npm install`.

Fix: delete `package-lock.json` and regenerate it from a clean `npm install` environment that uses HTTPS. While regenerating, update the pin to `#v1.0.1` (see LOW-I1 below).

**[HIGH-I4] The integration script hard-codes fabricated Kaggle usability scores and medals.**

`TIER_USABILITY = {"intro": "9.4", "intermediate": "9.1", "advanced": "8.9"}` and `TIER_MEDAL = {"intro": "Gold", "intermediate": "Silver", "advanced": "Bronze"}` are hard-coded in the script and emitted into per-tier configs as `kaggleUsability`/`kaggleMedals`. On real Kaggle, the usability score is computed by the platform from completeness/license signals, and medals are earned through community engagement — neither is maintainer-settable. These values do not appear to be rendered by the current ShmuggingFaceCore version (the committed HTML does not show them), making them dead config today but latent misinformation if a future framework version renders them. Remove both constants from the integration script. (See rebuttal for a parallel request to the framework.)

---

### MEDIUM

**[MEDIUM-I1] `make_dataset_config` silently defaults on missing or malformed manifest/metrics fields instead of raising.**

Specific default-masking patterns:

- `manifest.get("n_leads", 5000)` — silently shows "5,000 leads" if the manifest is malformed.
- `manifest.get("snapshot_day", 30)` — silently shows "Day 30" if missing.
- `manifest.get("tasks", {}).get(TASK, {})` with `train_rows/valid_rows/test_rows` defaulting to `0` — silently shows "0 rows" per split.
- `metrics.get("medians", {})` — the root `release/metrics.json` nests under `tiers.<tier>.medians`, not at the top level; if per-tier files follow the root shape, every headline metric on the preview silently shows `0.0`.

For a tool whose job is faithful preview, missing required fields should raise rather than silently substitute plausible-looking defaults.

Fix: validate required manifest and metrics keys at load time; raise with a clear error on missing or shape-mismatched fields; add tests for malformed manifests and metrics.

**[MEDIUM-I2] All three tier pages share the same global README as their body — tier-specific content is buried as a downloadable file rather than driving the page.**

The script renders `release/README.md` once and passes the same `readme_html` into every tier config as `descriptionHtml`. The per-tier `dataset_card.md` that already exists for each tier is listed only as a downloadable file and never used as the page body. A visitor to the "Advanced" tier page sees the same generic cross-tier prose as the "Intro" tier.

Fix: use each tier's `dataset_card.md` as that tier's primary `descriptionHtml`; include the global README as a separate "Release overview" file entry.

**[MEDIUM-I3] Link rewriting in `_rewrite_links` has systematic gaps.**

Known gaps:

- Only handles `../`-prefixed relative links and one hard-coded validation path.
- Bare relative links such as `[LICENSE](LICENSE)` remain relative and will 404 on the static host.
- Rewritten `.github/ISSUE_TEMPLATE/*.yml` links 404 because those files are not included in the pack (see §1 HIGH-C2).
- The `GITHUB_BLOB_BASE` constant hard-codes the org and branch; if either changes, all preview links silently 404 (see LOW-I2).

Fix: treat all relative Markdown links consistently; validate rewritten targets exist before writing them into the config.

**[MEDIUM-I4] The `split` column exists in `lead_scoring.csv` but is absent from the feature dictionary.**

The integration script's `lead_scoring.csv` "about" text asserts a `split` column. The CSV contains it, but the authoritative `feature_dictionary.csv` (and the feature spec in `leadforge/schema/features.py`) does not declare it. This makes the preview more accurate than the canonical spec.

Fix: either add `split` to the feature dictionary (as a meta-column, not a lead-scoring feature), or remove it from the flat CSV and update quick-start examples accordingly.

**[MEDIUM-I5] The build script is untested, and two preview systems coexist — making the "byte-exact preview" claim unverifiable.**

`scripts/build_shmuggingface_site.py` (the ShmuggingFaceCore path) has no test file, while the older `scripts/preview_hf_page.py` and `scripts/preview_kaggle_page.py` *are* tested (`tests/scripts/test_preview_hf_page.py`, `tests/scripts/test_preview_kaggle_page.py`). It is therefore ambiguous whether the committed `release/_preview_committed/*.html` files were produced by the tested per-page scripts or the untested ShmuggingFaceCore build — so "byte-exact rendering" cannot be confirmed.

Fix: pick one preview-generation path; add tests for `build_shmuggingface_site.py` covering at least: config-field presence, the file listing, the per-split row counts, and link rewriting; add a CI check that the committed preview HTML matches a fresh build.

**[MEDIUM-I6] Config generation uses static `export default {...}` construction — brittle if the framework ever requires dynamic JavaScript.**

The script constructs `shmuggingface.config.mjs` by dumping a Python dict to JSON and prepending `export default`. This works as long as ShmuggingFaceCore's config format stays plain-data; if a future release requires JS module imports or dynamic expressions, all integrators break. (Partially disputed — see rebuttal: the framework should commit to a plain-data config contract.)

Fix (our side): add a syntax check by running the generated config through `node --input-type=module` or the ShmuggingFace CLI in tests, so any breakage is caught immediately rather than at deploy time.

**[MEDIUM-I7] The script passes only `df.head(8)` as preview rows — but the framework renders these as distribution summaries, not as a sample.**

`build_shmuggingface_site.py` reads the full CSV but passes only `df.head(8)` as `rows` to ShmuggingFaceCore. The visible consequence (§3 MEDIUM-P13): the Shmaggle explorer shows `converted_within_90_days` with "136 unique values" — because it derives distribution stats from 8 rows that happen to have varied values, then presents them without any "sample only" caveat. (Partially disputed — see rebuttal: the framework should accept precomputed stats or label sample-based summaries explicitly.)

Fix (our side): pass precomputed full-file profile stats once ShmuggingFaceCore supports a `profileStats` config field; in the interim, derive summary statistics from the full CSV before calling the framework, or at minimum pass a larger sample (e.g., all 5,000 rows) rather than 8.

**[MEDIUM-I8] The deploy workflow pushes directly to the production Cloudflare slot with no preview-branch gate.**

`deploy_site` runs `wrangler pages deploy ... --branch main --commit-dirty=true`, deploying whatever is on disk to the public production URL. Additional fragility: the Cloudflare token is sourced from a hardcoded personal path (`~/.config/adanim/cloudflare_api_token.env`), and `wrangler` is not declared in `package.json` — the deploy depends on an unpinned global binary.

Fix:

- Pin `wrangler` as a `devDependency`.
- Default deploys to a per-branch preview slot; require an explicit flag to publish to production.
- Read the Cloudflare token from an environment variable, with the personal env file as a local-only fallback.
- Add a preflight: verify a clean working tree and that the cover image exists before deploying.
- Keep ShmuggingFaceCore's own GitHub Actions pattern (using `npm ci` and repository secrets) as the CI-oriented baseline.

---

### LOW

**[LOW-I1] Downstream package pin should move from v1.0.0 to v1.0.1.**

`package.json` pins `github:ShmuggingFace/ShmuggingFaceCore#v1.0.0`. ShmuggingFaceCore v1.0.1 removes the stale socks/laundry demo copy that currently appears on the Shmaggle page (§3 HIGH-P2). Fix: update the pin to `#v1.0.1`, regenerate the lockfile (combined with HIGH-I3 above to do both in one `npm install`).

**[LOW-I2] `_rewrite_links` hardcodes the GitHub org and default branch.**

The `GITHUB_BLOB_BASE` constant embeds both. If the repo moves orgs or the default branch changes, all preview links silently 404. Fix: read the base from config or environment at build time.

**[LOW-I3] `deploy_site` does not capture Wrangler `stderr` into raised exceptions.**

Non-zero exit codes are caught, but `stderr` is not surfaced in the Python exception output, making Cloudflare deployment failures hard to diagnose. Fix: capture and include both `stdout` and `stderr` in raised exceptions; preserve Wrangler's deployment URL output on success.

**[LOW-I4] Good integration ergonomics worth preserving.**

Keep: `ensure_smf_core` resolution order (`--smf-core` override → npm-pinned package) with actionable error messages; Markdown rendering through `markdown-it` with `linkify` disabled; deterministic `shmuggingface.config.mjs` output as `export default {...}`; the prominent mock-notice boundary on the landing page.

---

## Cross-Cutting Observations

**What is already strong and should be preserved:**
- The leakage/exposure machinery: single-sourced constants in `leakage_probes.py`, re-imported by writer, manifest builder, and validator — enforced at the API level with a hard `ValueError`.
- Deterministic generation: `core/rng.py` SHA-256 substream derivation, verified by `verify_hash_determinism.py`.
- The claims register and generation-method docs: unusually rigorous and honest.
- Known-Limitations section: unusually candid about flat AUC, GBM≤LR, weak channel signal, and account overlap — most synthetic-data releases would bury these.
- The notebook learning arc: logical, builds on itself, includes a leakage lesson and a calibration/value-ranking lesson.

**The highest-priority fixes before publish (consensus across all models + integration triage):**
1. Fix `has_open_opportunity`/`opportunity_estimated_acv` post-snapshot leak (code fix in `snapshots.py` + new leakage probe) — §1 HIGH-C1.
2. Fix `isPrivate: true` in `release/kaggle/dataset-metadata.json` — §3 HIGH-P1.
3. Update the integration script to read `release/kaggle/dataset-metadata.json` and `release/huggingface/README.md` as its source of truth, or add a lint/diff gate — §4 HIGH-I1.
4. Expand the preview file listing to include the full release artifact tree — §4 HIGH-I2.
5. Regenerate `package-lock.json` over HTTPS and bump the pin to v1.0.1 (fixes SSH lockfile + socks/laundry text in one step) — §4 HIGH-I3, LOW-I1.
6. Remove hard-coded fabricated Kaggle usability scores and medals from the config — §4 HIGH-I4.

**The most important framing fix:**
Reframe the difficulty axis — either make the tiers genuinely differ in signal learnability (AUC, not just AP/Brier), or relabel them as a prevalence/precision/noise axis throughout the README, dataset card, notebook headers, and Kaggle/HF metadata.
