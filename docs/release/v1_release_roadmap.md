# v1 Lead-Scoring Dataset Release Roadmap

**Target:** Publish `leadforge-lead-scoring-v1` to Kaggle and Hugging Face as a best-in-class educational synthetic CRM dataset family.
**Source of truth:** This roadmap is derived from `docs/external_review/summaries/recommendations_pass.md` (signed off 2026-05-05).
**Companion docs:** `v1_release_design.md`, `v1_acceptance_gates.md`, `post_v1_roadmap.md`.
**Naming convention:** the *dataset* is `leadforge-lead-scoring-v1`. The leadforge *package* remains at `1.x` and is decoupled from this dataset version (resolves recommendation #15).

## Vision

Six external reviews, two reviewers, two iterations each, surface one shared verdict: leadforge is much further along than greenfield, and the v1 milestone is **release hardening + adversarial validation**, not core implementation. The single most important blocker is that `student_public` bundles currently leak `converted_within_90_days` end-to-end through public relational tables (verified locally by ChatGPT v2 in a 500-lead smoke bundle). Everything else is a quality-bar issue, not a correctness one.

The v1 release ships:
- a public family — intro / intermediate / advanced flat task splits + snapshot-safe relational tables
- a separate research/instructor companion — full hidden graph, latent registry, mechanism summary, full-horizon relational tables
- a release-grade validation report with figures, lift curves, calibration, leakage probes, and cross-seed bands
- a 4-notebook teaching sequence (baseline → relational FE → leakage demo → lift/calibration/value)
- a Kaggle dataset and a Hugging Face dataset, both packaged programmatically and dry-run-tested
- a public adversarial framing (issue templates, break-me guide, v2 decision log)

## v1-ready definition (operational)

A release candidate is v1-ready when **all** of the following hold. Concrete bands and probes live in `v1_acceptance_gates.md`.

1. Fresh release candidate generates from code, byte-identical to the previous build modulo the pinned `generation_timestamp`.
2. Structural and FK validation pass on every bundle in the family.
3. **Relational leakage probe**: no public-only join path reconstructs the target above tolerance.
4. **Direct leakage probes**: no model trained on suspect-only feature subsets reconstructs the target above tolerance.
5. **Split leakage**: account/contact split-overlap audit is intentional and documented.
6. **Cohort/time-shift split**: AUC degradation under cohort split is within configured band.
7. **Calibration**: Brier score and reliability curve within tier bands.
8. **Lift / P@K / value capture**: difficulty signal visible across tiers in AP, P@K, lift, and model-family deltas (LR vs GBM).
9. **Public/instructor diff**: every difference is intentional and listed in the manifest.
10. **Platform packages**: Kaggle `dataset-metadata.json` validates against current platform requirements; HF `README.md` loads via local `load_dataset()` for every config; cover image meets Kaggle minimums.
11. **Notebooks**: all four notebooks run top-to-bottom and reproduce validation report metrics within tolerance.
12. **LLM critique**: one-shot pass produces structured findings; no unresolved high-severity findings.

## Phase summary

| Phase | Title | Size | PRs | Depends on | Status |
|---|---|---|---:|---|---|
| 1 | Audit and naming | S | 1 | — | ✓ done |
| 2 | Snapshot-safe relational export | M | 2 | 1 | ✓ done |
| 3 | Release validation hardening | L | 3 | 2 | ✓ done |
| 4 | Channel-signal audit + dataset card | M-S | 1 | 3 | ✓ done |
| 5 | Platform packaging | M | 2 | 4 | ✓ done |
| 6 | Notebook sequence + adversarial framing | M-L | 3 | 5 | ✓ done |
| 7 | LLM critique + preview site | M | 4+1† | 6 | in progress (7.1–7.2.2 ✓; 7.3 blocked on Phase 8) |
| 8 | Pre-publish review-driven fixes | M-L | 4 | 7 (partial) | not started |

† Phase 7 grew from 3 to 4+1 PRs: 7.2.1 (agent-reviewable artifacts) and 7.2.2 (ShmuggingFace preview site) were added; 7.3 (publish) is deferred until Phase 8 completes.

**Total: 19 PRs** (15 original + 4 Phase-8 additions). Each PR follows the `CLAUDE.md` workflow: branch → commit → update `.agent-plan.md` → PR with type+layer labels → milestone assignment (`dataset: leadforge-lead-scoring-v1`). PR-level decomposition is in the **PR breakdown** section immediately below.

## PR breakdown

First-cut decomposition of the 7 phases into ~15 PRs. The numbering `phase.seq` is a planning ID, not a GitHub PR number. Sizes are estimates; we may merge or split during implementation. Within a phase, PRs are typically sequential (later sub-PRs depend on earlier ones); cross-phase dependencies follow the phase summary above.

### Phase 1 — Audit and naming (1 PR)

- **PR 1.1** — `docs: Phase 1 audit + dataset name decision`
  - `docs/release/v1_current_state_audit.md` (the reproduction of the relational-leakage finding)
  - `scripts/probe_relational_leakage.py` (small probe script; also seeds the Phase 3 leakage_probes module)
  - Updates `v1_acceptance_gates.md` to lock G1.1 (dataset name `leadforge-lead-scoring-v1`)
  - Labels: `type: docs`
  - Size: S (~300 lines)

### Phase 2 — Snapshot-safe relational export (2 PRs)

- **PR 2.1** — `feat(render): snapshot-safe relational export + leakage validator`
  - `leadforge/render/relational_snapshot_safe.py` (new)
  - `leadforge/validation/relational_leakage.py` (new)
  - `tests/render/test_relational_snapshot_safe.py`, `tests/validation/test_relational_leakage.py`
  - Labels: `type: feature`, `layer: render`, `layer: validation`
  - Size: M (~600 lines)

- **PR 2.2** — `feat(exposure): route student_public through snapshot-safe export; bundle schema v5`
  - Wire `relational_snapshot_safe` into `leadforge/exposure/filters.py`, `leadforge/api/bundle.py`
  - `BUNDLE_SCHEMA_VERSION` 4 → 5; manifest field `relational_snapshot_safe`
  - `leadforge/validation/bundle_checks.py` calls relational-leakage probes
  - Regenerate alpha bundles under `release/` with pinned timestamp; hash-determinism check
  - Labels: `type: feature`, `layer: exposure`, `layer: render`
  - Size: M (~500 lines + regenerated parquet bundles)
  - Depends on PR 2.1

### Phase 3 — Release validation hardening (3 PRs)

- **PR 3.1** — `feat(validation): leakage_probes module`
  - `leadforge/validation/leakage_probes.py` — direct + time-window + relational + split + model-realism probes (per Guid §8 taxonomy)
  - Tests with synthetic minimal bundles
  - Labels: `type: feature`, `layer: validation`
  - Size: M (~600 lines)

- **PR 3.2** — `feat(validation): release_quality + reporting modules`
  - `leadforge/validation/release_quality.py` — calibration, lift, P@K, value capture, model-family deltas, cross-seed bands
  - `leadforge/validation/reporting.py` — JSON+MD report rendering + matplotlib figures
  - Tests
  - Labels: `type: feature`, `layer: validation`
  - Size: L (~900 lines)

- **PR 3.3** — `feat(scripts): validate_release_candidate driver + acceptance bands resolved`
  - `scripts/validate_release_candidate.py` (the driver)
  - Update `leadforge/validation/difficulty.py` with new band checks (AP, P@K, GBM-vs-LR delta, calibration)
  - Resolve `TBD-*` bands in `v1_acceptance_gates.md` using baseline measurements
  - Generate first `release/validation/validation_report.{json,md}` + figures
  - Labels: `type: feature`, `layer: validation`, `layer: cli`
  - Size: M (~500 lines)
  - Depends on PR 3.1, PR 3.2

### Phase 4 — Channel-signal audit + dataset card hardening (1 PR)

- **PR 4.1** — `docs/feat: channel-signal audit + release-grade dataset card`
  - `scripts/audit_channel_signal.py` (analysis script)
  - `docs/release/channel_signal_audit.md` (audit results vs gemini_v2 industry benchmarks)
  - `docs/release/generation_method.md` (standalone DGP summary for external readers)
  - `docs/release/feature_dictionary.md` (narrative companion to feature dict CSV)
  - `release/README.md` rewrite (release-grade dataset card; macro-framing paragraph; simulation-simplifications section)
  - Labels: `type: docs`
  - Size: M-S (~700 lines, mostly prose)

### Phase 5 — Platform packaging (2 PRs)

- **PR 5.1** — `feat(scripts): Kaggle release packager + cover image`
  - `scripts/package_kaggle_release.py` — generates and validates `release/kaggle/dataset-metadata.json`
  - `release/dataset-cover-image.png` (≥560×280; design TBD per roadmap open question)
  - `release/kaggle/dataset-metadata.json` (generated)
  - Kaggle dry-run package validation
  - Labels: `type: feature`, `layer: cli`
  - Size: M (~500 lines)

- **PR 5.2** — `feat(scripts): HF release packager + load_dataset smoke test`
  - `scripts/package_hf_release.py` — generates `release/huggingface/README.md` with full YAML metadata
  - Local `load_dataset()` smoke test for every config
  - Companion repo packaging stub for `leadforge-lead-scoring-v1-instructor`
  - Labels: `type: feature`, `layer: cli`
  - Size: M (~500 lines)

### Phase 6 — Notebook sequence + adversarial framing (3 PRs)

- **PR 6.1** — `notebooks: 01 baseline (refresh) + 02 relational feature engineering`
  - Update `release/notebooks/01_baseline_lead_scoring.ipynb` to reproduce Phase 3 validation report metrics within tolerance
  - New `release/notebooks/02_relational_feature_engineering.ipynb` — uses snapshot-safe relational tables; demonstrates legal joins
  - Labels: `type: feature`, `layer: recipes`
  - Size: M (~400 lines committed JSON; conceptually large)

- **PR 6.2** — `notebooks: 03 leakage + 04 lift/calibration/value`
  - New `release/notebooks/03_leakage_and_time_windows.ipynb` — leakage trap demo + walkthrough
  - New `release/notebooks/04_lift_calibration_value_ranking.ipynb` — value-aware ranking + calibration + cohort-shift evaluation
  - Labels: `type: feature`, `layer: recipes`
  - Size: M (~400 lines committed JSON; conceptually large)

- **PR 6.3** — `docs/feat(github): adversarial framing — issue templates + break-me guide`
  - `.github/ISSUE_TEMPLATE/dataset_breakage_report.yml`
  - `.github/ISSUE_TEMPLATE/realism_feedback.yml`
  - `docs/release/break_me_guide.md`
  - `docs/release/v2_decision_log.md` (empty stub)
  - `release/README.md` updated to link to the break-me guide
  - Labels: `type: docs`
  - Size: S (~300 lines)

### Phase 7 — LLM critique + publish (3 PRs)

- **PR 7.1** — `feat(validation): llm_critique module + prompt + driver`
  - `leadforge/validation/llm_critique.py` — single-provider, env-var creds, skip-cleanly without
  - `docs/release/llm_critique_prompt.md` — the rubric document
  - `scripts/run_llm_critique.py` — driver script
  - First critique run committed to `release/validation/llm_critique_*.{json,md}`
  - Adjudicate any high-severity findings (resolve in code in this or a follow-up PR; or document in `v2_decision_log.md`)
  - Labels: `type: feature`, `layer: validation`
  - Size: M (~500 lines)

- **PR 7.2** — `feat(scripts): local Kaggle + HuggingFace mock-page preview` ⚠️ **must land before PR 7.3**
  - **Goal:** before any real Kaggle/HF publish, the maintainer can render a faithful local preview of how each platform will display the dataset and click through it in a browser. Catch styling, link, embed, and YAML-rendering issues *before* they land on the live page where rollback is expensive (Kaggle and HF both keep cached previews around).
  - `scripts/preview_kaggle_page.py` — reads `release/kaggle/dataset-metadata.json` + the inlined README + the cover image, renders an offline HTML mock that mimics the public Kaggle dataset page (header, description, schema/columns table, file tree, license footer). Serves on `http://localhost:8765` via `python -m http.server` or a small Flask shim.
  - `scripts/preview_hf_page.py` — reads `release/huggingface/README.md` (YAML frontmatter + body), renders an offline HTML mock that mimics the HF dataset page (frontmatter pills, configs dropdown, README body, file tree). Serves on `http://localhost:8766`.
  - Both scripts: `--release-dir`, `--port`, `--variant=public|instructor` (HF only), `--open-browser`. Dry-run / no-network.
  - Both must round-trip the *exact* artefacts the publish PR will upload — same metadata JSON, same README, same cover image — so the preview is faithful, not a sketch.
  - Tests: `tests/scripts/test_preview_kaggle_page.py` + `tests/scripts/test_preview_hf_page.py`. Each renders the page once and asserts: required field labels appear, every Markdown link in the source resolves to a non-404 URL pattern, every config block is present, the Kaggle schema table lists every CSV/parquet column.
  - Pedagogically: this is the staging gate. The release runbook (`docs/release/v1_release_notes.md` in PR 7.3) cites both preview commands as required steps before `kaggle datasets create` / `huggingface-cli upload`.
  - Labels: `type: feature`, `layer: cli`
  - Size: M (~600 lines — two HTML templates + two render scripts + two test files)

- **PR 7.3** — `feat(scripts): publish_kaggle + publish_hf + tag v1 release` ⚠️ **depends on Phase 8**
  - `scripts/publish_kaggle.py`
  - `scripts/publish_hf.py`
  - `docs/release/v1_release_notes.md`
  - Dry-run → private/draft → public publish (manual step performed by maintainer with credentials, within the PR or as a follow-up release tag). The runbook references PR 7.2's preview commands as a required pre-flight.
  - Tag `leadforge-lead-scoring-v1`
  - Labels: `type: feature`, `layer: cli`
  - Size: S (~300 lines code + manual publish step)

---

### Phase 8 — Pre-publish review-driven fixes (4 PRs)

_Source: external three-model review (Claude, ChatGPT, Gemini) of the v1 preview site, 2026-05-25. Full synthesis in `docs/external_review/summaries/v1_release_review_synthesis.md`. Blocked by Phase 7 (partial: PRs 7.1–7.2.2). Blocks PR 7.3._

- **PR 8.1** — `fix(render,validation,schema): snapshot fixes + noise clamps + schema cleanup + bundle regen`
  - **`has_open_opportunity` / `opportunity_estimated_acv` post-snapshot leak**: `render/snapshots.py` — the open/closed gate currently uses `close_outcome.isna()`, a full-horizon terminal field; correct it to `closed_at is null OR closed_at > lead_created_at + snapshot_day`. Highest-severity correctness finding across all three external models; directly violates CLAUDE.md Hard Constraints.
  - **Flat-feature snapshot-consistency probe**: `validation/leakage_probes.py` — new probe that recomputes opportunity-derived features under the corrected semantics and asserts equality with the shipped column values. Closes the probe gap that allowed the bug above to ship undetected.
  - **`to_dataframes_snapshot_safe` guard**: assert `"lead_id" in events.columns` per `SNAPSHOT_FILTERED_TABLES` entry; fail loud rather than silently producing all-NaT cutoffs.
  - **Clamp Gaussian noise**: post-distortion clamp per column type in `_apply_difficulty_distortions` (`days_since_x ≥ 0`, monetary ≥ 0). Removes the non-physical-values known limitation at essentially zero cost.
  - **Exempt `total_touches_all` from distortion**: remove from `_NUMERIC_DISTORTION_COLS`. Up to 18% NaN injected into the trap in Advanced muddies the leakage lesson it's supposed to deliver cleanly.
  - **Drop `first_touch_channel`**: remove from `LEAD_SNAPSHOT_FEATURES`, flat export, and feature dictionary. Byte-identical to `lead_source` in v1 — documented redundancy that removes itself.
  - **Rename `touches_week_1` → `touches_days_0_7`**: the implementation spans days 0–7 inclusive (8 day values); the name implies 7.
  - **Label window `<` → `<=`**: `engine.py` — "converted within 90 days" is inclusive; the break-me guide explicitly invites students to audit this boundary.
  - Regenerate all three public bundles; rerun `validate_release_candidate`; sync claims register; measure and document AUC delta from the snapshot fix.
  - Labels: `type: bugfix`, `layer: render`, `layer: validation`, `layer: schema`
  - Size: M-L (~400 lines + regenerated bundles)

- **PR 8.2** — `docs(release): difficulty-axis reframe + disclosure hardening`
  - **Reframe difficulty axis throughout all copy**: README, dataset card, Kaggle/HF metadata, tier table, notebook headers. AUC is flat (0.879/0.886/0.886) across tiers; what differs is conversion rate (42.7/21.6/8.4%), AP, P@K, Brier, missingness, and noise. Reframe as prevalence/noise tiers: "Intro = high-prevalence classroom warm-up; Intermediate = default benchmark; Advanced = low-prevalence, calibration, and noise-handling exercise — not harder nonlinear modelling."
  - **Add `calibration_max_bin_error` to README calibration table**: advanced tier is at 0.52; the current table shows only Brier (which *improves* with falling prevalence, actively misleading). One row.
  - **Clarify acceptance bands are descriptive regression fences**: `docs/release/v1_acceptance_gates.md` — state plainly that bands are fitted to the generator's output, not external realism thresholds. The YAML inline comments already say this; the README does not.
  - **Fix `isPrivate: true`**: `release/kaggle/dataset-metadata.json` — one character; absolute publish blocker.
  - **Change HF default config to `intro`**: `release/huggingface/README.md` — `intermediate` is currently `default: true`; students executing `load_dataset(...)` with no arguments should land in the easiest tier.
  - **Remove `intermediate_instructor/` from public README tree**: instructor bundle reconstructs the label by construction; listing it in the public-facing tree is a redaction bypass risk.
  - **Elevate 93% account overlap to primary evaluation warning**: move above the tier table in README; add cross-reference to `GroupKFold(account_id)` notebook section.
  - **Add "non-physical values" bullet to known limitations**: "Advanced-tier noise can produce negative duration/monetary values; treat as synthetic distortion artifacts."
  - **Reconcile CLAUDE.md package layout**: delete or annotate aspirational modules; add modules that exist but are missing.
  - Labels: `type: docs`
  - Size: M (~250 lines across multiple files)

- **PR 8.3** — `docs(notebooks): teaching improvements`
  - **Fix stale internal forward-references**: Notebooks 01 and 02 still contain "Notebook 03 *(coming in PR 6.2)*" etc. Internal PR numbers in published teaching material.
  - **Add banner to Notebook 01**: nb01 deliberately keeps `total_touches_all` to reproduce the validation panel; a banner is needed: "⚠️ This notebook reproduces the published validation panel and intentionally includes the leakage trap — do not use its feature selection block as a modelling template."
  - **Add "switch to Advanced, watch calibration break" cell to Notebook 04**: nb04 teaches calibration on Intermediate (max-bin error ~0.13, looks fine); the Advanced tier at 0.52 is never shown to students who are invited to "graduate" to it.
  - **Add `GroupKFold(account_id)` section to Notebook 02 or 04**: 93% account overlap is the README's top disclosed limitation; not showing it in any notebook leaves the stated "group-split evaluation" intended use invisible.
  - Labels: `type: docs`
  - Size: S (~200 lines across 4 notebooks)

- **PR 8.4** — `feat(scripts): integration script + preview hardening`
  - **Regenerate lockfile + bump to v1.0.1**: delete `package-lock.json`, update `package.json` pin to `github:ShmuggingFace/ShmuggingFaceCore#v1.0.1`, regenerate. Fixes SSH lockfile breakage and picks up the upstream socks/laundry copy fix in one step.
  - **Remove fabricated Kaggle usability/medals**: delete `TIER_USABILITY` and `TIER_MEDAL` constants from `build_shmuggingface_site.py`. Dead config today; latent misinformation.
  - **Make build script read and diff against canonical metadata files**: load `release/kaggle/dataset-metadata.json` and `release/huggingface/README.md`; compare `isPrivate`, tags, license, task, and per-split row counts against the generated config; exit non-zero on mismatch. This is the structural gap that made the `isPrivate: true` bug invisible.
  - **Raise on missing/malformed manifest+metrics fields**: replace `manifest.get("n_leads", 5000)` style silent defaults with explicit key lookups and clear error messages.
  - **Use per-tier `dataset_card.md` as tier page body**: currently all three tier pages render the same global README; one-line change per tier in the config builder.
  - **Pin `wrangler` as devDependency + default to preview branch**: add `wrangler` to `package.json`; change default `--branch` from `main` to a preview branch; add `--production` flag. Prevents clobbering production on every local run.
  - **Add smoke tests for `build_shmuggingface_site.py`**: minimum: all three tier configs generated with non-empty file lists and correct split row counts; the diff gate catches an injected `isPrivate: true`.
  - **Fix `_rewrite_links` bare relative links** and hardcoded org/branch constant.
  - **Single-source the two README copies** (or auto-sync Kaggle copy from canonical).
  - **Add `split` to feature dictionary** (or remove from CSV).
  - Rebuild + redeploy preview site after changes.
  - Labels: `type: feat`, `type: bugfix`, `layer: cli`
  - Size: M (~350 lines + npm changes)

## PR breakdown — totals

- **15 PRs** across 7 phases.
- Estimated total LoC: ~6,500 (excluding regenerated parquet bundles and notebook JSON).
- All 14 PRs target the `dataset: leadforge-lead-scoring-v1` GitHub milestone.
- Calendar duration is not committed; depends on iteration cadence and review feedback.

---

## Phase 1 — Audit and naming

**Goal:** Reproduce the relational-leakage finding on the alpha bundles to confirm severity. Lock the dataset release name. Zero code changes.

**Work items:**
- Run a leakage-probe script against `release/intermediate/tables/` to verify the v2 finding: train a join-only model using `opportunities.close_outcome` + customer/subscription existence; confirm it reconstructs `converted_within_90_days` with the predicted accuracy.
- Document the reproduction in `docs/release/v1_current_state_audit.md`.
- Confirm dataset release name `leadforge-lead-scoring-v1`; record decision in `v1_acceptance_gates.md`.

**Files touched:** `docs/release/v1_current_state_audit.md` (new). No code.

**Acceptance:**
- The relational-leakage finding is reproduced with a numeric AUC/accuracy.
- Dataset release name is committed.

**PR labels:** `type: docs`.
**Milestone:** create new `dataset: leadforge-lead-scoring-v1` (or similar).

---

## Phase 2 — Snapshot-safe relational export

**Goal:** Eliminate the relational-leakage blocker. Public relational tables become snapshot-safe; full-horizon stays in the instructor companion only.

**Work items:**
- New `leadforge/render/relational_snapshot_safe.py`:
  - Filter `touches`, `sessions`, `sales_activities` to `timestamp <= lead_created_at + snapshot_day` per lead.
  - Filter `opportunities` to `created_at <= lead_created_at + snapshot_day`; drop `close_outcome` and `closed_at` columns from public.
  - Drop `converted_within_90_days` and `conversion_timestamp` from public `leads.parquet`.
  - Omit `customers` and `subscriptions` from public bundles entirely (they exist only for converted leads — their presence is leakage).
- New `leadforge/validation/relational_leakage.py`:
  - Probe: train a target-reconstruction model using only public relational features; assert AUC/accuracy below tolerance.
  - Probe: assert no public table contains banned columns (configurable list).
  - Probe: assert event tables contain no rows with `timestamp > lead_created_at + snapshot_day`.
- Update `leadforge/exposure/filters.py` and `leadforge/api/bundle.py` to route `student_public` through `relational_snapshot_safe`.
- Bundle schema bump: `BUNDLE_SCHEMA_VERSION` 4 → 5. Manifest records `relational_snapshot_safe: true` for `student_public`. Document the contract change in the dataset card.
- Update `leadforge/validation/bundle_checks.py` to call `relational_leakage.run_all_probes()` and fail the bundle on any violation.
- Tests: `tests/render/test_relational_snapshot_safe.py`, `tests/validation/test_relational_leakage.py`. Hash-determinism preserved.
- Regenerate alpha bundles using the new export; verify byte-identical regeneration with pinned timestamp.

**Files touched:**
- `leadforge/render/relational_snapshot_safe.py` (new)
- `leadforge/validation/relational_leakage.py` (new)
- `leadforge/exposure/filters.py`
- `leadforge/api/bundle.py`
- `leadforge/render/relational.py` (refactor)
- `leadforge/render/manifests.py` (add `relational_snapshot_safe` flag, bump schema version)
- `leadforge/validation/bundle_checks.py`
- `tests/render/test_relational_snapshot_safe.py` (new)
- `tests/validation/test_relational_leakage.py` (new)
- `release/{intro,intermediate,advanced}/` regenerated; `release/intermediate_instructor/` retains full-horizon

**Acceptance:**
- The Phase 1 leakage probe drops from "reconstructs target with 100% accuracy" to "below configured tolerance" on regenerated bundles.
- All existing tests pass; new tests added.
- Hash-determinism preserved across two builds with pinned timestamp.
- `instructor` companion still contains full-horizon tables for legitimate teaching use.

**PR labels:** `type: feature`, `layer: render`, `layer: exposure`, `layer: validation`.
**Note:** This is the structural fix. Treat the regenerated bundles as a *fresh alpha*, not yet v1-ready.

---

## Phase 3 — Release validation hardening

**Goal:** Move beyond `leadforge validate` to a single reproducible release-grade validation artifact with charts, leakage probes, calibration, lift, and cross-seed bands.

**Work items:**
- New `leadforge/validation/release_quality.py`:
  - Computes ROC-AUC, PR-AUC, log loss, Brier score, calibration bins, lift@1/5/10%, P@50/100, recall@K, top-decile rate, expected ACV captured at K, model-family deltas (LR vs GBM vs source-only vs engagement-only vs leakage-probe vs ID-only vs stage-only vs post-snapshot-aggregates).
  - Cross-seed stability (run N seeds; compute spread bands per metric).
  - Cross-tier difficulty ordering check (AP, P@K, model-family delta — not AUC).
- New `leadforge/validation/leakage_probes.py`:
  - 8.1 Direct leakage (per recommendations Guid §8.1): all-features vs no-suspect-cols vs IDs vs post-snapshot-aggregates deltas.
  - 8.2 Time-window leakage: every public feature derives from events ≤ snapshot_day.
  - 8.3 Relational leakage: re-runs Phase 2 probes over the RC bundles.
  - 8.4 Split leakage: account/contact overlap, near-duplicate row detection.
- New `leadforge/validation/reporting.py`:
  - Renders `validation_report.json` (machine-readable) and `validation_report.md` (human-readable).
  - Renders figures: lift curves per tier, calibration reliability per tier, leakage delta bar chart, cohort-shift comparison, value-capture curves.
- New `scripts/validate_release_candidate.py` — reads RC bundles → runs all checks → writes `release/validation/validation_report.{json,md}` and `release/validation/figures/*.png`.
- Update `leadforge/validation/difficulty.py` to define tier bands per the new metrics (AP, P@K, GBM-vs-LR delta), not just conversion rates.
- Bands defined and documented in `v1_acceptance_gates.md`.
- Tests: synthetic minimal bundles to exercise each probe path.

**Files touched:**
- `leadforge/validation/release_quality.py` (new)
- `leadforge/validation/leakage_probes.py` (new)
- `leadforge/validation/reporting.py` (new)
- `leadforge/validation/difficulty.py`
- `scripts/validate_release_candidate.py` (new)
- `tests/validation/test_release_quality.py`, `test_leakage_probes.py`, `test_reporting.py` (new)
- `release/validation/` (output directory, gitignored or committed depending on file sizes)

**Acceptance:**
- `python scripts/validate_release_candidate.py release/` produces a `validation_report.{json,md}` and figures with no critical findings.
- All metrics on RC bundles fall within configured tier bands.
- Cross-seed bands established for every reported metric.

**PR labels:** `type: feature`, `layer: validation`.

---

## Phase 4 — Channel-signal audit + dataset card hardening

**Goal:** Audit how strongly `source_channel` already signals conversion in the alpha bundles (per recommendation #8 v1 scope). Bring the dataset card to release-grade.

**Work items:**
- New analysis script `scripts/audit_channel_signal.py`:
  - For each tier, compute conversion rate by `source_channel`.
  - Compute univariate AUC of `source_channel` against the target.
  - Compare to gemini_v2's industry benchmarks (SEO ~51%, PPC ~26%, Email <1% MQL→SQL).
  - Output `docs/release/channel_signal_audit.md`.
- Update `release/README.md` to a release-grade dataset card:
  - Macro framing paragraph (one paragraph on 2024-2026 SaaS context — recommendation #19).
  - Simulation simplifications section (per chatgpt v2 §2.6 — what's modeled / approximate / not modeled).
  - Calibration documentation (link to validation report).
  - Public-vs-companion redaction policy (concrete column lists).
  - Intended use vs out-of-scope use.
  - Known limitations.
  - Adversarial framing pointer (link to break-me guide once Phase 6 lands).
- New `docs/release/generation_method.md` — full DGP summary written for external readers, separate from the release README. References the architecture spec but stands alone.
- New `docs/release/feature_dictionary.md` — narrative companion to the existing CSV feature dictionary.
- Validate all dataset-card content against Datasheets-for-Datasets / Data Cards Playbook checklist (provenance, motivation, content, quality, privacy, biases/limitations, intended use, out-of-scope use, maintenance).

**Files touched:**
- `scripts/audit_channel_signal.py` (new)
- `docs/release/channel_signal_audit.md` (new)
- `docs/release/generation_method.md` (new)
- `docs/release/feature_dictionary.md` (new)
- `release/README.md` (substantial rewrite)

**Acceptance:**
- Channel-signal audit is conclusive: clear statement of how the alpha's channel signal compares to industry benchmarks.
- Dataset card passes Datasheets-for-Datasets template.
- A new reader (no leadforge context) can understand the dataset, its provenance, and its limitations from the README + linked docs alone.

**PR labels:** `type: docs`.

---

## Phase 5 — Platform packaging

**Goal:** Generate Kaggle and Hugging Face upload artifacts programmatically. Dry-run validate both.

**Work items:**
- New `scripts/package_kaggle_release.py`:
  - Reads bundle manifests and feature dictionaries.
  - Generates `release/kaggle/dataset-metadata.json` validated against current Kaggle constraints (title 6-50 chars, subtitle 20-80, slug 3-50, single license, schema fields in order, `expectedUpdateFrequency` from approved values, image ≥560×280).
  - Copies / generates `release/dataset-cover-image.png` (≥560×280, 2:1 header crop, 1:1 thumbnail crop).
  - Produces a Kaggle-shaped upload directory under `release/kaggle/`.
  - Supports `--dry-run` mode: no upload, validates structure only.
- New `scripts/package_hf_release.py`:
  - Generates `release/huggingface/README.md` with full YAML metadata: `pretty_name`, `license`, `language: en`, `task_categories: [tabular-classification]`, `size_categories`, `tags: [tabular, lead-scoring, synthetic-data, crm, b2b, datasets, pandas]`, `configs` for intro/intermediate/advanced with `default: true` on intermediate (or whichever tier is the recommended entry point).
  - Symlinks/copies bundle files into a HF-loadable structure under `release/huggingface/`.
  - Runs a local `load_dataset(local_path, "intro")`, `("intermediate")`, `("advanced")` smoke test.
- New `release/dataset-cover-image.png` — funnel-themed cover (procurement SaaS visual). Source: TBD (could be auto-generated from the validation figures or hand-designed).
- Sanity test: zip the Kaggle upload dir; verify `kaggle datasets create -p <dir> --dir-mode zip` would succeed (dry-run with credentials available, or shape-validate without).

**Files touched:**
- `scripts/package_kaggle_release.py` (new)
- `scripts/package_hf_release.py` (new)
- `release/kaggle/` (new)
- `release/huggingface/` (new)
- `release/dataset-cover-image.png` (new)
- `release/HF_DATASET_CARD.md` superseded — moved to `docs/release/hf_dataset_card_legacy.md` or deleted (decide during PR)

**Acceptance:**
- Both packagers run cleanly on a fresh build.
- Kaggle metadata passes constraint validation.
- HF `load_dataset()` smoke test passes for every config.
- Cover image meets Kaggle minimums.

**PR labels:** `type: feature`, `layer: cli`, `layer: render`.

---

## Phase 6 — Notebook sequence + adversarial framing

**Goal:** Ship the 4-notebook teaching sequence (recommendation #7) and the public adversarial framing (recommendation #16).

**Work items:**
- Update `release/notebooks/01_baseline_lead_scoring.ipynb`:
  - Reproduce Phase 3 validation report metrics within tolerance.
  - LR + GBM + value-aware ranking baseline.
  - Decile lift chart, calibration plot, P@K table.
- New `release/notebooks/02_relational_feature_engineering.ipynb`:
  - Load snapshot-safe relational tables.
  - Demonstrate legal joins and feature engineering.
  - Show that with relational features GBM lift over the flat-CSV baseline is meaningful.
- New `release/notebooks/03_leakage_and_time_windows.ipynb`:
  - Deliberately add a leakage trap (instructor-side feature) to the student data.
  - Train a model and show inflated AUC.
  - Walk through why it's invalid; reference the recommendation pass / break-me guide.
- New `release/notebooks/04_lift_calibration_value_ranking.ipynb`:
  - `expected_acv` × `P(convert)` — value-aware ranking.
  - Calibration curves and reliability diagrams.
  - Threshold selection for top-K capacity.
  - Cohort-shift evaluation as the final stress test.
- New `.github/ISSUE_TEMPLATE/dataset_breakage_report.yml` — structured form for "I broke the dataset" reports.
- New `.github/ISSUE_TEMPLATE/realism_feedback.yml` — structured form for realism critiques.
- New `docs/release/break_me_guide.md`:
  - Explicit invitations to: find direct leakage, reconstruct labels through joins, beat baseline lift legitimately, show unrealistic distributions, identify documentation ambiguity, find platform issues, propose better calibration sources.
  - Triage labels: `critical-leakage` / `realism` / `difficulty` / `documentation` / `platform` / `notebook` / `pedagogy` / `v2-idea` / `out-of-scope-v1`.
- New `docs/release/v2_decision_log.md` — starts empty; populated post-launch as feedback flows in.

**Files touched:**
- `release/notebooks/0{1,2,3,4}_*.ipynb`
- `.github/ISSUE_TEMPLATE/dataset_breakage_report.yml` (new)
- `.github/ISSUE_TEMPLATE/realism_feedback.yml` (new)
- `docs/release/break_me_guide.md` (new)
- `docs/release/v2_decision_log.md` (new)
- `release/README.md` (link to break-me guide)

**Acceptance:**
- All four notebooks run top-to-bottom from a clean environment.
- Notebook outputs reproduce validation report metrics within tolerance.
- Issue templates render correctly on the GitHub web UI.

**PR labels:** `type: feature`, `layer: recipes` (notebooks), `type: docs`.

---

## Phase 7 — LLM critique + publish

**Goal:** Run a structured one-shot LLM critique over the RC; resolve high-severity findings; tag and publish.

**Work items:**
- New `leadforge/validation/llm_critique.py`:
  - Single-provider abstraction (Anthropic Claude as default; provider chosen via env var).
  - Reads creds via env vars; skips cleanly with a clear message if absent (no failure).
  - Prompt loaded from `docs/release/llm_critique_prompt.md`.
  - Output schema (per Guid §12): release_id, model, run_timestamp, overall_score, findings[severity/category/claim/evidence/reproducer/suggested_fix], missing_sections[], questions_for_maintainer[].
  - Includes "Effective Semantic Diversity" as one rubric dimension (recommendation #12 v1 scope).
- New `docs/release/llm_critique_prompt.md` — the rubric document, structured as the prompt the script feeds.
- New `scripts/run_llm_critique.py` — driver: builds the input bundle (README.md, dataset card, generation method, manifest, feature dictionary, validation report, first 100 public rows, public/instructor diff summary, public-safe mechanism summary) → calls the critique → writes `release/validation/llm_critique_raw_*.json` and `release/validation/llm_critique_summary.md`.
- Adjudicate any high-severity findings; resolve in code or document acknowledgment in `v2_decision_log.md` if intentional-and-accepted.
- **Local mock-page preview (PR 7.2 — must land before publish):** maintainer renders Kaggle and HF dataset pages locally from the actual upload artefacts (the same metadata JSON, README, cover image the publish PR will use) and clicks through them in a browser before any platform upload, so styling / link / YAML-rendering issues are caught before they hit cached previews on the live page.
  - `scripts/preview_kaggle_page.py` — reads `release/kaggle/dataset-metadata.json` + inlined README + cover image, renders an offline HTML page that mimics the public Kaggle dataset view.
  - `scripts/preview_hf_page.py` — reads `release/huggingface/README.md` (frontmatter + body), renders the analogous HF view.
  - Both serve over `python -m http.server` (or a small Flask shim) and accept `--variant=public|instructor` (HF), `--port`, `--open-browser`.
  - Tests: required field labels appear, every Markdown link resolves to a non-404 URL pattern, every config block is present, the Kaggle schema table lists every CSV/parquet column.
- New `scripts/publish_kaggle.py` — uses `kagglehub.dataset_upload()` with `version_notes` containing the commit hash and tag.
- New `scripts/publish_hf.py` — uses `huggingface_hub.HfApi().upload_folder()` with the dataset repo type.
- Tag the release: `leadforge-lead-scoring-v1`. Tag the leadforge package release if a coordinated package version bump is needed (TBD — likely just a patch bump).
- `docs/release/v1_release_notes.md` — public-facing release notes; references the PR 7.2 preview commands as a required pre-flight step.
- Both publish scripts exercised in **dry-run** before actual upload, **and the local mock-page previews from PR 7.2 reviewed in a browser**, then upload to **private/draft** repos for download smoke test, then promote to public.

**Files touched:**
- `leadforge/validation/llm_critique.py` (new)
- `docs/release/llm_critique_prompt.md` (new)
- `docs/release/v1_release_notes.md` (new)
- `scripts/run_llm_critique.py`, `scripts/preview_kaggle_page.py`, `scripts/preview_hf_page.py`, `scripts/publish_kaggle.py`, `scripts/publish_hf.py` (new)
- `tests/scripts/test_preview_kaggle_page.py`, `tests/scripts/test_preview_hf_page.py` (new)
- `release/validation/llm_critique_raw_*.json`, `release/validation/llm_critique_summary.md` (output artifacts)

**Acceptance:**
- LLM critique runs successfully with credentials; produces structured findings.
- No unresolved high-severity findings before tag.
- Local Kaggle and HF preview pages render against the as-shipped upload artefacts and are reviewed in a browser before any platform upload.
- Both platform publishes succeed in dry-run.
- Both private/draft uploads succeed; download smoke test passes from a clean environment.
- Public Kaggle and HF pages render the dataset; `load_dataset()` from a clean env works.
- Feedback channels (issue templates, break-me guide) are linked from Kaggle, HF, and README.

**PR labels:** `type: feature`, `layer: validation`, `layer: cli`.
**Note:** the publish step is the only step that requires manual approval and credentials.

---

## Out-of-scope for this roadmap

Out-of-scope items live in `post_v1_roadmap.md`. Highlights:

- Channel-conditional MQL→SQL rates as a real generative axis (audit only in v1; full encoding deferred).
- Log-normal / Weibull sales-cycle distributions.
- Demographic noise injection (job title permutations forcing NLP).
- Quantitative semantic-diversity validator.
- Multi-provider LLM critique CI integration.
- CI workflow for release-candidate packaging.
- `leadforge release ...` CLI subcommand consolidation.
- Per-vertical industry calibration (cybersecurity, fintech).
- Second vertical, LTV labels, leaderboard mini-site.

These are valuable but not v1-load-bearing. Most are post-v1-but-pre-v2-dataset; some are v2-vertical territory.

## Open questions

These need resolution during the roadmap, not before:

1. **Difficulty bands** — concrete numeric ranges for AP, P@K, calibration, GBM-vs-LR delta per tier. To be set in `v1_acceptance_gates.md` during Phase 3.
2. **Cover image source** — generated from validation figures, hand-designed, or licensed stock. Decide during Phase 5.
3. **Should the instructor companion ship to HF as a separate config or as a separate repo?** Reviewer recommendation is "separate." This roadmap defaults to separate GitHub Release artifact + separate HF repo; revisit if HF tooling makes a single-repo split clean.
4. **Coordinated package version bump?** If new modules ship significant API surface (e.g., `leadforge.release` namespace), bump leadforge to 1.1.0 alongside the dataset tag. If purely internal, no bump.
5. **Where do regenerated bundles live during v1 work?** Options: continue using `release/` in-repo, branch-only; or switch to `leadforge-datasets` repo as the source of truth with the leadforge repo only producing the build script. Decide before Phase 5.

## Status tracker

Phase status is tracked in `.agent-plan.md` and updated on each PR merge per the branch workflow.
