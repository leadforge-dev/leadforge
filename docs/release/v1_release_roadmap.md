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

| Phase | Title | Size | Depends on | Status |
|---|---|---|---|---|
| 1 | Audit and naming | S | — | not started |
| 2 | Snapshot-safe relational export | M | 1 | not started |
| 3 | Release validation hardening | L | 2 | not started |
| 4 | Channel-signal audit + dataset card | M-S | 3 | not started |
| 5 | Platform packaging | M | 4 | not started |
| 6 | Notebook sequence + adversarial framing | M-L | 5 | not started |
| 7 | LLM critique + publish | M | 6 | not started |

Each phase = one PR (or a small cluster of PRs against a feature branch). PRs follow `CLAUDE.md` workflow: branch → commit → update `.agent-plan.md` → PR with type+layer labels → milestone assignment.

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
**Milestone:** create new `v1.1.0 — Curated dataset v1 release` (or similar).

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
- New `scripts/publish_kaggle.py` — uses `kagglehub.dataset_upload()` with `version_notes` containing the commit hash and tag.
- New `scripts/publish_hf.py` — uses `huggingface_hub.HfApi().upload_folder()` with the dataset repo type.
- Tag the release: `leadforge-lead-scoring-v1`. Tag the leadforge package release if a coordinated package version bump is needed (TBD — likely just a patch bump).
- `docs/release/v1_release_notes.md` — public-facing release notes.
- Both publish scripts exercised in **dry-run** before actual upload, then upload to **private/draft** repos for download smoke test, then promote to public.

**Files touched:**
- `leadforge/validation/llm_critique.py` (new)
- `docs/release/llm_critique_prompt.md` (new)
- `docs/release/v1_release_notes.md` (new)
- `scripts/run_llm_critique.py`, `scripts/publish_kaggle.py`, `scripts/publish_hf.py` (new)
- `release/validation/llm_critique_raw_*.json`, `release/validation/llm_critique_summary.md` (output artifacts)

**Acceptance:**
- LLM critique runs successfully with credentials; produces structured findings.
- No unresolved high-severity findings before tag.
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
