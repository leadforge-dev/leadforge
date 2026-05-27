# v1 Release Notes — `leadforge-lead-scoring-v1`

**Release date:** 2026-05-27
**Package version:** leadforge 1.0.0
**Dataset version:** leadforge-lead-scoring-v1 (initial release)
**Kaggle:** https://www.kaggle.com/datasets/leadforge/leadforge-lead-scoring-v1
**Hugging Face:** https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1
**Instructor companion (HF):** https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1-instructor

---

## What is this dataset?

`leadforge-lead-scoring-v1` is a synthetic B2B CRM lead-scoring dataset generated from a
simulated mid-market SaaS procurement world. It ships as a family of three difficulty tiers
(intro / intermediate / advanced), each with 5,000 leads, split into train / valid / test
Parquet task splits and a flat `lead_scoring.csv` convenience export.

A companion `leadforge-lead-scoring-v1-instructor` dataset ships the full hidden world
(latent graph, mechanism summary, latent registry) for research and pedagogy.

See `release/README.md` (dataset card) for full documentation.

---

## Pre-publish runbook (required before `kaggle datasets create` or HF upload)

Run these steps **in order** from the repo root. Every step must exit 0.

### 1. Rebuild release bundles (if not already current)

```bash
python scripts/build_public_release.py
```

### 2. Regenerate release validation report

```bash
python scripts/validate_release_candidate.py --no-rebuild
```

### 3. Run Kaggle dry-run (package + lint)

```bash
python scripts/publish_kaggle.py --dry-run
```

Expected output: `Dry-run complete — all pre-flight checks passed.`

### 4. Run HuggingFace dry-run (package + lint + load_dataset G12.3)

```bash
python scripts/publish_hf.py --dry-run
python scripts/publish_hf.py --dry-run --variant=instructor
```

Expected output for each: `Dry-run complete — all pre-flight checks passed.`

### 5. Build and review the ShmuggingFace preview site (required)

```bash
npm install          # first time only
python scripts/build_shmuggingface_site.py --release-dir release
open release/_shmuggingface/dist/index.html
```

Review all three tiers on both the Shmaggle (Kaggle mock) and ShmuggingFace (HF mock) tabs.
Confirm: metadata accuracy, column preview, file listings, link resolution, description copy.

### 6. Preview Kaggle page

```bash
python scripts/preview_kaggle_page.py --open-browser
```

### 7. Preview Hugging Face page

```bash
python scripts/preview_hf_page.py --open-browser
python scripts/preview_hf_page.py --open-browser --variant=instructor
```

---

## Publish steps (private → review → public)

### Kaggle

```bash
# Upload as private (requires ~/.kaggle/kaggle.json or KAGGLE_USERNAME+KAGGLE_KEY)
python scripts/publish_kaggle.py

# Review at: https://www.kaggle.com/datasets/leadforge/leadforge-lead-scoring-v1
# Then flip to public via Kaggle web UI (Settings → Visibility → Public)
# or use: python scripts/publish_kaggle.py --public  (single-step public upload)
```

### Hugging Face (public dataset)

```bash
# Requires HF_TOKEN env var or: huggingface-cli login
python scripts/publish_hf.py

# Review at: https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1
# Then flip to public:
python scripts/publish_hf.py --go-public
```

### Hugging Face (instructor companion)

```bash
python scripts/publish_hf.py --variant=instructor

# Review, then:
python scripts/publish_hf.py --go-public --variant=instructor
```

---

## Tag and announce

After both platforms are live and public:

```bash
git tag -a leadforge-lead-scoring-v1 -m "leadforge-lead-scoring-v1: initial public release"
git push origin leadforge-lead-scoring-v1
```

Then update `docs/release/post_v1_roadmap.md` with the live URLs and announce.

---

## What changed since the alpha bundles (2026-05-05)

### Critical fixes

- **Relational leakage closed** (Phase 2): `student_public` relational tables were
  reconstructing `converted_within_90_days` at 100% accuracy via five join paths (A–E).
  All five paths are now closed by the snapshot-safe export filter.
- **Post-snapshot feature leak fixed** (PR 8.1): `has_open_opportunity` and
  `opportunity_estimated_acv` were using `close_outcome.isna()` (a full-horizon terminal
  field) as the open/closed gate; corrected to `closed_at is null OR closed_at > snapshot_day`.
- **Noise clamp applied** (PR 8.1): `lead_score_raw` and `lead_score_percentile` were
  carrying full-precision latent scores; now clamped to ±3σ and binned to 5 percentile bands.

### Platform hardening

- **Snapshot-safe relational export** (Phase 2): all event timestamps satisfy
  `<= lead_created_at + snapshot_day`; terminal-state fields removed from public leads /
  opportunities; conversion-conditional entities (customers, subscriptions) excluded.
- **Release validation report** (Phase 3): calibration curves, lift curves, P@K,
  cross-seed stability bands, cohort-shift probes — all gated on `v1_acceptance_gates.md`.
- **Dataset card** (Phase 4): full Datasheets-for-Datasets / Data Cards Playbook checklist;
  simulation simplifications; known limitations; intended-use / out-of-scope-use.
- **Agent-reviewable artifacts** (PR 7.2.1): `release/metrics.json` (root + per-tier),
  `release/docs/` vendored copies, `release/claims_register.{md,json}` (26 claims).
- **ShmuggingFace preview site** (PR 8.4): hardened site builder with no fabricated
  metadata, `_require()` for schema drift, per-tier dataset cards, `--config-only` flag,
  `--branch preview` default.

### Dataset

- `feature_dictionary.csv` includes the `split` column (documented in `split_metadata` category).
- Cover image generated at ≥ 560 × 280 (Kaggle minimum).
- All acceptance gates G1–G15 pass.

---

## Known limitations

See `release/README.md §Known limitations` and `docs/release/v2_decision_log.md` for
accepted-with-rationale findings from the external review (Claude, ChatGPT, Gemini).

Key items:
- GBM−LR sign flip on one feature in the intermediate tier (documented, under investigation for v2).
- Weak channel signal (`marketing_channel` AUC improvement ~0.01 over baseline).
- Flat AUC across tiers (by design — difficulty is modulated via noise and feature availability,
  not by artificially degrading the signal).
