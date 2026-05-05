# Post-v1 Roadmap

The work that should happen after `leadforge-lead-scoring-v1` ships, derived
from `docs/external_review/summaries/recommendations_pass.md`. This roadmap
is unscheduled — items are grouped by category and rationale, not by phase.
Most of these are accepted recommendations whose v1 scope was minimal or
deferred outright; a few are framework polish.

## Categories

- **DGP-deepening** — engine work that makes the *next* dataset version more realistic
- **Validation maturity** — moving from minimal v1 gates to full release-quality CI
- **Framework polish** — DX improvements that don't gate the dataset
- **v2 territory** — second vertical, LTV, leaderboard

## DGP-deepening (feeds the next dataset version)

### Channel-conditional MQL→SQL rates as a generative axis (recommendation #8 — full scope)
**v1 scope (already in v1 roadmap Phase 4):** audit how strongly `source_channel` signals conversion in alpha bundles; document realistic vs unrealistic mix.
**Post-v1 scope:** extend the recipe to declare per-channel transition probabilities; rework `assign_mechanisms()` to layer channel-conditional hazards on top of motif hazards; re-run difficulty-band calibration; re-baseline. Targets the gemini_v2 channel-mix benchmarks (SEO ~51%, PPC ~26%, Email <1% MQL→SQL); pedagogically validated by the Frontiers 2025 paper (`lead_source` is among the top important features in real CRM).
**Files likely touched:** `leadforge/recipes/b2b_saas_procurement_v1/recipe.yaml`, `leadforge/mechanisms/policies.py`, `leadforge/simulation/engine.py`, `leadforge/validation/difficulty.py`.
**Risk:** rebuilds part of the engine; requires re-validation of difficulty bands across all tiers.
**Reward:** significantly stronger, more realistic differential predictor; the most-cited feature in real-CRM literature.
**Trigger:** plan after v1 ships and the channel-signal audit (Phase 4) tells us how far the alpha already is from target.

### Log-normal / Weibull sales-cycle distributions (recommendation #20 — sales cycles)
**Post-v1 scope:** target a specific sales-cycle distribution (median ~84 days, top quartile 46-75 days) by tuning per-stage hazard rates or switching to an explicit sampling model. No leakage-safety payoff; pure realism.
**Files likely touched:** `leadforge/mechanisms/transitions.py`, `leadforge/mechanisms/hazards.py`, `leadforge/recipes/b2b_saas_procurement_v1/recipe.yaml`.
**Risk:** changes the funnel velocity; need to verify difficulty bands hold.
**Reward:** realistic delayed-conversion long tail; lift-curve realism for time-series teaching.

### Demographic noise injection — tier-modulated (recommendation #13)
**Post-v1 scope:** noisy job-title permutations ("Head of Ops" / "Director of Global Ops" / "Operations VP" instead of standardized "VP of Operations"), conditional address-format variation, occasional missing-field patterns. Modulated by difficulty tier (intro stays clean; intermediate/advanced get the noise) using the existing `_apply_difficulty_distortions()` extension point.
**Files likely touched:** `leadforge/render/snapshots.py`, `leadforge/recipes/b2b_saas_procurement_v1/difficulty_profiles.yaml`.
**Risk:** could distract students from the lesson if applied to intro tier.
**Reward:** forces NLP / categorical embedding cleanup; closer to real CRM messiness.
**Trigger:** plan once we have v1 user feedback on which tiers students actually use.

## Validation maturity

### Quantitative semantic-diversity validator (recommendation #12 — full scope)
**v1 scope:** "Effective Semantic Diversity" as one rubric dimension in the v1 LLM critique.
**Post-v1 scope:** dedicated quantitative validator — cohort embedding distance distribution, trajectory n-gram entropy, mode-coverage metrics. Engine-side, runs in CI on every recipe change.
**Files likely touched:** `leadforge/validation/diversity.py` (new).
**Trigger:** after running the v1 LLM critique a few times and learning what "diverse enough" looks like operationally.

### Multi-provider LLM critique CI integration (recommendation #11 — full scope)
**v1 scope:** single-provider one-shot critique pass before tag.
**Post-v1 scope:** multi-provider adjudication (≥2 model families); CI gate that fails on high-severity findings; periodic re-runs against new bundles; archive of raw outputs across runs for trend analysis.
**Files likely touched:** `leadforge/validation/llm_critique.py`, `.github/workflows/release_critique.yml` (new).
**Trigger:** after v1 ships and the prompt design / threshold tuning has stabilized.

### CI release-candidate workflow (recommendation #17)
**Post-v1 scope:** GitHub Actions workflow that runs `scripts/validate_release_candidate.py` on demand, uploads the validation report and figures as artifacts, and gates merging on no-critical-findings.
**Files likely touched:** `.github/workflows/release_candidate.yml` (new).

## Framework polish

### `leadforge release ...` CLI subcommands (recommendation #18)
**Post-v1 scope:** consolidate `scripts/{build,validate,package_kaggle,package_hf,publish_*}.py` under a single `leadforge release` namespace with subcommands. Add `--json` to all release commands. Add credential-presence checks. Add `--dry-run` to publish commands.
**Files likely touched:** `leadforge/cli/commands/release.py` (new), Click/Typer wiring in `leadforge/cli/main.py`.

### `--json` output across remaining CLI commands
`leadforge inspect --json` shipped in M12 (PR #60). `leadforge validate --json` and the new release commands should follow.

## v2 territory (later)

### Per-vertical industry calibration (recommendation #21)
File as v2-track issue. Industry-specific MQL→SQL rates from gemini_v2 (Cybersecurity 15-18%, Fintech 11-19%) should be retained as the seed numbers for whichever vertical lands first.

### Second vertical
Already in agent-plan post-v1 list. Likely candidates from the existing roadmap: cybersecurity SaaS, martech.

### LTV labels as first-class outputs
Customer/subscription entities exist in v1 internals already; the work is wiring them through to a labeled task and adding the appropriate task manifest. Out-of-scope for v1 by hard constraint in CLAUDE.md; tracked for v2.

### Leaderboard mini-site
Out-of-scope. If we ship one, it would consume v1 dataset feedback to build the v2 dataset rather than being a v1 sibling.

### Continuous-time engine
Already in agent-plan post-v1 list. Engine-level work; not coupled to dataset releases.

### Plugin architecture
Already in agent-plan post-v1 list. Framework architecture work.

### External-API enrichment
Already in agent-plan post-v1 list. Optional behind extras per existing hard constraint.

### Web UI / dashboard
Already in agent-plan post-v1 list.

## Out-of-roadmap

Items the corpus is silent on but that v1 launch will surface:

- Engineering-cost prioritization between competing post-v1 items.
- What difficulty bands the post-v1 generative changes target (depends on v1 baseline numbers).
- Cover-image content guidelines if we redesign for v1.1+.

These are decisions to make with v1 metrics in hand.
