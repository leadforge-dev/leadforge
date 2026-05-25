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

_Items marked `[Phase-8 deferred]` were surfaced in the 2026-05-25 release preview review (synthesis in `docs/external_review/summaries/v1_release_review_synthesis.md`) and explicitly judged too large for the pre-publish Phase 8 fixes._

## DGP-deepening (feeds the next dataset version)

### [Phase-8 deferred] Difficulty axis as genuine modelling complexity — AUC separation, not just prevalence
**Why deferred:** Fixing this requires re-tuning `signal_strength` in `difficulty_profiles.yaml` and the `assign_mechanisms()` policy so the weaker signal knobs actually suppress rank-discrimination (i.e., lower LR AUC), rather than only recalibrating the per-tier conversion-rate band. Currently LR AUC is flat (0.879/0.886/0.886) across all tiers — the tiers are a prevalence/AP/Brier axis, not a modelling-difficulty axis.
**v2 scope:** Tune `signal_strength` so AUC meaningfully separates across tiers (target: intro ~0.88, intermediate ~0.82, advanced ~0.75 or lower). Introduce at least one genuine non-linear interaction in the advanced tier so that GBM outperforms LR there (currently GBM−LR is negative in all three tiers). Suggested mechanisms: stronger cross-feature interactions between `employee_band` × `motif_family` latent score, a non-linear hazard that decays exponentially rather than linearly with latent fit score, and a time-varying engagement signal.
**Files likely touched:** `leadforge/recipes/b2b_saas_procurement_v1/difficulty_profiles.yaml`, `leadforge/mechanisms/policies.py`, `leadforge/mechanisms/scores.py`, `leadforge/validation/difficulty.py` (re-calibrate acceptance bands).
**Risk:** requires full bundle regeneration + re-validation; cross-seed bands must be re-fitted. The v1 acceptance band for `gbm_minus_lr_auc` was explicitly widened to accommodate negative values — that gate needs to be tightened once the DGP produces positive deltas.
**Reward:** "Advanced" tier becomes a genuinely harder modelling problem; the "comparing model families" intended use delivers a live lesson instead of a flat one; students graduating from Intermediate to Advanced face real AUC degradation, not just a class-imbalance adjustment.
**Trigger:** plan after v1 ships; design the non-linear interaction before touching any code.

### [Phase-8 deferred] Account-level population stratification — `GroupKFold` split as first-class task variant
**Why deferred:** The full fix requires partitioning account generation *before* lead sampling so that the three task splits are account-disjoint by construction, not just lead-disjoint. This changes `simulation/population.py`, the task-split writer in `render/tasks.py`, `schema/tasks.py` (`TaskManifest` gains a `split_key` field), the manifest schema, and the validation report (new `group_split_metrics` block). Requires full bundle regeneration.
**v2 scope:** Add a `split_key: account_id` option to `SplitSpec`; when set, `write_task_splits()` stratifies by account rather than by lead. Ship the account-stratified split as a second task variant alongside the existing random split: `tasks/converted_within_90_days_group/{train,valid,test}.parquet` + `task_manifest.json`. Expose group-split metrics in `validate_release_candidate` output.
**Files likely touched:** `leadforge/simulation/population.py`, `leadforge/render/tasks.py`, `leadforge/schema/tasks.py`, `leadforge/validation/release_quality.py`, `docs/release/v1_acceptance_gates.md`.
**Note:** PR 8.3's `GroupKFold` notebook section (which ships with v1) demonstrates the concept by having students split post-hoc; this item makes it a first-class default.
**Reward:** removes the "93% account overlap" known limitation from the dataset card; published headline metrics become honest group-split metrics; the "teaching B2B generalisation" intended use is fully supported.

### [Phase-8 deferred] Missing B2B lead-scoring signals — v2 feature set
**Why deferred:** Each signal requires new schema entities, simulation mechanisms, and generator logic — not a documentation or tuning change.
**v2 scope (prioritised list from the 2026-05-25 review):**
1. **Channel-conditional hazards** (partially overlaps with "Channel-conditional MQL→SQL rates" below): `lead_source` should drive different baseline conversion probabilities, not just label a lead's origin. Currently the channel audit shows per-channel AUC ~0.50–0.52 across all tiers.
2. **SDR/rep capacity and territory**: `sales_rep_id`, `territory`, `rep_open_pipeline_count` — adds a supply-side signal that is canonical in real CRM (rep at capacity → longer follow-up latency → lower conversion).
3. **SLA / follow-up latency**: `hours_to_first_contact`, `days_to_first_meeting` — the most consistently predictive signals in real CRM studies; trivial to derive from the existing `sales_activities` simulation.
4. **Email engagement**: `email_opens`, `email_clicks`, `email_replies` — standard outbound signals; add as `touches` subtypes in the simulation.
5. **BANT proxies**: `budget_confirmed`, `authority_level`, `need_score`, `timeline_days` — even weak/noisy proxies of these signal genuine B2B qualification.
6. **Technographics / competitor install**: `current_erp`, `competitor_install` — static firmographic enrichment; correlates with motif family in the hidden graph.
7. **Negative behaviours**: `unsubscribed`, `career_page_visit` — churn signals that currently have no representation.
**Files likely touched:** `leadforge/schema/entities.py`, `leadforge/schema/features.py`, `leadforge/simulation/engine.py`, `leadforge/mechanisms/`, `leadforge/recipes/b2b_saas_procurement_v1/`.

### [Phase-8 deferred] Hidden DAG as executable causal engine — per-node structural equations
**Why deferred:** This is a major simulation rearchitecture. Currently the world graph is validated and exported as a DAG, but the simulation primarily uses `world_graph.motif_family` to select mechanism parameters; the rewired graph topology and edge weights do not drive per-node structural equations in the daily loop. Making the graph causally executable would require designing a structural-equation interpreter that maps each graph edge to a conditional probability transformation.
**v2 scope:** Design a `StructuralEquationInterpreter` that maps `(parent_node, child_node, edge_weight)` → conditional distribution transform applied during the daily simulation step. The result: two seeds with the same motif family but different rewired topologies would produce genuinely different lead populations, rather than slightly differently-weighted versions of the same mechanisms. This would be the technical foundation for the "five motif families + stochastic rewiring" story to be causally true rather than narratively true.
**Files likely touched:** `leadforge/simulation/engine.py`, `leadforge/mechanisms/policies.py`, `leadforge/structure/graph.py`, new `leadforge/structure/interpreter.py`.
**Risk:** high. Changes the DGP fundamentally; requires re-validation of all tiers and rewriting the generation method documentation.
**Reward:** the central architectural claim becomes accurate; the hidden DAG becomes a genuinely auditable causal structure rather than a documented-but-inert artifact.

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
