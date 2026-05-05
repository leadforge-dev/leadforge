# Process and Recommendations Pass

For each of the 22 numbered findings in `key_findings.md`, an action code,
a one-line rationale, and a target roadmap (first-release vs after-release
vs out-of-scope). Items 1-7 (CRITICAL + HIGH) and items where there was
cross-source agreement are pre-accepted into the first-release-roadmap per
your direction; the heavier reasoning is on the Gemini-unique DGP items
where the v1 vs after-v1 split is a real call.

## Action codes

- **ACCEPT** — adopt as proposed; goes to specified roadmap
- **ACCEPT-DIFF** — adopt the *intent* with a different scope/approach
- **REJECT** — do not adopt
- **OOS-ISSUE** — out of scope for either roadmap; file a tracking issue
- **DEFER** — adopt later; goes to after-release-roadmap

## Roadmap targets

- **v1** = first-release-roadmap (what ships to Kaggle/HF as the inaugural release)
- **post-v1** = after-release-roadmap (engine/DGP improvements feeding the next dataset version, framework v1.x → v2)
- **issue** = file a GitHub issue, no roadmap commitment

---

## CRITICAL

### #1 — Public relational tables reconstruct the label
**Action:** ACCEPT → v1
**Rationale:** Pre-accepted. THE blocker. Reproduce locally on alpha bundles, then build snapshot-safe relational export + validator (item #6 is the same workstream; folded together in the roadmap).

---

## HIGH

### #2 — Difficulty signal too flat on AUC across alpha tiers
**Action:** ACCEPT → v1
**Rationale:** Cross-source agreement (chatgpt v2 directly; gemini implicit through difficulty-tier framework). Folds into validation hardening — the difficulty gate must include AP, P@K, calibration, lift, and model-family deltas, not just AUC. No new DGP work needed; the alpha already produces meaningful AP/P@K differentials, we just need the report to surface them.

### #3 — No Kaggle `dataset-metadata.json` generator
**Action:** ACCEPT → v1
**Rationale:** Cross-source agreement. Required to upload. Use ChatGPT-critique's verified field list (`expectedUpdateFrequency`, 6-50 char title, 20-80 char subtitle, 3-50 char slug, image ≥560×280).

### #4 — HF README needs hardening to be a real dataset card
**Action:** ACCEPT → v1
**Rationale:** Cross-source agreement. `pretty_name`, `tags: [tabular, lead-scoring, synthetic-data, crm, b2b, datasets, pandas]`, `configs` with `default: true`, local `load_dataset()` smoke test.

### #5 — Release validation must move beyond `leadforge validate`
**Action:** ACCEPT → v1
**Rationale:** Cross-source agreement. New modules: `leadforge/validation/{release_quality,leakage_probes,reporting}.py` + `scripts/validate_release_candidate.py`. Output: `release/validation/validation_report.{json,md}` + figures. Acceptance: zero critical leakage findings, metrics in bands, charts auto-generated. This is the single biggest piece of v1 work and absorbs most of the "release-grade gates" demanded by all reviewers.

### #6 — Snapshot-safe relational export design
**Action:** ACCEPT → v1
**Rationale:** Direct fix for #1. New module `leadforge/render/relational_snapshot_safe.py` + new validator `leadforge/validation/relational_leakage.py`. Filter event tables to `timestamp <= lead_created_at + snapshot_day`; drop terminal-state fields from public `opportunities`; omit `customers`/`subscriptions` from public bundles; full-horizon goes only to instructor companion.

### #7 — Notebook sequence (4 notebooks; only 1 exists today)
**Action:** ACCEPT → v1
**Rationale:** Cross-source agreement. v7 lecture sequence already exists in `lead_scoring_intro/RELEASE_v7.md` — operationalize as `02_relational_feature_engineering`, `03_leakage_and_time_windows`, `04_lift_calibration_value_ranking`. All run top-to-bottom; outputs match validation report.

---

## MEDIUM — most depth on Gemini-unique DGP items here

### #8 — Channel-conditional MQL→SQL rates as a strong differential predictor
**Action:** ACCEPT-DIFF → v1 (audit only) + post-v1 (full encoding)
**Rationale:** Gemini's strongest single DGP recommendation. Industry data (G2: SEO ~51%, PPC ~26%, Email <1%) shows lead source should be a top-tier conditional probability, and the Frontiers 2025 paper confirms `lead_source` is among the top important features in real CRM data. Genuinely valuable.
**But:** the leadforge engine drives conversion through motif-family-specific hazards keyed off latent traits, not through explicit channel-conditional probabilities. Properly encoding channel-conditional rates means (a) extending the recipe to declare per-channel transition probabilities, (b) reworking `assign_mechanisms()` to layer channel hazards on top of motif hazards, (c) re-running difficulty-band calibration across all three tiers, (d) re-baselining. That's an engine project, not release hardening, and risks rebuilding the DGP at exactly the wrong time.
**v1 scope:** audit how strongly `source_channel` already signals conversion in the alpha bundles; document realistic vs unrealistic mix in the dataset card; flag `lead_source` as a high-leverage feature that students should explore.
**post-v1 scope:** real channel-conditional encoding as a first-class generative axis. Worth a v1.1 dataset (same recipe, regenerated bundles) once the release is out and we can iterate on calibration without release-pressure.

### #9 — Train/test split policy: cohort/time + account-overlap audit
**Action:** ACCEPT → v1 (folded into #5 + #7)
**Rationale:** Cross-source. Account/contact overlap probe is small work and belongs in `leakage_probes.py`. Cohort-time-shift split should ship as one of the evaluation axes in notebook #4 + the validation report. v7 already has cohort-split AUC drop measurement (`RELEASE_v7.md`); port the pattern.

### #10 — v7 teaching lessons ported into v1
**Action:** ACCEPT → v1 (folded into #7 + dataset card)
**Rationale:** Cross-source. Mostly documentation work — the lecture sequence and pedagogical patterns are already proven; we just operationalize them in the multi-table v1 release.

### #11 — LLM-as-a-judge integration as release-quality gate
**Action:** ACCEPT-DIFF → v1 (minimal one-shot) + post-v1 (full CI integration)
**Rationale:** Cross-source agreement on the principle. But the gap between "minimal viable" and "full release-quality gate with multi-provider adjudication" is large.
**v1 scope:** `leadforge/validation/llm_critique.py` with a single provider abstraction (env-var creds, skip cleanly without). One-shot critique pass over dataset card + sample rows + validation report, structured output per Guid §12 schema (severity / category / claim / evidence / reproducer / suggested_fix). Run manually before tagging the release; high-severity findings adjudicated by hand. Output archived to `release/validation/llm_critique_*.json`.
**post-v1 scope:** multi-provider adjudication, CI gate, automated fail-on-high-severity, periodic re-runs against new bundles.
**Why split:** getting LLM critique right (prompt engineering, rubric design, threshold tuning, false-positive handling) takes meaningful iteration that we shouldn't gate v1 on. A minimal pass is enough to catch obvious gaps for the public release.

### #12 — Mode-collapse / semantic-diversity validation
**Action:** ACCEPT-DIFF → v1 (LLM-judge rubric dimension) + post-v1 (quantitative validator)
**Rationale:** Gemini-unique. Real concern — heavily-aligned synthetic generators do produce homogenized "happy path" trajectories that lose pedagogical breadth.
**v1 scope:** include "Effective Semantic Diversity" as one of the rubric dimensions in the v1 LLM critique (item #11). Cohort sample → "does this set cover the full firmographic / behavioral space?" → severity-tagged finding.
**post-v1 scope:** dedicated quantitative validator — cohort embedding distance distribution, trajectory n-gram entropy, or similar. Engine-side work that depends on knowing what "diverse enough" looks like, which itself depends on running the v1 LLM critique a few times.

### #13 — Demographic noise injection (job title permutations forcing NLP)
**Action:** DEFER → post-v1 (with tier-modulated approach)
**Rationale:** Gemini-unique. Real CRM messiness, but adding it to v1 risks distracting students from the core lead-scoring lessons — many will spend energy fighting NLP issues that aren't the lesson. Better: stage it as a difficulty-tier knob (intro stays clean; intermediate/advanced get the noise) once we have v1 feedback on which tiers students actually use. Note: difficulty profiles already modulate noise/missingness/outliers via `_apply_difficulty_distortions()` — this is an extension of that mechanism, not a new system.

### #14 — Cover image asset
**Action:** ACCEPT → v1
**Rationale:** Required for Kaggle. ≥560×280, with 2:1 header and 1:1 thumbnail crops. Sourcing/design TBD — recommend a stylized funnel diagram conveying "synthetic" + "B2B SaaS procurement". Cheap.

### #15 — Versioning / naming clarification
**Action:** ACCEPT → v1 (do early)
**Rationale:** ChatGPT-unique but obviously right. `leadforge` package stays at 1.x; the curated dataset release is named `leadforge-lead-scoring-v1` (or similar). Decoupling avoids the "package says Production/Stable but the data is alpha" confusion.

### #16 — Issue templates + break-me guide + v2 decision log
**Action:** ACCEPT → v1
**Rationale:** Cross-source. `.github/ISSUE_TEMPLATE/{dataset_breakage_report,realism_feedback}.yml` + `docs/release/break_me_guide.md` + `docs/release/v2_decision_log.md` (starts empty, populated post-launch). Required for the adversarial public framing both reviewers demand.

---

## LOW / Defer / Out-of-scope

### #17 — CI workflow for release-candidate packaging
**Action:** DEFER → post-v1
**Rationale:** Manual run of `validate_release_candidate.py` covers the v1 use case. Add CI workflow once the release process is stable.

### #18 — `leadforge release ...` CLI subcommands
**Action:** DEFER → post-v1
**Rationale:** Scripts in `scripts/` cover v1 needs. Subcommand consolidation is polish, not load-bearing.

### #19 — Macro framing in dataset card (CAC ratios, growth decline)
**Action:** ACCEPT-DIFF → v1
**Rationale:** Gemini-unique. Cheap to add a short "Why lead scoring matters in 2026 SaaS" paragraph to the dataset card; high pedagogical value (motivates the dataset for students). Don't build a whole "industry context" section. One paragraph + a citation or two.

### #20 — Channel-conditional / log-normal sales cycles / demographic noise (catch-all)
**Action:** Split per components:
- Channel-conditional → see #8 (audit in v1, full encoding post-v1)
- Log-normal / Weibull sales-cycle distributions → DEFER → post-v1
- Demographic noise → see #13 (DEFER → post-v1)
**Rationale on log-normal sales cycles:** Gemini-unique. The engine's daily-step simulation produces whatever cycle distribution falls out of the hazard rates; explicitly targeting log-normal (median ~84d, top quartile 46-75d) requires either tuning hazards per stage to hit the target distribution or switching to a different sampling model. Real work, no leakage-safety payoff. Defer until post-v1 DGP overhaul.

### #21 — Per-vertical industry calibration (cybersecurity, fintech)
**Action:** OOS-ISSUE → file as v2 second-vertical work
**Rationale:** v1 vertical is locked: B2B SaaS procurement. Per-vertical calibration is exactly v2 territory. File the issue with G2's industry-specific rates (cyber 15-18% MQL→SQL, fintech 11-19%) so it's not lost.

### #22 — LTV labels / leaderboard / second vertical
**Action:** OOS-ISSUE
**Rationale:** Explicitly out per `.agent-plan.md` and Guid §3.6 / C2 §8. Already tracked in deferred items.

---

## Summary

| Roadmap | Items | Items pulled in |
|---|---:|---|
| **v1 (first-release-roadmap)** | 14 | #1, #2, #3, #4, #5, #6, #7, #8 (audit), #9, #10, #11 (minimal), #12 (LLM rubric), #14, #15, #16, #19 |
| **post-v1 (after-release-roadmap)** | 8 | #8 (full encoding), #11 (full CI), #12 (quantitative), #13, #17, #18, #20 (log-normal) |
| **out-of-scope-issue** | 2 | #21, #22 |

(Some items appear in both roadmaps because they were split into a v1 minimal scope and a post-v1 full scope.)

## Strategic frame for Gemini's deeper DGP work

Gemini's strongest unique contributions are channel-conditional rates (#8), log-normal sales cycles (#20-log-normal), demographic noise (#13), and mode-collapse validation (#12). Three observations on the v1-vs-post-v1 split:

1. **None of Gemini's DGP items are leakage-safety-load-bearing.** They make the dataset more pedagogically rich and more realistic, but none of them block v1 from being safe to publish. The blocker is structural (relational tables leak the label); Gemini didn't catch that. So the v1-vs-post-v1 split here is "ship a leak-safe v1 first, deepen the DGP for the next dataset version" — not "delay v1 for DGP work."

2. **Channel-conditional rates have the highest pedagogical ROI** of the four. The Frontiers 2025 paper directly identifies `lead_source` as a top important feature in real CRM data. Students using v1 *should* find that `source_channel` is one of the strongest predictors. The v1 audit step (#8) tells us how close we already are. If the audit shows the alpha bundles already have a strong channel signal (because motif families cover similar territory implicitly), the post-v1 work is calibration tuning rather than a rebuild.

3. **The LLM-judge work is the single highest-leverage post-v1 investment.** A working multi-provider critique gate would catch every Gemini concern (mode collapse, narrative incoherence, demographic flatness) and every realism issue we haven't anticipated. Worth investing in once v1 is out and we can iterate on prompt design without release pressure.

## Suggested ordering for the v1 roadmap (preview, not the roadmap itself)

A dependency-respecting sequence for the v1 work, just to surface ordering questions before drafting:

1. Reproduce the relational-leakage finding locally (sanity check, no roadmap)
2. Versioning / naming decision (#15 — cheap, do first)
3. Snapshot-safe relational export + validator (#1 + #6) — fixes the blocker
4. Release validation hardening (#2 + #5 + #9 probes) — depends on #6 validators existing
5. Channel-signal audit + macro framing paragraph (#8 audit + #19) — light docs work
6. Platform packaging: Kaggle metadata + HF README + cover image (#3 + #4 + #14)
7. Notebook sequence (#7 + #10 + #9 cohort split) — depends on validation report figures to reproduce
8. Issue templates + break-me guide (#16) — readiness for public adversarial framing
9. Minimal LLM critique pass (#11 + #12 rubric) — final quality gate before tagging the release

Sign-off question: do you want me to push back on any of these recommendations before drafting the actual roadmap?
