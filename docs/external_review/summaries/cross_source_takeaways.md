# Cross-Source Takeaways

A consolidation of the six external review files. Each theme below tracks
where the agreement is strong, where reviewers diverge, and where one
source surfaces something the others miss.

Source codes used:
- `G1` = gemini_v1
- `G2` = gemini_v2
- `C1` = chatgpt_v1
- `Crit` = chatgpt v1 critique
- `Guid` = chatgpt second-attempt guidance
- `C2` = chatgpt_v2 (the substantive one)

---

## 1. Strongly agreed themes (act on these)

### 1.1 Temporal leakage prevention is the foundational concern
All sources lead with this. (G1, G2, C1, C2)
- Strict `prediction_timestamp` / snapshot boundary
- Aggregations strictly bounded to events ≤ snapshot
- Label resolution uses full label horizon but not as a feature source

### 1.2 LLM-as-a-judge integration belongs in CI
(G1, G2, C1, C2, Guid)
- Reference-less rubric scoring of synthetic trajectories
- Logical coherence + behavioral plausibility + semantic diversity + syntax validity
- Strict numeric thresholds halt the build on failure
- C2/Guid contribute the concrete output JSON schema (severity / category / claim / evidence / reproducer / suggested_fix)
- G2 contributes bias mitigation (forced-rationale prompting)
- G1 contributes DeepEval as a candidate framework
- C2 recommends ≥2 model families with adjudication of high-severity findings before release

### 1.3 Lift / calibration / P@K / value-aware ranking, not raw AUC
(G1, G2, C2)
- Decile lift charts as a headline metric (G1, G2)
- Calibration curves + Brier + log loss (C2)
- Top-K precision and recall (C2)
- Expected-value-at-K — `P(convert) × expected_acv` (C2 from v7 lineage)

### 1.4 Industry-calibrated funnel benchmarks
(G1, G2)
- Channel-conditional MQL→SQL rates (G2: SEO 51%, PPC 26%, Email <1%) — strongest differential predictor design
- Top-quartile vs baseline contrast across all funnel stages
- Sales-cycle distributions sampled from log-normal/Weibull (G2)

### 1.5 Release as a family, not a single CSV
(C2 explicit; G1/G2 implicit)
- intro / intermediate / advanced public tiers + instructor companion
- Public bundle = flat task splits + snapshot-safe relational tables + feature dict + validation report + notebooks + data card + break-me guide
- Instructor companion = full hidden graph, latent registry, mechanisms, full-horizon relational tables, leakage-trap materials

### 1.6 Platform packaging must be programmatic
(G1, G2, C2, Guid)
- Kaggle: `dataset-metadata.json` generator, dry-run command, cover image
- HF: README.md with YAML configs/default/pretty_name/tabular tag, `load_dataset()` smoke test
- CI/CD: GitHub Actions with `HF_TOKEN`/`KAGGLE_USERNAME`/`KAGGLE_KEY` secrets, dry-run publishing
- Use `huggingface_hub` library and `kagglehub` library (not raw CLI) for Python integration

### 1.7 Companion notebook(s) are non-negotiable
- G1: "masterclass starter notebook" — single deep notebook
- G2: Kaggle Solution Write-Up rubric (Context / Overview / Details / Sources)
- C2: 4-notebook sequence — baseline → relational FE → leakage demo → lift/calibration/value
- C2's sequence wins on pedagogical depth and aligns with v7 lecture sequencing

### 1.8 Adversarial public framing + feedback loop
- G1: explicit "challenge community to break it"
- C2: explicit issue templates + break-me guide + triage labels + v2 decision log
- Public invitation to find leakage / break baselines / report unrealistic distributions
- Triage taxonomy: critical-leakage / realism / difficulty / documentation / platform / notebook / pedagogy / v2-idea / out-of-scope-v1

### 1.9 Dataset card adheres to Datasheets / Data Cards Playbook
(G1, G2, C1, C2)
- Provenance, motivation, content, quality, privacy, biases/limitations, intended use, out-of-scope use

---

## 2. Divergent themes (resolve before roadmap)

### 2.1 What's the biggest v1 risk?
- **C2:** Public relational tables leak the target with 100% accuracy via join paths through `opportunities.close_outcome` + `customers`/`subscriptions` existence. THE blocker.
- **G1/G2:** Don't surface this; their leakage worry is the temporal one, which the engine has partially addressed.
- **Resolution:** C2 is right; G1/G2 missed it because they didn't open the bundles. Verify locally as the very first thing.

### 2.2 How much DGP work is needed before release?
- **G1/G2:** Significant. "Inject non-linear complexity," "deeper funnel calibration," "channel-conditional probabilities," "demographic noise injection."
- **C2:** "Leadforge is much further along than greenfield. v1 is release hardening + adversarial validation, not core implementation."
- **Resolution:** C2 has the evidence (937 tests, vertical-slice generator, alpha bundles). G1/G2's DGP recommendations are still useful inputs but should be prioritized against current state, not assumed greenfield.

### 2.3 Is a single masterclass notebook enough, or do we need a sequence?
- **G1:** One masterclass starter notebook with baseline + decile lift chart.
- **C2:** Four notebooks (baseline, relational FE, leakage demo, lift/value).
- **Resolution:** C2's sequence is stronger pedagogically and matches v7-track lessons (4 lectures already designed in `RELEASE_v7.md`). Use the sequence.

### 2.4 Should the instructor companion ship to Kaggle?
- **C2:** No — separate GitHub Release artifact or HF repo/config. Don't put hidden truth on the public Kaggle page.
- **G1/G2:** Don't address the instructor-companion question explicitly.
- **Resolution:** C2's instinct is sound — keep it separate to preserve the leakage trap's pedagogical value.

### 2.5 What should the LLM judge actually score?
- **G1:** Logical coherence + behavioral plausibility + narrative consistency, scored 1-10.
- **G2:** Adds effective semantic diversity (mode collapse check) and syntax validity.
- **C2:** Adds severity/category/evidence/reproducer/suggested-fix structure for findings; treats it as a release-quality gate not a per-row scorer.
- **Resolution:** Combine — per-trajectory rubric scoring (G1+G2) AND per-release findings document (C2). The latter is more important for v1.

---

## 3. Items only one source surfaces (worth absorbing)

### Only in G1
- DeepEval as a concrete LLM-judge framework name
- "Hidden Gems" Kaggle notebook quality reference

### Only in G2
- Channel-conditional MQL→SQL rates as differential predictor design
- Log-normal / Weibull sales-cycle long-tail distributions
- Demographic noise injection (job title permutations forcing NLP)
- Mode collapse / semantic diversity validation as an explicit dimension
- LLM-judge bias mitigation via forced rationale (analytical decomposition before scoring)
- Group/similarity leakage from latent-seed duplication
- 2024-2026 SaaS macroeconomic framing (CAC ratio +14%, growth decline) as pedagogical motivation
- Kaggle Solution Write-Up rubric (Context / Overview / Details / Sources)

### Only in C1 (mostly historical)
- Simula framing (datasets-as-functions; programmable diversity / complexity / quality axes)
- Reference to BlueGen AI's Data Plagiarism Index / authenticity scores

### Only in Crit
- Acceptance rubric for evaluating any future report (8 dimensions)
- Corrected platform facts (Kaggle 560×280 minimum, `expectedUpdateFrequency` field name)
- Framework-vs-dataset lane separation argument

### Only in Guid
- Mandatory dataset-forensics probe taxonomy (5 categories) — the spec that surfaced the relational-leakage blocker
- Required release-tree layout (`leadforge-v1-lead-scoring/` structure)
- 12-criterion acceptance rubric for the report itself
- Pitfalls list (don't ignore v7 track, don't treat HF card as absent, etc.)

### Only in C2
- The relational-leakage blocker finding (verified via local smoke bundle)
- File:line-cited current-state audit
- Gap matrix with severities
- Snapshot-safe relational export design
- Public-Kaggle / instructor-companion split recommendation
- Concrete v1-ready definition
- v2 feedback triage labels
- Frontiers 2025 paper as a real B2B realism citation (23154 records, 15 classifiers)
- "Pin the timestamp; verify byte-equal regeneration" as a release-readiness check
- Alpha LR/HistGBM gap finding (model-family deltas should be larger to reward sophistication)

---

## 4. Where the corpus is silent (worth flagging in the roadmap)

- **No reviewer addresses the engineering cost** of any recommendation against the current state of the codebase.
- **No reviewer offers prioritization between LLM critique investment and snapshot-safe relational** — both are recommended as critical, but with different cost/risk profiles.
- **No reviewer specifies the cover-image content or sourcing.**
- **No reviewer addresses how v1 lessons should feed back into the framework (vs into a v2 dataset)** — i.e., should a release-blocking issue cause a major framework version bump? C2 separates package version from dataset release name, which is the closest answer.
- **No reviewer specifies what difficulty bands the release validation should enforce** (only that bands should exist).
- **No reviewer engages with the v0.1.0-alpha datasets repo's reviewer-targeted artifacts** (`build.sh`, `provenance.json`, `BASELINES.md`, `EXPOSURE_DELTA.md`, `validation.log`) beyond surface mention. These already model some of what's recommended.
