# Summary — gemini_report_v2.md

**Source:** `docs/external_review/gemini/gemini_report_v2.md` (246 lines, ~46 KB)
**Author:** Gemini (revised second attempt)
**Verdict in one line:** Same shape as v1 but sharper — adds macroeconomic motivation, channel-conditional rates, sales-cycle distributions, and LLM-judge bias mitigation; still does not audit the repo.

## Document role

A second pass at the same brief. Tighter, more empirically anchored than v1, but with the same blind spot toward the existing codebase. Adds 2024-2026 SaaS macro framing as pedagogical motivation.

## Top points (ranked by what's new vs v1)

1. **Macro framing as pedagogical justification.** 2024-2026 SaaS environment: median growth rate dropped from 30% (2023) to 25% (2025); New CAC Ratio rose 14% in 2024 (~$2 spent per $1 ARR). Frames lead scoring as survival-critical, motivating realism investment.
2. **Channel-conditional MQL→SQL rates** as a strong differential predictor:
   - SEO ~51%
   - PPC ~26%
   - Email <1%
   - Cybersecurity 15-18%, Fintech 11-19%
   - DGP should produce sharply different conversion probabilities by channel; `lead_source` becomes a meaningful feature, not a uniform 13% prior.
3. **Top-quartile vs baseline contrast** to make difficulty tiers meaningful:
   - MQL→SQL baseline 13-15%, top-quartile 28-40%
   - SQL→Opp baseline 10-12%, top-quartile 45-60%
   - Opp→Won baseline 6-9%, top-quartile 20-35%
4. **Sales-cycle distributions:** median ~84 days, optimized 46-75 day window; sample with log-normal or Weibull to produce a realistic delayed-conversion long tail that confounds linear time-series forecasting.
5. **Demographic noise injection:** instead of standardizing "VP of Operations," produce variants ("Head of Ops", "Director of Global Operations", "Operations VP") to force NLP / categorical embedding cleanup before modeling.
6. **Mode collapse risk in synthetic generators:** explicitly validate effective semantic diversity of generated cohorts so every "happy-path" trajectory isn't identical. Without this, synthetic data loses pedagogical breadth.
7. **LLM-judge bias mitigation:** verbosity bias and self-preference bias are known LLM-evaluator failure modes. Mitigate by **forced-rationale** prompts — model emits step-by-step analytical decomposition before assigning a numerical score.
8. **Group/similarity leakage:** synthetic engines can produce near-duplicates from similar latent seeds; if these end up split across train/test, models memorize. Require time-based / temporally-shifted splits over random shuffle.
9. **HF `huggingface/hub-sync` GitHub Action** + Kaggle `kagglehub.dataset_upload()` Python library (preferred over CLI) with `version_notes` parameter for commit-hash lineage.
10. **Kaggle Solution Write-Up rubric (4 pillars):** Context → Overview of Approach → Details of Data → Sources. Notebook should follow this structure with mathematical exploration of DGP and explicit anti-leakage explanation.

## Useful artifacts / templates / schemas

- Channel × stage transition probability matrix
- HF YAML metadata key list (with `configs`, `default: true`)
- 5-phase roadmap: stat calibration → anti-leakage → LLM judge → CI/CD + docs → starter notebook

## Limitations / blind spots

- Same as v1 — no repo audit, no awareness of current code state, does not detect the relational-table leakage.
- Funnel benchmark numbers source-cited but not always cross-referenced; some industry sources are vendor-blog quality.

## Items unique to this source (relative to v1 and chatgpt v2)

- Channel-conditional conversion rates (SEO 51% vs Email <1% MQL→SQL)
- Log-normal / Weibull sales-cycle long-tail distributions
- Demographic noise injection forcing NLP cleanup
- Mode collapse / semantic diversity validation as an explicit dimension
- LLM-judge verbosity-bias / self-preference-bias mitigation via forced rationale
- Group/similarity leakage from latent-seed duplication
- 2024-2026 SaaS macroeconomic framing as pedagogical justification
- Kaggle Solution Write-Up rubric (Context / Overview / Details / Sources)
