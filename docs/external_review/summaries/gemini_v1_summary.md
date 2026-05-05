# Summary — gemini_report_v1.md

**Source:** `docs/external_review/gemini/gemini_report_v1.md` (244 lines, ~51 KB)
**Author:** Gemini (research-report style with academic-form citations)
**Verdict in one line:** Useful research synthesis on temporal leakage, funnel realism, LLM-as-judge, and platform packaging — but does not audit the actual repo and assumes work is needed where work has shipped.

## Document role

A "what should a best-in-class synthetic CRM dataset look like?" report. Mostly external-evidence-driven (industry benchmarks, academic citations) with a 5-phase roadmap. Treats leadforge primarily as ambition, not as code.

## Top points (ranked by usefulness)

1. **Temporal leakage as the single biggest threat.** Demands a strict `prediction_timestamp` boundary; flat-CSV projection of relational data must filter out events with timestamps after the boundary. Programmatic guarantee, not just documentation.
2. **Industry funnel benchmarks (table form):**
   - Visitor → Lead: 2–3% median, 4–6% top quartile
   - Lead → MQL: 23% median, 31–41% top quartile
   - MQL → SQL: 13% median, 28–40% top quartile (the modeling battleground)
   - SQL → Opportunity: 56% median, 73% top quartile
   - Lead → Customer: 1.3% (enterprise) – 2.7% (SMB), 5%+ top
3. **Decile lift / top-decile capture as the headline metric**, not raw classification accuracy. Basic demo+behavioral counting on flat tables typically yields 2-4× lift on top decile in B2B; complex relational data should obscure top signals so simple LR doesn't trivially win.
4. **LLM-as-a-judge integration.** DeepEval-style framework with GPT-4-class or Claude 3.5 Sonnet as instruction-tuned judge. Sample lead trajectories → 1–10 score on logical coherence, behavioral plausibility, narrative consistency. CI fails on threshold breach or flagrant logical impossibility.
5. **HF dataset card YAML schema (specific keys):** `language`, `license`, `task_categories: tabular-classification`, `tags: [synthetic, crm, lead-scoring, b2b]`, `pretty_name`. Kaggle: `dataset-metadata.json` with title, slug, license, file paths.
6. **CI/CD via GitHub Actions** wraps Hugging Face `huggingface_hub` (create_repo, upload_file/folder) and Kaggle CLI (`kaggle datasets create / version`) with `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` as repository secrets.
7. **Companion "masterclass" starter notebook** is non-negotiable: EDA → temporal-leakage explainer → LR baseline → LightGBM/XGBoost → decile lift chart establishing the community baseline.
8. **Adversarial public framing:** publicly invite the community to break the DGP — fastest path to v2 robustness.

## Useful artifacts / templates / schemas

- HF YAML schema table (fields × required/recommended)
- Funnel benchmark table (median vs top-quartile, by stage)
- 5-phase roadmap: DGP refinement → LLM judge → metadata + data card → CI/CD → starter notebook + adversarial challenge

## Limitations / blind spots

- Does not inspect the leadforge repo. Treats simulation engine, validation, CLI, HF card as "to build" when they exist.
- Does not catch the relational-table leakage chatgpt v2 surfaces as THE blocker.
- Cited Kaggle image dimensions ("1200×400") and outdated metadata field names; chatgpt v1 critique flagged these.
- Citations are bracketed reference IDs (`[10]`, `[11]`, …) without portable URLs anchored to the report body.

## Items unique to this source (not duplicated as strongly elsewhere)

- Funnel benchmarks expressed by quartile (median vs top-quartile vs SMB vs enterprise)
- DeepEval as a concrete framework recommendation
- "Masterclass starter notebook" framing for the launch deliverable
