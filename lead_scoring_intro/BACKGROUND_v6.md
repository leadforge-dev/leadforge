# BACKGROUND v6 — Lead Scoring Intro Dataset

## Business context

You are a data scientist at **ProcureFlow**, a mid-market B2B SaaS company selling AP automation and procurement workflow software. ProcureFlow targets companies with 200–2,000+ employees in manufacturing, logistics, healthcare, financial services, and professional services.

The sales team generates leads through three channels:
- **Inbound marketing**: content downloads, webinars, website forms
- **SDR outbound**: cold outreach by sales development representatives
- **Partner referral**: introductions through consulting and technology partners

### The lead scoring problem

The sales team can only actively work a limited number of leads per quarter. Your job is to build a **lead scoring model** that predicts which leads are most likely to convert to paying customers within 90 days of entering the pipeline.

A good lead score helps sales prioritize their time — contacting high-probability leads first and deprioritizing unlikely conversions.

## Dataset description

The dataset contains **1,000 leads** observed at **day 14** of their lifecycle (the "snapshot day"). All features are computed from activity that occurred during the first 14 days. The target variable (`converted`) indicates whether the lead converted to a paying customer within 90 days.

### Deal sizes

ProcureFlow's annual contract value (ACV) ranges from **$18,000** (starter plan, small companies) to **$120,000** (enterprise plan, large companies). The `expected_acv` column provides an estimate of the deal size for each lead based on company size and any existing opportunity data.

This variation in deal size means that not all conversions are equally valuable — a model that identifies high-value conversions may be more useful than one that maximizes the number of conversions.

### Acquisition waves

Leads enter the pipeline in three cohorts (`acquisition_wave`): A (earliest), B (middle), C (most recent). These roughly correspond to different time periods. The market conditions and lead mix may vary across cohorts, which is relevant for thinking about how models perform on future data.

## What to expect

- **Base conversion rate**: ~30%
- **Baseline AUC**: A simple logistic regression achieves ~0.63 AUC
- **Missingness**: 5 columns have missing values (2–7% each) due to different data collection processes across lead sources
- **Feature interactions**: The relationship between engagement and conversion is nonlinear — tree-based models capture this better than linear models

## Key columns

| Column | What it measures |
|---|---|
| `industry` | Business sector |
| `region` | Geography |
| `company_size` | Employee headcount band |
| `company_revenue` | Revenue band |
| `contact_role` | Job function of primary contact |
| `seniority` | Job level |
| `lead_source` | How the lead was acquired |
| `opportunity_created` | Whether sales opened an opportunity |
| `demo_completed` | Whether the lead viewed demo content |
| `expected_acv` | Estimated deal size (USD) |
| `inbound_touches` | Marketing touches received |
| `outbound_touches` | Sales touches initiated |
| `touches_week_1` | Touches in first 7 days |
| `touches_last_7_days` | Touches in days 8–14 (recent momentum) |
| `days_since_first_touch` | Time since first engagement |
| `web_sessions` | Website visits |
| `sales_activities` | Sales rep logged activities |
| `days_since_last_touch` | Recency of last engagement |
| `acquisition_wave` | Cohort (A, B, or C) |
| `converted` | **Target**: 1 = converted within 90 days |
