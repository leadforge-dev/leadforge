"""Customer snapshot feature catalog for the lifecycle (pLTV) scheme.

:data:`CUSTOMER_SNAPSHOT_FEATURES` is the canonical column spec for the
customer snapshot table built by
:func:`~leadforge.schemes.lifecycle.snapshots.build_customer_snapshot` —
the lifecycle analogue of the lead-scoring catalog
(:data:`~leadforge.schemes.lead_scoring.features.LEAD_SNAPSHOT_FEATURES`).
The same catalog serves both observation regimes (calendar-anchored and
tenure-anchored early-pLTV); only the cutoff differs (design.md §4).

Targets (design.md §8, D6/D7/D9):

- ``ltv_revenue_{90,365,730}d`` — gross revenue (paid + recovered invoices)
  in the forward window after the cutoff.  Continuous, zero-inflated,
  right-skewed (ZILN-shaped) regression targets.
- ``churned_within_180d`` — secondary churn label / ZILN zero-inflation
  indicator.

Leakage trap (design.md §7): ``mrr_change_full_period`` is computed through
the **end of simulation** — post-cutoff expansions, which directly drive the
pLTV targets, inflate it.  ``leakage_risk=True`` but never redacted: it is a
deliberate pedagogical trap, retained in all modes and exempt from difficulty
distortions.

Semantic nulls vs. injected missingness: ``last_nps_score`` and
``weeks_since_last_payment_failure`` are null *by meaning* (no survey response
yet / never failed).  Difficulty-tier MCAR missingness stacks on top of these,
so at distorted tiers a null no longer distinguishes "never happened" from
"not recorded" — deliberate (real CRM exports have both kinds of missingness)
and to be documented in the feature dictionary at publication (LTV-M6).

Deliberately absent (vs. design.md §8's draft list):

- ``current_plan`` — the engine has no plan-change mechanism; the column
  would duplicate ``initial_plan`` exactly.
- ``downgrade_count`` — no downgrade mechanism exists (see the engine module
  docstring); the column would be zero-variance, violating the published
  no-zero-variance-features invariant.  Revisit if a downgrade mechanism
  lands.
"""

from __future__ import annotations

from leadforge.schema.features import FeatureSpec

__all__ = ["CUSTOMER_SNAPSHOT_FEATURES"]

CUSTOMER_SNAPSHOT_FEATURES: list[FeatureSpec] = [
    # -- identifiers --------------------------------------------------------
    FeatureSpec(
        name="customer_id",
        dtype="string",
        description="Stable customer identifier (join key; not a feature).",
        category="customer_meta",
    ),
    FeatureSpec(
        name="account_id",
        dtype="string",
        description="Owning account identifier (join key; not a feature).",
        category="customer_meta",
    ),
    # -- account firmographics ----------------------------------------------
    FeatureSpec(
        name="industry",
        dtype="string",
        description="Account industry vertical.",
        category="account",
    ),
    FeatureSpec(
        name="region",
        dtype="string",
        description="Account geographic region.",
        category="account",
    ),
    FeatureSpec(
        name="employee_band",
        dtype="string",
        description="Account employee-count band.",
        category="account",
    ),
    FeatureSpec(
        name="estimated_revenue_band",
        dtype="string",
        description="Account estimated-annual-revenue band.",
        category="account",
    ),
    # -- customer / subscription state at the cutoff -------------------------
    FeatureSpec(
        name="tenure_weeks",
        dtype="Int64",
        description="Whole weeks from customer start to the snapshot cutoff.",
        category="subscription",
        non_negative=True,
    ),
    FeatureSpec(
        name="initial_plan",
        dtype="string",
        description="Plan tier at signing (starter / growth / enterprise).",
        category="subscription",
    ),
    FeatureSpec(
        name="initial_mrr",
        dtype="Int64",
        description="Monthly recurring revenue (USD) at signing.",
        category="subscription",
        non_negative=True,
    ),
    FeatureSpec(
        name="current_mrr",
        dtype="Int64",
        description="Monthly recurring revenue (USD) as of the cutoff.",
        category="subscription",
        non_negative=True,
    ),
    FeatureSpec(
        name="mrr_change_at_snapshot",
        dtype="Int64",
        description=(
            "current_mrr - initial_mrr, both measured at the cutoff.  The "
            "leakage-safe counterpart of mrr_change_full_period."
        ),
        category="subscription",
    ),
    FeatureSpec(
        name="renewal_count",
        dtype="Int64",
        description="Contract renewals completed at or before the cutoff.",
        category="subscription",
        non_negative=True,
    ),
    FeatureSpec(
        name="expansion_count",
        dtype="Int64",
        description="Expansion (upsell) events at or before the cutoff.",
        category="subscription",
        non_negative=True,
    ),
    FeatureSpec(
        name="contract_term_months",
        dtype="Int64",
        description="Contract term length in months (12 or 24).",
        category="subscription",
        non_negative=True,
    ),
    FeatureSpec(
        name="weeks_to_next_renewal",
        dtype="Int64",
        description="Weeks from the cutoff to the next contract anniversary.",
        category="subscription",
        non_negative=True,
    ),
    # -- health aggregates (last 12 weeks before the cutoff) -----------------
    FeatureSpec(
        name="avg_active_users_l12w",
        dtype="Float64",
        description="Mean weekly active users over the 12 weeks before the cutoff.",
        category="health",
        non_negative=True,
    ),
    FeatureSpec(
        name="active_user_trend_l12w",
        dtype="Float64",
        description=(
            "OLS slope of weekly active users over the 12 weeks before the "
            "cutoff (users per week; negative = declining usage)."
        ),
        category="health",
    ),
    FeatureSpec(
        name="avg_feature_depth_l12w",
        dtype="Float64",
        description="Mean feature-depth score over the 12 weeks before the cutoff.",
        category="health",
        non_negative=True,
    ),
    FeatureSpec(
        name="support_ticket_count_l12w",
        dtype="Int64",
        description="Support tickets filed in the 12 weeks before the cutoff.",
        category="health",
        non_negative=True,
    ),
    FeatureSpec(
        name="last_nps_score",
        dtype="Int64",
        description=(
            "Most recent NPS response (0-10) at or before the cutoff; null if "
            "the customer has not yet answered a quarterly survey."
        ),
        category="health",
        non_negative=True,
    ),
    # -- financial -----------------------------------------------------------
    FeatureSpec(
        name="payment_failure_count",
        dtype="Int64",
        description="Invoice payment failures at or before the cutoff.",
        category="financial",
        non_negative=True,
    ),
    FeatureSpec(
        name="weeks_since_last_payment_failure",
        dtype="Int64",
        description=(
            "Weeks from the most recent payment failure to the cutoff; null "
            "if the customer has never had one."
        ),
        category="financial",
        non_negative=True,
    ),
    # -- leakage trap (deliberate, all modes; see module docstring) -----------
    FeatureSpec(
        name="mrr_change_full_period",
        dtype="Int64",
        description=(
            "MRR delta from signing to END OF SIMULATION.  LEAKAGE TRAP: "
            "post-cutoff expansions, which directly drive the pLTV targets, "
            "inflate this column.  Use mrr_change_at_snapshot instead."
        ),
        category="subscription",
        leakage_risk=True,
    ),
    # -- targets (D6 windows; D7 gross revenue; D9 secondary churn) ----------
    FeatureSpec(
        name="ltv_revenue_90d",
        dtype="Float64",
        description=(
            "Gross revenue (USD, paid + recovered invoices) in the 90 days "
            "after the cutoff.  Primary pLTV regression target (warm-up)."
        ),
        category="target",
        is_target=True,
        non_negative=True,
    ),
    FeatureSpec(
        name="ltv_revenue_365d",
        dtype="Float64",
        description=(
            "Gross revenue (USD, paid + recovered invoices) in the 365 days "
            "after the cutoff.  Primary pLTV regression target (standard)."
        ),
        category="target",
        is_target=True,
        non_negative=True,
    ),
    FeatureSpec(
        name="ltv_revenue_730d",
        dtype="Float64",
        description=(
            "Gross revenue (USD, paid + recovered invoices) in the 730 days "
            "after the cutoff.  Primary pLTV regression target (hard)."
        ),
        category="target",
        is_target=True,
        non_negative=True,
    ),
    FeatureSpec(
        name="churned_within_180d",
        dtype="boolean",
        description=(
            "True iff the customer churned within 180 days after the cutoff.  "
            "Secondary task / ZILN zero-inflation indicator."
        ),
        category="target",
        is_target=True,
    ),
]
