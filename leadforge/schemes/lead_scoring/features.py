"""Lead-scoring (``b2b_saas_procurement_v1``) feature spec.

:data:`LEAD_SNAPSHOT_FEATURES` is the canonical ordered list of features
present in the primary lead-scoring task export.  The shared
:class:`~leadforge.schema.features.FeatureSpec` primitive stays in
:mod:`leadforge.schema.features`.
"""

from __future__ import annotations

from leadforge.core.enums import ExposureMode
from leadforge.schema.features import FeatureSpec

# ---------------------------------------------------------------------------
# Canonical feature list — lead snapshot
# ---------------------------------------------------------------------------

LEAD_SNAPSHOT_FEATURES: tuple[FeatureSpec, ...] = (
    # -- Account features --
    FeatureSpec("account_id", "string", "Opaque account identifier.", "account"),
    FeatureSpec(
        "industry",
        "string",
        "Industry vertical of the buying organization.",
        "account",
    ),
    FeatureSpec(
        "region",
        "string",
        "Geographic region of the account's headquarters.",
        "account",
    ),
    FeatureSpec(
        "employee_band",
        "string",
        "Banded employee headcount of the account.",
        "account",
    ),
    FeatureSpec(
        "estimated_revenue_band",
        "string",
        "Banded estimated annual revenue of the account.",
        "account",
    ),
    FeatureSpec(
        "process_maturity_band",
        "string",
        "Banded internal process maturity score (latent).",
        "account",
        leakage_risk=False,
    ),
    # -- Contact features --
    FeatureSpec("contact_id", "string", "Opaque contact identifier.", "contact"),
    FeatureSpec(
        "role_function",
        "string",
        "Functional area of the primary contact (e.g. finance, ops).",
        "contact",
    ),
    FeatureSpec(
        "seniority",
        "string",
        "Seniority band of the primary contact.",
        "contact",
    ),
    FeatureSpec(
        "buyer_role",
        "string",
        "Buyer role classification (economic_buyer, champion, etc.).",
        "contact",
    ),
    # -- Lead metadata features --
    FeatureSpec("lead_id", "string", "Opaque lead identifier.", "lead_meta"),
    FeatureSpec(
        "lead_created_at",
        "string",
        "ISO-8601 timestamp when the lead was created.",
        "lead_meta",
    ),
    FeatureSpec(
        "lead_source",
        "string",
        "Origination source of the lead (e.g. inbound_form, sdr_outbound).",
        "lead_meta",
    ),
    # Note: ``first_touch_channel`` is absent from this list.  In v1 the
    # simulation sets it to the same value as ``lead_source`` (both derive
    # from the channel drawn during lead creation), making it byte-identical
    # and zero-information.  It is retained in the relational ``leads``
    # table for completeness; it is excluded from the flat snapshot because
    # a duplicate column would be actively misleading in a teaching dataset.
    FeatureSpec(
        "current_stage",
        "string",
        "Funnel stage at snapshot anchor date. WARNING: at full-horizon "
        "(90-day) snapshots this contains terminal stages (closed_won / "
        "closed_lost) that encode the label. Exclude from modeling or use "
        "a windowed snapshot.",
        "lead_meta",
        leakage_risk=True,
        redact_in_modes=frozenset({ExposureMode.student_public}),
    ),
    # Note: ``is_mql`` was removed from the canonical feature list (issue #57)
    # because every lead is initialised at MQL stage in
    # ``leadforge/schemes/lead_scoring/simulation/population.py``, making the
    # column constant ``True`` and zero-variance.  The underlying
    # ``LeadRow.is_mql`` field still lives on the relational ``leads.parquet``
    # table.
    FeatureSpec(
        "is_sql",
        "boolean",
        "Whether the lead had achieved SQL status at snapshot date. "
        "Strongly correlated with the label: the simulator only converts "
        "non-SQL leads via a rare direct-conversion path, so "
        "is_sql=False predicts non-conversion with very high probability "
        "(P(conv | is_sql=False) ≈ 0.04 / 0.015 / 0.006 across difficulty "
        "tiers).  Redacted from student_public bundles.",
        "lead_meta",
        leakage_risk=True,
        redact_in_modes=frozenset({ExposureMode.student_public}),
    ),
    # -- Engagement features --
    FeatureSpec(
        "touch_count",
        "Int64",
        "Total number of marketing/sales touches recorded before snapshot.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "inbound_touch_count",
        "Int64",
        "Number of inbound touches before snapshot.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "outbound_touch_count",
        "Int64",
        "Number of outbound touches before snapshot.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "session_count",
        "Int64",
        "Number of web/trial sessions recorded before snapshot.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "pricing_page_views",
        "Int64",
        "Cumulative pricing page views across all sessions before snapshot.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "demo_page_views",
        "Int64",
        "Cumulative demo page views across all sessions before snapshot.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "total_session_duration_seconds",
        "Int64",
        "Sum of session durations (seconds) before snapshot.",
        "engagement",
        non_negative=True,
    ),
    # -- Momentum features --
    FeatureSpec(
        "touches_days_0_7",
        "Int64",
        "Number of touches in days 0–7 (inclusive) after lead creation.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "touches_last_7_days",
        "Int64",
        "Number of touches in the last 7 days before snapshot cutoff.",
        "engagement",
        non_negative=True,
    ),
    FeatureSpec(
        "days_since_first_touch",
        "Float64",
        "Days between first touch and snapshot cutoff (NaN if no touches).",
        "engagement",
        non_negative=True,
    ),
    # -- Sales activity features --
    FeatureSpec(
        "activity_count",
        "Int64",
        "Number of sales activities logged before snapshot.",
        "sales",
        non_negative=True,
    ),
    FeatureSpec(
        "days_since_last_touch",
        "Float64",
        "Days elapsed between most recent touch and snapshot cutoff.",
        "sales",
        non_negative=True,
    ),
    FeatureSpec(
        "opportunity_created",
        "boolean",
        "Whether any opportunity was created by snapshot date (open or closed).",
        "sales",
    ),
    FeatureSpec(
        "has_open_opportunity",
        "boolean",
        "Whether an open opportunity existed at snapshot date.",
        "sales",
    ),
    FeatureSpec(
        "opportunity_estimated_acv",
        "Float64",
        "Estimated ACV of the most recent open opportunity (NaN if none).",
        "sales",
        non_negative=True,
    ),
    FeatureSpec(
        "expected_acv",
        "Float64",
        "Expected ACV: opportunity ACV if available by snapshot, else "
        "revenue band midpoint heuristic (NaN if neither available).",
        "sales",
        non_negative=True,
    ),
    # -- Pedagogical leakage trap (deliberately retained in all modes) --
    FeatureSpec(
        "total_touches_all",
        "Int64",
        "Total touches over full 90-day window. LEAKAGE TRAP: uses "
        "post-snapshot data. Included for pedagogical purposes only.",
        "engagement",
        leakage_risk=True,
    ),
    # -- Target --
    FeatureSpec(
        "converted_within_90_days",
        "boolean",
        "Label: True if a closed_won event occurred within 90 days of "
        "the snapshot anchor date. Derived from simulated events.",
        "target",
        is_target=True,
    ),
)


def redacted_columns_for(
    mode: ExposureMode,
    features: tuple[FeatureSpec, ...] = LEAD_SNAPSHOT_FEATURES,
) -> frozenset[str]:
    """Return the set of column names that must be stripped from *mode* exports.

    Args:
        mode: The exposure mode being published.
        features: Feature spec tuple to consult.  Defaults to the canonical
            :data:`LEAD_SNAPSHOT_FEATURES` list.
    """
    return frozenset(f.name for f in features if mode in f.redact_in_modes)
