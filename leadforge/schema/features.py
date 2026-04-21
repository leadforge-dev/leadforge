"""Feature specification for the lead snapshot task table.

:data:`LEAD_SNAPSHOT_FEATURES` is the canonical ordered list of features
present in the primary task export (``tasks/converted_within_90_days/``).
Every feature here is anchored at or before the snapshot date — no
post-anchor data is included (leakage rule, §4 of the architecture spec).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one column in the lead snapshot table.

    Attributes:
        name: Column name as it appears in the Parquet file.
        dtype: Pandas-compatible dtype string (``"string"``, ``"Int64"``,
            ``"Float64"``, ``"boolean"``).
        description: Human-readable explanation of what the column captures.
        category: Logical grouping (``"account"``, ``"contact"``,
            ``"lead_meta"``, ``"engagement"``, ``"sales"``, ``"target"``).
        is_target: True for the label column only.
        leakage_risk: True if the column could contain post-snapshot-anchor
            information and must be excluded from student_public exports.
    """

    name: str
    dtype: str
    description: str
    category: str
    is_target: bool = False
    leakage_risk: bool = False


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
    FeatureSpec(
        "first_touch_channel",
        "string",
        "Marketing channel responsible for the first recorded touch.",
        "lead_meta",
    ),
    FeatureSpec(
        "current_stage",
        "string",
        "Funnel stage at snapshot anchor date.",
        "lead_meta",
    ),
    FeatureSpec(
        "is_mql",
        "boolean",
        "Whether the lead had achieved MQL status at snapshot date.",
        "lead_meta",
    ),
    FeatureSpec(
        "is_sql",
        "boolean",
        "Whether the lead had achieved SQL status at snapshot date.",
        "lead_meta",
    ),
    # -- Engagement features --
    FeatureSpec(
        "touch_count",
        "Int64",
        "Total number of marketing/sales touches recorded before snapshot.",
        "engagement",
    ),
    FeatureSpec(
        "inbound_touch_count",
        "Int64",
        "Number of inbound touches before snapshot.",
        "engagement",
    ),
    FeatureSpec(
        "outbound_touch_count",
        "Int64",
        "Number of outbound touches before snapshot.",
        "engagement",
    ),
    FeatureSpec(
        "session_count",
        "Int64",
        "Number of web/trial sessions recorded before snapshot.",
        "engagement",
    ),
    FeatureSpec(
        "pricing_page_views",
        "Int64",
        "Cumulative pricing page views across all sessions before snapshot.",
        "engagement",
    ),
    FeatureSpec(
        "demo_page_views",
        "Int64",
        "Cumulative demo page views across all sessions before snapshot.",
        "engagement",
    ),
    FeatureSpec(
        "total_session_duration_seconds",
        "Int64",
        "Sum of session durations (seconds) before snapshot.",
        "engagement",
    ),
    # -- Sales activity features --
    FeatureSpec(
        "activity_count",
        "Int64",
        "Number of sales activities logged before snapshot.",
        "sales",
    ),
    FeatureSpec(
        "days_since_last_touch",
        "Float64",
        "Days elapsed between most recent touch and snapshot anchor date.",
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
