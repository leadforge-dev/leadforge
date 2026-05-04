"""Feature specification for the lead snapshot task table.

:data:`LEAD_SNAPSHOT_FEATURES` is the canonical ordered list of features
present in the primary task export (``tasks/converted_within_90_days/``).
Every feature here is anchored at or before the snapshot date — no
post-anchor data is included (leakage rule, §4 of the architecture spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from leadforge.core.enums import ExposureMode


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one column in the lead snapshot table.

    Two concerns are kept deliberately separate:

    - :attr:`leakage_risk` is *descriptive*: the value of this column is
      computed from events that may post-date the snapshot anchor and so
      correlates with the label.  It is informational metadata for
      downstream consumers and is preserved in the published feature
      dictionary.
    - :attr:`redact_in_modes` is *prescriptive*: the bundle writer must
      strip this column from any export whose mode is in this set.

    These can disagree: ``total_touches_all`` is ``leakage_risk=True``
    (it does encode post-snapshot information) but
    ``redact_in_modes=frozenset()`` (it is deliberately retained as a
    pedagogical trap).  Conversely a recipe could redact a column that
    is not itself leakage-risky for unrelated policy reasons.

    Attributes:
        name: Column name as it appears in the Parquet file.
        dtype: Pandas-compatible dtype string (``"string"``, ``"Int64"``,
            ``"Float64"``, ``"boolean"``).
        description: Human-readable explanation of what the column captures.
        category: Logical grouping (``"account"``, ``"contact"``,
            ``"lead_meta"``, ``"engagement"``, ``"sales"``, ``"target"``).
        is_target: True for the label column only.
        leakage_risk: Descriptive — this column is post-snapshot correlated.
        redact_in_modes: Prescriptive — exposure modes in which the
            bundle writer must strip this column from snapshot, task
            splits, and feature dictionary.
    """

    name: str
    dtype: str
    description: str
    category: str
    is_target: bool = False
    leakage_risk: bool = False
    redact_in_modes: frozenset[ExposureMode] = field(default_factory=frozenset)


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
        "Funnel stage at snapshot anchor date. WARNING: at full-horizon "
        "(90-day) snapshots this contains terminal stages (closed_won / "
        "closed_lost) that encode the label. Exclude from modeling or use "
        "a windowed snapshot.",
        "lead_meta",
        leakage_risk=True,
        redact_in_modes=frozenset({ExposureMode.student_public}),
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
    # -- Momentum features --
    FeatureSpec(
        "touches_week_1",
        "Int64",
        "Number of touches in the first 7 days after lead creation.",
        "engagement",
    ),
    FeatureSpec(
        "touches_last_7_days",
        "Int64",
        "Number of touches in the last 7 days before snapshot cutoff.",
        "engagement",
    ),
    FeatureSpec(
        "days_since_first_touch",
        "Float64",
        "Days between first touch and snapshot cutoff (NaN if no touches).",
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
        "Days elapsed between most recent touch and snapshot cutoff.",
        "sales",
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
    ),
    FeatureSpec(
        "expected_acv",
        "Float64",
        "Expected ACV: opportunity ACV if available by snapshot, else "
        "revenue band midpoint heuristic (NaN if neither available).",
        "sales",
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

    The redaction policy is encoded per-feature in
    :attr:`FeatureSpec.redact_in_modes`.  Callers (the bundle writer, the
    validation check) all derive their answer from this single function, so
    a single source of truth governs both producing and verifying bundles.

    Args:
        mode: The exposure mode being published.
        features: Feature spec tuple to consult.  Defaults to the canonical
            lead snapshot list; callable with a custom tuple for tests or
            future per-recipe feature sets.
    """
    return frozenset(f.name for f in features if mode in f.redact_in_modes)
