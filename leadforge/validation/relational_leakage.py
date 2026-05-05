"""Probes that detect public-bundle reconstruction of ``converted_within_90_days``.

The audit in ``docs/release/v1_current_state_audit.md`` enumerates four
deterministic paths (A-E) by which alpha public bundles reconstruct the
target via joins.  This module is the validator that asserts the
snapshot-safe contract — encoded in :data:`BANNED_LEAD_COLUMNS`,
:data:`BANNED_OPP_COLUMNS`, :data:`BANNED_TABLES`, and
:data:`SNAPSHOT_FILTERED_TABLES` — is in place on any bundle claiming to
be ``student_public``.  The matching writer-side enforcement lives in
:mod:`leadforge.render.relational_snapshot_safe` and imports the same
constants from this module.

Five probes, each producing zero or more :class:`LeakageFinding`:

* :func:`probe_banned_columns` — public ``leads`` and ``opportunities``
  must not contain :data:`BANNED_LEAD_COLUMNS` or
  :data:`BANNED_OPP_COLUMNS` respectively.
* :func:`probe_banned_tables` — public bundles must not include
  :data:`BANNED_TABLES`.
* :func:`probe_deterministic_reconstruction` — paths B / C / D from the
  audit must produce zero positive predictions.  **Path A is not
  checked here** — it is the column-presence violation already covered
  by :func:`probe_banned_columns`.
* :func:`probe_snapshot_window` — every event-table row must satisfy
  ``timestamp <= lead_created_at + snapshot_day``.
* :func:`probe_bonus_model_auc` — *opt-in* honest-feature baseline:
  trains LR + HistGBM on the legitimate aggregates ``n_opps`` /
  ``max_acv`` / ``mean_acv`` (plus ``n_customers`` /
  ``n_subscriptions`` if present) and asserts CV AUC stays below an
  explicit ``max_auc``.  The orchestrators skip this probe unless the
  caller passes ``bonus_model_max_auc=...``.

:func:`run_all_probes` is the file-based orchestrator (designed to be
called from :func:`leadforge.validation.bundle_checks.validate_bundle`).
:func:`run_all_probes_on_dataframes` is the same orchestrator without
the disk read, for unit tests against in-memory bundles.

The :func:`deterministic_relational_reconstruction` function is the
single source of truth for the join graph that defines paths A-E.  The
companion script ``scripts/probe_relational_leakage.py`` re-exports it
from here so the alpha-bundle audit and the validator agree by
construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from leadforge.core.exceptions import LeadforgeError

# ---------------------------------------------------------------------------
# Snapshot-safe contract — the single source of truth for "what is leakage".
# leadforge.render.relational_snapshot_safe imports these so the writer
# and the validator share one definition.
# ---------------------------------------------------------------------------

#: Columns dropped from public ``leads.parquet``.
BANNED_LEAD_COLUMNS: Final[tuple[str, ...]] = (
    "converted_within_90_days",
    "conversion_timestamp",
)

#: Columns dropped from public ``opportunities.parquet``.
BANNED_OPP_COLUMNS: Final[tuple[str, ...]] = (
    "close_outcome",
    "closed_at",
)

#: Tables omitted from public bundles entirely.
BANNED_TABLES: Final[tuple[str, ...]] = ("customers", "subscriptions")

#: Tables filtered per-lead by their timestamp column to
#: ``lead_created_at + snapshot_day``.  ``opportunities`` is included
#: even though it is an entity table, because its ``created_at``
#: anchors when the entity becomes observable in the funnel.
SNAPSHOT_FILTERED_TABLES: Final[tuple[tuple[str, str], ...]] = (
    ("touches", "touch_timestamp"),
    ("sessions", "session_timestamp"),
    ("sales_activities", "activity_timestamp"),
    ("opportunities", "created_at"),
)

#: Channel labels carried on :class:`LeakageFinding.channel`.  Constants
#: rather than an enum because findings serialise straight to JSON.
CHANNEL_BANNED_COLUMN: Final[str] = "banned_column"
CHANNEL_BANNED_TABLE: Final[str] = "banned_table"
CHANNEL_JOIN_RECONSTRUCTION: Final[str] = "join_reconstruction"
CHANNEL_SNAPSHOT_WINDOW: Final[str] = "snapshot_window"
CHANNEL_BONUS_MODEL: Final[str] = "bonus_model"

_PUBLIC_TABLES: Final[tuple[str, ...]] = (
    "accounts",
    "contacts",
    "leads",
    "touches",
    "sessions",
    "sales_activities",
    "opportunities",
)


@dataclass(frozen=True)
class LeakageFinding:
    """One leakage-channel violation surfaced by a probe."""

    channel: str
    detail: str
    message: str


@dataclass(frozen=True)
class LeakageReport:
    """Aggregate result of a probe run.  Empty :attr:`findings` means OK."""

    findings: tuple[LeakageFinding, ...]

    @property
    def ok(self) -> bool:
        return len(self.findings) == 0

    def raise_if_failing(self) -> None:
        """Raise :class:`RelationalLeakageError` if any probe reported a finding."""
        if not self.ok:
            raise RelationalLeakageError(self)


class RelationalLeakageError(LeadforgeError):
    """Raised by :meth:`LeakageReport.raise_if_failing` on any finding.

    Carries the originating :class:`LeakageReport` on ``self.report`` so
    callers (e.g. ``leadforge validate``) can render the full set of
    findings rather than just the first one.
    """

    def __init__(self, report: LeakageReport) -> None:
        self.report = report
        rendered = "\n".join(f"  - [{f.channel}] {f.detail}: {f.message}" for f in report.findings)
        super().__init__(
            f"public bundle leaks `converted_within_90_days` "
            f"({len(report.findings)} finding(s)):\n{rendered}"
        )


# ---------------------------------------------------------------------------
# Deterministic reconstruction — the join graph that defines paths A-E.
# Lifted from the PR 1.1 audit script; the script now re-exports this
# function from here so there is one implementation.
# ---------------------------------------------------------------------------


def deterministic_relational_reconstruction(
    leads: pd.DataFrame,
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct ``converted_within_90_days`` from public relational joins.

    Returns a DataFrame indexed by ``lead_id`` with five boolean columns,
    one per reconstruction path (A-E). Path E is the union of B, C, D and
    is the headline relational-leakage prediction.

    No hidden state, no model fit — pure joins.

    Empty ``customers``/``subscriptions`` frames are accepted (the
    post-fix expected state); the corresponding paths simply return
    all-False.

    Raises:
        ValueError: if ``leads.lead_id`` contains duplicates. A validator
            cannot operate safely on non-unique keys.
    """
    if not leads["lead_id"].is_unique:
        raise ValueError("leads.lead_id must be unique")

    leads_idx = leads.set_index("lead_id", drop=False)

    # Path A — the label itself, if present in public leads.
    # Plain ``astype(bool)`` would map NaN to True; route through pandas'
    # nullable boolean dtype so missing values fill cleanly to False without
    # triggering object-downcast warnings.
    if "converted_within_90_days" in leads.columns:
        path_a = leads_idx["converted_within_90_days"].astype("boolean").fillna(False).astype(bool)
    else:
        path_a = pd.Series(False, index=leads_idx.index, name="converted_within_90_days")

    # Path B — any opportunity with close_outcome == "closed_won".
    if "close_outcome" in opportunities.columns and len(opportunities) > 0:
        won_leads = set(
            opportunities.loc[opportunities["close_outcome"] == "closed_won", "lead_id"]
        )
    else:
        won_leads = set()
    path_b = leads_idx["lead_id"].isin(won_leads)

    # Path C — lead has any joined customer (via opportunity_id -> opportunity.lead_id).
    if len(opportunities) > 0:
        opp_to_lead = dict(
            zip(opportunities["opportunity_id"], opportunities["lead_id"], strict=False)
        )
    else:
        opp_to_lead = {}
    customer_leads = {
        opp_to_lead[opp_id] for opp_id in customers["opportunity_id"] if opp_id in opp_to_lead
    }
    path_c = leads_idx["lead_id"].isin(customer_leads)

    # Path D — lead has any joined subscription (sub -> customer -> opportunity -> lead).
    if len(customers) > 0:
        cust_to_opp = dict(zip(customers["customer_id"], customers["opportunity_id"], strict=False))
    else:
        cust_to_opp = {}
    sub_leads: set[str] = set()
    for cust_id in subscriptions["customer_id"]:
        opp_id = cust_to_opp.get(cust_id)
        if opp_id is None:
            continue
        lead_id = opp_to_lead.get(opp_id)
        if lead_id is not None:
            sub_leads.add(lead_id)
    path_d = leads_idx["lead_id"].isin(sub_leads)

    # Path E — deterministic OR of B, C, D (the headline join-only path).
    path_e = path_b | path_c | path_d

    return pd.DataFrame(
        {
            "path_a_direct_label": path_a.values,
            "path_b_opportunity_won": path_b.values,
            "path_c_customer_exists": path_c.values,
            "path_d_subscription_exists": path_d.values,
            "path_e_or_b_c_d": path_e.values,
        },
        index=leads_idx.index,
    )


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_banned_columns(tables: Mapping[str, pd.DataFrame]) -> list[LeakageFinding]:
    """Public ``leads``/``opportunities`` must not carry banned columns.

    Detects Path A (label column directly readable from ``leads``) and
    the ``opportunities.close_outcome`` / ``closed_at`` channels — i.e.
    leakage that any caller can spot by listing column names, no joins
    required.
    """
    findings: list[LeakageFinding] = []
    for table_name, banned in (
        ("leads", BANNED_LEAD_COLUMNS),
        ("opportunities", BANNED_OPP_COLUMNS),
    ):
        df = tables.get(table_name)
        if df is None:
            continue
        for col in banned:
            if col in df.columns:
                findings.append(
                    LeakageFinding(
                        channel=CHANNEL_BANNED_COLUMN,
                        detail=f"{table_name}.{col}",
                        message=(
                            f"public {table_name}.parquet must not contain "
                            f"the banned column '{col}'"
                        ),
                    )
                )
    return findings


def probe_banned_tables(table_names: Iterable[str]) -> list[LeakageFinding]:
    """Public bundles must not include conversion-conditional tables."""
    present = set(table_names)
    return [
        LeakageFinding(
            channel=CHANNEL_BANNED_TABLE,
            detail=name,
            message=(
                f"public bundles must not include '{name}.parquet' "
                "(it exists only for converted leads, so its presence "
                "reconstructs the label)"
            ),
        )
        for name in BANNED_TABLES
        if name in present
    ]


def probe_deterministic_reconstruction(
    tables: Mapping[str, pd.DataFrame],
) -> list[LeakageFinding]:
    """Audit paths B / C / D must produce zero positive predictions.

    This probe focuses exclusively on the **join-graph** reconstruction:

    * B — at least one opportunity with ``close_outcome == "closed_won"``;
    * C — a joinable customer row reachable via ``opportunity_id``;
    * D — a joinable subscription row reachable via ``customer_id``.

    Path A (direct read of ``leads.converted_within_90_days``) is *not*
    checked here — it is the column-presence violation already raised by
    :func:`probe_banned_columns`.  Re-emitting it here would double-count
    one defect across two channels.  Tests assert this delegation
    explicitly so that future maintainers don't widen the scope by
    accident.
    """
    leads = tables.get("leads")
    if leads is None or len(leads) == 0:
        return []

    opportunities = tables.get(
        "opportunities",
        _empty_frame({"opportunity_id": "string", "lead_id": "string"}),
    )
    customers = tables.get(
        "customers",
        _empty_frame({"customer_id": "string", "opportunity_id": "string", "account_id": "string"}),
    )
    subscriptions = tables.get(
        "subscriptions",
        _empty_frame({"subscription_id": "string", "customer_id": "string"}),
    )

    paths = deterministic_relational_reconstruction(leads, opportunities, customers, subscriptions)

    findings: list[LeakageFinding] = []
    for path_col, label in (
        ("path_b_opportunity_won", "B (opportunity.close_outcome == 'closed_won')"),
        ("path_c_customer_exists", "C (joined customer exists)"),
        ("path_d_subscription_exists", "D (joined subscription exists)"),
    ):
        positive = int(paths[path_col].sum())
        if positive > 0:
            findings.append(
                LeakageFinding(
                    channel=CHANNEL_JOIN_RECONSTRUCTION,
                    detail=path_col,
                    message=(
                        f"path {label} produced {positive}/{len(paths)} "
                        "positive predictions; a snapshot-safe public "
                        "bundle must produce zero"
                    ),
                )
            )
    return findings


def probe_snapshot_window(
    tables: Mapping[str, pd.DataFrame], snapshot_day: int
) -> list[LeakageFinding]:
    """Every event-table row must satisfy ``timestamp <= lead_created_at + snapshot_day``."""
    if snapshot_day < 0:
        raise ValueError(f"snapshot_day must be non-negative, got {snapshot_day}")
    leads = tables.get("leads")
    if leads is None or len(leads) == 0:
        return []
    if "lead_id" not in leads.columns or "lead_created_at" not in leads.columns:
        raise ValueError("leads must contain 'lead_id' and 'lead_created_at' columns")
    if not leads["lead_id"].is_unique:
        raise ValueError("leads.lead_id must be unique")

    anchor = leads[["lead_id", "lead_created_at"]].copy()
    anchor["lead_created_at"] = pd.to_datetime(anchor["lead_created_at"])
    horizon = pd.Timedelta(days=snapshot_day)

    findings: list[LeakageFinding] = []
    for name, ts_col in SNAPSHOT_FILTERED_TABLES:
        df = tables.get(name)
        if df is None or len(df) == 0 or ts_col not in df.columns:
            continue
        merged = df[["lead_id", ts_col]].merge(anchor, on="lead_id", how="left")
        ts = pd.to_datetime(merged[ts_col])
        cutoff = merged["lead_created_at"] + horizon
        violations = int((ts > cutoff).fillna(False).sum())
        if violations > 0:
            findings.append(
                LeakageFinding(
                    channel=CHANNEL_SNAPSHOT_WINDOW,
                    detail=f"{name}.{ts_col}",
                    message=(
                        f"{violations}/{len(df)} rows in {name}.parquet "
                        f"have {ts_col} > lead_created_at + {snapshot_day}d"
                    ),
                )
            )
    return findings


def probe_bonus_model_auc(
    tables: Mapping[str, pd.DataFrame],
    *,
    max_auc: float,
    seed: int = 42,
    label: pd.Series | None = None,
) -> list[LeakageFinding]:
    """Opt-in honest-feature baseline: 5-fold CV LR + HistGBM AUC.

    Trains on per-lead aggregates (``n_opps`` / ``max_acv`` /
    ``mean_acv``, plus ``n_customers`` / ``n_subscriptions`` if those
    tables are present) and asserts the mean cross-validated AUC stays
    below ``max_auc``.

    Caller responsibilities:

    * ``max_auc`` is required.  PR 2.1 ships this probe with no
      calibrated threshold; PR 3.3 will land per-tier bands.
    * ``label`` must be a :class:`pandas.Series` indexed by ``lead_id``
      (``index.name == "lead_id"``) **and** cover every lead in the
      bundle.  Both are enforced — a misaligned or partial label would
      silently neutralise the probe (via the binary-cardinality gate
      or NaN folds), which defeats the validator's purpose.

    The probe skips silently (no findings, no error) when:

    * scikit-learn is not installed;
    * ``leads`` is missing or empty;
    * the label is unavailable (no ``label`` argument and the public
      bundle has correctly redacted ``converted_within_90_days``);
    * the label has fewer than two classes after alignment;
    * the smaller class has fewer members than the minimum needed for
      stratified CV (``n_splits >= 2``).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return []

    leads = tables.get("leads")
    if leads is None or len(leads) == 0:
        return []

    y_series = _resolve_label(leads, label)
    if y_series is None:
        return []

    features = _build_relational_features(leads, tables)
    if features.empty or len(features.columns) == 0:
        return []

    aligned = y_series.reindex(features.index)
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(
            f"label is missing values for {missing} lead_id(s) present in the "
            "bundle; supply a complete label or omit it to read from leads"
        )
    y = aligned.astype(int)
    if y.nunique(dropna=True) < 2:
        return []

    # Stratified CV needs at least n_splits members in each class.  If the
    # smaller class is below that, the probe can't run — skip silently
    # (this is a sample-size constraint, not a leakage finding).
    min_class = int(y.value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return []

    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "hist_gbm": Pipeline([("clf", HistGradientBoostingClassifier(random_state=seed))]),
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    findings: list[LeakageFinding] = []
    for name, pipe in models.items():
        aucs: list[float] = []
        for train_idx, test_idx in skf.split(features.values, y.values):
            x_tr, x_te = features.values[train_idx], features.values[test_idx]
            y_tr, y_te = y.values[train_idx], y.values[test_idx]
            pipe.fit(x_tr, y_tr)
            proba = pipe.predict_proba(x_te)[:, 1]
            aucs.append(float(roc_auc_score(y_te, proba)))
        auc_mean = sum(aucs) / len(aucs)
        if auc_mean > max_auc:
            findings.append(
                LeakageFinding(
                    channel=CHANNEL_BONUS_MODEL,
                    detail=name,
                    message=(
                        f"{n_splits}-fold CV AUC {auc_mean:.3f} on join-derived "
                        f"features exceeds max_auc={max_auc:.3f}; honest "
                        "aggregates carry stronger signal than the band allows"
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def run_all_probes_on_dataframes(
    tables: Mapping[str, pd.DataFrame],
    *,
    snapshot_day: int,
    bonus_model_max_auc: float | None = None,
    label: pd.Series | None = None,
) -> LeakageReport:
    """Run every structural probe; run the bonus probe iff ``bonus_model_max_auc`` is set."""
    findings: list[LeakageFinding] = []
    findings += probe_banned_columns(tables)
    findings += probe_banned_tables(tables.keys())
    findings += probe_deterministic_reconstruction(tables)
    findings += probe_snapshot_window(tables, snapshot_day=snapshot_day)
    if bonus_model_max_auc is not None:
        findings += probe_bonus_model_auc(tables, max_auc=bonus_model_max_auc, label=label)
    return LeakageReport(findings=tuple(findings))


def run_all_probes(
    bundle_dir: Path,
    *,
    snapshot_day: int,
    bonus_model_max_auc: float | None = None,
    label: pd.Series | None = None,
) -> LeakageReport:
    """Run every structural probe against ``<bundle_dir>/tables/*.parquet``.

    Args:
        bundle_dir: Bundle root (must contain ``tables/leads.parquet``).
        snapshot_day: Snapshot window for the timestamp probe.  The
            caller (typically ``validate_bundle``) is expected to read
            it from ``manifest.json``.
        bonus_model_max_auc: Pass a numeric threshold to enable the
            opt-in :func:`probe_bonus_model_auc`.  ``None`` (default)
            skips it — the calibrated band ships in PR 3.3.
        label: Optional ground-truth labels for the bonus probe when
            ``leads.converted_within_90_days`` has been redacted.  Must
            be indexed by ``lead_id`` (see :func:`probe_bonus_model_auc`).
            Ignored when ``bonus_model_max_auc`` is ``None``.

    Raises:
        FileNotFoundError: if ``<bundle_dir>/tables/`` is missing or
            ``leads.parquet`` is not present.
    """
    tables_dir = bundle_dir / "tables"
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"missing tables/ under {bundle_dir}")
    if not (tables_dir / "leads.parquet").exists():
        raise FileNotFoundError(f"missing required leads.parquet under {tables_dir}")

    tables: dict[str, pd.DataFrame] = {}
    for name in (*_PUBLIC_TABLES, *BANNED_TABLES):
        path = tables_dir / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
    return run_all_probes_on_dataframes(
        tables,
        snapshot_day=snapshot_day,
        bonus_model_max_auc=bonus_model_max_auc,
        label=label,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_frame(dtype_map: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in dtype_map.items()})


def _resolve_label(
    leads: pd.DataFrame,
    label: pd.Series | None,
) -> pd.Series | None:
    """Pick a label series to score against, or ``None`` to skip the probe.

    A caller-supplied ``label`` must be indexed by ``lead_id``
    (``index.name == "lead_id"``).  Without that guarantee a misaligned
    label would silently skip the probe via the binary-cardinality gate
    downstream — exactly the kind of hidden no-op a leakage validator
    must not have.
    """
    if label is not None:
        if label.index.name != "lead_id":
            raise ValueError(
                "label must be a pandas.Series indexed by lead_id "
                f"(got index.name={label.index.name!r})"
            )
        return label.astype("boolean").fillna(False).astype(int)
    if "converted_within_90_days" in leads.columns:
        return (
            leads.set_index("lead_id")["converted_within_90_days"]
            .astype("boolean")
            .fillna(False)
            .astype(int)
        )
    return None


def _build_relational_features(
    leads: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Per-lead aggregates from joinable public/optional relational tables.

    Honest features only — no aggregate of ``close_outcome``.  Customers
    and subscriptions counts are included only when the corresponding
    tables exist (i.e. on a tampered bundle); on a clean public bundle
    they default to 0 and the model can discard the column.
    """
    opps = tables.get("opportunities")
    customers = tables.get("customers")
    subscriptions = tables.get("subscriptions")

    feats = leads[["lead_id"]].copy()

    if opps is not None and len(opps) > 0:
        agg: dict[str, tuple[str, str]] = {"n_opps": ("opportunity_id", "count")}
        if "estimated_acv" in opps.columns:
            agg["max_acv"] = ("estimated_acv", "max")
            agg["mean_acv"] = ("estimated_acv", "mean")
        opp_agg = opps.groupby("lead_id").agg(**agg).reset_index()
        feats = feats.merge(opp_agg, on="lead_id", how="left")
        opp_to_lead = dict(zip(opps["opportunity_id"], opps["lead_id"], strict=False))
    else:
        opp_to_lead = {}

    if customers is not None and len(customers) > 0:
        cust = customers.copy()
        cust["lead_id"] = cust["opportunity_id"].map(opp_to_lead)
        cust_agg = cust.groupby("lead_id").size().rename("n_customers").reset_index()
        feats = feats.merge(cust_agg, on="lead_id", how="left")
        cust_to_opp = dict(zip(customers["customer_id"], customers["opportunity_id"], strict=False))
    else:
        cust_to_opp = {}

    if subscriptions is not None and len(subscriptions) > 0:
        subs = subscriptions.copy()
        subs["opportunity_id"] = subs["customer_id"].map(cust_to_opp)
        subs["lead_id"] = subs["opportunity_id"].map(opp_to_lead)
        sub_agg = subs.groupby("lead_id").size().rename("n_subscriptions").reset_index()
        feats = feats.merge(sub_agg, on="lead_id", how="left")

    fill_defaults: dict[str, float] = {
        "n_opps": 0.0,
        "max_acv": 0.0,
        "mean_acv": 0.0,
        "n_customers": 0.0,
        "n_subscriptions": 0.0,
    }
    for col, default in fill_defaults.items():
        if col in feats.columns:
            feats[col] = feats[col].fillna(default)
        else:
            feats[col] = default

    feature_cols = list(fill_defaults.keys())
    return feats.set_index("lead_id")[feature_cols].astype(float)


__all__ = [
    "BANNED_LEAD_COLUMNS",
    "BANNED_OPP_COLUMNS",
    "BANNED_TABLES",
    "CHANNEL_BANNED_COLUMN",
    "CHANNEL_BANNED_TABLE",
    "CHANNEL_BONUS_MODEL",
    "CHANNEL_JOIN_RECONSTRUCTION",
    "CHANNEL_SNAPSHOT_WINDOW",
    "LeakageFinding",
    "LeakageReport",
    "RelationalLeakageError",
    "SNAPSHOT_FILTERED_TABLES",
    "deterministic_relational_reconstruction",
    "probe_banned_columns",
    "probe_banned_tables",
    "probe_bonus_model_auc",
    "probe_deterministic_reconstruction",
    "probe_snapshot_window",
    "run_all_probes",
    "run_all_probes_on_dataframes",
]
