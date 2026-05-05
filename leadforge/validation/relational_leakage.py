"""Probes that detect public-bundle reconstruction of ``converted_within_90_days``.

The audit in ``docs/release/v1_current_state_audit.md`` enumerates four
deterministic paths (A-E) by which alpha public bundles reconstruct the
target via joins.  The structural fix lives in
:mod:`leadforge.render.relational_snapshot_safe`; this module is the
matching validator that asserts the fix is in place on any bundle
claiming to be ``student_public``.

Five probes, each producing zero or more :class:`LeakageFinding`
instances:

* :func:`probe_banned_columns` — public ``leads`` and ``opportunities``
  tables must not contain :data:`~leadforge.render.relational_snapshot_safe.BANNED_LEAD_COLUMNS`
  or :data:`~leadforge.render.relational_snapshot_safe.BANNED_OPP_COLUMNS`
  respectively.
* :func:`probe_banned_tables` — public bundles must not include the
  conversion-conditional tables ``customers`` or ``subscriptions``.
* :func:`probe_deterministic_reconstruction` — paths B / C / D from the
  audit must produce zero positive predictions.
* :func:`probe_snapshot_window` — every event-table row must satisfy
  ``timestamp <= lead_created_at + snapshot_day``.
* :func:`probe_bonus_model_auc` — optional honest-feature baseline:
  trains LR + HistGBM on the legitimate aggregates ``n_opps`` / ``max_acv``
  / ``mean_acv`` (plus ``n_customers`` / ``n_subscriptions`` if present)
  and asserts CV AUC stays below ``max_auc``.

:func:`run_all_probes` is the file-based orchestrator that PR 2.2 will
call from :func:`leadforge.validation.bundle_checks.validate_bundle`.
:func:`run_all_probes_on_dataframes` is the same orchestrator without
the disk read, so unit tests can exercise the probes against synthetic
bundles built in-memory.

The :func:`deterministic_relational_reconstruction` function is the
single source of truth for the join graph that defines paths A-E.  The
companion script ``scripts/probe_relational_leakage.py`` re-exports it
unchanged so the alpha-bundle audit and the validator agree by
construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from leadforge.render.relational_snapshot_safe import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
    EVENT_TABLES,
)

#: Channel labels carried on :class:`LeakageFinding.channel`.  Constants
#: rather than an enum because findings serialise straight to JSON in
#: PR 3.2's reporting layer.
CHANNEL_BANNED_COLUMN: Final[str] = "banned_column"
CHANNEL_BANNED_TABLE: Final[str] = "banned_table"
CHANNEL_DETERMINISTIC_PATH: Final[str] = "deterministic_path"
CHANNEL_SNAPSHOT_WINDOW: Final[str] = "snapshot_window"
CHANNEL_BONUS_MODEL: Final[str] = "bonus_model"

#: Default ceiling for the bonus-model AUC probe.  Honest aggregates
#: (``n_opps`` / ACV) on the v0.1.0-alpha intermediate tier produce a
#: legitimate signal in the high-0.5s under the post-fix shape, so 0.65
#: is a conservative placeholder until PR 3.3 calibrates a per-tier
#: band against measured baselines.
DEFAULT_MAX_BONUS_AUC: Final[float] = 0.65
# TODO(PR 3.3): tighten this band against a measured honest-feature baseline.

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


class RelationalLeakageError(AssertionError):
    """Raised by :meth:`LeakageReport.raise_if_failing` on any finding.

    Carries the originating :class:`LeakageReport` on ``self.report`` so
    callers (e.g. ``leadforge validate``) can render the full set of
    findings in their output rather than just the first one.
    """

    def __init__(self, report: LeakageReport) -> None:
        self.report = report
        first_lines = "\n".join(
            f"  - [{f.channel}] {f.detail}: {f.message}" for f in report.findings
        )
        super().__init__(
            f"public bundle leaks `converted_within_90_days` "
            f"({len(report.findings)} finding(s)):\n{first_lines}"
        )


# ---------------------------------------------------------------------------
# Deterministic reconstruction — the join graph that defines paths A-E.
#
# Lifted verbatim from ``scripts/probe_relational_leakage.py`` (PR 1.1) so
# the package and the script share one implementation.  The script now
# re-exports this function from here.
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
    """Public ``leads``/``opportunities`` must not carry banned columns."""
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
    """Paths B / C / D from the audit must produce zero positive predictions.

    Path A is intentionally not checked here — it is fully covered by
    :func:`probe_banned_columns` (Path A reads
    ``leads.converted_within_90_days`` directly, which is a
    :data:`BANNED_LEAD_COLUMNS` violation).
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
                    channel=CHANNEL_DETERMINISTIC_PATH,
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
    for name, ts_col in EVENT_TABLES:
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
    max_auc: float = DEFAULT_MAX_BONUS_AUC,
    seed: int = 42,
    label: pd.Series | None = None,
) -> list[LeakageFinding]:
    """5-fold CV LR + HistGBM AUC on honest relational aggregates.

    If the public bundle has been correctly redacted, ``leads`` no longer
    carries ``converted_within_90_days`` — in that case the caller must
    supply the held-back ``label`` (typically read from the task split)
    so we can score against ground truth.  When neither is available the
    probe is skipped silently (no finding, no error): there is simply no
    truth to compare against.

    Skipped (no finding) if scikit-learn is unavailable.
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
    y = y_series.reindex(features.index).astype(int)
    if y.nunique(dropna=True) < 2:
        return []

    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "hist_gbm": Pipeline([("clf", HistGradientBoostingClassifier(random_state=seed))]),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

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
                        f"5-fold CV AUC {auc_mean:.3f} on join-derived features "
                        f"exceeds max_auc={max_auc:.3f}; honest aggregates "
                        "carry stronger signal than the band allows"
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
    max_auc: float = DEFAULT_MAX_BONUS_AUC,
    label: pd.Series | None = None,
) -> LeakageReport:
    """Run every probe against an in-memory tables dict."""
    findings: list[LeakageFinding] = []
    findings += probe_banned_columns(tables)
    findings += probe_banned_tables(tables.keys())
    findings += probe_deterministic_reconstruction(tables)
    findings += probe_snapshot_window(tables, snapshot_day=snapshot_day)
    findings += probe_bonus_model_auc(tables, max_auc=max_auc, label=label)
    return LeakageReport(findings=tuple(findings))


def run_all_probes(
    bundle_dir: Path,
    *,
    snapshot_day: int,
    max_auc: float = DEFAULT_MAX_BONUS_AUC,
    label: pd.Series | None = None,
) -> LeakageReport:
    """Run every probe against ``<bundle_dir>/tables/*.parquet``.

    Args:
        bundle_dir: Bundle root (must contain ``tables/leads.parquet``).
        snapshot_day: Snapshot window for the timestamp probe.
        max_auc: Threshold for the bonus-model probe.
        label: Optional ground-truth labels to feed the bonus-model
            probe when ``leads.converted_within_90_days`` has been
            redacted.  Not loading them automatically (e.g. from the
            task splits) keeps this module independent of task layout —
            PR 2.2's wiring layer is the right place for that lookup.

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
        tables, snapshot_day=snapshot_day, max_auc=max_auc, label=label
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

    When ``label`` is supplied the caller is responsible for aligning it
    to ``lead_id`` (either as the index name or in a way that
    ``Series.reindex(features.index)`` resolves).  When it is not
    supplied we read ``leads.converted_within_90_days`` directly — this
    branch is exercised by tampered bundles in tests.
    """
    if label is not None:
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
    they default to 0 and become uninformative columns the model can
    discard.
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
    "CHANNEL_BANNED_COLUMN",
    "CHANNEL_BANNED_TABLE",
    "CHANNEL_BONUS_MODEL",
    "CHANNEL_DETERMINISTIC_PATH",
    "CHANNEL_SNAPSHOT_WINDOW",
    "DEFAULT_MAX_BONUS_AUC",
    "LeakageFinding",
    "LeakageReport",
    "RelationalLeakageError",
    "deterministic_relational_reconstruction",
    "probe_banned_columns",
    "probe_banned_tables",
    "probe_bonus_model_auc",
    "probe_deterministic_reconstruction",
    "probe_snapshot_window",
    "run_all_probes",
    "run_all_probes_on_dataframes",
]
