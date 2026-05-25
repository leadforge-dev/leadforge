"""Unified leakage taxonomy for ``leadforge-lead-scoring-v1`` validation.

Subsumes ``leadforge.validation.relational_leakage`` (PR 2.1) and broadens
it to the full taxonomy from ``docs/release/v1_release_design.md`` /
``docs/leadforge_design_doc.md``: direct, time-window, relational, split,
and model-realism.  PR 3.2's ``release_quality.py`` and PR 3.3's
``validate_release_candidate.py`` driver consume this module.

Probe families
--------------

* **Direct** — :func:`probe_banned_columns`, :func:`probe_banned_tables`.
  A published bundle must not carry the label or conversion-conditional
  artefacts.  Defaults match the snapshot-safe contract; both probes
  accept caller-supplied banned sets so future publication channels
  (Kaggle / HF / instructor companion) can declare their own.
* **Time-window** — :func:`probe_snapshot_window`.  Every event row must
  satisfy ``timestamp <= lead_created_at + horizon`` for the relevant
  per-table timestamp column.
* **Relational** — :func:`probe_deterministic_reconstruction`,
  :func:`deterministic_relational_reconstruction`.  Pure-join paths
  (B/C/D from the v1 audit) must produce zero positive predictions;
  Path A is delegated to the banned-column probe.
* **Split** — :func:`probe_split_id_overlap`,
  :func:`probe_split_near_duplicates`, :func:`probe_split_label_drift`.
  Cross-split contamination via shared IDs, near-duplicate rows, or
  drifted label rates.
* **Model realism** — :func:`probe_bonus_model_auc`,
  :func:`probe_id_only_baseline`, :func:`probe_feature_subset_baseline`.
  Calibrated baselines that flag *under-realistic* leakage (e.g. an
  ID-only model that scores well, or post-snapshot aggregates that
  saturate).  All opt-in: PR 3.3 supplies per-tier ``max_auc`` bands.

Orchestrators
-------------

* :func:`run_all_probes_on_dataframes` / :func:`run_all_probes` — the
  bundle-level structural orchestrator that PR 2.2 wires into
  ``validate_bundle``.  Skips opt-in probes unless explicitly asked.
* :func:`run_split_probes` — split-level orchestrator over a
  ``{split_name: DataFrame}`` mapping.  PR 3.2/3.3 will plumb this
  through the release-quality driver.

Probe registry
--------------

:data:`PROBE_REGISTRY` maps every probe name to its taxonomy and input
needs.  The orchestrators iterate it; the meta-test
``test_probe_registry_covers_every_module_level_probe`` enforces that any new ``probe_*``
function is registered, so a future "I added a probe but forgot to wire it"
regression fails loudly rather than silently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from leadforge.core.exceptions import LeadforgeError

# ---------------------------------------------------------------------------
# Snapshot-safe contract — single source of truth for "what is leakage".
# ``leadforge.render.relational_snapshot_safe`` (writer) and
# ``leadforge.render.manifests`` (manifest's structural_redactions) import
# from here so the writer and the validator share one definition.
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

#: Tables omitted from public bundles entirely.  Conversion-conditional —
#: their mere presence reconstructs the label.
BANNED_TABLES: Final[tuple[str, ...]] = ("customers", "subscriptions")

#: Default banned-columns map for the snapshot-safe contract, suitable as
#: the ``banned`` argument to :func:`probe_banned_columns`.
DEFAULT_BANNED_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "leads": BANNED_LEAD_COLUMNS,
    "opportunities": BANNED_OPP_COLUMNS,
}

#: Tables filtered per-lead by their timestamp column to
#: ``lead_created_at + snapshot_day``.  ``opportunities`` is included
#: even though it is an entity table, because its ``created_at`` anchors
#: when the entity becomes observable in the funnel.
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
CHANNEL_SPLIT_ID_OVERLAP: Final[str] = "split_id_overlap"
CHANNEL_SPLIT_NEAR_DUPLICATE: Final[str] = "split_near_duplicate"
CHANNEL_SPLIT_LABEL_DRIFT: Final[str] = "split_label_drift"
CHANNEL_ID_ONLY_BASELINE: Final[str] = "id_only_baseline"
CHANNEL_FEATURE_SUBSET_BASELINE: Final[str] = "feature_subset_baseline"
CHANNEL_OPPORTUNITY_SNAPSHOT_CONSISTENCY: Final[str] = "opportunity_snapshot_consistency"

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

    The name is retained from PR 2.1 for backward compatibility — the
    error class now spans every taxonomy in this module, not just
    relational join reconstruction.  Carries the originating
    :class:`LeakageReport` on ``self.report`` so callers (e.g.
    ``leadforge validate``) can render the full set of findings rather
    than just the first one.
    """

    def __init__(self, report: LeakageReport) -> None:
        self.report = report
        rendered = "\n".join(f"  - [{f.channel}] {f.detail}: {f.message}" for f in report.findings)
        super().__init__(
            f"leakage probe(s) failed ({len(report.findings)} finding(s)):\n{rendered}"
        )


# ---------------------------------------------------------------------------
# Deterministic reconstruction — the join graph that defines paths A-E.
# ``scripts/probe_relational_leakage.py`` re-exports this function so the
# alpha-bundle audit and the validator agree by construction.
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
# §8.1 Direct leakage — banned columns / banned tables.
# ---------------------------------------------------------------------------


def probe_banned_columns(
    tables: Mapping[str, pd.DataFrame],
    *,
    banned: Mapping[str, Iterable[str]] | None = None,
) -> list[LeakageFinding]:
    """Public tables must not carry caller-banned columns.

    Args:
        tables: Mapping of table name → DataFrame.
        banned: Mapping of table name → banned column names.  Defaults to
            :data:`DEFAULT_BANNED_COLUMNS` (the snapshot-safe contract:
            ``leads`` drops :data:`BANNED_LEAD_COLUMNS`, ``opportunities``
            drops :data:`BANNED_OPP_COLUMNS`).  Pass an explicit mapping
            to widen the contract for non-relational publication channels
            (e.g. flat-CSV exports with their own redaction list).

    Detects Path A (label column directly readable from ``leads``) and the
    ``opportunities.close_outcome`` / ``closed_at`` channels — i.e. leakage
    that any caller can spot by listing column names, no joins required.
    """
    spec = banned if banned is not None else DEFAULT_BANNED_COLUMNS
    findings: list[LeakageFinding] = []
    for table_name, banned_cols in spec.items():
        df = tables.get(table_name)
        if df is None:
            continue
        for col in banned_cols:
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


def probe_banned_tables(
    table_names: Iterable[str],
    *,
    banned: Iterable[str] | None = None,
) -> list[LeakageFinding]:
    """Public bundles must not include caller-banned tables.

    Args:
        table_names: Names of tables present in the bundle.
        banned: Iterable of banned table names.  Defaults to
            :data:`BANNED_TABLES` (the conversion-conditional
            ``customers`` / ``subscriptions``).
    """
    banned_set = tuple(banned) if banned is not None else BANNED_TABLES
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
        for name in banned_set
        if name in present
    ]


# ---------------------------------------------------------------------------
# §8.2 Time-window leakage — events past the snapshot anchor.
# ---------------------------------------------------------------------------


def probe_snapshot_window(
    tables: Mapping[str, pd.DataFrame],
    snapshot_day: int,
    *,
    filtered_tables: Iterable[tuple[str, str]] | None = None,
) -> list[LeakageFinding]:
    """Every event-table row must satisfy ``timestamp <= lead_created_at + snapshot_day``.

    Args:
        tables: Mapping of table name → DataFrame.
        snapshot_day: Number of days after ``lead_created_at`` beyond
            which event rows are forbidden.  Negative values raise.
        filtered_tables: Iterable of ``(table_name, timestamp_column)``
            pairs to audit.  Defaults to :data:`SNAPSHOT_FILTERED_TABLES`.
            Pass an explicit list to widen the contract (e.g. for new
            event tables added by a future task).
    """
    if snapshot_day < 0:
        raise ValueError(f"snapshot_day must be non-negative, got {snapshot_day}")
    pairs = tuple(filtered_tables) if filtered_tables is not None else SNAPSHOT_FILTERED_TABLES
    leads = tables.get("leads")
    if leads is None or len(leads) == 0:
        return []
    if "lead_id" not in leads.columns or "lead_created_at" not in leads.columns:
        raise ValueError("leads must contain 'lead_id' and 'lead_created_at' columns")
    if not leads["lead_id"].is_unique:
        raise ValueError("leads.lead_id must be unique")

    anchor = leads[["lead_id", "lead_created_at"]].copy()
    anchor["lead_created_at"] = pd.to_datetime(anchor["lead_created_at"], errors="coerce")
    # NaT in the anchor would propagate to NaT cutoffs, then ``ts > NaT``
    # is NaN, and the violation count's ``fillna(False)`` would silently
    # drop those rows — masking a data-quality bug in the bundle.  Refuse
    # to operate on a malformed anchor, same posture as the duplicate-
    # lead_id check above.
    nat_mask = anchor["lead_created_at"].isna()
    if nat_mask.any():
        sample = anchor.loc[nat_mask, "lead_id"].head(5).tolist()
        raise ValueError(
            f"leads.lead_created_at has {int(nat_mask.sum())} unparseable / null "
            f"value(s); offending lead_id sample: {sample}"
        )
    horizon = pd.Timedelta(days=snapshot_day)

    findings: list[LeakageFinding] = []
    for name, ts_col in pairs:
        df = tables.get(name)
        if df is None or len(df) == 0 or ts_col not in df.columns:
            continue
        merged = df[["lead_id", ts_col]].merge(anchor, on="lead_id", how="left")
        # An event row whose lead_id has no match in leads gets NaT for
        # ``lead_created_at`` after the left-merge; that row's cutoff is
        # NaT and the violation count would silently miss it.  An orphan
        # event row is a structural FK violation (and a leakage attack
        # surface — a tampered bundle could insert post-snapshot events
        # tied to lead_ids absent from the public leads table).  Refuse
        # to bless it.
        orphan_mask = merged["lead_created_at"].isna()
        if orphan_mask.any():
            sample = merged.loc[orphan_mask, "lead_id"].head(5).tolist()
            raise ValueError(
                f"{name}.parquet has {int(orphan_mask.sum())} row(s) referencing "
                f"lead_id(s) absent from leads; sample: {sample}"
            )
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


# ---------------------------------------------------------------------------
# §8.3 Relational leakage — deterministic join paths.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# §8.4 Split leakage — ID overlap / near-duplicates / label drift across
# train / valid / test splits.
# ---------------------------------------------------------------------------


def probe_split_id_overlap(
    splits: Mapping[str, pd.DataFrame],
    *,
    id_columns: Iterable[str] = ("lead_id",),
) -> list[LeakageFinding]:
    """Report any value of ``id_columns`` appearing in more than one split.

    For ``lead_id`` (the default) any overlap is a hard contamination
    finding — each lead is one row, so a clean random-split task should
    never duplicate a lead_id across splits.

    For ``account_id`` / ``contact_id`` overlap is informational at this
    layer: G6.1 / G6.2 in ``v1_acceptance_gates.md`` say it must be
    *documented as intentional or absent*.  PR 3.3 will set whether the
    finding is a release blocker; this probe just surfaces it.

    Splits with no ``id_columns`` present are skipped (e.g. a flat task
    table without ``account_id``).  Empty splits are also skipped.
    """
    findings: list[LeakageFinding] = []
    split_names = list(splits.keys())
    for col in id_columns:
        # Build the union of (split_name, id_value) pairs once per column.
        per_split: dict[str, set[Any]] = {}
        for name, df in splits.items():
            if col not in df.columns or len(df) == 0:
                continue
            per_split[name] = set(df[col].dropna().tolist())
        # Pairwise overlap across splits (deduplicated by ordering names).
        for i, a in enumerate(split_names):
            if a not in per_split:
                continue
            for b in split_names[i + 1 :]:
                if b not in per_split:
                    continue
                shared = per_split[a] & per_split[b]
                if not shared:
                    continue
                # Sort the full overlap before slicing — set iteration order
                # is implementation-defined, so ``list(shared)[:5]`` would
                # yield non-reproducible sample messages across runs.
                sample = sorted(str(s) for s in shared)[:5]
                findings.append(
                    LeakageFinding(
                        channel=CHANNEL_SPLIT_ID_OVERLAP,
                        detail=f"{col}:{a}∩{b}",
                        message=(
                            f"{len(shared)} {col} value(s) appear in both "
                            f"'{a}' and '{b}' splits; sample={sample}"
                        ),
                    )
                )
    return findings


def probe_split_near_duplicates(
    splits: Mapping[str, pd.DataFrame],
    *,
    feature_columns: Iterable[str],
    decimals: int = 4,
    max_findings: int = 5,
) -> list[LeakageFinding]:
    """Detect rows in different splits that match after rounding numeric features.

    Pragmatic, deterministic, no-sklearn-needed approximation of
    cosine-similarity ≈ 1 near-duplicate detection: round each numeric
    feature in ``feature_columns`` to ``decimals`` places, stringify the
    rounded vector, and look for collisions across splits.  Catches the
    common cases:

    * exact duplicates of the same record landing in two splits;
    * records whose features are indistinguishable within sensible
      numeric precision (e.g. two leads with identical aggregates).

    The probe deliberately under-reports rather than over-reports —
    cosine-similarity machinery would flag spurious near-duplicates on
    sparse one-hot-encoded data and flake under sklearn version drift.
    Rounded-vector hashing is reproducible and orthogonal to model
    choice.

    Skips silently (empty findings) when ``feature_columns`` is empty,
    every split is empty, or every requested column is non-numeric or
    all-NaN after coercion.  Caller-provided columns missing from a
    split are ignored on a per-split basis.

    Rows whose rounded signature is entirely ``"nan"`` after coercion
    (e.g. all-non-numeric inputs, or rows with missing values for every
    requested feature) are excluded from the comparison — they carry no
    information and would otherwise collide as a single saturating
    false-positive across splits.
    """
    cols = list(feature_columns)
    if not cols:
        return []

    rounded: dict[str, pd.Series] = {}
    for name, df in splits.items():
        if len(df) == 0:
            continue
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        # Coerce to numeric; non-numeric columns become NaN.  We then
        # drop rows whose entire signature is NaN — those carry no
        # information and would otherwise saturate as a single
        # collision across splits, which is a false positive (not a
        # near-duplicate).
        numeric = df[present].apply(pd.to_numeric, errors="coerce").round(decimals)
        non_empty = numeric.notna().any(axis=1)
        if not non_empty.any():
            continue
        rounded[name] = numeric.loc[non_empty].astype(str).agg("|".join, axis=1)

    findings: list[LeakageFinding] = []
    split_names = list(rounded.keys())
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            shared = set(rounded[a]) & set(rounded[b])
            if not shared:
                continue
            # Sort the full overlap before slicing — set iteration order
            # is implementation-defined, so ``list(shared)[:N]`` would
            # yield non-reproducible sample messages across runs.
            sample = sorted(shared)[:max_findings]
            findings.append(
                LeakageFinding(
                    channel=CHANNEL_SPLIT_NEAR_DUPLICATE,
                    detail=f"{a}∩{b}",
                    message=(
                        f"{len(shared)} row signature(s) (numeric features rounded "
                        f"to {decimals} dp) match between '{a}' and '{b}' splits; "
                        f"sample={sample}"
                    ),
                )
            )
    return findings


def probe_split_label_drift(
    splits: Mapping[str, pd.DataFrame],
    *,
    label_col: str,
    max_drift: float,
) -> list[LeakageFinding]:
    """Per-split positive rate must not drift beyond ``max_drift`` between any two splits.

    Drift is measured as the absolute difference of mean(label) across
    each pair of splits.  A drifted positive rate signals a non-IID
    split (e.g. a time-cohort split that wasn't rebalanced) — useful
    info for the release-quality report; opt-in because cohort splits
    are *intentionally* drifted.

    Splits without ``label_col`` are skipped on a per-split basis.
    Skips silently when fewer than two splits carry the label.
    """
    if max_drift < 0:
        raise ValueError(f"max_drift must be non-negative, got {max_drift}")
    rates: dict[str, float] = {}
    for name, df in splits.items():
        if label_col not in df.columns or len(df) == 0:
            continue
        rates[name] = float(df[label_col].astype("boolean").fillna(False).astype(int).mean())
    if len(rates) < 2:
        return []

    findings: list[LeakageFinding] = []
    names = list(rates.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            drift = abs(rates[a] - rates[b])
            if drift > max_drift:
                findings.append(
                    LeakageFinding(
                        channel=CHANNEL_SPLIT_LABEL_DRIFT,
                        detail=f"{a}↔{b}",
                        message=(
                            f"|rate({a}) - rate({b})| = "
                            f"|{rates[a]:.3f} - {rates[b]:.3f}| = {drift:.3f} "
                            f"exceeds max_drift={max_drift:.3f}"
                        ),
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# §8.5 Model-realism — calibrated baselines.
# All opt-in: PR 3.3 supplies per-tier ``max_auc`` bands.
# ---------------------------------------------------------------------------


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
    sk = _import_sklearn()
    if sk is None:
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

    models: dict[str, Any] = {
        "logistic_regression": sk.Pipeline(
            [("scaler", sk.StandardScaler()), ("clf", sk.LogisticRegression(max_iter=1000))]
        ),
        "hist_gbm": sk.Pipeline([("clf", sk.HistGradientBoostingClassifier(random_state=seed))]),
    }
    skf = sk.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    findings: list[LeakageFinding] = []
    for name, pipe in models.items():
        aucs: list[float] = []
        for train_idx, test_idx in skf.split(features.values, y.values):
            x_tr, x_te = features.values[train_idx], features.values[test_idx]
            y_tr, y_te = y.values[train_idx], y.values[test_idx]
            pipe.fit(x_tr, y_tr)
            proba = pipe.predict_proba(x_te)[:, 1]
            aucs.append(float(sk.roc_auc_score(y_te, proba)))
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


def probe_id_only_baseline(
    splits: Mapping[str, pd.DataFrame],
    *,
    label_col: str,
    max_auc: float,
    id_columns: Iterable[str] = ("lead_id", "account_id", "contact_id"),
    seed: int = 42,
) -> list[LeakageFinding]:
    """Opt-in: a model trained on IDs alone must not predict the label.

    A passing baseline (mean test AUC ≤ ``max_auc``, expected near 0.5)
    confirms IDs carry no signal.  A failing baseline reveals
    ID-encoded leakage (e.g. IDs reflecting creation order correlated
    with conversion, or IDs leaking through the ordinal hash of an
    upstream sort key).

    Trains on the ``train`` split, evaluates on each of ``valid`` and
    ``test`` if present.  Splits provide the train/test boundary, so no
    cross-validation here — split contamination is the next concern up
    and a separate probe.

    Skips silently when sklearn is unavailable, when ``train`` is
    missing, when no ``id_columns`` are present in the train split,
    when the train label has fewer than two classes, or when no
    evaluation split is available.
    """
    sk = _import_sklearn()
    if sk is None:
        return []

    train = splits.get("train")
    if train is None or len(train) == 0 or label_col not in train.columns:
        return []
    cols = [c for c in id_columns if c in train.columns]
    if not cols:
        return []
    y_train = train[label_col].astype("boolean").fillna(False).astype(int)
    if y_train.nunique() < 2:
        return []

    # Hash the IDs to integers — the model needs numeric input but we
    # don't want to one-hot every ID (would explode dimension and would
    # *guarantee* AUC near 1.0 on the train split via memorisation).
    # The hash is deterministic across runs and small enough for HistGBM.
    x_train = _hash_id_columns(train[cols])

    eval_splits = {
        name: df
        for name, df in splits.items()
        if name != "train"
        and len(df) > 0
        and label_col in df.columns
        and all(c in df.columns for c in cols)
    }
    if not eval_splits:
        return []

    model = sk.HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    model.fit(x_train.values, y_train.values)

    findings: list[LeakageFinding] = []
    for name, df in eval_splits.items():
        y_eval = df[label_col].astype("boolean").fillna(False).astype(int)
        if y_eval.nunique() < 2:
            continue
        x_eval = _hash_id_columns(df[cols])
        proba = model.predict_proba(x_eval.values)[:, 1]
        auc = float(sk.roc_auc_score(y_eval.values, proba))
        if auc > max_auc:
            findings.append(
                LeakageFinding(
                    channel=CHANNEL_ID_ONLY_BASELINE,
                    detail=f"split={name},cols={','.join(cols)}",
                    message=(
                        f"HistGBM trained on hashed {cols} alone reaches "
                        f"AUC {auc:.3f} on '{name}' (max_auc={max_auc:.3f}); "
                        "IDs carry signal they should not"
                    ),
                )
            )
    return findings


def probe_feature_subset_baseline(
    splits: Mapping[str, pd.DataFrame],
    *,
    feature_columns: Iterable[str],
    label_col: str,
    max_auc: float,
    name: str = "subset",
    seed: int = 42,
) -> list[LeakageFinding]:
    """Opt-in: a model trained on a feature subset must not predict the label above ``max_auc``.

    The umbrella probe behind G5.1 (post-snapshot aggregates) and G5.2
    (suspect-stage columns).  Caller declares the suspect subset; a
    mean-evaluation-AUC > ``max_auc`` is a finding.

    Train on ``train``, evaluate on each non-``train`` split present.
    All numeric coercion is via :func:`pandas.to_numeric` with
    ``errors="coerce"`` — non-numeric / missing values become NaN and
    are passed straight to HistGBM (which handles NaN natively); LR
    would not, so we use HistGBM only.

    Skips silently when sklearn is unavailable, when ``train`` is
    missing or has < 2 classes, when no requested columns are present,
    or when no evaluation split is available.
    """
    sk = _import_sklearn()
    if sk is None:
        return []

    train = splits.get("train")
    if train is None or len(train) == 0 or label_col not in train.columns:
        return []
    cols = [c for c in feature_columns if c in train.columns]
    if not cols:
        return []
    y_train = train[label_col].astype("boolean").fillna(False).astype(int)
    if y_train.nunique() < 2:
        return []

    x_train = train[cols].apply(pd.to_numeric, errors="coerce")

    eval_splits = {
        eval_name: df
        for eval_name, df in splits.items()
        if eval_name != "train"
        and len(df) > 0
        and label_col in df.columns
        and all(c in df.columns for c in cols)
    }
    if not eval_splits:
        return []

    model = sk.HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    model.fit(x_train.values, y_train.values)

    findings: list[LeakageFinding] = []
    for eval_name, df in eval_splits.items():
        y_eval = df[label_col].astype("boolean").fillna(False).astype(int)
        if y_eval.nunique() < 2:
            continue
        x_eval = df[cols].apply(pd.to_numeric, errors="coerce")
        proba = model.predict_proba(x_eval.values)[:, 1]
        auc = float(sk.roc_auc_score(y_eval.values, proba))
        if auc > max_auc:
            findings.append(
                LeakageFinding(
                    channel=CHANNEL_FEATURE_SUBSET_BASELINE,
                    detail=f"name={name},split={eval_name}",
                    message=(
                        f"HistGBM trained on '{name}' subset ({len(cols)} cols) "
                        f"reaches AUC {auc:.3f} on '{eval_name}' "
                        f"(max_auc={max_auc:.3f})"
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# §8.6 Flat-feature snapshot-consistency — opportunity-derived columns.
# Opt-in: requires both the flat leads snapshot AND the full-horizon
# opportunities table, so it can only run when both are available.
# ---------------------------------------------------------------------------


def probe_opportunity_snapshot_consistency(
    leads_df: pd.DataFrame,
    opportunities_df: pd.DataFrame,
    snapshot_day: int,
) -> list[LeakageFinding]:
    """Verify that ``has_open_opportunity`` / ``opportunity_estimated_acv`` use correct semantics.

    Recomputes what these columns *should* be using the correct
    ``closed_at > cutoff`` semantics (an opportunity is open at snapshot time
    iff it was not yet closed as of ``lead_created_at + snapshot_day``), then
    asserts that the shipped column values match.

    A mismatch indicates that the snapshot builder used the wrong open/closed
    gate — e.g. checking ``close_outcome.isna()`` instead of
    ``closed_at > cutoff``, which treats opportunities closed after the snapshot
    window as already closed at snapshot time and artificially reduces the
    ``has_open_opportunity`` positive rate.

    Opt-in: this probe requires both the flat leads snapshot (with
    ``has_open_opportunity``, ``opportunity_estimated_acv``, and
    ``lead_created_at``) and the full-horizon ``opportunities`` table (with
    ``closed_at`` and ``estimated_acv``).  Callers that only have public
    bundles (where ``close_outcome`` / ``closed_at`` are banned) cannot run
    this probe — pass the instructor table.

    Args:
        leads_df: Flat snapshot DataFrame.  Must contain ``lead_id``,
            ``lead_created_at``, ``has_open_opportunity``, and
            ``opportunity_estimated_acv``.
        opportunities_df: Full-horizon opportunities table.  Must contain
            ``lead_id``, ``created_at``, ``estimated_acv``, and ``closed_at``.
        snapshot_day: Snapshot window in days.

    Returns:
        Empty list if the columns match; one :class:`LeakageFinding` per
        mismatched column otherwise.
    """
    required_leads = {"lead_id", "lead_created_at", "has_open_opportunity"}
    missing_leads = required_leads - set(leads_df.columns)
    if missing_leads:
        raise ValueError(f"leads_df is missing required columns: {sorted(missing_leads)}")

    required_opps = {"lead_id", "created_at", "estimated_acv", "closed_at"}
    missing_opps = required_opps - set(opportunities_df.columns)
    if missing_opps:
        raise ValueError(
            f"opportunities_df is missing required columns: {sorted(missing_opps)}; "
            "pass the full-horizon (instructor) opportunities table, not the public one."
        )

    if len(leads_df) == 0:
        return []

    # Build per-lead cutoffs.
    leads_copy = leads_df[["lead_id", "lead_created_at", "has_open_opportunity"]].copy()
    if "opportunity_estimated_acv" in leads_df.columns:
        leads_copy["opportunity_estimated_acv"] = leads_df["opportunity_estimated_acv"]
    else:
        leads_copy["opportunity_estimated_acv"] = float("nan")

    leads_copy["_created_at_ts"] = pd.to_datetime(leads_copy["lead_created_at"], errors="coerce")
    leads_copy["_snapshot_cutoff"] = leads_copy["_created_at_ts"] + pd.Timedelta(days=snapshot_day)

    # Filter opportunities to those created on/before the snapshot cutoff.
    opps = opportunities_df[["lead_id", "created_at", "estimated_acv", "closed_at"]].copy()
    opps["_opp_created_ts"] = pd.to_datetime(opps["created_at"], errors="coerce")
    opps["_closed_at_ts"] = pd.to_datetime(opps["closed_at"], errors="coerce")

    # Merge cutoffs onto opportunities.
    cutoffs = leads_copy[["lead_id", "_snapshot_cutoff"]]
    opps = opps.merge(cutoffs, on="lead_id", how="left")

    # Keep only opportunities visible at snapshot time.
    opps_at_snapshot = opps[opps["_opp_created_ts"] <= opps["_snapshot_cutoff"]]

    # An opportunity is open at snapshot time iff closed_at is NaT or > cutoff.
    opps_open = opps_at_snapshot[
        opps_at_snapshot["_closed_at_ts"].isna()
        | (opps_at_snapshot["_closed_at_ts"] > opps_at_snapshot["_snapshot_cutoff"])
    ]

    # Compute expected has_open_opportunity and opportunity_estimated_acv per lead.
    # Sort by created_at before groupby so first() is deterministic when a lead
    # has multiple open opportunities — matches the production sort in snapshots.py.
    expected_open = (
        opps_open.sort_values("created_at")
        .groupby("lead_id")["estimated_acv"]
        .first()
        .reset_index()
        .rename(columns={"estimated_acv": "_expected_acv"})
    )
    expected_open["_expected_has_open"] = True

    # Merge expected values back onto leads.
    check = leads_copy.merge(expected_open, on="lead_id", how="left")
    check["_expected_has_open"] = check["_expected_has_open"].fillna(False)

    # Compare has_open_opportunity.
    shipped_open = check["has_open_opportunity"].astype("boolean").fillna(False).astype(bool)
    expected_open_bool = check["_expected_has_open"].astype(bool)
    mismatch_open = (shipped_open != expected_open_bool).sum()

    # Compare opportunity_estimated_acv (both NaN → match; only one NaN → mismatch).
    shipped_acv = pd.to_numeric(check["opportunity_estimated_acv"], errors="coerce")
    expected_acv_series = pd.to_numeric(check["_expected_acv"], errors="coerce")
    both_nan = shipped_acv.isna() & expected_acv_series.isna()
    neither_nan = shipped_acv.notna() & expected_acv_series.notna()
    acv_match = both_nan | (neither_nan & (shipped_acv == expected_acv_series))
    mismatch_acv = int((~acv_match).sum())

    findings: list[LeakageFinding] = []
    if mismatch_open > 0:
        n = len(check)
        findings.append(
            LeakageFinding(
                channel=CHANNEL_OPPORTUNITY_SNAPSHOT_CONSISTENCY,
                detail="has_open_opportunity",
                message=(
                    f"{mismatch_open}/{n} leads have incorrect has_open_opportunity; "
                    "the snapshot builder likely used close_outcome.isna() instead of "
                    "closed_at > lead_created_at + snapshot_day"
                ),
            )
        )
    if mismatch_acv > 0:
        n = len(check)
        findings.append(
            LeakageFinding(
                channel=CHANNEL_OPPORTUNITY_SNAPSHOT_CONSISTENCY,
                detail="opportunity_estimated_acv",
                message=(
                    f"{mismatch_acv}/{n} leads have incorrect opportunity_estimated_acv; "
                    "the shipped ACV does not match the recomputed value under correct "
                    "closed_at > cutoff semantics"
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Probe registry — single source of truth for "what probes exist and what
# do they need".  The orchestrators iterate it; the meta-test asserts that
# every module-level ``probe_*`` function is registered.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeSpec:
    """Metadata for one probe.

    Attributes:
        name: Bare function name (matches the registered key).
        callable: The probe function itself.
        taxonomy: One of ``"direct"``, ``"time_window"``, ``"relational"``,
            ``"split"``, ``"model_realism"``.
        opt_in: True iff the probe needs caller-supplied calibrated
            thresholds (currently every model-realism probe and the
            label-drift split probe).
    """

    name: str
    callable: Callable[..., list[LeakageFinding]]
    taxonomy: str
    opt_in: bool


PROBE_REGISTRY: Final[Mapping[str, ProbeSpec]] = {
    "banned_columns": ProbeSpec("banned_columns", probe_banned_columns, "direct", opt_in=False),
    "banned_tables": ProbeSpec("banned_tables", probe_banned_tables, "direct", opt_in=False),
    "snapshot_window": ProbeSpec(
        "snapshot_window", probe_snapshot_window, "time_window", opt_in=False
    ),
    "deterministic_reconstruction": ProbeSpec(
        "deterministic_reconstruction",
        probe_deterministic_reconstruction,
        "relational",
        opt_in=False,
    ),
    "split_id_overlap": ProbeSpec(
        "split_id_overlap", probe_split_id_overlap, "split", opt_in=False
    ),
    "split_near_duplicates": ProbeSpec(
        "split_near_duplicates", probe_split_near_duplicates, "split", opt_in=False
    ),
    "split_label_drift": ProbeSpec(
        "split_label_drift", probe_split_label_drift, "split", opt_in=True
    ),
    "bonus_model_auc": ProbeSpec(
        "bonus_model_auc", probe_bonus_model_auc, "model_realism", opt_in=True
    ),
    "id_only_baseline": ProbeSpec(
        "id_only_baseline", probe_id_only_baseline, "model_realism", opt_in=True
    ),
    "feature_subset_baseline": ProbeSpec(
        "feature_subset_baseline",
        probe_feature_subset_baseline,
        "model_realism",
        opt_in=True,
    ),
    "opportunity_snapshot_consistency": ProbeSpec(
        "opportunity_snapshot_consistency",
        probe_opportunity_snapshot_consistency,
        "snapshot_consistency",
        opt_in=True,  # Requires full-horizon opportunities table (not public bundle)
    ),
}


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
    """Run every structural relational/time-window probe; bonus probe iff threshold given.

    This is the bundle-level orchestrator wired into ``validate_bundle``
    via :func:`run_all_probes`.  Split-level probes have their own
    orchestrator (:func:`run_split_probes`) because they consume the
    task-split files, not the relational ``tables/`` dict.
    """
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


def run_split_probes(
    splits: Mapping[str, pd.DataFrame],
    *,
    label_col: str = "converted_within_90_days",
    id_columns: Iterable[str] = ("lead_id",),
    near_duplicate_columns: Iterable[str] | None = None,
    near_duplicate_decimals: int = 4,
    label_drift_max: float | None = None,
    id_only_max_auc: float | None = None,
    id_only_columns: Iterable[str] = ("lead_id", "account_id", "contact_id"),
    feature_subsets: Mapping[str, tuple[float, Iterable[str]]] | None = None,
) -> LeakageReport:
    """Run split-level leakage probes over a ``{split_name: DataFrame}`` mapping.

    Args:
        splits: Mapping of split name (``train`` / ``valid`` / ``test``)
            to the corresponding DataFrame.  Empty splits are skipped.
        label_col: Label column name.  Defaults to the v1 task target.
        id_columns: ID columns audited for cross-split overlap.  Defaults
            to ``("lead_id",)``.
        near_duplicate_columns: Numeric feature columns to use for
            near-duplicate detection.  ``None`` (default) skips the
            probe entirely.
        near_duplicate_decimals: Rounding precision for the near-
            duplicate signature (see :func:`probe_split_near_duplicates`).
        label_drift_max: Pass a positive float to enable
            :func:`probe_split_label_drift`.  ``None`` skips it.
        id_only_max_auc: Pass a numeric threshold to enable
            :func:`probe_id_only_baseline`.  ``None`` skips it.
        id_only_columns: ID columns to feed the ID-only baseline.
        feature_subsets: Optional mapping ``name → (max_auc, columns)``.
            For each entry, runs :func:`probe_feature_subset_baseline`
            with the given subset.  Used by PR 3.3 to wire up the
            post-snapshot-aggregate / suspect-stage / etc. baselines.
    """
    findings: list[LeakageFinding] = []
    findings += probe_split_id_overlap(splits, id_columns=id_columns)
    if near_duplicate_columns is not None:
        findings += probe_split_near_duplicates(
            splits,
            feature_columns=near_duplicate_columns,
            decimals=near_duplicate_decimals,
        )
    if label_drift_max is not None:
        findings += probe_split_label_drift(splits, label_col=label_col, max_drift=label_drift_max)
    if id_only_max_auc is not None:
        findings += probe_id_only_baseline(
            splits,
            label_col=label_col,
            max_auc=id_only_max_auc,
            id_columns=id_only_columns,
        )
    if feature_subsets:
        for subset_name, (max_auc, cols) in feature_subsets.items():
            findings += probe_feature_subset_baseline(
                splits,
                feature_columns=cols,
                label_col=label_col,
                max_auc=max_auc,
                name=subset_name,
            )
    return LeakageReport(findings=tuple(findings))


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


def _hash_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map opaque ID strings to deterministic 32-bit hashes (stored as int64) per column.

    HistGBM handles integer features natively; one-hot encoding every
    distinct ID would explode dimension and let the model memorise the
    train split (guaranteeing AUC ~ 1.0 on train and ~ 0.5 elsewhere,
    which is *not* the leakage signal we want to surface).  Hashing
    keeps the same train/eval semantics as the production model
    pipeline while bounding feature width.
    """
    import hashlib

    def _h(value: object) -> int:
        # Fixed-output stable hash; ``hash()`` is salted per process.
        digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big", signed=False)

    return pd.DataFrame({col: df[col].map(_h).astype("int64") for col in df.columns})


@dataclass(frozen=True)
class _SklearnHandles:
    Pipeline: Any
    StandardScaler: Any
    LogisticRegression: Any
    HistGradientBoostingClassifier: Any
    StratifiedKFold: Any
    roc_auc_score: Any


def _import_sklearn() -> _SklearnHandles | None:
    """Lazy import for sklearn; returns ``None`` if not installed.

    Centralised so every model-realism probe agrees on which symbols it
    needs and so the skip-cleanly-without-sklearn behaviour is uniform.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    return _SklearnHandles(
        Pipeline=Pipeline,
        StandardScaler=StandardScaler,
        LogisticRegression=LogisticRegression,
        HistGradientBoostingClassifier=HistGradientBoostingClassifier,
        StratifiedKFold=StratifiedKFold,
        roc_auc_score=roc_auc_score,
    )


__all__ = [
    "BANNED_LEAD_COLUMNS",
    "BANNED_OPP_COLUMNS",
    "BANNED_TABLES",
    "CHANNEL_BANNED_COLUMN",
    "CHANNEL_BANNED_TABLE",
    "CHANNEL_BONUS_MODEL",
    "CHANNEL_FEATURE_SUBSET_BASELINE",
    "CHANNEL_ID_ONLY_BASELINE",
    "CHANNEL_JOIN_RECONSTRUCTION",
    "CHANNEL_OPPORTUNITY_SNAPSHOT_CONSISTENCY",
    "CHANNEL_SNAPSHOT_WINDOW",
    "CHANNEL_SPLIT_ID_OVERLAP",
    "CHANNEL_SPLIT_LABEL_DRIFT",
    "CHANNEL_SPLIT_NEAR_DUPLICATE",
    "DEFAULT_BANNED_COLUMNS",
    "LeakageFinding",
    "LeakageReport",
    "PROBE_REGISTRY",
    "ProbeSpec",
    "RelationalLeakageError",
    "SNAPSHOT_FILTERED_TABLES",
    "deterministic_relational_reconstruction",
    "probe_banned_columns",
    "probe_banned_tables",
    "probe_bonus_model_auc",
    "probe_deterministic_reconstruction",
    "probe_feature_subset_baseline",
    "probe_id_only_baseline",
    "probe_opportunity_snapshot_consistency",
    "probe_snapshot_window",
    "probe_split_id_overlap",
    "probe_split_label_drift",
    "probe_split_near_duplicates",
    "run_all_probes",
    "run_all_probes_on_dataframes",
    "run_split_probes",
]
