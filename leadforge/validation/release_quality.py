"""Release-grade quality metrics for ``leadforge-lead-scoring-v1`` bundles.

Sits one layer above
:mod:`leadforge.validation.{realism,difficulty,drift,lead_scoring}` and
produces a single :class:`ReleaseQualityReport` covering G7.* (per-tier
performance), G8.* (cross-seed stability), and G6.4 (cohort/time-shift
degradation) of ``docs/release/v1_acceptance_gates.md``.

PR 3.2 measures and serialises; PR 3.3 calibrates per-tier band literals
in :mod:`leadforge.validation.difficulty` and gates on them.  This module
deliberately stays band-free so the same numbers feed both the JSON
report and the (future) gating layer.

Public surface
--------------

* :func:`measure_tier_from_bundle` — full metric panel for one bundle.
* :func:`measure_cohort_shift_from_bundle` — random-vs-cohort split AUC.
* :func:`regenerate_tier_for_seeds` — orchestrate cross-seed rebuilds.
* :func:`measure_release_quality` — top-level orchestrator producing the
  full :class:`ReleaseQualityReport`.

Result dataclasses (:class:`TierMetrics`, :class:`CrossSeedTierMetrics`,
:class:`CohortShiftMetrics`, :class:`CrossTierOrdering`,
:class:`ReleaseQualityReport`) are JSON-primitive end-to-end so
:func:`leadforge.validation.reporting.render_report` can ``asdict`` →
``json.dumps`` without custom encoders.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadforge.core.serialization import load_json
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Label column on the snapshot task splits.  Mirrors
#: :data:`leadforge.validation.realism._LABEL_COLUMN`; duplicated here to
#: keep this module standalone (the realism module uses a different
#: label-resolution path that goes through the bundle manifest).
LABEL_COLUMN = "converted_within_90_days"

#: Default generation-seed alias used by the driver when invoking
#: :func:`measure_tier_from_bundle` directly on a single bundle.  The
#: seed is the bundle's *generation* seed, NOT the model's
#: ``random_state`` — see :data:`DEFAULT_MODEL_RANDOM_STATE` for that.
DEFAULT_SEED: int = 42

#: ``random_state`` used for every sklearn estimator inside this module.
#: Held constant across the cross-seed sweep so the AUC variance the
#: report attributes to "data variance" is *only* data variance — not
#: data-seed × model-seed interaction.  Decoupling this from the
#: bundle's generation seed is a real correctness concern: with the
#: previous design, two consecutive seeds happening to align with a
#: HistGBM tree-split tie-break could masquerade as cross-seed
#: instability.
DEFAULT_MODEL_RANDOM_STATE: int = 0

#: K values for ``precision_at_k`` / ``recall_at_k``.  Matches G7.*.6 in
#: ``v1_acceptance_gates.md`` (P@100 is the headline; P@50 carries the
#: tighter top-of-funnel bound).
PRECISION_KS: tuple[int, ...] = (50, 100)

#: Lift percentages (top-X% of predictions, by score).  Matches the
#: design-doc §"Release validation" call for "lift@1/5/10%".
LIFT_PCTS: tuple[float, ...] = (1.0, 5.0, 10.0)

#: Cumulative-gains curve sampling points (top-X% of leads, by score).
#: 11 points at 0%, 10%, …, 100% — coarse enough for a deterministic
#: byte-stable PNG, fine enough that the plotted curve actually traces
#: the ranking quality (rather than 3 measured points connected by a
#: straight line, which was misleading).
CUMULATIVE_GAINS_PCTS: tuple[float, ...] = tuple(float(p) for p in range(0, 101, 10))

#: Number of equal-width bins for the calibration / reliability diagram.
N_CALIBRATION_BINS: int = 10

#: Fraction of the chronologically-ordered combined train+test used as
#: training data for the cohort-shift comparison; the remainder is the
#: cohort test set.  85/15 mirrors the bundle's own valid+test fraction
#: (15%) so the two splits are roughly comparable in test size.
COHORT_TRAIN_FRAC: float = 0.85


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBin:
    """One row of a reliability diagram."""

    bin_lower: float
    bin_upper: float
    n: int
    mean_predicted: float
    mean_actual: float


@dataclass(frozen=True)
class TierMetrics:
    """Full metric panel for one (tier, seed) pair.

    Mirrors the gates declared in ``v1_acceptance_gates.md`` G7.*; field
    names are stable because PR 3.3 wires them straight into the JSON
    report and the markdown citation table (G10.6).  Add new metrics
    only at the bottom of this list, never in the middle.
    """

    tier: str
    seed: int
    n_train: int
    n_test: int
    base_rate: float

    conversion_rate_train: float
    conversion_rate_test: float

    # Headline pair: LR (interpretable) vs HistGBM (sophistication-rewarding).
    # ``lr_average_precision`` is the canonical "AP" reported in the
    # dataset card — the previous version of this dataclass also carried
    # an ``average_precision`` field that was an exact alias; that field
    # has been removed because two fields with the same value confused
    # readers and added JSON noise.
    lr_auc: float
    gbm_auc: float
    gbm_minus_lr_auc: float
    lr_average_precision: float
    gbm_average_precision: float

    precision_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    lift_at_pct: dict[str, float]
    top_decile_rate: float

    # Cumulative-gains curve sampled at :data:`CUMULATIVE_GAINS_PCTS`.
    # Each entry maps ``"<pct>"`` → fraction of positives captured among
    # the top-pct% of leads (sorted descending by predicted P(convert)).
    # The renderer plots this directly — earlier versions fabricated the
    # curve by interpolating between the three lift@pct measurements,
    # which lied about the shape between data points.
    cumulative_gains: dict[str, float]

    # Value-aware ranking (G7.*.5 / design-doc "expected ACV captured at K").
    expected_acv_capture_at_k: dict[str, float]

    # Calibration (G7.*.7 / G7.*.8).
    brier_score: float
    log_loss: float
    calibration_max_bin_error: float
    calibration_bins: list[CalibrationBin]

    # Model-family / feature-subset baselines.  Names are well-known
    # constants, not free strings, so the reporting layer can render
    # them deterministically.  AUCs are absolute; deltas (full_lr_auc -
    # baseline_auc) are computed once at serialisation time.
    baselines: dict[str, float]


@dataclass(frozen=True)
class CrossSeedTierMetrics:
    """Aggregate of :class:`TierMetrics` across one tier's seed sweep."""

    tier: str
    seeds: list[int]
    per_seed: list[TierMetrics]
    medians: dict[str, float]
    spreads: dict[str, float]


@dataclass(frozen=True)
class CohortShiftMetrics:
    """Random vs chronological cohort split AUC for one tier (G6.4)."""

    tier: str
    seed: int
    random_split_auc: float
    cohort_split_auc: float
    auc_degradation: float


@dataclass(frozen=True)
class CrossTierOrdering:
    """Cross-tier difficulty ordering (G7.4.*).

    Each ``*_gt_*`` boolean is ``True`` / ``False`` when both tiers in
    the comparison are present in the report, and ``None`` when one or
    both are missing.  The previous design defaulted these to ``True``
    on missing data, which silently green-lit partial releases at the
    PR 3.3 gating layer; ``None`` forces the gating layer to make an
    explicit decision (skip vs fail) per tier pair.
    """

    by_average_precision: list[str]
    by_precision_at_100: list[str]
    by_gbm_minus_lr: list[str]
    by_conversion_rate: list[str]
    average_precision_intro_gt_intermediate: bool | None
    average_precision_intermediate_gt_advanced: bool | None
    precision_at_100_intro_gt_intermediate: bool | None
    precision_at_100_intermediate_gt_advanced: bool | None
    conversion_rate_intro_gt_intermediate: bool | None
    conversion_rate_intermediate_gt_advanced: bool | None
    gbm_minus_lr_positive_in_every_tier: bool | None


@dataclass(frozen=True)
class ReleaseQualityReport:
    """Top-level structured result.  JSON-primitive end-to-end."""

    release_id: str
    package_version: str
    generation_timestamp: str
    seeds: list[int]
    tiers: dict[str, CrossSeedTierMetrics]
    cohort_shift: dict[str, CohortShiftMetrics]
    cross_tier_ordering: CrossTierOrdering


@dataclass(frozen=True)
class TierBuildSpec:
    """Recipe configuration to regenerate a tier across seeds.

    Fields default to manifest values via :meth:`from_bundle`.  PR 3.3's
    driver builds one of these per tier and hands them to
    :func:`measure_release_quality` along with the seed list.
    """

    name: str
    recipe_id: str
    difficulty: str
    n_leads: int
    n_accounts: int
    n_contacts: int
    snapshot_day: int | None
    primary_task: str = "converted_within_90_days"
    label_window_days: int = 90
    exposure_mode: str = "student_public"

    @classmethod
    def from_bundle(cls, bundle_dir: Path, *, name: str | None = None) -> TierBuildSpec:
        """Build a spec by reading a bundle's manifest.json."""
        manifest = load_json(bundle_dir / "manifest.json")
        return cls(
            name=name or str(manifest.get("difficulty", bundle_dir.name)),
            recipe_id=str(manifest["recipe_id"]),
            difficulty=str(manifest["difficulty"]),
            n_leads=int(manifest["n_leads"]),
            n_accounts=int(manifest["n_accounts"]),
            n_contacts=int(manifest["n_contacts"]),
            snapshot_day=int(manifest["snapshot_day"]) if manifest.get("snapshot_day") else None,
            primary_task=str(manifest.get("primary_task", "converted_within_90_days")),
            label_window_days=int(manifest.get("label_window_days", 90)),
            exposure_mode=str(manifest.get("exposure_mode", "student_public")),
        )


# ---------------------------------------------------------------------------
# Single-tier measurement
# ---------------------------------------------------------------------------


def measure_tier_from_bundle(
    bundle_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    tier_name: str | None = None,
    model_random_state: int = DEFAULT_MODEL_RANDOM_STATE,
) -> TierMetrics:
    """Compute the full :class:`TierMetrics` panel for one bundle.

    Reads the primary task's ``train.parquet`` / ``test.parquet`` (the
    ``valid`` split is intentionally unused here — it is reserved for
    hyperparameter selection by downstream consumers, and including it
    in the test set would conflate model selection with reporting).

    Args:
        bundle_dir: Path to a single-seed bundle root.
        seed: Bundle's *generation* seed; recorded on the result for
            traceability and used as the row label in reports.  The
            sklearn estimator's ``random_state`` is governed by
            ``model_random_state`` instead — they MUST be independent
            so the cross-seed sweep measures only data variance, not
            data-seed × model-seed interaction.
        tier_name: Override the tier label.  Defaults to the bundle's
            declared difficulty.
        model_random_state: ``random_state`` for every sklearn
            estimator fitted by this call.  Held constant across the
            sweep by :func:`measure_release_quality`.

    Raises:
        FileNotFoundError: when the manifest or task files are missing.
        ValueError: when the train split has only one class (an honest
            degeneracy that breaks every downstream metric, surfaced
            loudly rather than silently producing NaNs).
    """
    sk = _import_sklearn()
    manifest = load_json(bundle_dir / "manifest.json")
    primary_task = str(manifest.get("primary_task", "converted_within_90_days"))

    train_path = bundle_dir / f"tasks/{primary_task}/train.parquet"
    test_path = bundle_dir / f"tasks/{primary_task}/test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"missing train.parquet or test.parquet under {bundle_dir}/tasks/{primary_task}/"
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    if LABEL_COLUMN not in train.columns or LABEL_COLUMN not in test.columns:
        raise ValueError(f"task splits must contain the {LABEL_COLUMN!r} label column")

    y_train = train[LABEL_COLUMN].astype("boolean").fillna(False).astype(int)
    y_test = test[LABEL_COLUMN].astype("boolean").fillna(False).astype(int)
    if y_train.nunique() < 2:
        raise ValueError(
            "train split has fewer than two classes; refusing to fit "
            "(a single-class regime breaks every downstream metric)"
        )
    if y_test.nunique() < 2:
        raise ValueError("test split has fewer than two classes; refusing to score")

    cat_cols, num_cols = _partition_columns(train, exclude={LABEL_COLUMN})
    x_train = _sanitize_categoricals(train[cat_cols + num_cols], cat_cols)
    x_test = _sanitize_categoricals(test[cat_cols + num_cols], cat_cols)

    lr_pipe = _build_pipeline(num_cols, cat_cols, model="lr", seed=model_random_state, sk=sk)
    gbm_pipe = _build_pipeline(num_cols, cat_cols, model="gbm", seed=model_random_state, sk=sk)

    lr_pipe.fit(x_train, y_train.values)
    gbm_pipe.fit(x_train, y_train.values)
    lr_probs = lr_pipe.predict_proba(x_test)[:, 1]
    gbm_probs = gbm_pipe.predict_proba(x_test)[:, 1]

    lr_auc = float(sk.roc_auc_score(y_test.values, lr_probs))
    gbm_auc = float(sk.roc_auc_score(y_test.values, gbm_probs))
    lr_ap = float(sk.average_precision_score(y_test.values, lr_probs))
    gbm_ap = float(sk.average_precision_score(y_test.values, gbm_probs))

    p_at_k: dict[str, float] = {}
    r_at_k: dict[str, float] = {}
    for k in PRECISION_KS:
        p_at_k[str(k)] = _precision_at_k(lr_probs, y_test.values, k)
        r_at_k[str(k)] = _recall_at_k(lr_probs, y_test.values, k)
    lift_at_pct = {f"{p:g}": _lift_at_pct(lr_probs, y_test.values, p) for p in LIFT_PCTS}
    top_decile = _top_decile_rate(lr_probs, y_test.values)
    cumulative_gains = _cumulative_gains_curve(lr_probs, y_test.values, CUMULATIVE_GAINS_PCTS)

    acv_capture: dict[str, float] = {}
    if "expected_acv" in test.columns:
        acv = pd.to_numeric(test["expected_acv"], errors="coerce").fillna(0.0).values
        for k in PRECISION_KS:
            acv_capture[str(k)] = _expected_acv_capture(lr_probs, y_test.values, acv, k)

    brier = float(sk.brier_score_loss(y_test.values, lr_probs))
    eps = 1e-15
    clipped = np.clip(lr_probs, eps, 1.0 - eps)
    log_loss = float(sk.log_loss(y_test.values, clipped, labels=[0, 1]))
    bins, max_bin_err = _calibration_bins(lr_probs, y_test.values, n_bins=N_CALIBRATION_BINS)

    baselines = _compute_baselines(
        train=train,
        test=test,
        y_train=y_train.values,
        y_test=y_test.values,
        seed=model_random_state,
        sk=sk,
    )

    return TierMetrics(
        tier=tier_name or str(manifest.get("difficulty", bundle_dir.name)),
        seed=seed,
        n_train=int(len(train)),
        n_test=int(len(test)),
        base_rate=float(y_test.mean()),
        conversion_rate_train=float(y_train.mean()),
        conversion_rate_test=float(y_test.mean()),
        lr_auc=lr_auc,
        gbm_auc=gbm_auc,
        gbm_minus_lr_auc=gbm_auc - lr_auc,
        lr_average_precision=lr_ap,
        gbm_average_precision=gbm_ap,
        precision_at_k=p_at_k,
        recall_at_k=r_at_k,
        lift_at_pct=lift_at_pct,
        top_decile_rate=top_decile,
        cumulative_gains=cumulative_gains,
        expected_acv_capture_at_k=acv_capture,
        brier_score=brier,
        log_loss=log_loss,
        calibration_max_bin_error=max_bin_err,
        calibration_bins=bins,
        baselines=baselines,
    )


def measure_cohort_shift_from_bundle(
    bundle_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    tier_name: str | None = None,
    model_random_state: int = DEFAULT_MODEL_RANDOM_STATE,
) -> CohortShiftMetrics:
    """Random-vs-chronological-cohort split AUC degradation (G6.4).

    Uses the bundle's existing train/test as the random-split AUC and
    re-splits the union chronologically by ``lead_created_at`` for the
    cohort-split AUC.  HistGBM is used for both — it handles NaN
    natively so we don't have to thread a separate imputation pipeline
    through the chronological resplit.

    See :func:`measure_tier_from_bundle` for the seed / model-seed
    decoupling rationale.
    """
    sk = _import_sklearn()
    manifest = load_json(bundle_dir / "manifest.json")
    primary_task = str(manifest.get("primary_task", "converted_within_90_days"))
    label = tier_name or str(manifest.get("difficulty", bundle_dir.name))

    train = pd.read_parquet(bundle_dir / f"tasks/{primary_task}/train.parquet")
    test = pd.read_parquet(bundle_dir / f"tasks/{primary_task}/test.parquet")

    cat_cols, num_cols = _partition_columns(train, exclude={LABEL_COLUMN})
    x_train = _sanitize_categoricals(train[cat_cols + num_cols], cat_cols)
    x_test = _sanitize_categoricals(test[cat_cols + num_cols], cat_cols)
    y_train = train[LABEL_COLUMN].astype("boolean").fillna(False).astype(int).values
    y_test = test[LABEL_COLUMN].astype("boolean").fillna(False).astype(int).values

    rand_pipe = _build_pipeline(num_cols, cat_cols, model="gbm", seed=model_random_state, sk=sk)
    rand_pipe.fit(x_train, y_train)
    rand_probs = rand_pipe.predict_proba(x_test)[:, 1]
    random_auc = float(sk.roc_auc_score(y_test, rand_probs))

    def _no_cohort() -> CohortShiftMetrics:
        # Surface NaN rather than inventing a value when chronological
        # resplit is unsupported (no timestamp column / unparseable
        # timestamps / single-class early or late half / empty late
        # half).  PR 3.3's gating layer can then treat NaN as "skip"
        # rather than silently scoring 0.
        return CohortShiftMetrics(
            tier=label,
            seed=seed,
            random_split_auc=random_auc,
            cohort_split_auc=float("nan"),
            auc_degradation=float("nan"),
        )

    if "lead_created_at" not in train.columns:
        return _no_cohort()

    pooled = pd.concat([train, test], ignore_index=True)
    ts = pd.to_datetime(pooled["lead_created_at"], errors="coerce")
    if ts.isna().any():
        return _no_cohort()

    # Stable primary key = ``lead_created_at``; deterministic
    # tie-breaker = ``lead_id`` so that bundles with many leads sharing
    # one timestamp (common with synthetic generators that anchor every
    # day) split the same way across pandas versions and concat orders.
    if "lead_id" in pooled.columns:
        sort_frame = pd.DataFrame({"_ts": ts.values, "_lid": pooled["lead_id"].astype(str).values})
        order = sort_frame.sort_values(["_ts", "_lid"], kind="stable").index.to_numpy()
    else:
        order = np.argsort(ts.values, kind="stable")
    cutoff = int(round(len(pooled) * COHORT_TRAIN_FRAC))
    early_idx = order[:cutoff]
    late_idx = order[cutoff:]
    if len(late_idx) == 0:
        return _no_cohort()

    early = pooled.iloc[early_idx]
    late = pooled.iloc[late_idx]
    y_early = early[LABEL_COLUMN].astype("boolean").fillna(False).astype(int).values
    y_late = late[LABEL_COLUMN].astype("boolean").fillna(False).astype(int).values
    if np.unique(y_early).size < 2 or np.unique(y_late).size < 2:
        return _no_cohort()

    x_early = _sanitize_categoricals(early[cat_cols + num_cols], cat_cols)
    x_late = _sanitize_categoricals(late[cat_cols + num_cols], cat_cols)
    cohort_pipe = _build_pipeline(num_cols, cat_cols, model="gbm", seed=model_random_state, sk=sk)
    cohort_pipe.fit(x_early, y_early)
    cohort_probs = cohort_pipe.predict_proba(x_late)[:, 1]
    cohort_auc = float(sk.roc_auc_score(y_late, cohort_probs))

    return CohortShiftMetrics(
        tier=label,
        seed=seed,
        random_split_auc=random_auc,
        cohort_split_auc=cohort_auc,
        auc_degradation=random_auc - cohort_auc,
    )


# ---------------------------------------------------------------------------
# Cross-seed orchestration
# ---------------------------------------------------------------------------


def regenerate_tier_for_seeds(
    spec: TierBuildSpec,
    seeds: Sequence[int],
    workdir: Path,
) -> dict[int, Path]:
    """Generate one bundle per seed under ``workdir``.

    Idempotent: if ``workdir / "<tier>__seed{seed}"`` already contains a
    valid manifest, that bundle is reused.  Used by
    :func:`measure_release_quality` and the round-trip test; PR 3.3's
    driver can use it directly to keep cross-seed sweep state on disk
    between runs.
    """
    from leadforge.api.generator import Generator

    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    for seed in seeds:
        target = workdir / f"{spec.name}__seed{seed}"
        if (target / "manifest.json").exists():
            out[seed] = target
            continue
        gen = Generator.from_recipe(
            spec.recipe_id,
            seed=seed,
            exposure_mode=spec.exposure_mode,
            difficulty=spec.difficulty,
            n_accounts=spec.n_accounts,
            n_contacts=spec.n_contacts,
            n_leads=spec.n_leads,
            primary_task=spec.primary_task,
            label_window_days=spec.label_window_days,
            snapshot_day=spec.snapshot_day,
        )
        gen.generate().save(str(target))
        out[seed] = target
    return out


def measure_release_quality(
    tier_bundles: Mapping[str, Mapping[int, Path]],
    *,
    cohort_canonical_seed: int | None = None,
    release_id: str = "leadforge-lead-scoring-v1",
    package_version: str | None = None,
    generation_timestamp: str | None = None,
    model_random_state: int = DEFAULT_MODEL_RANDOM_STATE,
) -> ReleaseQualityReport:
    """Aggregate per-(tier, seed) measurements into a full report.

    Args:
        tier_bundles: Mapping ``tier_name -> {seed: bundle_dir}``.
            Tier names are arbitrary strings; the cross-tier ordering
            check looks for the canonical ``intro``/``intermediate``/
            ``advanced`` names but tolerates their absence (the
            corresponding ordering-bool fields default to ``True`` so a
            partial release does not over-report ordering failures).
        cohort_canonical_seed: Seed at which to run the cohort-shift
            evaluation per tier.  When ``None``, the smallest seed
            present per tier is used.  Cohort shift is reported for one
            seed per tier — running it on every seed would multiply the
            sweep cost without producing extra signal at this layer.
        release_id: Identifier baked into the JSON.
        package_version: leadforge package version.  Defaults to the
            installed version from :mod:`leadforge.version`.
        generation_timestamp: Pinned timestamp for the report.  Defaults
            to current UTC.
    """
    from leadforge.version import __version__

    cross_seed: dict[str, CrossSeedTierMetrics] = {}
    cohort: dict[str, CohortShiftMetrics] = {}
    for tier_name, by_seed in tier_bundles.items():
        seeds = sorted(by_seed.keys())
        per_seed_metrics = [
            measure_tier_from_bundle(
                by_seed[s],
                seed=s,
                tier_name=tier_name,
                model_random_state=model_random_state,
            )
            for s in seeds
        ]
        medians, spreads = _aggregate_cross_seed(per_seed_metrics)
        cross_seed[tier_name] = CrossSeedTierMetrics(
            tier=tier_name,
            seeds=list(seeds),
            per_seed=per_seed_metrics,
            medians=medians,
            spreads=spreads,
        )
        canonical: int = (
            cohort_canonical_seed
            if cohort_canonical_seed is not None and cohort_canonical_seed in by_seed
            else seeds[0]
        )
        cohort[tier_name] = measure_cohort_shift_from_bundle(
            by_seed[canonical],
            seed=canonical,
            tier_name=tier_name,
            model_random_state=model_random_state,
        )

    ordering = _compute_cross_tier_ordering(cross_seed)

    if generation_timestamp is None:
        from datetime import UTC, datetime

        generation_timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()

    return ReleaseQualityReport(
        release_id=release_id,
        package_version=package_version or __version__,
        generation_timestamp=generation_timestamp,
        seeds=sorted({s for d in tier_bundles.values() for s in d}),
        tiers=cross_seed,
        cohort_shift=cohort,
        cross_tier_ordering=ordering,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_HEADLINE_FIELDS: tuple[str, ...] = (
    "lr_auc",
    "gbm_auc",
    "gbm_minus_lr_auc",
    "lr_average_precision",
    "gbm_average_precision",
    "brier_score",
    "log_loss",
    "calibration_max_bin_error",
    "top_decile_rate",
    "conversion_rate_test",
)


def _aggregate_cross_seed(
    per_seed: list[TierMetrics],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute medians and (max - min) spreads for the headline fields.

    Spreads here are the simple max-min range, not a standard deviation —
    G8.1 declares the band on ``±TBD`` of the median, which is most
    naturally expressed as a half-range.  PR 3.3 reads this directly.
    """
    medians: dict[str, float] = {}
    spreads: dict[str, float] = {}
    if not per_seed:
        return medians, spreads
    for fld in _HEADLINE_FIELDS:
        values = [float(getattr(m, fld)) for m in per_seed]
        medians[fld] = float(np.median(values))
        spreads[fld] = float(max(values) - min(values))
    return medians, spreads


def _compute_cross_tier_ordering(
    cross_seed: Mapping[str, CrossSeedTierMetrics],
) -> CrossTierOrdering:
    """Derive G7.4.* ordering booleans + descending tier rankings.

    Each ``*_gt_*`` boolean is ``None`` when one or both compared tiers
    are absent from the report (or carry NaN medians).  The previous
    design defaulted to ``True`` on missing data, which silently
    green-lit partial releases at the PR 3.3 gating layer; ``None``
    forces an explicit decision per pair.

    The ``intro`` / ``intermediate`` / ``advanced`` tier names are
    hardcoded because they are the v1 dataset family (per
    ``docs/release/v1_release_design.md`` §"Dataset family architecture");
    this function is therefore not a general N-tier comparator.
    """
    if not cross_seed:
        return CrossTierOrdering(
            by_average_precision=[],
            by_precision_at_100=[],
            by_gbm_minus_lr=[],
            by_conversion_rate=[],
            average_precision_intro_gt_intermediate=None,
            average_precision_intermediate_gt_advanced=None,
            precision_at_100_intro_gt_intermediate=None,
            precision_at_100_intermediate_gt_advanced=None,
            conversion_rate_intro_gt_intermediate=None,
            conversion_rate_intermediate_gt_advanced=None,
            gbm_minus_lr_positive_in_every_tier=None,
        )

    # Build per-tier representative numbers from the median across seeds.
    median_ap: dict[str, float] = {}
    median_p100: dict[str, float] = {}
    median_gbm_lr: dict[str, float] = {}
    median_rate: dict[str, float] = {}
    for tier, csm in cross_seed.items():
        # Median P@100 is computed from the per-seed dicts directly —
        # the headline aggregator only carries scalars.
        p100s = [float(m.precision_at_k.get("100", float("nan"))) for m in csm.per_seed]
        median_ap[tier] = csm.medians.get("lr_average_precision", float("nan"))
        if p100s and not all(math.isnan(p) for p in p100s):
            median_p100[tier] = float(np.median(p100s))
        else:
            median_p100[tier] = float("nan")
        median_gbm_lr[tier] = csm.medians.get("gbm_minus_lr_auc", float("nan"))
        median_rate[tier] = csm.medians.get("conversion_rate_test", float("nan"))

    def _sorted_desc(d: Mapping[str, float]) -> list[str]:
        # NaN sorts last so it doesn't artificially top the ranking.
        return sorted(d, key=lambda k: (math.isnan(d[k]), -d[k] if not math.isnan(d[k]) else 0.0))

    def _gt(d: Mapping[str, float], a: str, b: str) -> bool | None:
        # Missing tier or NaN median → undefined, surface as ``None``.
        if a not in d or b not in d:
            return None
        if math.isnan(d[a]) or math.isnan(d[b]):
            return None
        return d[a] > d[b]

    finite_gbm_lr = [v for v in median_gbm_lr.values() if not math.isnan(v)]
    gbm_minus_lr_positive: bool | None = (
        all(v > 0 for v in finite_gbm_lr) if finite_gbm_lr else None
    )

    return CrossTierOrdering(
        by_average_precision=_sorted_desc(median_ap),
        by_precision_at_100=_sorted_desc(median_p100),
        by_gbm_minus_lr=_sorted_desc(median_gbm_lr),
        by_conversion_rate=_sorted_desc(median_rate),
        average_precision_intro_gt_intermediate=_gt(median_ap, "intro", "intermediate"),
        average_precision_intermediate_gt_advanced=_gt(median_ap, "intermediate", "advanced"),
        precision_at_100_intro_gt_intermediate=_gt(median_p100, "intro", "intermediate"),
        precision_at_100_intermediate_gt_advanced=_gt(median_p100, "intermediate", "advanced"),
        conversion_rate_intro_gt_intermediate=_gt(median_rate, "intro", "intermediate"),
        conversion_rate_intermediate_gt_advanced=_gt(median_rate, "intermediate", "advanced"),
        gbm_minus_lr_positive_in_every_tier=gbm_minus_lr_positive,
    )


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _precision_at_k(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(y):
        return float("nan")
    order = np.argsort(-np.asarray(probs), kind="stable")
    return float(np.asarray(y)[order[:k]].mean())


def _recall_at_k(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(y):
        return float("nan")
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probs), kind="stable")
    return float(np.asarray(y)[order[:k]].sum() / n_pos)


def _lift_at_pct(probs: np.ndarray, y: np.ndarray, pct: float) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    base_rate = float(np.mean(y))
    if base_rate <= 0:
        return float("nan")
    k = max(1, int(round(n * pct / 100.0)))
    order = np.argsort(-np.asarray(probs), kind="stable")
    top_k_y = np.asarray(y)[order[:k]]
    return float(top_k_y.mean() / base_rate)


def _top_decile_rate(probs: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    return _precision_at_k(probs, y, max(1, int(round(n * 0.1))))


def _expected_acv_capture(probs: np.ndarray, y: np.ndarray, acv: np.ndarray, k: int) -> float:
    """Fraction of total converted-ACV captured in the top-k by score."""
    if k <= 0 or k > len(y):
        return float("nan")
    order = np.argsort(-np.asarray(probs), kind="stable")
    captured = float(np.sum(np.asarray(acv)[order[:k]] * np.asarray(y)[order[:k]]))
    total = float(np.sum(np.asarray(acv) * np.asarray(y)))
    if total <= 0:
        return float("nan")
    return captured / total


def _cumulative_gains_curve(
    probs: np.ndarray,
    y: np.ndarray,
    pcts: tuple[float, ...],
) -> dict[str, float]:
    """Fraction of positives captured at each top-pct% cut-off.

    Stored on :class:`TierMetrics` so the renderer plots the actual
    ranking-quality curve instead of fabricating one by interpolating
    between three lift@pct measurements.

    For ``pct == 0`` returns 0.0 (an empty selection captures nothing);
    for ``pct == 100`` returns 1.0.  When there are no positives in
    ``y`` the entire curve is NaN — there's no denominator.
    """
    n = len(y)
    n_pos = int(np.sum(y))
    out: dict[str, float] = {}
    if n == 0 or n_pos == 0:
        for p in pcts:
            out[f"{p:g}"] = float("nan")
        return out
    order = np.argsort(-np.asarray(probs), kind="stable")
    y_sorted = np.asarray(y)[order]
    cum = np.cumsum(y_sorted)
    for p in pcts:
        if p <= 0:
            out[f"{p:g}"] = 0.0
            continue
        if p >= 100:
            out[f"{p:g}"] = 1.0
            continue
        k = max(1, int(round(n * p / 100.0)))
        out[f"{p:g}"] = float(cum[k - 1] / n_pos)
    return out


def _calibration_bins(
    probs: np.ndarray, y: np.ndarray, *, n_bins: int = 10
) -> tuple[list[CalibrationBin], float]:
    """Equal-width reliability bins + max absolute calibration error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[CalibrationBin] = []
    max_err = 0.0
    probs_arr = np.asarray(probs, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < n_bins - 1:
            mask = (probs_arr >= lo) & (probs_arr < hi)
        else:
            mask = (probs_arr >= lo) & (probs_arr <= hi)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        mean_p = float(probs_arr[mask].mean())
        mean_y = float(y_arr[mask].mean())
        err = abs(mean_p - mean_y)
        if err > max_err:
            max_err = err
        out.append(
            CalibrationBin(
                bin_lower=lo, bin_upper=hi, n=n_in, mean_predicted=mean_p, mean_actual=mean_y
            )
        )
    return out, max_err


# ---------------------------------------------------------------------------
# Pipeline construction (consumes the leadforge.pipelines.ml conventions
# but rebuilds the column partition because the bundle's task split has
# different column names than the v6/v7 flat CSVs).
# ---------------------------------------------------------------------------


_EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset(
    {LABEL_COLUMN, "lead_id", "account_id", "contact_id", "lead_created_at"}
)


def _partition_columns(df: pd.DataFrame, *, exclude: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split bundle-snapshot columns into (categorical, numeric) by dtype.

    IDs and timestamp anchors are always excluded — they are not
    legitimate predictive features for this task and they would balloon
    the OneHotEncoder's vocabulary.  Boolean / nullable-int / nullable-
    float columns count as numeric.
    """
    excl = set(exclude) | _EXCLUDE_FROM_FEATURES
    cat: list[str] = []
    num: list[str] = []
    for col in df.columns:
        if col in excl:
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            num.append(col)
        else:
            cat.append(col)
    return cat, num


def _sanitize_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Convert pd.NA in categorical columns to None for sklearn compatibility.

    Mirrors :func:`leadforge.pipelines.ml.sanitize_categoricals` — kept
    here as a private helper to avoid importing the v6/v7 pipeline
    constants (this module's column lists are derived from the bundle
    schema, not the flat-CSV schema).
    """
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def _build_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    *,
    model: str,
    seed: int,
    sk: _SklearnHandles,
) -> Any:
    """LR or HistGBM pipeline on top of the canonical preprocessor.

    The preprocessor mirrors :func:`leadforge.pipelines.ml.build_preprocessor`
    (median-impute + standard-scale numeric; most-frequent-impute +
    one-hot encode categorical) so the metric panel agrees by
    construction with the canonical baseline used elsewhere in the
    package.  It is rebuilt locally because the bundle column set
    differs from the flat-CSV column set.
    """
    numeric_t = sk.Pipeline(
        [("imputer", sk.SimpleImputer(strategy="median")), ("scaler", sk.StandardScaler())]
    )
    categorical_t = sk.Pipeline(
        [
            ("imputer", sk.SimpleImputer(strategy="most_frequent")),
            ("encoder", sk.OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = sk.ColumnTransformer(
        transformers=[("num", numeric_t, num_cols), ("cat", categorical_t, cat_cols)],
        remainder="drop",
    )
    if model == "lr":
        clf: Any = sk.LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    elif model == "gbm":
        clf = sk.HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"unknown model: {model!r}")
    return sk.Pipeline([("preprocessor", pre), ("classifier", clf)])


# ---------------------------------------------------------------------------
# Baseline subsets
# ---------------------------------------------------------------------------


_SOURCE_COLUMNS: tuple[str, ...] = ("lead_source", "first_touch_channel")
_STAGE_COLUMNS: tuple[str, ...] = ("current_stage", "is_sql")
_ID_COLUMNS: tuple[str, ...] = ("lead_id", "account_id", "contact_id")
_POST_SNAPSHOT_AGGREGATES: tuple[str, ...] = ("total_touches_all",)


def _engagement_columns() -> tuple[str, ...]:
    """Column names tagged ``category="engagement"`` in the snapshot spec."""
    return tuple(f.name for f in LEAD_SNAPSHOT_FEATURES if f.category == "engagement")


def _compute_baselines(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    sk: _SklearnHandles,
) -> dict[str, float]:
    """AUC of HistGBM trained on each well-known feature subset.

    Keys present in the result are exactly those whose source columns
    exist in the bundle.  Stage-only is typically absent from public
    bundles (G5.2's columns are redacted under student_public); the
    omission is the result, not an error.
    """
    out: dict[str, float] = {}
    for name, cols in (
        ("source_only", _SOURCE_COLUMNS),
        ("engagement_only", _engagement_columns()),
        ("stage_only", _STAGE_COLUMNS),
        ("post_snapshot_aggregates", _POST_SNAPSHOT_AGGREGATES),
    ):
        present = [c for c in cols if c in train.columns]
        if not present:
            continue
        auc = _subset_auc(train, test, y_train, y_test, present, seed=seed, sk=sk)
        if auc is not None:
            out[name] = auc

    id_present = [c for c in _ID_COLUMNS if c in train.columns]
    if id_present:
        auc = _id_only_auc(train, test, y_train, y_test, id_present, seed=seed, sk=sk)
        if auc is not None:
            out["id_only"] = auc
    return out


def _subset_auc(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cols: list[str],
    *,
    seed: int,
    sk: _SklearnHandles,
) -> float | None:
    """HistGBM on a feature subset; returns None when scoring is impossible."""
    cat_in_subset = [
        c
        for c in cols
        if not pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])
    ]
    num_in_subset = [c for c in cols if c not in cat_in_subset]
    x_tr = _sanitize_categoricals(train[cols], cat_in_subset)
    x_te = _sanitize_categoricals(test[cols], cat_in_subset)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return None
    pipe = _build_pipeline(num_in_subset, cat_in_subset, model="gbm", seed=seed, sk=sk)
    pipe.fit(x_tr, y_train)
    probs = pipe.predict_proba(x_te)[:, 1]
    return float(sk.roc_auc_score(y_test, probs))


def _id_only_auc(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    id_cols: list[str],
    *,
    seed: int,
    sk: _SklearnHandles,
) -> float | None:
    """Hash IDs to ints and feed HistGBM directly.

    Mirrors :func:`leadforge.validation.leakage_probes._hash_id_columns`
    so the leakage-probe baseline and the release-quality baseline
    produce comparable numbers.  Expected ≈ 0.5 + ε on a clean bundle.
    """
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return None
    x_tr = _hash_id_columns(train[id_cols])
    x_te = _hash_id_columns(test[id_cols])
    model = sk.HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    model.fit(x_tr.values, y_train)
    probs = model.predict_proba(x_te.values)[:, 1]
    return float(sk.roc_auc_score(y_test, probs))


def _hash_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _h(value: object) -> int:
        digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big", signed=False)

    return pd.DataFrame({col: df[col].map(_h).astype("int64") for col in df.columns})


# ---------------------------------------------------------------------------
# sklearn handles — lazy bundle so import-time failures stay loud and the
# probe-style "skip cleanly when missing" pattern from leakage_probes is
# inverted here (release quality REQUIRES sklearn to do anything useful).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SklearnHandles:
    Pipeline: Any
    ColumnTransformer: Any
    SimpleImputer: Any
    StandardScaler: Any
    OneHotEncoder: Any
    LogisticRegression: Any
    HistGradientBoostingClassifier: Any
    roc_auc_score: Any
    average_precision_score: Any
    brier_score_loss: Any
    log_loss: Any


def _import_sklearn() -> _SklearnHandles:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return _SklearnHandles(
        Pipeline=Pipeline,
        ColumnTransformer=ColumnTransformer,
        SimpleImputer=SimpleImputer,
        StandardScaler=StandardScaler,
        OneHotEncoder=OneHotEncoder,
        LogisticRegression=LogisticRegression,
        HistGradientBoostingClassifier=HistGradientBoostingClassifier,
        roc_auc_score=roc_auc_score,
        average_precision_score=average_precision_score,
        brier_score_loss=brier_score_loss,
        log_loss=log_loss,
    )


# ---------------------------------------------------------------------------
# JSON serialisation helpers (used by reporting.py).  Centralised here so
# any caller that imports the dataclasses also gets a deterministic
# JSON-conversion entry point.
# ---------------------------------------------------------------------------


def report_to_dict(report: ReleaseQualityReport) -> dict[str, Any]:
    """Convert a :class:`ReleaseQualityReport` into a JSON-primitive dict.

    Wraps :func:`dataclasses.asdict` and walks the result to coerce
    floats that ``json.dumps`` would otherwise reject (NaN / ±Inf) into
    ``None``.  PR 3.2 produces NaN deliberately (e.g. cohort-shift on a
    bundle with no ``lead_created_at``); turning them into ``null`` is
    the cheapest contract change for downstream JSON consumers.
    """
    raw = dataclasses.asdict(report)
    # ``_json_safe`` walks dicts/lists in place; the top-level result of
    # ``asdict`` on a dataclass is always a dict, so the return type
    # narrows to ``dict[str, Any]``.  ``cast`` is the right tool here —
    # we're not asserting an invariant, we're declaring a known shape.
    return cast(dict[str, Any], _json_safe(raw))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def report_to_json(report: ReleaseQualityReport, *, indent: int = 2) -> str:
    """Stable JSON dump of the report (sorted keys, fixed indent)."""
    return json.dumps(report_to_dict(report), indent=indent, sort_keys=True)


__all__ = [
    "COHORT_TRAIN_FRAC",
    "CUMULATIVE_GAINS_PCTS",
    "DEFAULT_MODEL_RANDOM_STATE",
    "DEFAULT_SEED",
    "LABEL_COLUMN",
    "LIFT_PCTS",
    "N_CALIBRATION_BINS",
    "PRECISION_KS",
    "CalibrationBin",
    "CohortShiftMetrics",
    "CrossSeedTierMetrics",
    "CrossTierOrdering",
    "ReleaseQualityReport",
    "TierBuildSpec",
    "TierMetrics",
    "measure_cohort_shift_from_bundle",
    "measure_release_quality",
    "measure_tier_from_bundle",
    "regenerate_tier_for_seeds",
    "report_to_dict",
    "report_to_json",
]
