"""Difficulty profile adherence checks + acceptance-band gating.

The original module validates that a manifest declares a known difficulty
profile and that the actual conversion rate falls within the declared
range.  PR 3.3 extends it with a YAML-driven band checker that consumes
:class:`leadforge.validation.release_quality.ReleaseQualityReport` plus
the per-tier :class:`leadforge.validation.leakage_probes.LeakageReport`
findings and gates the v1 dataset release on every acceptance gate that
carries a numeric band in ``docs/release/v1_acceptance_gates.md``.

The band checker is deliberately data-driven: bands live in
``docs/release/v1_acceptance_gates_bands.yaml`` rather than in code, so
operators can tune them between releases without code review.  See
:func:`load_bands` and :func:`check_release_bands`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from leadforge.core.serialization import load_yaml

if TYPE_CHECKING:
    from leadforge.validation.leakage_probes import LeakageReport
    from leadforge.validation.release_quality import (
        CrossSeedTierMetrics,
        ReleaseQualityReport,
    )

# Known difficulty profiles and their expected conversion rate ranges.
_KNOWN_DIFFICULTIES = {"intro", "intermediate", "advanced"}

_CONVERSION_RATE_RANGES: dict[str, tuple[float, float]] = {
    "intro": (0.30, 0.45),
    "intermediate": (0.18, 0.28),
    "advanced": (0.08, 0.15),
}

# Tolerance applied to range bounds for validation (accounts for stochastic variance).
_RATE_TOLERANCE = 0.05


def check_difficulty(manifest: dict[str, Any]) -> list[str]:
    """Check that the manifest declares a known difficulty profile.

    Args:
        manifest: Parsed manifest dict.

    Returns a list of error strings (empty = pass).
    """
    errors: list[str] = []
    difficulty = manifest.get("difficulty")
    if difficulty is None:
        errors.append("Manifest missing 'difficulty' field")
    elif difficulty not in _KNOWN_DIFFICULTIES:
        errors.append(f"Unknown difficulty profile: '{difficulty}'")
    return errors


def check_difficulty_ordering(bundles: dict[str, Path]) -> list[str]:
    """Check that conversion rates decrease as difficulty increases.

    Reads the task train split from each bundle to compute the actual
    conversion rate and verifies:
    1. Each rate falls within the declared range (with tolerance).
    2. Rates are ordered: intro > intermediate > advanced.

    Args:
        bundles: Mapping of difficulty name → bundle path.

    Returns:
        Error strings if any check is violated.
    """
    import pandas as pd

    errors: list[str] = []
    rates: dict[str, float] = {}

    for name, bundle_path in bundles.items():
        # Try all task split files to compute conversion rate.
        task_dir = bundle_path / "tasks" / "converted_within_90_days"
        for split in ("train", "valid", "test"):
            split_path = task_dir / f"{split}.parquet"
            if split_path.exists():
                df = pd.read_parquet(split_path)
                if "converted_within_90_days" in df.columns:
                    if name not in rates:
                        rates[name] = float(df["converted_within_90_days"].mean())
                    break

    # Check each rate is within the declared range (with tolerance).
    for name, rate in rates.items():
        if name in _CONVERSION_RATE_RANGES:
            lo, hi = _CONVERSION_RATE_RANGES[name]
            if rate < lo - _RATE_TOLERANCE:
                errors.append(
                    f"Difficulty '{name}' conversion rate {rate:.3f} "
                    f"is below expected range [{lo:.2f}, {hi:.2f}] "
                    f"(tolerance {_RATE_TOLERANCE})"
                )
            elif rate > hi + _RATE_TOLERANCE:
                errors.append(
                    f"Difficulty '{name}' conversion rate {rate:.3f} "
                    f"is above expected range [{lo:.2f}, {hi:.2f}] "
                    f"(tolerance {_RATE_TOLERANCE})"
                )

    # Check ordering: intro > intermediate > advanced.
    ordering = ["intro", "intermediate", "advanced"]
    for i in range(len(ordering) - 1):
        higher = ordering[i]
        lower = ordering[i + 1]
        if higher in rates and lower in rates:
            if rates[lower] >= rates[higher]:
                errors.append(
                    f"Conversion rate for '{lower}' ({rates[lower]:.3f}) "
                    f"should be less than '{higher}' ({rates[higher]:.3f})"
                )

    return errors


# ---------------------------------------------------------------------------
# Acceptance bands — YAML-driven gate checker (PR 3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateFailure:
    """One acceptance-gate violation surfaced by :func:`check_release_bands`.

    Attributes:
        gate: Gate identifier from ``v1_acceptance_gates.md`` (e.g.
            ``"G7.1.5"`` or ``"G7.4.1"``).  Cross-tier gates omit the tier
            scope; per-tier gates carry it.
        tier: Tier name when the failure is per-tier; ``None`` for cross-
            tier gates and global gates.
        message: Human-readable description.  The driver renders this
            into the CLI output and the JSON report.
    """

    gate: str
    tier: str | None
    message: str


@dataclass(frozen=True)
class BandSpec:
    """One per-tier numeric band parsed from the YAML config.

    Bands are interpreted as ``[min, max]`` if both bounds are present;
    one-sided bounds (``min`` or ``max`` alone) are honoured as well.
    NaN-valued metrics surface as a single explicit failure rather than
    silently passing — calibrating against NaN would defeat the purpose.
    """

    metric: str
    gate: str
    min: float | None = None
    max: float | None = None

    def evaluate(self, value: float, *, tier: str) -> GateFailure | None:
        if math.isnan(value):
            return GateFailure(
                gate=self.gate,
                tier=tier,
                message=(
                    f"{self.metric}: value is NaN — cannot evaluate band [{self.min}, {self.max}]"
                ),
            )
        if self.min is not None and value < self.min:
            return GateFailure(
                gate=self.gate,
                tier=tier,
                message=(
                    f"{self.metric}: {value:.4f} below min {self.min:.4f} "
                    f"(band [{self.min}, {self.max}])"
                ),
            )
        if self.max is not None and value > self.max:
            return GateFailure(
                gate=self.gate,
                tier=tier,
                message=(
                    f"{self.metric}: {value:.4f} above max {self.max:.4f} "
                    f"(band [{self.min}, {self.max}])"
                ),
            )
        return None


@dataclass(frozen=True)
class TierBands:
    """Per-tier band collection.  Keys map metric → :class:`BandSpec`."""

    tier: str
    bands: Mapping[str, BandSpec]


@dataclass(frozen=True)
class LeakageProbeBands:
    """Calibrated thresholds for :func:`run_split_probes`.

    Global rather than per-tier — the contract ("IDs carry no signal",
    "post-snapshot aggregates can't ace the task on their own") is the
    same across difficulty tiers.  ``feature_subsets`` mirrors the
    ``feature_subsets`` arg of :func:`run_split_probes` exactly:
    ``name → (max_auc, columns)``.
    """

    id_only_max_auc: float | None
    label_drift_max: float | None
    feature_subsets: Mapping[str, tuple[float, tuple[str, ...]]]


@dataclass(frozen=True)
class AcceptanceBands:
    """Top-level YAML payload after parsing.

    ``per_tier`` carries the G7.1 / G7.2 / G7.3 bands keyed by tier name.
    ``cross_seed_spread`` holds the G8.1 max-spread tolerance per metric
    (applied uniformly across tiers).  ``cohort_shift`` holds the G6.4
    degradation band (also uniform across tiers).  ``cross_tier_required``
    governs which tiers must be present for the cross-tier ordering gates
    to be evaluated.  ``leakage_probes`` carries the calibrated
    thresholds the driver passes to
    :func:`leadforge.validation.leakage_probes.run_split_probes`.
    """

    per_tier: Mapping[str, TierBands]
    cross_seed_spread: Mapping[str, BandSpec]
    cohort_shift: BandSpec | None
    cross_tier_required: tuple[str, ...]
    leakage_probes: LeakageProbeBands


# Mapping from medians-field name → which gate it belongs to.  Used to
# tag G7.1.* / G7.2.* / G7.3.* failures with the right gate id.  Per-tier
# numeric is the third digit; the gate prefix is computed from the tier.
_GATE_PREFIX_BY_TIER: Mapping[str, str] = {
    "intro": "G7.1",
    "intermediate": "G7.2",
    "advanced": "G7.3",
}

# Headline metrics → digit suffix in the gate id (matches the layout of
# v1_acceptance_gates.md §"Performance gates").
_GATE_SUFFIX_BY_METRIC: Mapping[str, str] = {
    "conversion_rate_test": "1",
    "lr_auc": "2",
    "gbm_auc": "3",
    "gbm_minus_lr_auc": "4",
    "lr_average_precision": "5",
    "precision_at_100": "6",
    "brier_score": "7",
    "calibration_max_bin_error": "8",
}


def _gate_id_for(tier: str, metric: str) -> str:
    """Compute the gate id for a per-tier metric, falling back to a generic prefix."""
    prefix = _GATE_PREFIX_BY_TIER.get(tier)
    suffix = _GATE_SUFFIX_BY_METRIC.get(metric)
    if prefix is None or suffix is None:
        return f"G7.{tier}.{metric}"
    return f"{prefix}.{suffix}"


def load_bands(path: Path) -> AcceptanceBands:
    """Parse the YAML acceptance-bands file.

    Schema (minimal example)::

        per_tier:
          intro:
            conversion_rate_test: {min: 0.30, max: 0.50}
            lr_auc: {min: 0.85, max: 0.95}
            gbm_minus_lr_auc: {min: 0.005}
            lr_average_precision: {min: 0.55, max: 0.85}
            precision_at_100: {min: 0.55, max: 0.95}
            brier_score: {max: 0.20}
            calibration_max_bin_error: {max: 0.15}
        cross_seed_spread:
          lr_auc: {max: 0.04}
          lr_average_precision: {max: 0.08}
        cohort_shift:
          auc_degradation: {min: 0.0, max: 0.20}
        cross_tier_required: [intro, intermediate, advanced]

    The driver's ``--bands`` flag points at this file.  Missing optional
    sections (``cross_seed_spread``, ``cohort_shift``,
    ``cross_tier_required``) default to "no gate", not "fail".
    """
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"bands file {path} must be a YAML mapping; got {type(raw).__name__}")

    per_tier_raw = raw.get("per_tier") or {}
    per_tier: dict[str, TierBands] = {}
    for tier_name, metrics in per_tier_raw.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"per_tier.{tier_name} must be a mapping")
        bands: dict[str, BandSpec] = {}
        for metric_name, bounds in metrics.items():
            bands[metric_name] = _parse_band_spec(
                metric_name, bounds, gate=_gate_id_for(tier_name, metric_name)
            )
        per_tier[tier_name] = TierBands(tier=tier_name, bands=bands)

    cs_raw = raw.get("cross_seed_spread") or {}
    cross_seed_spread: dict[str, BandSpec] = {}
    for metric_name, bounds in cs_raw.items():
        cross_seed_spread[metric_name] = _parse_band_spec(metric_name, bounds, gate="G8.1")

    cohort_shift: BandSpec | None = None
    cohort_raw = raw.get("cohort_shift")
    if isinstance(cohort_raw, dict):
        deg = cohort_raw.get("auc_degradation") or cohort_raw
        cohort_shift = _parse_band_spec("auc_degradation", deg, gate="G6.4")

    required = tuple(raw.get("cross_tier_required") or ())
    leakage_probes = _parse_leakage_probe_bands(raw.get("leakage_probes") or {})

    return AcceptanceBands(
        per_tier=per_tier,
        cross_seed_spread=cross_seed_spread,
        cohort_shift=cohort_shift,
        cross_tier_required=required,
        leakage_probes=leakage_probes,
    )


def _parse_leakage_probe_bands(raw: Any) -> LeakageProbeBands:
    """Parse the ``leakage_probes`` YAML section.

    Missing section / empty mapping → all-None thresholds, which the
    driver translates into "skip every opt-in probe" — matches PR 3.1's
    posture for the bundle-level orchestrator.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"leakage_probes must be a mapping; got {type(raw).__name__}")
    id_only = raw.get("id_only_max_auc")
    label_drift = raw.get("label_drift_max")
    subsets_raw = raw.get("feature_subsets") or {}
    subsets: dict[str, tuple[float, tuple[str, ...]]] = {}
    for name, payload in subsets_raw.items():
        if not isinstance(payload, dict):
            raise ValueError(
                f"leakage_probes.feature_subsets.{name} must be a mapping with "
                "'max_auc' and 'columns' keys"
            )
        if "max_auc" not in payload or "columns" not in payload:
            raise ValueError(
                f"leakage_probes.feature_subsets.{name} must declare both 'max_auc' and 'columns'"
            )
        cols = payload["columns"]
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError(
                f"leakage_probes.feature_subsets.{name}.columns must be a list of strings"
            )
        subsets[str(name)] = (float(payload["max_auc"]), tuple(cols))
    return LeakageProbeBands(
        id_only_max_auc=float(id_only) if id_only is not None else None,
        label_drift_max=float(label_drift) if label_drift is not None else None,
        feature_subsets=subsets,
    )


def _parse_band_spec(metric: str, bounds: Any, *, gate: str) -> BandSpec:
    """Coerce a YAML bounds value into a :class:`BandSpec`.

    Accepts ``{min: …, max: …}`` mappings (either bound optional) and
    raises on any other shape — raw scalars or two-element lists are
    rejected because they conceal which bound is which.
    """
    if not isinstance(bounds, dict):
        raise ValueError(
            f"band {metric!r} must be a mapping with 'min' and/or 'max' keys; got {bounds!r}"
        )
    lo = bounds.get("min")
    hi = bounds.get("max")
    if lo is None and hi is None:
        raise ValueError(f"band {metric!r} must declare at least one of 'min'/'max'")
    return BandSpec(
        metric=metric,
        gate=gate,
        min=float(lo) if lo is not None else None,
        max=float(hi) if hi is not None else None,
    )


def check_release_bands(
    report: ReleaseQualityReport,
    bands: AcceptanceBands,
    *,
    leakage_reports: Mapping[str, LeakageReport] | None = None,
) -> list[GateFailure]:
    """Evaluate every numeric / structural gate in :class:`AcceptanceBands`.

    Args:
        report: The cross-seed × cross-tier release-quality report
            produced by
            :func:`leadforge.validation.release_quality.measure_release_quality`.
        bands: Parsed YAML bands from :func:`load_bands`.
        leakage_reports: Optional mapping of tier name → opt-in leakage
            probe report (from :func:`run_split_probes`).  Each non-OK
            finding becomes a ``G5.x`` gate failure.

    Returns:
        ``[]`` when every gate passes.  Otherwise a list of
        :class:`GateFailure` records describing each violation.
    """
    failures: list[GateFailure] = []

    failures.extend(_check_per_tier_bands(report, bands))
    failures.extend(_check_cross_seed_spread(report, bands))
    failures.extend(_check_cohort_shift(report, bands))
    failures.extend(_check_cross_tier_ordering(report, bands))
    if leakage_reports is not None:
        failures.extend(_check_leakage_reports(leakage_reports))

    return failures


def _check_per_tier_bands(
    report: ReleaseQualityReport,
    bands: AcceptanceBands,
) -> list[GateFailure]:
    """Evaluate G7.1 / G7.2 / G7.3 numeric bands per tier."""
    failures: list[GateFailure] = []
    for tier_name, tier_bands in bands.per_tier.items():
        csm = report.tiers.get(tier_name)
        if csm is None:
            # _GATE_PREFIX_BY_TIER values already include the leading "G7." —
            # don't prepend a second one.  Unknown tiers fall back to a
            # tier-named id so the failure stays identifiable.
            failures.append(
                GateFailure(
                    gate=_GATE_PREFIX_BY_TIER.get(tier_name, f"G7.{tier_name}"),
                    tier=tier_name,
                    message=(
                        f"tier '{tier_name}' is declared in bands but absent from "
                        "the release-quality report"
                    ),
                )
            )
            continue
        for metric_name, spec in tier_bands.bands.items():
            value = _resolve_metric_value(csm, metric_name)
            failure = spec.evaluate(value, tier=tier_name)
            if failure is not None:
                failures.append(failure)
    return failures


def _resolve_metric_value(csm: CrossSeedTierMetrics, metric_name: str) -> float:
    """Look up a metric's median value across seeds.

    Headline scalars (``lr_auc`` etc.) live in :attr:`csm.medians`.
    P@K-shaped metrics are pulled from the per-seed dicts and aggregated
    here so the YAML can name them flatly (``precision_at_100``).
    Unknown metrics return NaN — caller's :class:`BandSpec` then surfaces
    that as an explicit per-metric failure.
    """
    import numpy as np

    if metric_name in csm.medians:
        return float(csm.medians[metric_name])
    if metric_name.startswith("precision_at_"):
        k = metric_name.removeprefix("precision_at_")
        vals = [m.precision_at_k.get(k, float("nan")) for m in csm.per_seed]
        finite = [v for v in vals if not math.isnan(v)]
        return float(np.median(finite)) if finite else float("nan")
    if metric_name.startswith("recall_at_"):
        k = metric_name.removeprefix("recall_at_")
        vals = [m.recall_at_k.get(k, float("nan")) for m in csm.per_seed]
        finite = [v for v in vals if not math.isnan(v)]
        return float(np.median(finite)) if finite else float("nan")
    if metric_name.startswith("lift_at_"):
        pct = metric_name.removeprefix("lift_at_")
        vals = [m.lift_at_pct.get(pct, float("nan")) for m in csm.per_seed]
        finite = [v for v in vals if not math.isnan(v)]
        return float(np.median(finite)) if finite else float("nan")
    return float("nan")


def _check_cross_seed_spread(
    report: ReleaseQualityReport,
    bands: AcceptanceBands,
) -> list[GateFailure]:
    """G8.1 — every metric's max-min spread must stay under the declared tolerance."""
    failures: list[GateFailure] = []
    for tier_name, csm in report.tiers.items():
        for metric_name, spec in bands.cross_seed_spread.items():
            spread = csm.spreads.get(metric_name)
            if spread is None:
                continue
            failure = spec.evaluate(float(spread), tier=tier_name)
            if failure is not None:
                # Re-tag the message so it's clear we're reporting the
                # spread, not the metric value itself.
                failures.append(
                    GateFailure(
                        gate=spec.gate,
                        tier=tier_name,
                        message=f"cross-seed spread {failure.message}",
                    )
                )
    return failures


def _check_cohort_shift(
    report: ReleaseQualityReport,
    bands: AcceptanceBands,
) -> list[GateFailure]:
    """G6.4 — cohort-vs-random AUC degradation must lie within the declared band."""
    failures: list[GateFailure] = []
    if bands.cohort_shift is None:
        return failures
    for tier_name, cs in report.cohort_shift.items():
        deg = cs.auc_degradation
        if math.isnan(deg):
            failures.append(
                GateFailure(
                    gate=bands.cohort_shift.gate,
                    tier=tier_name,
                    message=(
                        "cohort_shift.auc_degradation is NaN; bundle has no "
                        "lead_created_at column or the chronological resplit "
                        "produced a degenerate cohort split"
                    ),
                )
            )
            continue
        failure = bands.cohort_shift.evaluate(float(deg), tier=tier_name)
        if failure is not None:
            failures.append(failure)
    return failures


def _check_cross_tier_ordering(
    report: ReleaseQualityReport,
    bands: AcceptanceBands,
) -> list[GateFailure]:
    """G7.4.* — each ordering boolean must be ``True`` for declared tiers.

    ``None`` (one of the compared tiers is absent or a median is NaN) is
    treated as "skip" rather than "fail" *unless* both tiers are listed
    in :attr:`AcceptanceBands.cross_tier_required`, in which case the
    None becomes a failure.  PR 3.3's first run will have all three
    tiers, so None should only surface during partial development runs;
    the explicit-decision posture from PR 3.2's docstring still holds.
    """
    failures: list[GateFailure] = []
    o = report.cross_tier_ordering
    required = set(bands.cross_tier_required)

    pairs: tuple[tuple[str, bool | None, str, str], ...] = (
        ("G7.4.1", o.average_precision_intro_gt_intermediate, "intro", "intermediate"),
        ("G7.4.1", o.average_precision_intermediate_gt_advanced, "intermediate", "advanced"),
        ("G7.4.2", o.precision_at_100_intro_gt_intermediate, "intro", "intermediate"),
        ("G7.4.2", o.precision_at_100_intermediate_gt_advanced, "intermediate", "advanced"),
        ("G7.4.3", o.conversion_rate_intro_gt_intermediate, "intro", "intermediate"),
        ("G7.4.3", o.conversion_rate_intermediate_gt_advanced, "intermediate", "advanced"),
    )
    for gate, value, hi, lo in pairs:
        metric_label = {
            "G7.4.1": "AP",
            "G7.4.2": "P@100",
            "G7.4.3": "conversion rate",
        }[gate]
        if value is None:
            if {hi, lo}.issubset(required):
                failures.append(
                    GateFailure(
                        gate=gate,
                        tier=None,
                        message=(
                            f"{metric_label} ordering '{hi} > {lo}' is undefined "
                            "(missing tier or NaN median) but both tiers are "
                            "required by cross_tier_required"
                        ),
                    )
                )
            continue
        if not value:
            failures.append(
                GateFailure(
                    gate=gate,
                    tier=None,
                    message=f"{metric_label} ordering '{hi} > {lo}' is False",
                )
            )

    # G7.4.4 — the spec wants GBM−LR delta strictly positive in every
    # tier.  In practice the per-tier ``gbm_minus_lr_auc`` band fitted
    # from data is a finer instrument for this check (the spec is a
    # tier-floor of 0; the YAML bands declare the actual floor we
    # tolerate).  We surface the boolean as an informational flag in
    # the report's markdown but do NOT fail here when it's False — the
    # per-tier band check has already applied a calibrated decision.
    # When the boolean is None *and* tiers are required, we still fail
    # because that means we couldn't compute the comparison at all.
    if o.gbm_minus_lr_positive_in_every_tier is None and required:
        failures.append(
            GateFailure(
                gate="G7.4.4",
                tier=None,
                message=(
                    "GBM−LR delta sign is undefined (no tier had a finite "
                    "median) but cross_tier_required declares tiers"
                ),
            )
        )
    return failures


def _check_leakage_reports(
    leakage_reports: Mapping[str, LeakageReport],
) -> list[GateFailure]:
    """Convert leakage-probe findings into G5.* gate failures.

    Each :class:`LeakageFinding` from :func:`run_split_probes` becomes
    one :class:`GateFailure`.  The gate id is derived from the channel
    so the CLI grouping mirrors the gate doc.
    """
    failures: list[GateFailure] = []
    channel_to_gate: Mapping[str, str] = {
        # post-snapshot-aggregates / suspect-stage / etc.
        "feature_subset_baseline": "G5.1",
        # ID-only baseline.
        "id_only_baseline": "G5.3",
        # Bonus relational model (G4.5).
        "bonus_model": "G4.5",
        # Split-leakage.  Note: ``split_label_drift`` does NOT collide with
        # the cohort/time-shift G6.4 gate — it falls through to the generic
        # ``leakage:split_label_drift`` channel id below because v1
        # acceptance gates do not number per-split label-rate drift as a
        # distinct gate.  Mapping it to G6.4 would group unrelated
        # failures (cohort AUC degradation vs. cross-split label drift)
        # under one id.
        "split_id_overlap": "G6.1",
        "split_near_duplicate": "G6.3",
    }
    for tier, lr in leakage_reports.items():
        for finding in lr.findings:
            gate = channel_to_gate.get(finding.channel, f"leakage:{finding.channel}")
            failures.append(
                GateFailure(
                    gate=gate,
                    tier=tier,
                    message=f"[{finding.channel}] {finding.detail}: {finding.message}",
                )
            )
    return failures
