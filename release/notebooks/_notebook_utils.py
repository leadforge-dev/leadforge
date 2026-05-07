"""Shared helpers for the public release notebooks.

The release notebooks are downloaded by Kaggle / HF consumers alongside
the parquet bundle.  They cannot rely on ``leadforge`` being installed
in the consumer's environment, so the helpers needed inside the
notebooks live here as a small sibling module with no project imports.

The metric helpers mirror ``leadforge.validation.release_quality`` so
notebook-side numbers and validation-report numbers compare apples to
apples — same ranking convention, same tie-breaking, same calibration
binning — and the ``assert_within_tolerance`` gate (G13.2) is meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def precision_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Mean label of the top-``k`` rows by descending score.

    Mirrors ``leadforge.validation.release_quality._precision_at_k``:
    stable argsort so ties resolve identically to the validation report.
    """
    scores = np.asarray(scores)
    y = np.asarray(y)
    if k <= 0 or k > len(y):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    return float(y[order[:k]].mean())


def top_decile_rate(scores: np.ndarray, y: np.ndarray) -> float:
    """Precision at the top 10% of ranked scores."""
    n = len(y)
    if n == 0:
        return float("nan")
    return precision_at_k(scores, y, max(1, int(round(n * 0.1))))


def assert_within_tolerance(
    observed: Mapping[str, float],
    target: Mapping[str, float],
    tolerances: Mapping[str, float] | float,
    *,
    label: str = "metrics",
) -> None:
    """Assert ``|observed[k] - target[k]| <= tol`` for every key in ``target``.

    Backs the G13.2 acceptance gate inside the notebooks: once the
    notebook has computed its own metrics it pins them against the
    cross-seed-median values from ``release/validation/validation_report.md``
    and fails loudly if the notebook drifts out of band.

    The gate is intentionally strict about silent-pass paths:

    * Non-finite ``observed`` or ``target`` values fail (rather than
      slipping through because ``NaN > tol`` evaluates ``False``).
    * When ``tolerances`` is a mapping, every key in ``target`` must
      have an explicit tolerance — a missing entry is treated as a
      configuration error and aborts the gate up front, instead of
      defaulting to ``+inf`` and silently disabling the check for
      that metric.

    Args:
        observed: Notebook-computed metrics, keyed by metric name.
        target: Reference values from the validation report.
        tolerances: Either a per-metric tolerance map (every key in
            ``target`` must be present), or a single float applied to
            every metric (G13.2's default is 0.05).
        label: Human-readable name for the metric panel; appears in the
            error message so the failing assertion identifies its source.

    Raises:
        AssertionError: when any metric falls outside its tolerance,
            ``observed`` is missing a key listed in ``target``, an
            ``observed`` / ``target`` value is non-finite, or
            ``tolerances`` is a mapping that omits a required key.
    """
    if isinstance(tolerances, int | float):
        per_key: Mapping[str, float] = {k: float(tolerances) for k in target}
    else:
        per_key = tolerances
        missing_tolerances = [k for k in target if k not in per_key]
        if missing_tolerances:
            raise AssertionError(
                f"{label}: tolerances mapping is missing entries for "
                f"target metrics: {sorted(missing_tolerances)}.  "
                "Falling back to +inf would silently disable the gate "
                "for these metrics; declare an explicit tolerance per key."
            )
    failures: list[str] = []
    for key, target_value in target.items():
        if key not in observed:
            failures.append(f"  {key}: missing from observed metrics")
            continue
        observed_f = float(observed[key])
        target_f = float(target_value)
        if not (math.isfinite(observed_f) and math.isfinite(target_f)):
            failures.append(
                f"  {key}: non-finite value (observed={observed_f}, "
                f"target={target_f}) — gate refuses to silently pass NaN/inf"
            )
            continue
        tol = float(per_key[key])
        diff = abs(observed_f - target_f)
        if diff > tol:
            failures.append(
                f"  {key}: observed={observed_f:.4f} target={target_f:.4f} "
                f"|diff|={diff:.4f} > tol={tol:.4f}"
            )
    if failures:
        raise AssertionError(f"{label} drifted outside tolerance:\n" + "\n".join(failures))
