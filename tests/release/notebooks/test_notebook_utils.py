"""Unit tests for ``release/notebooks/_notebook_utils.py``.

The notebook helpers ship inside the public bundle (consumers download
them alongside the parquet tables), so they cannot live inside the
``leadforge`` package import tree.  These tests load the module through
``importlib`` and exercise it the way a notebook cell would.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "release" / "notebooks" / "_notebook_utils.py"
_spec = importlib.util.spec_from_file_location("_notebook_utils", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
nbu = importlib.util.module_from_spec(_spec)
sys.modules["_notebook_utils"] = nbu
_spec.loader.exec_module(nbu)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_simple() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 1, 0, 0, 0])
    assert nbu.precision_at_k(scores, y, 2) == pytest.approx(1.0)
    assert nbu.precision_at_k(scores, y, 5) == pytest.approx(0.4)


def test_precision_at_k_handles_ties_via_stable_sort() -> None:
    """Ties resolved by original order — same convention as
    ``release_quality._precision_at_k`` so notebook and validation
    report agree on tied-score rows.
    """
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([1, 0, 1, 0])
    # Stable argsort of -scores preserves [0,1,2,3] order, so top-2 = y[:2]
    assert nbu.precision_at_k(scores, y, 2) == pytest.approx(0.5)


def test_precision_at_k_invalid_k_returns_nan() -> None:
    scores = np.array([0.9, 0.8])
    y = np.array([1, 0])
    assert np.isnan(nbu.precision_at_k(scores, y, 0))
    assert np.isnan(nbu.precision_at_k(scores, y, 3))


def test_top_decile_rate_uses_10_percent_cut() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(100)
    y = (scores > 0.9).astype(int)
    # Top-10 by score = exactly the 10 positives → top decile rate = 1.0
    assert nbu.top_decile_rate(scores, y) == pytest.approx(1.0)


def test_top_decile_rate_empty_returns_nan() -> None:
    assert np.isnan(nbu.top_decile_rate(np.array([]), np.array([])))


# ---------------------------------------------------------------------------
# assert_within_tolerance
# ---------------------------------------------------------------------------


def test_assert_within_tolerance_passes_when_inside_band() -> None:
    nbu.assert_within_tolerance(
        observed={"auc": 0.88, "ap": 0.57},
        target={"auc": 0.886, "ap": 0.575},
        tolerances=0.05,
    )


def test_assert_within_tolerance_fails_when_outside_band() -> None:
    with pytest.raises(AssertionError) as exc:
        nbu.assert_within_tolerance(
            observed={"auc": 0.50},
            target={"auc": 0.886},
            tolerances=0.05,
        )
    msg = str(exc.value)
    assert "auc" in msg
    assert "observed=0.5000" in msg
    assert "target=0.8860" in msg


def test_assert_within_tolerance_per_metric_tolerances() -> None:
    nbu.assert_within_tolerance(
        observed={"auc": 0.83, "brier": 0.105},
        target={"auc": 0.886, "brier": 0.110},
        tolerances={"auc": 0.10, "brier": 0.05},
    )


def test_assert_within_tolerance_reports_missing_key() -> None:
    with pytest.raises(AssertionError, match="missing from observed metrics"):
        nbu.assert_within_tolerance(
            observed={"auc": 0.88},
            target={"auc": 0.886, "brier": 0.110},
            tolerances=0.05,
        )


def test_assert_within_tolerance_label_appears_in_error() -> None:
    with pytest.raises(AssertionError, match="notebook 01"):
        nbu.assert_within_tolerance(
            observed={"auc": 0.5},
            target={"auc": 0.886},
            tolerances=0.05,
            label="notebook 01",
        )


def test_assert_within_tolerance_aggregates_multiple_failures() -> None:
    with pytest.raises(AssertionError) as exc:
        nbu.assert_within_tolerance(
            observed={"auc": 0.50, "ap": 0.10},
            target={"auc": 0.886, "ap": 0.575},
            tolerances=0.05,
        )
    msg = str(exc.value)
    assert "auc" in msg
    assert "ap" in msg


def test_assert_within_tolerance_ignores_extra_observed_keys() -> None:
    """Observed metrics may carry extras (e.g. GBM AUC); the gate only
    enforces the keys present in ``target``.
    """
    nbu.assert_within_tolerance(
        observed={"auc": 0.88, "ap": 0.57, "extra": 999.0},
        target={"auc": 0.886, "ap": 0.575},
        tolerances=0.05,
    )


# ---------------------------------------------------------------------------
# Silent-pass paths: NaN / inf observed and target, missing tolerance keys
# ---------------------------------------------------------------------------


def test_assert_within_tolerance_fails_on_nan_observed() -> None:
    """``NaN > tol`` is ``False`` in IEEE 754; without an explicit
    finiteness check a NaN-valued observed metric would slip through
    the gate.
    """
    with pytest.raises(AssertionError, match="non-finite"):
        nbu.assert_within_tolerance(
            observed={"auc": float("nan")},
            target={"auc": 0.886},
            tolerances=0.05,
        )


def test_assert_within_tolerance_fails_on_inf_observed() -> None:
    with pytest.raises(AssertionError, match="non-finite"):
        nbu.assert_within_tolerance(
            observed={"auc": float("inf")},
            target={"auc": 0.886},
            tolerances=0.05,
        )


def test_assert_within_tolerance_fails_on_nan_target() -> None:
    """A NaN target would also produce a NaN diff and bypass the gate."""
    with pytest.raises(AssertionError, match="non-finite"):
        nbu.assert_within_tolerance(
            observed={"auc": 0.85},
            target={"auc": float("nan")},
            tolerances=0.05,
        )


def test_assert_within_tolerance_rejects_incomplete_tolerance_mapping() -> None:
    """A per-metric tolerances dict missing keys present in ``target``
    used to default each missing key to ``+inf``, silently disabling the
    gate for those metrics.  The fix is to fail up front with the list
    of missing keys, treating it as a configuration error.
    """
    with pytest.raises(AssertionError, match="missing entries for target metrics"):
        nbu.assert_within_tolerance(
            observed={"auc": 0.88, "brier": 0.10},
            target={"auc": 0.886, "brier": 0.110},
            tolerances={"auc": 0.05},  # ``brier`` deliberately missing
        )


# ---------------------------------------------------------------------------
# precision_at_k must mirror release_quality._precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_mirrors_release_quality() -> None:
    """The notebook helper's docstring claims byte-equivalence with
    ``leadforge.validation.release_quality._precision_at_k`` (same
    stable argsort, same tie-breaking).  This test pins that claim:
    if either implementation drifts, the notebook's reproduction gate
    silently drifts with it.
    """
    from leadforge.validation import release_quality

    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    y = (rng.random(1000) > 0.7).astype(int)

    for k in (1, 10, 50, 100, 250, 500, 999):
        nbu_value = nbu.precision_at_k(scores, y, k)
        rq_value = release_quality._precision_at_k(scores, y, k)
        assert nbu_value == pytest.approx(rq_value), (
            f"divergence at k={k}: notebook helper {nbu_value}, release_quality {rq_value}"
        )

    # Tied scores — the convention drift this is most likely to surface.
    tied_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3])
    tied_y = np.array([1, 0, 1, 0, 1, 1, 0])
    for k in (1, 2, 4, 6, 7):
        assert nbu.precision_at_k(tied_scores, tied_y, k) == pytest.approx(
            release_quality._precision_at_k(tied_scores, tied_y, k)
        )


def test_top_decile_rate_mirrors_release_quality() -> None:
    """``release_quality._top_decile_rate`` and the notebook helper share
    the same ``max(1, int(round(n * 0.1)))`` k-selection rule today.
    Lock that in: if either side ever changes (e.g. switches to
    ``ceil`` or ``floor`` on edge cases), the gate would silently
    diverge from the validation report.  Includes the exact `n` the
    intermediate tier ships (``n_test = 750``) and a few small `n`
    where banker's rounding bites.
    """
    from leadforge.validation import release_quality

    rng = np.random.default_rng(0)
    for n in (5, 10, 25, 99, 100, 750, 1234):
        scores = rng.random(n)
        y = (rng.random(n) > 0.7).astype(int)
        assert nbu.top_decile_rate(scores, y) == pytest.approx(
            release_quality._top_decile_rate(scores, y)
        ), f"divergence at n={n}"
