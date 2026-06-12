"""Tests for the shared difficulty-distortion helper (render/distortions.py)."""

import pandas as pd

from leadforge.core.models import DifficultyParams
from leadforge.render.distortions import apply_difficulty_distortions
from leadforge.schema.features import FeatureSpec

_SPECS = [
    FeatureSpec(name="feat_f", dtype="Float64", description="", category="x", non_negative=True),
    FeatureSpec(name="trap", dtype="Float64", description="", category="x", non_negative=True),
    FeatureSpec(
        name="target",
        dtype="Float64",
        description="",
        category="target",
        is_target=True,
        non_negative=True,
    ),
]

_PARAMS = DifficultyParams(
    signal_strength=1.0,
    noise_scale=1.0,
    missing_rate=0.0,
    outlier_rate=0.0,
    conversion_rate_lo=0.02,
    conversion_rate_hi=0.4,
    committee_friction=0.5,
)


def _frame() -> pd.DataFrame:
    # Deliberately out-of-contract negatives in the target and trap columns:
    # the helper must not "repair" columns it is forbidden from touching.
    return pd.DataFrame(
        {
            "feat_f": pd.array([1.0, 2.0, 3.0, 4.0], dtype="Float64"),
            "trap": pd.array([-5.0, 1.0, 2.0, 3.0], dtype="Float64"),
            "target": pd.array([-7.0, 0.0, 10.0, 20.0], dtype="Float64"),
        }
    )


def test_targets_never_clamped_or_distorted() -> None:
    """Regression (Copilot review on #119): the non-negative clamp lists must
    exclude targets — 'targets are never distorted' has to hold by
    construction, including the clip step."""
    out = apply_difficulty_distortions(_frame(), _PARAMS, seed=3, feature_specs=_SPECS)
    pd.testing.assert_series_equal(out["target"], _frame()["target"])


def test_exempt_columns_never_clamped_or_distorted() -> None:
    out = apply_difficulty_distortions(
        _frame(), _PARAMS, seed=3, feature_specs=_SPECS, exempt_cols=frozenset({"trap"})
    )
    pd.testing.assert_series_equal(out["trap"], _frame()["trap"])


def test_nonneg_feature_columns_still_clamped() -> None:
    out = apply_difficulty_distortions(_frame(), _PARAMS, seed=3, feature_specs=_SPECS)
    assert (out["feat_f"].dropna() >= 0).all()
    assert not out["feat_f"].equals(_frame()["feat_f"])  # noise actually applied
