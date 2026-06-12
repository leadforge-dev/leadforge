"""Scheme-agnostic difficulty distortions for snapshot tables.

:func:`apply_difficulty_distortions` injects Gaussian noise, MCAR missingness,
and outliers into the numeric feature columns of a snapshot DataFrame,
parameterized by a scheme's :class:`~leadforge.schema.features.FeatureSpec`
catalog.  Extracted from the lead-scoring snapshot builder (verbatim op order
and RNG substream, so existing outputs stay byte-identical) so the lifecycle
scheme can share it.

Known wart (inherited, locked by byte-identity with shipped lead-scoring
bundles): missingness injection converts an Int64 column to Float64 **only if
at least one of its cells is masked**, so the post-distortion dtype of integer
columns varies with seed and missing_rate.  Consumers must not rely on
integer dtypes surviving distortion.

Column eligibility is derived from the feature catalog rather than runtime
dtype sniffing — categoricals, booleans, IDs, and target columns are never
distorted even if their runtime dtype happens to be numeric.  Callers exempt
pedagogical leakage-trap columns explicitly (distorting a trap muddies the
lesson the trap exists to teach).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from leadforge.core.rng import RNGRoot

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from leadforge.core.models import DifficultyParams
    from leadforge.schema.features import FeatureSpec

__all__ = ["apply_difficulty_distortions"]

_FLOAT_DTYPES = ("Float64", "float64")
_NUMERIC_DTYPES = ("Float64", "float64", "Int64", "int64")


def apply_difficulty_distortions(
    df: pd.DataFrame,
    params: DifficultyParams,
    seed: int,
    *,
    feature_specs: Sequence[FeatureSpec],
    exempt_cols: frozenset[str] = frozenset(),
    rng_substream: str = "snapshot_distortions",
) -> pd.DataFrame:
    """Apply noise, missingness, and outliers to numeric snapshot features.

    Args:
        df: The snapshot table.  Not mutated — a new DataFrame is returned.
        params: Difficulty knobs (``noise_scale``, ``missing_rate``,
            ``outlier_rate``); a knob at 0 disables that distortion.
        seed: Seed for the distortion RNG substream.  Pass the generation
            seed so distortions are deterministic per run.
        feature_specs: The scheme's snapshot feature catalog.  Float-dtyped,
            non-target, non-exempt features receive noise and outliers; all
            numeric non-target, non-exempt features receive missingness.
            Targets are never distorted.
        exempt_cols: Columns excluded from every distortion — deliberate
            leakage traps whose signal must survive intact.
        rng_substream: Name of the numpy child stream.  Schemes with multiple
            distortion call sites must use distinct names.

    Returns:
        A distorted copy of *df*.
    """
    float_distortion_cols = [
        f.name
        for f in feature_specs
        if f.dtype in _FLOAT_DTYPES and not f.is_target and f.name not in exempt_cols
    ]
    numeric_distortion_cols = [
        f.name
        for f in feature_specs
        if f.dtype in _NUMERIC_DTYPES and not f.is_target and f.name not in exempt_cols
    ]
    # Post-noise physical-range clamps, derived from FeatureSpec.non_negative
    # so the lists stay in sync automatically when features are added/renamed.
    nonneg_float_cols = frozenset(
        f.name for f in feature_specs if f.dtype in _FLOAT_DTYPES and f.non_negative
    )
    nonneg_int_cols = frozenset(
        f.name for f in feature_specs if f.dtype in ("Int64", "int64") and f.non_negative
    )

    df = df.copy()
    rng_root = RNGRoot(seed)
    np_rng = rng_root.numpy_child(rng_substream)

    # Filter to columns actually present (guards against feature spec drift).
    float_cols = [c for c in float_distortion_cols if c in df.columns]
    all_numeric_cols = [c for c in numeric_distortion_cols if c in df.columns]

    # 1. Gaussian noise on float features only (avoids int casting issues).
    if params.noise_scale > 0:
        for col in float_cols:
            valid_mask = df[col].notna()
            if valid_mask.sum() == 0:
                continue
            col_std = float(df.loc[valid_mask, col].std())
            if col_std == 0 or np.isnan(col_std):
                continue
            noise = np_rng.normal(0, params.noise_scale * col_std, size=len(df))
            # Add noise only where values are valid.
            values = df[col].copy()
            values[valid_mask] = values[valid_mask] + noise[valid_mask.values]
            df[col] = values

    # 1b. Post-noise clamp to physical ranges.
    # Non-negative float columns: clip to >= 0.
    for col in nonneg_float_cols:
        if col in df.columns and df[col].notna().any():
            df[col] = df[col].clip(lower=0)
    # Non-negative int columns: clip to >= 0.  clip() preserves Int64 dtype.
    for col in nonneg_int_cols:
        if col in df.columns and df[col].notna().any():
            df[col] = df[col].clip(lower=0)

    # 2. MCAR missingness injection (all numeric columns).
    if params.missing_rate > 0:
        mask = np_rng.random(size=(len(df), len(all_numeric_cols))) < params.missing_rate
        for i, col in enumerate(all_numeric_cols):
            col_mask = mask[:, i]
            if col_mask.any():
                # Convert int columns to float to support NaN.
                if df[col].dtype in ("int64", "Int64"):
                    df[col] = df[col].astype("Float64")
                df.loc[col_mask, col] = np.nan

    # 3. Outlier injection (float columns only).  Uses 5σ to produce values
    #    clearly distinguishable from natural variation.
    if params.outlier_rate > 0:
        for col in float_cols:
            valid_mask = df[col].notna()
            col_std = float(df.loc[valid_mask, col].std())
            if col_std == 0 or np.isnan(col_std):
                continue
            col_median = float(df[col].median())
            outlier_mask = np_rng.random(size=len(df)) < params.outlier_rate
            signs = np_rng.choice([-1, 1], size=len(df)).astype(float)
            outlier_values = col_median + signs * 5 * col_std
            combined = outlier_mask & valid_mask.values
            if combined.any():
                df.loc[combined, col] = outlier_values[combined]

    return df
