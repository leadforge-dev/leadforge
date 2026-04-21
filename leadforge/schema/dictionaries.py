"""Feature dictionary builder.

Converts :data:`~leadforge.schema.features.LEAD_SNAPSHOT_FEATURES` into a
``pd.DataFrame`` and optionally writes it as ``feature_dictionary.csv`` — one
of the three files required in every bundle output mode (§14.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES, FeatureSpec

if TYPE_CHECKING:
    pass

_COLUMNS = ("name", "dtype", "description", "category", "is_target", "leakage_risk")


def feature_dictionary_df(
    features: tuple[FeatureSpec, ...] = LEAD_SNAPSHOT_FEATURES,
) -> pd.DataFrame:
    """Return the feature dictionary as a ``pd.DataFrame``.

    Columns: name, dtype, description, category, is_target, leakage_risk.

    Args:
        features: Ordered tuple of :class:`~leadforge.schema.features.FeatureSpec`
            objects.  Defaults to the canonical lead snapshot feature list.

    Returns:
        A ``pd.DataFrame`` with one row per feature.  String columns
        (``name``, ``dtype``, ``description``, ``category``) use
        ``pd.StringDtype``; flag columns (``is_target``, ``leakage_risk``)
        use ``pd.BooleanDtype``.
    """
    rows = [
        {
            "name": f.name,
            "dtype": f.dtype,
            "description": f.description,
            "category": f.category,
            "is_target": f.is_target,
            "leakage_risk": f.leakage_risk,
        }
        for f in features
    ]
    df = pd.DataFrame(rows, columns=list(_COLUMNS))
    for col in ("name", "dtype", "description", "category"):
        df[col] = df[col].astype("string")
    df["is_target"] = df["is_target"].astype("boolean")
    df["leakage_risk"] = df["leakage_risk"].astype("boolean")
    return df


def write_feature_dictionary(
    path: Path,
    features: tuple[FeatureSpec, ...] = LEAD_SNAPSHOT_FEATURES,
) -> None:
    """Write the feature dictionary CSV to *path*.

    Args:
        path: Destination file path (created with ``parents=True``).
        features: Feature list to serialize.  Defaults to the canonical list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_dictionary_df(features).to_csv(path, index=False)
