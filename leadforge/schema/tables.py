"""Parquet serialization helpers for schema-conformant DataFrames.

These utilities are used by the rendering layer (M7+) and by tests to verify
that empty tables can be round-tripped through Parquet without schema loss.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to a Parquet file at *path*, creating parent directories.

    Args:
        df: DataFrame to serialize.  Should be created via an entity class's
            ``empty_dataframe()`` or populated with real simulation rows.
        path: Destination ``.parquet`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file back into a DataFrame.

    Args:
        path: Path to the ``.parquet`` file.

    Returns:
        A ``pd.DataFrame`` with columns as stored in the file.
    """
    return pd.read_parquet(path, engine="pyarrow")
