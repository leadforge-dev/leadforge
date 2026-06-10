"""Shared relational-table writer (bundle-output envelope).

:func:`write_relational_tables` is the scheme-agnostic step that serialises a
``{table_name: DataFrame}`` dict to a bundle's ``tables/`` directory.  Each
generation scheme decides the relational *shape* (which tables, any
snapshot-safe projection) and then calls this to write them.  The lead-scoring
table *assembler* (``to_dataframes``) lives with its scheme in
:mod:`leadforge.schemes.lead_scoring.render.relational`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path


def write_relational_tables(
    dfs: dict[str, pd.DataFrame],
    tables_dir: Path,
    *,
    redacted: Collection[str] = frozenset(),
) -> dict[str, int]:
    """Write a ``{table_name: DataFrame}`` dict to *tables_dir* as Parquet.

    A shared, scheme-agnostic envelope step used by each scheme's
    ``write_bundle``: it drops any *redacted* columns present in a table,
    writes one ``<name>.parquet`` per entry, and returns ``{table_name:
    row_count}``.  The relational *shape* (which tables, snapshot-safe
    projection) is the scheme's concern and is decided before calling this.

    Args:
        dfs: Mapping of table name → DataFrame, already projected to the
            published shape (e.g. snapshot-safe for ``student_public``).
        tables_dir: Destination directory (created if absent).
        redacted: Column names to strip from any table that contains them.

    Returns:
        Row count per written table, in *dfs* iteration order.
    """
    from leadforge.schema.tables import write_parquet

    tables_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for table_name, df in dfs.items():
        if redacted:
            cols_to_drop = [c for c in redacted if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        write_parquet(df, tables_dir / f"{table_name}.parquet")
        row_counts[table_name] = len(df)
    return row_counts
