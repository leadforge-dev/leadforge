"""Relational export — convert SimulationResult to typed DataFrames.

:func:`to_dataframes` is the single entry point.  It produces one
``pd.DataFrame`` per relational table, with dtypes matching the
:attr:`~leadforge.schema.entities.AccountRow.DTYPE_MAP` of each entity
class.  The resulting dict is consumed by the bundle writer to produce
the ``tables/`` directory in the output bundle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import pandas as pd

from leadforge.schema.entities import (
    AccountRow,
    ContactRow,
    CustomerRow,
    EntityRowProtocol,
    LeadRow,
    OpportunityRow,
    SalesActivityRow,
    SessionRow,
    SubscriptionRow,
    TouchRow,
)

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from leadforge.schemes.lead_scoring.simulation.engine import SimulationResult
    from leadforge.schemes.lead_scoring.simulation.population import PopulationResult

_Source = Literal["population", "simulation"]


class _TableSource(NamedTuple):
    cls: type[EntityRowProtocol]
    origin: _Source  # which object holds the rows
    attr: str  # attribute name on that object


# Maps table name → source descriptor.
_TABLE_SOURCES: dict[str, _TableSource] = {
    AccountRow.TABLE_NAME: _TableSource(AccountRow, "population", "accounts"),
    ContactRow.TABLE_NAME: _TableSource(ContactRow, "population", "contacts"),
    LeadRow.TABLE_NAME: _TableSource(LeadRow, "simulation", "leads"),
    TouchRow.TABLE_NAME: _TableSource(TouchRow, "simulation", "touches"),
    SessionRow.TABLE_NAME: _TableSource(SessionRow, "simulation", "sessions"),
    SalesActivityRow.TABLE_NAME: _TableSource(SalesActivityRow, "simulation", "sales_activities"),
    OpportunityRow.TABLE_NAME: _TableSource(OpportunityRow, "simulation", "opportunities"),
    CustomerRow.TABLE_NAME: _TableSource(CustomerRow, "simulation", "customers"),
    SubscriptionRow.TABLE_NAME: _TableSource(SubscriptionRow, "simulation", "subscriptions"),
}


def to_dataframes(
    result: SimulationResult,
    population: PopulationResult,
) -> dict[str, pd.DataFrame]:
    """Convert simulation output to one typed DataFrame per relational table.

    Args:
        result: Output of :func:`~leadforge.schemes.lead_scoring.simulation.engine.simulate_world`.
        population: Output of
            :func:`~leadforge.schemes.lead_scoring.simulation.population.build_population`.

    Returns:
        Dict mapping table name → ``pd.DataFrame`` with dtypes matching the
        entity class's ``DTYPE_MAP``.  Empty tables are returned as zero-row
        DataFrames with the correct schema.
    """
    dfs: dict[str, pd.DataFrame] = {}
    for table_name, src in _TABLE_SOURCES.items():
        obj = population if src.origin == "population" else result
        rows = getattr(obj, src.attr)  # AttributeError surfaces missing attrs immediately
        if rows:
            df = pd.DataFrame([row.to_dict() for row in rows])
            for col, dtype in src.cls.DTYPE_MAP.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
        else:
            df = src.cls.empty_dataframe()
        dfs[table_name] = df
    return dfs


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
