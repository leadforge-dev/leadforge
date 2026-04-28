"""Relational export — convert SimulationResult to typed DataFrames.

:func:`to_dataframes` is the single entry point.  It produces one
``pd.DataFrame`` per relational table, with dtypes matching the
:attr:`~leadforge.schema.entities.AccountRow.DTYPE_MAP` of each entity
class.  The resulting dict is consumed by the bundle writer to produce
the ``tables/`` directory in the output bundle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    from leadforge.simulation.engine import SimulationResult
    from leadforge.simulation.population import PopulationResult

_Source = Literal["population", "simulation"]

# Maps table name → (entity class, data source, attribute name on source object).
_TABLE_SOURCES: dict[str, tuple[type[EntityRowProtocol], _Source, str]] = {
    AccountRow.TABLE_NAME: (AccountRow, "population", "accounts"),
    ContactRow.TABLE_NAME: (ContactRow, "population", "contacts"),
    LeadRow.TABLE_NAME: (LeadRow, "simulation", "leads"),
    TouchRow.TABLE_NAME: (TouchRow, "simulation", "touches"),
    SessionRow.TABLE_NAME: (SessionRow, "simulation", "sessions"),
    SalesActivityRow.TABLE_NAME: (SalesActivityRow, "simulation", "sales_activities"),
    OpportunityRow.TABLE_NAME: (OpportunityRow, "simulation", "opportunities"),
    CustomerRow.TABLE_NAME: (CustomerRow, "simulation", "customers"),
    SubscriptionRow.TABLE_NAME: (SubscriptionRow, "simulation", "subscriptions"),
}


def to_dataframes(
    result: SimulationResult,
    population: PopulationResult,
) -> dict[str, pd.DataFrame]:
    """Convert simulation output to one typed DataFrame per relational table.

    Args:
        result: Output of :func:`~leadforge.simulation.engine.simulate_world`.
        population: Output of
            :func:`~leadforge.simulation.population.build_population`.

    Returns:
        Dict mapping table name → ``pd.DataFrame`` with dtypes matching the
        entity class's ``DTYPE_MAP``.  Empty tables are returned as zero-row
        DataFrames with the correct schema.
    """
    dfs: dict[str, pd.DataFrame] = {}
    for table_name, (cls, source, attr) in _TABLE_SOURCES.items():
        obj = population if source == "population" else result
        rows = getattr(obj, attr, [])
        if rows:
            df = pd.DataFrame([row.to_dict() for row in rows])
            for col, dtype in cls.DTYPE_MAP.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
        else:
            df = cls.empty_dataframe()
        dfs[table_name] = df
    return dfs
