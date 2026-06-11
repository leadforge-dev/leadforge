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

from leadforge.schemes.lead_scoring.entities import (
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
