"""Lifecycle relational export — one typed DataFrame per relational table.

:func:`to_dataframes` flattens a lifecycle population + simulation result into
the six relational tables: ``accounts`` (shared firmographics) plus the five
lifecycle entity tables (``customers``, ``subscriptions``,
``subscription_events``, ``health_signals``, ``invoices``).  Mirrors the
lead-scoring :func:`~leadforge.schemes.lead_scoring.render.relational.to_dataframes`
pattern: a table-source registry drives a uniform row → DataFrame conversion
with dtypes from each entity's ``DTYPE_MAP``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from leadforge.schema.entities import AccountRow, EntityRowProtocol
from leadforge.schemes.lifecycle.entities import (
    CustomerLifecycleRow,
    HealthSignalRow,
    InvoiceRow,
    SubscriptionEventRow,
    SubscriptionLifecycleRow,
)

if TYPE_CHECKING:
    from leadforge.schemes.lifecycle.engine import LifecycleSimulationResult
    from leadforge.schemes.lifecycle.population import CustomerPopulationResult

__all__ = ["to_dataframes"]


@dataclass(frozen=True)
class _TableSource:
    cls: type[EntityRowProtocol]
    origin: str  # "population" or "simulation"
    attr: str


# Table name → where its rows come from.  Insertion order is the table order.
_TABLE_SOURCES: dict[str, _TableSource] = {
    AccountRow.TABLE_NAME: _TableSource(AccountRow, "population", "accounts"),
    CustomerLifecycleRow.TABLE_NAME: _TableSource(CustomerLifecycleRow, "population", "customers"),
    SubscriptionLifecycleRow.TABLE_NAME: _TableSource(
        SubscriptionLifecycleRow, "simulation", "subscriptions"
    ),
    SubscriptionEventRow.TABLE_NAME: _TableSource(
        SubscriptionEventRow, "simulation", "subscription_events"
    ),
    HealthSignalRow.TABLE_NAME: _TableSource(HealthSignalRow, "simulation", "health_signals"),
    InvoiceRow.TABLE_NAME: _TableSource(InvoiceRow, "simulation", "invoices"),
}


def to_dataframes(
    result: LifecycleSimulationResult,
    population: CustomerPopulationResult,
) -> dict[str, pd.DataFrame]:
    """Convert lifecycle output to one typed DataFrame per relational table.

    Returns:
        Dict mapping table name → ``pd.DataFrame`` with dtypes from each entity
        class's ``DTYPE_MAP``.  Empty tables are zero-row DataFrames with the
        correct schema.
    """
    dfs: dict[str, pd.DataFrame] = {}
    for table_name, src in _TABLE_SOURCES.items():
        obj = population if src.origin == "population" else result
        rows = getattr(obj, src.attr)
        if rows:
            df = pd.DataFrame([row.to_dict() for row in rows])
            for col, dtype in src.cls.DTYPE_MAP.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
        else:
            df = src.cls.empty_dataframe()
        dfs[table_name] = df
    return dfs
