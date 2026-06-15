"""Snapshot-safe relational export for ``student_public`` lifecycle bundles.

:func:`to_dataframes_snapshot_safe` projects the full-horizon dict from
:func:`leadforge.schemes.lifecycle.render.relational.to_dataframes` onto the
shape published in public bundles, enforcing the design.md §5 contract against
the absolute calendar ``cutoff`` (the world ``observation_date``):

* event tables (``subscription_events`` / ``health_signals`` / ``invoices``)
  are row-filtered to ``timestamp <= cutoff`` — no post-cutoff events;
* ``subscriptions`` drops its stateful/terminal columns
  (:data:`LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS`), keeping only the at-signing
  identity (plan, term, start) — current MRR / status / counts / churn fields
  all hold end-of-simulation values that leak the pLTV / churn targets;
* ``accounts`` / ``customers`` pass through (firmographic / at-signing, no
  post-cutoff state).

The public **task** parquets are already snapshot-safe by construction (their
features are computed at/before the cutoff and each carries only its own
target); this module only governs the relational ``tables/``.

The cutoff is the calendar regime's ``observation_date``.  The early-pLTV
(tenure-anchored) task family is therefore **omitted from public bundles**
(``LifecycleScheme.write_bundle``): its forward window precedes
``observation_date``, so its targets would be reconstructible by joining the
public event tables (the invoices between the early cutoff and
``observation_date`` *are* the early target window).  A single
``observation_date``-anchored relational export cannot serve both regimes; the
early family stays instructor-only.

``research_instructor`` keeps the full-horizon
:func:`~leadforge.schemes.lifecycle.render.relational.to_dataframes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from leadforge.validation.leakage_probes import (
    LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS,
    LIFECYCLE_SNAPSHOT_FILTERED_TABLES,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

__all__ = [
    "LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS",
    "LIFECYCLE_SNAPSHOT_FILTERED_TABLES",
    "to_dataframes_snapshot_safe",
]

# Canonical output order (parity with the full-horizon to_dataframes).
_OUTPUT_ORDER = (
    "accounts",
    "customers",
    "subscriptions",
    "subscription_events",
    "health_signals",
    "invoices",
)


def to_dataframes_snapshot_safe(
    dfs: Mapping[str, pd.DataFrame],
    *,
    cutoff: str,
) -> dict[str, pd.DataFrame]:
    """Project the full-horizon lifecycle relational dict to its public shape.

    Args:
        dfs: Output of
            :func:`leadforge.schemes.lifecycle.render.relational.to_dataframes`.
            Input frames are never mutated.
        cutoff: Absolute ISO date (the world ``observation_date``); event rows
            with a timestamp strictly after it are dropped.

    Returns:
        A new dict in canonical order.  ``subscriptions`` has its
        stateful/terminal columns removed; the event tables are row-filtered to
        ``<= cutoff``; ``accounts`` / ``customers`` pass through.

    Raises:
        ValueError: if *cutoff* is empty.
    """
    if not cutoff:
        raise ValueError("cutoff (observation_date) must be a non-empty ISO date string")

    filtered_tables = dict(LIFECYCLE_SNAPSHOT_FILTERED_TABLES)
    banned = set(LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS)

    out: dict[str, pd.DataFrame] = {}
    for name in _OUTPUT_ORDER:
        if name not in dfs:
            continue
        df = dfs[name]
        if name == "subscriptions":
            out[name] = df.drop(columns=[c for c in banned if c in df.columns])
        elif name in filtered_tables:
            ts_col = filtered_tables[name]
            # ISO date strings compare correctly lexicographically.
            out[name] = df[df[ts_col] <= cutoff].reset_index(drop=True)
        else:
            out[name] = df
    return out
