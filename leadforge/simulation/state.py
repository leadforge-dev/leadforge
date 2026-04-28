"""Per-lead mutable state for the discrete-time simulation engine.

:class:`LeadSimState` is the only mutable object touched by
:func:`~leadforge.simulation.engine.simulate_world`.  After the simulation
loop completes, the final state of each instance is used to populate the
:class:`~leadforge.schema.entities.LeadRow` and any post-conversion entity
rows (opportunity, customer, subscription).
"""

from __future__ import annotations

from dataclasses import dataclass

# Funnel stages that are at or past the SQL qualification gate.
# Used by advance_stage() to record sql_day regardless of the exact
# stage name, so opportunity timestamps are correct even if the stage
# sequence evolves in future milestones.
_SQL_OR_DEEPER: frozenset[str] = frozenset(
    {
        "sql",
        "demo_scheduled",
        "demo_completed",
        "proposal_sent",
        "negotiation",
        "closed_won",
    }
)


@dataclass
class LeadSimState:
    """Mutable simulation state for one lead across the 90-day horizon.

    The engine updates this object on each daily step.  It is never written
    to disk directly — the final state is used to populate relational rows.

    Args:
        lead_id: Stable opaque lead identifier.
        current_stage: Funnel stage at initialisation (typically ``"mql"``).
    """

    lead_id: str
    current_stage: str
    dwell_days: int = 0
    """Days spent in the current stage; reset to 0 on each stage advance."""

    converted: bool = False
    conversion_day: int | None = None
    """0-based day index within the simulation horizon when conversion fired."""

    churned: bool = False
    churn_day: int | None = None
    """0-based day index when the lead was marked ``closed_lost``."""

    sql_day: int | None = None
    """First day the lead entered ``sql`` or any deeper funnel stage.
    Used to anchor opportunity creation timestamps."""

    @property
    def is_terminal(self) -> bool:
        """``True`` once the lead has converted or churned."""
        return self.converted or self.churned

    def advance_stage(self, new_stage: str, day: int) -> None:
        """Transition to *new_stage* on *day*, resetting the dwell counter.

        Records the first time the lead enters ``sql`` or any deeper stage
        (``demo_scheduled``, ``demo_completed``, ``proposal_sent``,
        ``negotiation``, ``closed_won``) so the engine can create an
        opportunity row at the correct timestamp regardless of which
        qualifying stage is reached first.

        Args:
            new_stage: The funnel stage to transition into.
            day: Current 0-based day index in the simulation horizon.
        """
        self.current_stage = new_stage
        self.dwell_days = 0
        if new_stage in _SQL_OR_DEEPER and self.sql_day is None:
            self.sql_day = day

    def mark_converted(self, day: int) -> None:
        """Record a ``closed_won`` conversion event on *day*."""
        self.converted = True
        self.conversion_day = day
        self.current_stage = "closed_won"

    def mark_churned(self, day: int) -> None:
        """Record a ``closed_lost`` churn event on *day*."""
        self.churned = True
        self.churn_day = day
        self.current_stage = "closed_lost"
