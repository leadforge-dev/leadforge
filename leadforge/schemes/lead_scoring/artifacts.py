"""In-memory artifacts produced by the lead-scoring pipeline.

:class:`LeadScoringArtifacts` is the scheme-owned payload carried by a
:class:`~leadforge.core.models.WorldBundle` for this scheme.  The bundle's
``artifacts`` field is typed ``Any`` in the shared core layer (it must not
reference a scheme); each scheme defines and unwraps its own container here,
so the core never depends on lead-scoring types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leadforge.schemes.lead_scoring.simulation.engine import SimulationResult
    from leadforge.schemes.lead_scoring.simulation.population import PopulationResult
    from leadforge.schemes.lead_scoring.structure.graph import WorldGraph

__all__ = ["LeadScoringArtifacts"]


@dataclass
class LeadScoringArtifacts:
    """The in-memory result of one lead-scoring generation run."""

    population: PopulationResult
    simulation_result: SimulationResult
    world_graph: WorldGraph
