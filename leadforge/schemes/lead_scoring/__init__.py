"""The ``lead_scoring`` generation scheme.

Wraps the existing population + simulation pipeline as a registered
:class:`~leadforge.schemes.base.GenerationScheme`.  This is the first scheme
extracted (LTV-Pd) and is the trunk that the lifecycle scheme parallels.

The implementation modules (``population``, ``engine``, mechanisms, structure,
render) still live under their original package paths; they are physically
relocated into this package in LTV-Pe.  Until then the methods delegate to the
current homes, keeping the lead-scoring bundle's output byte-for-byte identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from leadforge.schemes.base import register_scheme

if TYPE_CHECKING:
    from leadforge.core.models import GenerationConfig
    from leadforge.narrative.spec import NarrativeSpec
    from leadforge.simulation.engine import SimulationResult
    from leadforge.simulation.population import PopulationResult
    from leadforge.structure.graph import WorldGraph


class LeadScoringScheme:
    """The lead-scoring (``converted_within_90_days``) generation pipeline."""

    name = "lead_scoring"

    def build_population(
        self,
        config: GenerationConfig,
        narrative: NarrativeSpec,
        world_graph: WorldGraph,
        *,
        category_latent_correlations: dict | None = None,
    ) -> PopulationResult:
        from leadforge.simulation.population import build_population

        return build_population(
            config,
            narrative,
            world_graph,
            category_latent_correlations=category_latent_correlations,
        )

    def simulate(
        self,
        config: GenerationConfig,
        population: PopulationResult,
        world_graph: WorldGraph,
        *,
        latent_touch_intensity: bool = False,
    ) -> SimulationResult:
        from leadforge.simulation.engine import simulate_world

        return simulate_world(
            config,
            population,
            world_graph,
            latent_touch_intensity=latent_touch_intensity,
        )


LEAD_SCORING_SCHEME = LeadScoringScheme()
register_scheme(LEAD_SCORING_SCHEME)
