"""In-memory artifacts produced by the lifecycle (pLTV) pipeline.

:class:`LifecycleArtifacts` is the scheme-owned payload carried by a
:class:`~leadforge.core.models.WorldBundle` for the lifecycle scheme — the
lifecycle analogue of
:class:`~leadforge.schemes.lead_scoring.artifacts.LeadScoringArtifacts`.  The
bundle's ``artifacts`` field is typed ``Any`` in the shared core layer (it must
not reference a scheme); this scheme defines and unwraps its own container here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leadforge.schemes.lifecycle.engine import LifecycleSimulationResult
    from leadforge.schemes.lifecycle.population import CustomerPopulationResult

__all__ = ["LifecycleArtifacts"]


@dataclass
class LifecycleArtifacts:
    """The in-memory result of one lifecycle generation run.

    Attributes:
        population: Accounts, customers, and latent state from
            :func:`~leadforge.schemes.lifecycle.population.build_customer_population`.
        simulation_result: Subscriptions and the three event tables from
            :func:`~leadforge.schemes.lifecycle.engine.simulate_lifecycle`.
        motif_family: The retention motif family sampled for this world (also
            recorded on the population; carried here for convenience).
    """

    population: CustomerPopulationResult
    simulation_result: LifecycleSimulationResult
    motif_family: str
