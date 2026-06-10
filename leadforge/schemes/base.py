"""Generation-scheme abstraction — the registry of peer dataset pipelines.

leadforge hosts multiple *generation schemes* as peers (e.g. ``lead_scoring``
and, from the LTV workstream, ``lifecycle``).  Each scheme owns one end-to-end
pipeline shape — population → simulation → render → tasks — while the outer
envelope (RNG, config resolution, bundle layout, manifest, exposure dispatch,
CLI) is shared.  See ``docs/ltv/design.md`` §2.5.

A scheme is a small object registered by ``name`` in :data:`SCHEME_REGISTRY`
and resolved via :func:`get_scheme`.  The recipe declares which scheme it runs
via its ``scheme:`` field; :class:`~leadforge.api.generator.Generator` looks the
scheme up and runs its pipeline rather than branching on a recipe type.

Scope note
----------
This protocol currently covers the *generation* half (population + simulation)
that flows through ``Generator.generate()``.  Render dispatch (``to_dataframes``
/ snapshots / task splits) is added to the protocol as the lifecycle scheme is
built out (see ``docs/ltv/roadmap.md`` — LTV-M6); today the bundle writer still
calls the lead-scoring render functions directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from leadforge.core.exceptions import LeadforgeError

if TYPE_CHECKING:
    from leadforge.core.models import GenerationConfig
    from leadforge.narrative.spec import NarrativeSpec
    from leadforge.simulation.engine import SimulationResult
    from leadforge.simulation.population import PopulationResult
    from leadforge.structure.graph import WorldGraph


class UnknownSchemeError(LeadforgeError):
    """Raised when a generation-scheme name is not present in the registry."""


@runtime_checkable
class GenerationScheme(Protocol):
    """One end-to-end dataset generation pipeline shape.

    Implementations are registered by :attr:`name` and resolved at generation
    time.  The two methods below are the generation half of the pipeline; both
    must be deterministic given ``(config, ...)`` per the package's RNG
    contract.
    """

    name: str

    def build_population(
        self,
        config: GenerationConfig,
        narrative: NarrativeSpec,
        world_graph: WorldGraph,
        *,
        category_latent_correlations: dict | None = None,
    ) -> PopulationResult:
        """Generate the scheme's base population (entities + latent state)."""
        ...

    def simulate(
        self,
        config: GenerationConfig,
        population: PopulationResult,
        world_graph: WorldGraph,
        *,
        latent_touch_intensity: bool = False,
    ) -> SimulationResult:
        """Run the scheme's simulation over *population*, returning event tables."""
        ...


# Name → scheme instance.  Populated by importing ``leadforge.schemes`` (its
# package ``__init__`` imports each built-in scheme module, which self-register).
SCHEME_REGISTRY: dict[str, GenerationScheme] = {}


def register_scheme(scheme: GenerationScheme) -> None:
    """Register *scheme* under its ``name``.

    Idempotent for the same instance; raises if a *different* scheme is already
    registered under the same name (guards against accidental shadowing).
    """
    name = scheme.name
    existing = SCHEME_REGISTRY.get(name)
    if existing is not None and existing is not scheme:
        raise ValueError(f"A different generation scheme named {name!r} is already registered")
    SCHEME_REGISTRY[name] = scheme


def get_scheme(name: str) -> GenerationScheme:
    """Return the registered scheme named *name*.

    Raises:
        UnknownSchemeError: if no scheme is registered under *name*.
    """
    try:
        return SCHEME_REGISTRY[name]
    except KeyError:
        raise UnknownSchemeError(
            f"Unknown generation scheme {name!r}. Registered schemes: {sorted(SCHEME_REGISTRY)}"
        ) from None


def available_schemes() -> tuple[str, ...]:
    """Return the names of all registered schemes, sorted."""
    return tuple(sorted(SCHEME_REGISTRY))
