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

Where the seam sits
-------------------
A scheme owns the **whole** generation pipeline from ``(config, narrative)`` to
in-memory world artifacts: structure/graph sampling, difficulty interpretation,
population, simulation, and :class:`~leadforge.core.models.WorldBundle`
assembly.  These steps differ between schemes (the lead-scoring hidden DAG,
``DifficultyParams``, and touch emission are all lead-scoring-specific), so the
boundary is the single :meth:`GenerationScheme.build_world` method rather than a
set of lead-scoring-shaped sub-steps.  This keeps
:meth:`~leadforge.api.generator.Generator.generate` genuinely scheme-agnostic.

Scheme-specific options are passed through ``Generator.generate(**kwargs)`` to
``build_world`` and consumed by the scheme that understands them (e.g.
``latent_touch_intensity`` for lead scoring).

Scope note
----------
Render dispatch (``to_dataframes`` / snapshots / task splits, currently in
``WorldBundle.save`` → the bundle writer) is folded into the scheme as the
lifecycle scheme is built out (see ``docs/ltv/roadmap.md`` — LTV-M6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from leadforge.core.exceptions import LeadforgeError

if TYPE_CHECKING:
    from leadforge.core.models import GenerationConfig, WorldBundle
    from leadforge.narrative.spec import NarrativeSpec


class UnknownSchemeError(LeadforgeError):
    """Raised when a generation-scheme name is not present in the registry."""


@runtime_checkable
class GenerationScheme(Protocol):
    """One end-to-end dataset generation pipeline shape.

    Implementations are registered by :attr:`name` and resolved at generation
    time.  :meth:`build_world` must be deterministic given ``(config,
    narrative, options)`` per the package's RNG contract.
    """

    name: str

    def build_world(
        self,
        config: GenerationConfig,
        narrative: NarrativeSpec,
        **options: Any,
    ) -> WorldBundle:
        """Run the scheme's full pipeline and return an in-memory bundle.

        Implementations own structure sampling, difficulty interpretation,
        population, simulation, and bundle assembly.  ``options`` carries
        scheme-specific flags forwarded from ``Generator.generate(**kwargs)``;
        a scheme ignores options it does not recognise.
        """
        ...

    def write_bundle(
        self,
        bundle: WorldBundle,
        path: str,
        generation_timestamp: str | None = None,
    ) -> None:
        """Serialise *bundle* to *path* (the render half of the pipeline).

        Implementations own the bundle's on-disk shape: which relational tables
        are written, the task snapshot/splits, dataset card, feature dictionary,
        exposure filtering, and manifest.  Shared envelope helpers
        (``build_manifest``, ``apply_exposure``, ``get_filter``) are reused
        across schemes.
        """
        ...


# Name → scheme instance.  Populated by importing the built-in scheme modules
# (each self-registers on import).  ``_ensure_builtins`` triggers this lazily so
# resolution works even if a caller reaches for ``base`` directly without first
# importing the ``leadforge.schemes`` package.
SCHEME_REGISTRY: dict[str, GenerationScheme] = {}

_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import built-in scheme modules once, so the registry is populated.

    Importing the ``leadforge.schemes`` package runs its ``__init__``, which
    imports each shipped scheme for its registration side effect.  Guarded by a
    module flag and idempotent (``import_module`` returns the cached module), so
    this is safe to call on every resolution and re-entrant during package
    import.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    import importlib

    importlib.import_module("leadforge.schemes")


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
    _ensure_builtins()
    try:
        return SCHEME_REGISTRY[name]
    except KeyError:
        raise UnknownSchemeError(
            f"Unknown generation scheme {name!r}. Registered schemes: {sorted(SCHEME_REGISTRY)}"
        ) from None


def available_schemes() -> tuple[str, ...]:
    """Return the names of all registered schemes, sorted."""
    _ensure_builtins()
    return tuple(sorted(SCHEME_REGISTRY))
