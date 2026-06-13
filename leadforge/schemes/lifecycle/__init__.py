"""The ``lifecycle`` generation scheme (``b2b_saas_ltv_v1``) — scaffold.

Registers the second peer scheme alongside ``lead_scoring``.  Its entity rows
and FK constraints live here (``entities`` / ``relationships``); the pipeline
itself (``build_world`` / ``write_bundle``) is built out across LTV-M3…M6 and
currently raises :class:`NotImplementedError`.  Registering the stub now lets
the registry, recipe ``scheme:`` resolution, and tests treat lifecycle as a
first-class peer before its internals exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from leadforge.schemes.base import register_scheme

if TYPE_CHECKING:
    from pathlib import Path

    from leadforge.core.models import GenerationConfig, WorldBundle
    from leadforge.narrative.spec import NarrativeSpec

_NOT_IMPLEMENTED = (
    "the lifecycle (b2b_saas_ltv_v1) scheme is not implemented yet; "
    "its pipeline is built across LTV-M3…M6"
)


class LifecycleScheme:
    """Stub for the customer-lifetime-value (pLTV) generation pipeline."""

    name = "lifecycle"

    def build_world(
        self,
        config: GenerationConfig,
        narrative: NarrativeSpec,
        **options: Any,
    ) -> WorldBundle:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def write_bundle(
        self,
        bundle: WorldBundle,
        path: str,
        generation_timestamp: str | None = None,
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def write_metadata(self, bundle: WorldBundle, meta_dir: Path) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)


LIFECYCLE_SCHEME = LifecycleScheme()
register_scheme(LIFECYCLE_SCHEME)
