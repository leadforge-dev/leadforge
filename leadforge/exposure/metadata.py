"""Write hidden-truth metadata files for ``research_instructor`` mode.

:func:`write_metadata_dir` creates ``bundle_root/metadata/`` and populates
it with five files that expose the full hidden world:

- ``graph.json`` — world graph as JSON (nodes, edges, motif family)
- ``graph.graphml`` — world graph as GraphML for graph tools
- ``world_spec.json`` — generation config + narrative spec
- ``latent_registry.json`` — per-entity latent trait values
- ``mechanism_summary.json`` — mechanism assignment summary
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leadforge.core.models import WorldBundle


def write_metadata_dir(bundle: WorldBundle, bundle_root: Path) -> None:
    """Populate ``bundle_root/metadata/`` with all hidden-truth files.

    Args:
        bundle: Fully populated :class:`~leadforge.core.models.WorldBundle`.
        bundle_root: Root directory of the written bundle.
    """
    from leadforge.core.rng import RNGRoot
    from leadforge.mechanisms.policies import assign_mechanisms

    # Callers must only invoke this after full bundle assembly; world_graph
    # and population are guaranteed non-None at that point.
    assert bundle.world_graph is not None  # noqa: S101
    assert bundle.population is not None  # noqa: S101

    meta_dir = bundle_root / "metadata"
    meta_dir.mkdir(exist_ok=True)

    # graph.json + graph.graphml
    (meta_dir / "graph.json").write_text(bundle.world_graph.to_json())
    (meta_dir / "graph.graphml").write_text(bundle.world_graph.to_graphml())

    # latent_registry.json
    ls = bundle.population.latent_state
    latent_registry: dict[str, object] = {
        "account_latents": ls.account_latents,
        "contact_latents": ls.contact_latents,
        "lead_latents": ls.lead_latents,
    }
    (meta_dir / "latent_registry.json").write_text(json.dumps(latent_registry, indent=2))

    # world_spec.json — config + narrative (if present)
    config_dict = dataclasses.asdict(bundle.spec.config)
    narrative_dict = (
        dataclasses.asdict(bundle.spec.narrative) if bundle.spec.narrative is not None else None
    )
    world_spec_dict = {"config": config_dict, "narrative": narrative_dict}
    (meta_dir / "world_spec.json").write_text(json.dumps(world_spec_dict, indent=2))

    # mechanism_summary.json
    # Reconstruct the mechanism assignment with the same RNG substream that
    # was used during simulation — produces the identical parameter values.
    motif_family = bundle.world_graph.motif_family
    mech_rng = RNGRoot(bundle.spec.config.seed).child("mechanisms")
    assignment = assign_mechanisms(motif_family, mech_rng)
    (meta_dir / "mechanism_summary.json").write_text(
        json.dumps(assignment.summary().to_dict(), indent=2)
    )
