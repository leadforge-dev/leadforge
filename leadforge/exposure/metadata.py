"""Scheme-agnostic hidden-truth metadata for ``research_instructor`` mode.

The bundle's ``metadata/`` directory mixes scheme-agnostic provenance
(``world_spec.json`` — config + narrative) with scheme-specific hidden truth
(the lead-scoring world graph, latent registry, and mechanism summary; the
lifecycle scheme will emit its own).  Only the generic part lives here;
:func:`write_world_spec_json` writes it.  Each scheme owns the rest via its
:meth:`~leadforge.schemes.base.GenerationScheme.write_metadata` hook, called by
:func:`leadforge.exposure.modes.apply_exposure`.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from leadforge.core.models import WorldSpec

__all__ = ["write_world_spec_json"]


def write_world_spec_json(spec: WorldSpec, meta_dir: Path) -> None:
    """Write ``meta_dir/world_spec.json`` — the resolved config + narrative.

    Scheme-agnostic: depends only on the shared :class:`WorldSpec`, so it is
    identical across generation schemes.
    """
    config_dict = dataclasses.asdict(spec.config)
    narrative_dict = dataclasses.asdict(spec.narrative) if spec.narrative is not None else None
    world_spec_dict = {"config": config_dict, "narrative": narrative_dict}
    (meta_dir / "world_spec.json").write_text(json.dumps(world_spec_dict, indent=2))
