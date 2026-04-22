"""World graph sampler — draw a concrete hidden world from a motif + seed.

:func:`sample_hidden_graph` is the single entry point consumed by the
simulation layer.  It selects a motif family (deterministically from the
recipe, or randomly from the seed), applies stochastic rewiring, and
returns a validated :class:`~leadforge.structure.graph.WorldGraph`.
"""

from __future__ import annotations

import numpy as np

from leadforge.structure.graph import WorldGraph
from leadforge.structure.motifs import (
    ALL_MOTIF_FAMILIES,
    MotifFamily,
    get_motif_family,
)
from leadforge.structure.rewiring import rewire

# Maximum number of rewiring attempts before giving up.
_MAX_ATTEMPTS = 20


def sample_hidden_graph(
    seed: int,
    motif_family_name: str | None = None,
) -> WorldGraph:
    """Draw a validated hidden world graph.

    The function is fully deterministic given ``(seed, motif_family_name)``.

    Args:
        seed: Integer seed for the NumPy random generator.  All stochastic
            choices (motif selection if *motif_family_name* is ``None``,
            rewiring decisions, weight jitter) derive from this seed.
        motif_family_name: If provided, pin the motif family by name
            (must be one of :data:`~leadforge.structure.motifs.MOTIF_FAMILY_NAMES`).
            If ``None``, a family is chosen uniformly at random from the
            five v1 families.

    Returns:
        A validated :class:`~leadforge.structure.graph.WorldGraph`.

    Raises:
        KeyError: If *motif_family_name* is not a known motif family name.
        RuntimeError: If :data:`_MAX_ATTEMPTS` rewiring attempts all
            produce graphs that fail structural validation (should not
            happen in practice with well-formed motifs).
    """
    rng = np.random.default_rng(seed)

    motif = _select_motif(motif_family_name, rng)

    last_exc: Exception | None = None
    for _attempt in range(_MAX_ATTEMPTS):
        # Each attempt uses an independent sub-seed so that earlier
        # failures do not corrupt the RNG state of later attempts.
        attempt_seed = int(rng.integers(0, 2**31))
        attempt_rng = np.random.default_rng(attempt_seed)
        nodes, edges = rewire(motif, attempt_rng)
        try:
            return WorldGraph(nodes=nodes, edges=edges, motif_family=motif.name)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    raise RuntimeError(
        f"Failed to produce a valid WorldGraph from motif "
        f"{motif.name!r} after {_MAX_ATTEMPTS} rewiring attempts. "
        f"Last error: {last_exc}"
    )


def _select_motif(
    name: str | None,
    rng: np.random.Generator,
) -> MotifFamily:
    """Return the requested motif family, or pick one at random."""
    if name is not None:
        return get_motif_family(name)
    idx = int(rng.integers(0, len(ALL_MOTIF_FAMILIES)))
    return ALL_MOTIF_FAMILIES[idx]
