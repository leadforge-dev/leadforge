"""The ``lifecycle`` generation scheme (``b2b_saas_ltv_v1``).

The second peer scheme alongside ``lead_scoring``.  Its entity rows and FK
constraints live here (``entities`` / ``relationships``); the snapshot, feature,
and task definitions live in sibling modules.  :meth:`LifecycleScheme.build_world`
is implemented (LTV-Pn.4a); :meth:`write_bundle` / :meth:`write_metadata` are
built out in LTV-Pn.4b–c and currently raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from leadforge.schemes.base import register_scheme

if TYPE_CHECKING:
    from pathlib import Path

    from leadforge.core.models import GenerationConfig, WorldBundle
    from leadforge.narrative.spec import NarrativeSpec

_NOT_IMPLEMENTED = (
    "the lifecycle (b2b_saas_ltv_v1) write path is not implemented yet; "
    "it is built across LTV-Pn.4b–c"
)


def _sample_motif_family(rng: random.Random) -> str:
    """Deterministically pick a retention motif family for this world.

    Sampling per-seed (rather than hard-coding one family) honours the
    "world structure varies via named motif families" invariant — different
    seeds yield structurally different worlds.
    """
    from leadforge.schemes.lifecycle.population import LIFECYCLE_MOTIF_FAMILIES

    # Sort for a stable, order-independent candidate list before sampling.
    return rng.choice(sorted(LIFECYCLE_MOTIF_FAMILIES))


class LifecycleScheme:
    """The customer-lifetime-value (pLTV) generation pipeline."""

    name = "lifecycle"

    def build_world(
        self,
        config: GenerationConfig,
        narrative: NarrativeSpec,
        **options: Any,
    ) -> WorldBundle:
        """Sample a motif family, build the customer population, and simulate.

        Deterministic given ``config`` (the population and the per-customer
        weekly simulation derive from ``config.seed`` via distinct RNG
        substreams).  Consumes the lifecycle config fields: ``n_customers``,
        ``observation_date``, ``early_tenure_weeks``, and
        ``forward_windows_days`` (the engine simulates through the longest
        window so every pLTV target is fully covered).

        Not yet applied (tracked, not silent):

        - **Difficulty.**  ``config.difficulty`` / ``difficulty_params`` are
          NOT consumed here, so every difficulty tier currently yields the same
          world.  Two distinct pieces remain: resolving ``difficulty_params``
          from the active profile and threading it into the snapshot
          distortions (``LTV-Pn.4b``, where snapshots are built), and
          simulation-level difficulty scaling that actually makes harder tiers
          harder worlds (deferred — see ``mechanisms.py`` and the roadmap).
        - **Narrative.**  ``narrative`` is accepted for protocol parity but
          unused: the lifecycle population builder generates its own
          firmographics from internal distributions, so the recipe's
          ``narrative.yaml`` will not drive them until ``LTV-Po`` decides
          whether the lifecycle scheme should consume the narrative spec.
        """
        from leadforge.core.models import WorldBundle, WorldSpec
        from leadforge.core.rng import RNGRoot
        from leadforge.schemes.lifecycle.artifacts import LifecycleArtifacts
        from leadforge.schemes.lifecycle.engine import simulate_lifecycle
        from leadforge.schemes.lifecycle.population import build_customer_population

        motif_rng = RNGRoot(config.seed).child("lifecycle_motif")
        motif_family = _sample_motif_family(motif_rng)

        population = build_customer_population(
            config.n_customers,
            config.seed,
            motif_family=motif_family,
            observation_date=config.observation_date,
        )
        simulation_result = simulate_lifecycle(
            population,
            config.seed,
            forward_window_days=max(config.forward_windows_days),
            early_tenure_weeks=config.early_tenure_weeks,
        )

        spec = WorldSpec(config=config, narrative=narrative, scheme=self.name)
        return WorldBundle(
            spec=spec,
            artifacts=LifecycleArtifacts(
                population=population,
                simulation_result=simulation_result,
                motif_family=motif_family,
            ),
        )

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
