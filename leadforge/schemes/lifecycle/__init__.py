"""The ``lifecycle`` generation scheme (``b2b_saas_ltv_v1``).

The second peer scheme alongside ``lead_scoring``.  Its entity rows and FK
constraints live here (``entities`` / ``relationships``); the snapshot, feature,
and task definitions live in sibling modules.  ``build_world`` (LTV-Pn.4a) and
the instructor-mode ``write_bundle`` / ``write_metadata`` (LTV-Pn.4b) are
implemented; the ``student_public`` snapshot-safe export lands in LTV-Pn.4c.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from leadforge.schemes.base import register_scheme

if TYPE_CHECKING:
    from pathlib import Path

    from leadforge.core.models import GenerationConfig, WorldBundle
    from leadforge.narrative.spec import NarrativeSpec

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
        """Serialise a lifecycle *bundle* to *path* (instructor mode).

        Writes the six relational tables, both observation regimes' snapshots
        split into 8 task directories (3 pLTV regression + 1 churn
        classification per regime, the early regime prefixed ``early_``), a
        dataset card, the feature dictionary, the hidden-truth ``metadata/``
        (via :meth:`write_metadata`), and the manifest (recording
        ``generation_scheme`` + ``observation_date`` + the forward windows).

        ``config.difficulty_params`` is threaded into both snapshot builders —
        when set (LTV-Po resolves it from the recipe profile), it drives the
        snapshot distortions.

        Only ``research_instructor`` mode is supported here.  The
        ``student_public`` snapshot-safety projection (event-table cutoff
        filtering, terminal-column drops, per-task target projection) lands in
        LTV-Pn.4c; until then this refuses to write a public bundle rather than
        emit one that is not snapshot-safe.
        """
        from pathlib import Path

        from leadforge.core.enums import ExposureMode
        from leadforge.exposure.modes import apply_exposure
        from leadforge.render.manifests import build_manifest, write_manifest
        from leadforge.render.relational_io import write_relational_tables
        from leadforge.render.tasks import write_task_splits
        from leadforge.schema.dictionaries import write_feature_dictionary
        from leadforge.schemes.lifecycle.artifacts import LifecycleArtifacts
        from leadforge.schemes.lifecycle.features import CUSTOMER_SNAPSHOT_FEATURES
        from leadforge.schemes.lifecycle.render.dataset_card import render_lifecycle_dataset_card
        from leadforge.schemes.lifecycle.render.relational import to_dataframes
        from leadforge.schemes.lifecycle.snapshots import (
            build_customer_snapshot,
            build_early_pltv_snapshot,
        )
        from leadforge.schemes.lifecycle.tasks import (
            CALENDAR_REGIME,
            EARLY_REGIME,
            lifecycle_task_manifests,
        )

        artifacts = bundle.artifacts
        if not isinstance(artifacts, LifecycleArtifacts):
            raise RuntimeError(
                "WorldBundle is not populated with lifecycle artifacts. "
                "Call Generator.generate() / build_world() first."
            )
        config = bundle.spec.config
        if config.exposure_mode is not ExposureMode.research_instructor:
            raise NotImplementedError(
                f"lifecycle write_bundle currently supports only "
                f"research_instructor; {config.exposure_mode.value!r} (snapshot-safe "
                "public export) lands in LTV-Pn.4c"
            )

        population = artifacts.population
        sim = artifacts.simulation_result
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        # 1. Relational tables → tables/
        dfs = to_dataframes(sim, population)
        table_row_counts = write_relational_tables(dfs, root / "tables")

        # 2. Both regime snapshots → 8 task directories.
        #    difficulty_params (None until LTV-Po resolves it) drives distortions.
        snapshots = {
            CALENDAR_REGIME: build_customer_snapshot(
                population, sim, difficulty_params=config.difficulty_params, seed=config.seed
            ),
            EARLY_REGIME: build_early_pltv_snapshot(
                population,
                sim,
                early_tenure_weeks=config.early_tenure_weeks,
                difficulty_params=config.difficulty_params,
                seed=config.seed,
            ),
        }
        task_row_counts: dict[str, dict[str, int]] = {}
        all_tasks = []
        for regime, snapshot in snapshots.items():
            for task in lifecycle_task_manifests(regime):
                counts = write_task_splits(snapshot, root / "tasks", seed=config.seed, task=task)
                task_row_counts[task.task_id] = counts
                all_tasks.append(task)

        # 3. Dataset card + feature dictionary
        (root / "dataset_card.md").write_text(
            render_lifecycle_dataset_card(
                bundle.spec,
                table_counts=table_row_counts,
                tasks=tuple(all_tasks),
                observation_date=population.observation_date,
            )
        )
        write_feature_dictionary(
            root / "feature_dictionary.csv", features=tuple(CUSTOMER_SNAPSHOT_FEATURES)
        )

        # 4. Exposure metadata (delegates hidden truth to write_metadata)
        apply_exposure(bundle, root, config.exposure_mode)

        # 5. Manifest
        manifest = build_manifest(
            config=config,
            generation_scheme=self.name,
            motif_family=artifacts.motif_family,
            table_row_counts=table_row_counts,
            task_row_counts=task_row_counts,
            bundle_root=root,
            generation_timestamp=generation_timestamp,
            extra_fields={
                "observation_date": population.observation_date,
                "forward_windows_days": list(config.forward_windows_days),
                "early_tenure_weeks": config.early_tenure_weeks,
            },
        )
        write_manifest(manifest, root)

    def write_metadata(self, bundle: WorldBundle, meta_dir: Path) -> None:
        """Write the lifecycle hidden-truth files into *meta_dir*.

        Called by :func:`leadforge.exposure.modes.apply_exposure` after the
        shared ``world_spec.json``.  The lifecycle scheme has no hidden graph;
        its latent truth is the per-entity latent registry and the
        motif-derived mechanism parameters.
        """
        import json

        from leadforge.schemes.lifecycle.artifacts import LifecycleArtifacts
        from leadforge.schemes.lifecycle.render.metadata import (
            latent_registry_dict,
            mechanism_summary_dict,
        )

        artifacts = bundle.artifacts
        if not isinstance(artifacts, LifecycleArtifacts):
            raise RuntimeError("WorldBundle is not populated with lifecycle artifacts.")

        (meta_dir / "latent_registry.json").write_text(
            json.dumps(latent_registry_dict(artifacts.population.latent_state), indent=2)
        )
        (meta_dir / "mechanism_summary.json").write_text(
            json.dumps(mechanism_summary_dict(artifacts.motif_family), indent=2)
        )


LIFECYCLE_SCHEME = LifecycleScheme()
register_scheme(LIFECYCLE_SCHEME)
