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
        from leadforge.core.exceptions import InvalidConfigError
        from leadforge.core.models import WorldBundle, WorldSpec
        from leadforge.core.rng import RNGRoot
        from leadforge.schemes.lifecycle.artifacts import LifecycleArtifacts
        from leadforge.schemes.lifecycle.engine import simulate_lifecycle
        from leadforge.schemes.lifecycle.population import build_customer_population
        from leadforge.schemes.lifecycle.snapshots import FORWARD_WINDOWS_DAYS

        # config.forward_windows_days is not yet threaded into the snapshot
        # builder, which exports the fixed FORWARD_WINDOWS_DAYS targets.  Reject
        # an override now (clear, early) rather than emit a bundle whose manifest
        # disagrees with its task dirs, or under-simulate and fail opaquely later.
        # Threading config-driven windows through is tracked for a later step.
        if tuple(config.forward_windows_days) != tuple(FORWARD_WINDOWS_DAYS):
            raise InvalidConfigError(
                f"config.forward_windows_days={tuple(config.forward_windows_days)} differs "
                f"from the lifecycle scheme's exported windows {tuple(FORWARD_WINDOWS_DAYS)}; "
                "config-driven forward windows are not yet supported (the snapshot builder "
                "exports the fixed set).  Use the default until that wiring lands."
            )

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
        """Serialise a lifecycle *bundle* to *path*.

        Writes the six relational tables, both observation regimes' snapshots
        split into 8 task directories (3 pLTV regression + 1 churn
        classification per regime, the early regime prefixed ``early_``), a
        dataset card, the feature dictionary, the hidden-truth ``metadata/``
        (instructor only, via :meth:`write_metadata`), and the manifest
        (``generation_scheme`` + ``observation_date`` + forward windows).

        ``config.difficulty_params`` is threaded into both snapshot builders —
        when set (LTV-Po resolves it from the recipe profile), it drives the
        snapshot distortions.

        ``student_public`` bundles are projected snapshot-safe: the relational
        event tables are filtered to ``<= observation_date`` and the
        ``subscriptions`` table's stateful/terminal columns are dropped (see
        :mod:`leadforge.schemes.lifecycle.render.relational_snapshot_safe`); no
        ``metadata/`` is written; and the manifest records
        ``relational_snapshot_safe`` + ``structural_redactions``.  The per-task
        splits are single-target and cutoff-bounded by construction.
        """
        from pathlib import Path

        from leadforge.exposure.filters import get_filter
        from leadforge.exposure.modes import apply_exposure
        from leadforge.render.manifests import build_manifest, write_manifest
        from leadforge.render.relational_io import write_relational_tables
        from leadforge.render.tasks import write_task_splits
        from leadforge.schema.dictionaries import write_feature_dictionary
        from leadforge.schemes.lifecycle.artifacts import LifecycleArtifacts
        from leadforge.schemes.lifecycle.features import CUSTOMER_SNAPSHOT_FEATURES
        from leadforge.schemes.lifecycle.render.dataset_card import render_lifecycle_dataset_card
        from leadforge.schemes.lifecycle.render.relational import to_dataframes
        from leadforge.schemes.lifecycle.render.relational_snapshot_safe import (
            LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS,
            to_dataframes_snapshot_safe,
        )
        from leadforge.schemes.lifecycle.snapshots import (
            FORWARD_WINDOWS_DAYS,
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
        bundle_filter = get_filter(config.exposure_mode)

        population = artifacts.population
        sim = artifacts.simulation_result
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        # 1. Relational tables → tables/
        #    student_public is projected snapshot-safe (event tables filtered to
        #    <= observation_date; subscriptions' stateful/terminal columns
        #    dropped).  research_instructor keeps the full-horizon shape.
        dfs = to_dataframes(sim, population)
        structural_redactions: dict[str, object] | None = None
        if bundle_filter.relational_snapshot_safe:
            dfs = to_dataframes_snapshot_safe(dfs, cutoff=population.observation_date)
            structural_redactions = {
                "columns": {"subscriptions": sorted(LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS)},
                "omitted_tables": [],
            }
        table_row_counts = write_relational_tables(dfs, root / "tables")

        # 2. Regime snapshots → task directories.
        #    difficulty_params (None until LTV-Po resolves it) drives distortions.
        #
        # The early-pLTV (tenure-anchored) family is OMITTED from snapshot-safe
        # public bundles: its forward window (start + early_tenure_weeks + Nd)
        # precedes the relational cutoff (observation_date), so its targets are
        # reconstructible by joining the public event tables (invoices between
        # the early cutoff and observation_date *are* the early target window).
        # One observation_date-anchored relational export cannot serve both
        # regimes; the early family stays instructor-only.  The calendar family
        # is safe (its targets fall after observation_date, absent from the
        # public relational tables).
        snapshots = {
            CALENDAR_REGIME: build_customer_snapshot(
                population, sim, difficulty_params=config.difficulty_params, seed=config.seed
            ),
        }
        if not bundle_filter.relational_snapshot_safe:
            snapshots[EARLY_REGIME] = build_early_pltv_snapshot(
                population,
                sim,
                early_tenure_weeks=config.early_tenure_weeks,
                difficulty_params=config.difficulty_params,
                seed=config.seed,
            )
        # Each task is a standalone single-target split: drop every OTHER
        # target column so a task's parquet cannot leak the answer's siblings
        # (e.g. ltv_revenue_730d ⊇ ltv_revenue_90d).  The deliberate
        # mrr_change_full_period trap (leakage_risk but not a target) is kept.
        all_target_cols = {f.name for f in CUSTOMER_SNAPSHOT_FEATURES if f.is_target}
        task_row_counts: dict[str, dict[str, int]] = {}
        all_tasks = []
        for regime, snapshot in snapshots.items():
            for task in lifecycle_task_manifests(regime):
                other_targets = [
                    c for c in all_target_cols - {task.label_column} if c in snapshot.columns
                ]
                task_df = snapshot.drop(columns=other_targets)
                counts = write_task_splits(task_df, root / "tasks", seed=config.seed, task=task)
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
                # The actual exported target windows (source of truth), not
                # config.forward_windows_days — build_world rejects any mismatch.
                "forward_windows_days": list(FORWARD_WINDOWS_DAYS),
                "early_tenure_weeks": config.early_tenure_weeks,
            },
            relational_snapshot_safe=bundle_filter.relational_snapshot_safe,
            structural_redactions=structural_redactions,
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
