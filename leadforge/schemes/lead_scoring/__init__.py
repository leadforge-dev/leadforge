"""The ``lead_scoring`` generation scheme.

Owns the lead-scoring pipeline — hidden-DAG sampling, difficulty interpretation,
population, simulation, and bundle assembly — behind the single
:meth:`~leadforge.schemes.base.GenerationScheme.build_world` entry point.  This
is the first scheme extracted (LTV-Pd) and the trunk the lifecycle scheme
parallels.

The implementation modules (``population``, ``engine``, mechanisms, structure,
render) still live under their original package paths; they are physically
relocated into this package in LTV-Pe.  Until then ``build_world`` calls the
current homes, keeping the lead-scoring bundle's output byte-for-byte identical.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from leadforge.schemes.base import register_scheme

if TYPE_CHECKING:
    from leadforge.core.models import GenerationConfig, WorldBundle
    from leadforge.narrative.spec import NarrativeSpec


class LeadScoringScheme:
    """The lead-scoring (``converted_within_90_days``) generation pipeline."""

    name = "lead_scoring"

    def build_world(
        self,
        config: GenerationConfig,
        narrative: NarrativeSpec,
        **options: Any,
    ) -> WorldBundle:
        """Sample the hidden world, build the population, simulate, and assemble.

        Recognised ``options``:
            latent_touch_intensity (bool): use the latent-driven touch
                intensity mechanism instead of recency decay.  Default ``False``.
        """
        from leadforge.core.models import WorldBundle, WorldSpec
        from leadforge.core.rng import RNGRoot
        from leadforge.simulation.engine import simulate_world
        from leadforge.simulation.population import build_population
        from leadforge.structure.sampler import sample_hidden_graph

        latent_touch_intensity = bool(options.get("latent_touch_intensity", False))

        rng_root = RNGRoot(config.seed)
        world_graph = sample_hidden_graph(rng_root)

        config, category_latent_correlations = self._resolve_difficulty(config)

        population = build_population(
            config,
            narrative,
            world_graph,
            category_latent_correlations=category_latent_correlations,
        )
        result = simulate_world(
            config,
            population,
            world_graph,
            latent_touch_intensity=latent_touch_intensity,
        )

        spec = WorldSpec(config=config, narrative=narrative, scheme=self.name)
        return WorldBundle(
            spec=spec,
            population=population,
            simulation_result=result,
            world_graph=world_graph,
        )

    @staticmethod
    def _resolve_difficulty(
        config: GenerationConfig,
    ) -> tuple[GenerationConfig, dict | None]:
        """Attach :class:`DifficultyParams` to *config* and return category-latent
        correlations from the active difficulty profile.

        Returns ``(config, None)`` unchanged if the recipe has no
        difficulty-profiles file (e.g. ad-hoc configs in tests).
        """
        from leadforge.api.recipes import Recipe
        from leadforge.core.models import DifficultyParams
        from leadforge.recipes.registry import load_recipe

        try:
            raw = load_recipe(config.recipe_id)
            recipe = Recipe.from_dict(raw)
            profiles = recipe.load_difficulty_profiles()
        except (FileNotFoundError, KeyError):
            return config, None

        profile = profiles.get(config.difficulty.value, {})
        category_latent_correlations = profile.get("category_latent_correlations")

        # All keys are required — a missing key indicates a malformed profile
        # YAML and should fail loudly rather than silently defaulting.
        required_keys = (
            "signal_strength",
            "noise_scale",
            "missing_rate",
            "outlier_rate",
            "conversion_rate_range",
            "committee_friction",
        )
        missing = [k for k in required_keys if k not in profile]
        if missing:
            from leadforge.core.exceptions import InvalidRecipeError

            raise InvalidRecipeError(
                f"Difficulty profile '{config.difficulty.value}' is missing "
                f"required keys: {missing}"
            )
        cr_range = profile["conversion_rate_range"]
        difficulty_params = DifficultyParams(
            signal_strength=profile["signal_strength"],
            noise_scale=profile["noise_scale"],
            missing_rate=profile["missing_rate"],
            outlier_rate=profile["outlier_rate"],
            conversion_rate_lo=cr_range[0],
            conversion_rate_hi=cr_range[1],
            committee_friction=profile["committee_friction"],
        )
        return dataclasses.replace(config, difficulty_params=difficulty_params), (
            category_latent_correlations
        )

    def write_bundle(
        self,
        bundle: WorldBundle,
        path: str,
        generation_timestamp: str | None = None,
    ) -> None:
        """Serialise a lead-scoring *bundle* to *path*.

        This method currently owns the *entire* lead-scoring on-disk shape —
        relational export (snapshot-safe for ``student_public``), lead snapshot
        + task splits, dataset card, feature dictionary, exposure metadata, and
        manifest.  Only the genuinely scheme-agnostic relational-table write is
        factored out (``write_relational_tables``); the rest is intentionally
        *not* yet shared.

        The deeper envelope/scheme decomposition (a shared bundle orchestrator
        with scheme render hooks) is deferred to ``LTV-M6``: it requires
        generalising ``build_manifest`` (today it takes the lead-scoring
        ``world_graph``) and ``apply_exposure`` (today it writes the
        lead-scoring hidden graph + latent registry), and is best designed with
        a second scheme in hand.  Until then ``LifecycleScheme.write_bundle``
        will reuse ``write_relational_tables`` + the leaf helpers
        (``build_manifest`` / ``apply_exposure`` / ``get_filter``) but
        orchestrate them itself.
        """
        from pathlib import Path

        from leadforge.exposure.filters import get_filter
        from leadforge.exposure.modes import apply_exposure
        from leadforge.narrative.dataset_card import render_dataset_card
        from leadforge.render.manifests import build_manifest, write_manifest
        from leadforge.render.relational import to_dataframes, write_relational_tables
        from leadforge.render.relational_snapshot_safe import to_dataframes_snapshot_safe
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits
        from leadforge.schema.dictionaries import write_feature_dictionary
        from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES, redacted_columns_for
        from leadforge.schema.tasks import task_manifest_for_config

        if (
            bundle.simulation_result is None
            or bundle.population is None
            or bundle.world_graph is None
        ):
            raise RuntimeError(
                "WorldBundle is not fully populated. Call Generator.generate() first."
            )

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        config = bundle.spec.config
        result = bundle.simulation_result
        population = bundle.population
        world_graph = bundle.world_graph

        # The redaction set comes from the canonical feature spec — the same
        # source of truth the validator uses.  It is applied uniformly to
        # every published parquet file (relational tables AND task splits) so
        # users doing feature engineering off the raw tables (per the
        # README's "Option 3") cannot trivially reintroduce a redacted
        # column by joining ``tables/leads.parquet`` to their feature set.
        redacted = redacted_columns_for(config.exposure_mode)
        bundle_filter = get_filter(config.exposure_mode)

        # ------------------------------------------------------------------
        # 1. Relational tables → tables/
        #
        # The lead-scoring *shape* (9 tables; snapshot-safe projection for
        # student_public) is decided here; the redaction-drop + parquet-write +
        # row-count loop is the shared, scheme-agnostic envelope step.
        # ------------------------------------------------------------------
        dfs = to_dataframes(result, population)
        if bundle_filter.relational_snapshot_safe:
            if config.snapshot_day is None:
                raise ValueError(
                    f"exposure_mode={config.exposure_mode.value!r} requires "
                    "config.snapshot_day to be set (the snapshot-safe relational "
                    "export filters event tables to lead_created_at + snapshot_day); "
                    "got snapshot_day=None.  Pin a snapshot_day on the recipe or "
                    "pass it explicitly."
                )
            dfs = to_dataframes_snapshot_safe(dfs, snapshot_day=config.snapshot_day)
        table_row_counts = write_relational_tables(dfs, root / "tables", redacted=redacted)

        # ------------------------------------------------------------------
        # 2. Snapshot + task splits → tasks/
        # ------------------------------------------------------------------
        snapshot = build_snapshot(
            result,
            population,
            horizon_days=config.horizon_days,
            snapshot_day=config.snapshot_day,
            difficulty_params=config.difficulty_params,
            seed=config.seed,
        )
        if redacted:
            drop_cols = [c for c in redacted if c in snapshot.columns]
            if drop_cols:
                snapshot = snapshot.drop(columns=drop_cols)
        visible_features = tuple(f for f in LEAD_SNAPSHOT_FEATURES if f.name not in redacted)

        task = task_manifest_for_config(config.primary_task, config.label_window_days)
        task_row_counts = write_task_splits(snapshot, root / "tasks", seed=config.seed, task=task)

        # ------------------------------------------------------------------
        # 3. Dataset card and feature dictionary
        # ------------------------------------------------------------------
        (root / "dataset_card.md").write_text(
            render_dataset_card(
                bundle.spec,
                task_manifest=task,
                table_counts=table_row_counts,
                features=visible_features,
            )
        )
        write_feature_dictionary(root / "feature_dictionary.csv", features=visible_features)

        # ------------------------------------------------------------------
        # 4. Exposure metadata (research_instructor only)
        # ------------------------------------------------------------------
        apply_exposure(bundle, root, config.exposure_mode)

        # ------------------------------------------------------------------
        # 5. Manifest
        # ------------------------------------------------------------------
        manifest = build_manifest(
            config=config,
            world_graph=world_graph,
            table_row_counts=table_row_counts,
            task_row_counts={task.task_id: task_row_counts},
            bundle_root=root,
            generation_timestamp=generation_timestamp,
            redacted_columns=sorted(redacted),
            relational_snapshot_safe=bundle_filter.relational_snapshot_safe,
        )
        write_manifest(manifest, root)


LEAD_SCORING_SCHEME = LeadScoringScheme()
register_scheme(LEAD_SCORING_SCHEME)
