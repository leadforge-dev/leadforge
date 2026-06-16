"""The ``lead_scoring`` generation scheme.

Owns the lead-scoring pipeline — hidden-DAG sampling, difficulty interpretation,
population, simulation, and bundle assembly — behind the single
:meth:`~leadforge.schemes.base.GenerationScheme.build_world` entry point.  This
is the first scheme extracted, and the trunk the lifecycle scheme parallels.

The compute-core modules (``simulation``, ``mechanisms``, ``structure``) live
under this package as of LTV-Pf.  The render modules (``snapshots``,
``relational``, ``tasks``) still live under ``leadforge.render`` and are
relocated in a follow-up; ``build_world`` / ``write_bundle`` import from their
current homes.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from leadforge.schemes.base import register_scheme

if TYPE_CHECKING:
    from pathlib import Path

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
        from leadforge.schemes.lead_scoring.simulation.engine import simulate_world
        from leadforge.schemes.lead_scoring.simulation.population import build_population
        from leadforge.schemes.lead_scoring.structure.sampler import sample_hidden_graph

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

        from leadforge.schemes.lead_scoring.artifacts import LeadScoringArtifacts

        spec = WorldSpec(config=config, narrative=narrative, scheme=self.name)
        return WorldBundle(
            spec=spec,
            artifacts=LeadScoringArtifacts(
                population=population,
                simulation_result=result,
                world_graph=world_graph,
            ),
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

        ``build_manifest`` (``LTV-Pn.1``) and ``apply_exposure`` (``LTV-Pn.2``)
        are now scheme-agnostic: ``apply_exposure`` writes the generic
        ``world_spec.json`` and delegates this scheme's hidden-truth files
        (graph, latent registry, mechanism summary) to
        :meth:`write_metadata`.  The remaining shared-orchestrator decomposition
        (a bundle orchestrator with scheme render hooks lifted out of each
        ``write_bundle``) is deferred to ``LTV-Pn.4``, once the lifecycle
        ``write_bundle`` exists to reveal the real shared shape.
        """
        from pathlib import Path

        from leadforge.exposure.filters import get_filter
        from leadforge.narrative.dataset_card import render_dataset_card
        from leadforge.render.bundle import TaskExport, write_bundle_envelope
        from leadforge.schemes.lead_scoring.artifacts import LeadScoringArtifacts
        from leadforge.schemes.lead_scoring.features import (
            LEAD_SNAPSHOT_FEATURES,
            redacted_columns_for,
        )
        from leadforge.schemes.lead_scoring.render.relational import to_dataframes
        from leadforge.schemes.lead_scoring.render.relational_snapshot_safe import (
            to_dataframes_snapshot_safe,
        )
        from leadforge.schemes.lead_scoring.render.snapshots import build_snapshot
        from leadforge.schemes.lead_scoring.tasks import task_manifest_for_config

        artifacts = bundle.artifacts
        if not isinstance(artifacts, LeadScoringArtifacts):
            raise RuntimeError(
                "WorldBundle is not populated with lead-scoring artifacts. "
                "Call Generator.generate() first."
            )

        config = bundle.spec.config
        result = artifacts.simulation_result
        population = artifacts.population
        world_graph = artifacts.world_graph

        # The redaction set comes from the canonical feature spec — applied to
        # every published parquet (relational tables AND task splits) so a user
        # cannot reintroduce a redacted column by joining the raw tables.
        redacted = redacted_columns_for(config.exposure_mode)
        bundle_filter = get_filter(config.exposure_mode)

        # Relational shape (9 tables; snapshot-safe projection for public).
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
        # Row counts for the dataset card == write_relational_tables' counts
        # (redaction drops columns, not rows).
        table_counts = {name: len(df) for name, df in dfs.items()}

        # Lead snapshot + single task, redacted to the exposure mode.
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

        dataset_card = render_dataset_card(
            bundle.spec,
            task_manifest=task,
            table_counts=table_counts,
            features=visible_features,
        )

        write_bundle_envelope(
            bundle,
            Path(path),
            relational=dfs,
            tasks=[TaskExport(manifest=task, frame=snapshot)],
            dataset_card=dataset_card,
            feature_specs=visible_features,
            generation_scheme=self.name,
            redacted=redacted,
            motif_family=world_graph.motif_family,
            relational_snapshot_safe=bundle_filter.relational_snapshot_safe,
            generation_timestamp=generation_timestamp,
        )

    def write_metadata(self, bundle: WorldBundle, meta_dir: Path) -> None:
        """Write the lead-scoring hidden-truth files into *meta_dir*.

        Called by :func:`leadforge.exposure.modes.apply_exposure` for
        ``research_instructor`` mode (after the shared, scheme-agnostic
        ``world_spec.json`` is written).  Emits the hidden world graph
        (``graph.json`` / ``graph.graphml``), the per-entity latent registry
        (``latent_registry.json``), and the mechanism-assignment summary
        (``mechanism_summary.json``).
        """
        import json

        from leadforge.core.rng import RNGRoot
        from leadforge.schemes.lead_scoring.artifacts import LeadScoringArtifacts
        from leadforge.schemes.lead_scoring.mechanisms.policies import assign_mechanisms

        artifacts = bundle.artifacts
        if not isinstance(artifacts, LeadScoringArtifacts):
            raise RuntimeError("WorldBundle is not populated with lead-scoring artifacts.")
        world_graph = artifacts.world_graph

        (meta_dir / "graph.json").write_text(world_graph.to_json())
        (meta_dir / "graph.graphml").write_text(world_graph.to_graphml())

        ls = artifacts.population.latent_state
        latent_registry: dict[str, object] = {
            "account_latents": ls.account_latents,
            "contact_latents": ls.contact_latents,
            "lead_latents": ls.lead_latents,
        }
        (meta_dir / "latent_registry.json").write_text(json.dumps(latent_registry, indent=2))

        # Reconstruct the mechanism assignment with the same RNG substream used
        # during simulation — produces the identical parameter values.
        motif_family = world_graph.motif_family
        mech_rng = RNGRoot(bundle.spec.config.seed).child("mechanisms")
        assignment = assign_mechanisms(motif_family, mech_rng)
        (meta_dir / "mechanism_summary.json").write_text(
            json.dumps(assignment.summary().to_dict(), indent=2)
        )


LEAD_SCORING_SCHEME = LeadScoringScheme()
register_scheme(LEAD_SCORING_SCHEME)
