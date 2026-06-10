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


LEAD_SCORING_SCHEME = LeadScoringScheme()
register_scheme(LEAD_SCORING_SCHEME)
