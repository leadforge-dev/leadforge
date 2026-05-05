"""Public Generator API."""

from __future__ import annotations

from typing import Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.models import DifficultyParams, GenerationConfig, WorldBundle, WorldSpec
from leadforge.core.rng import RNGRoot
from leadforge.core.sentinels import _MISSING


class Generator:
    """High-level entry point for generating a synthetic CRM dataset bundle.

    Usage::

        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1",
            seed=42,
            exposure_mode="student_public",
        )
        bundle = gen.generate(n_leads=5000, difficulty="intermediate")
        bundle.save("./out/demo_bundle")

    ``from_recipe`` is implemented in Milestone 1–2. Full generation
    (``generate``) is implemented across Milestones 3–9.
    """

    def __init__(self, world_spec: WorldSpec) -> None:
        self._world_spec = world_spec
        self._rng = RNGRoot(world_spec.config.seed)

    @property
    def config(self) -> GenerationConfig:
        return self._world_spec.config

    @property
    def world_spec(self) -> WorldSpec:
        """The resolved world specification, including narrative."""
        return self._world_spec

    @classmethod
    def from_recipe(
        cls,
        recipe_id: str,
        *,
        seed: int = _MISSING,  # type: ignore[assignment]
        exposure_mode: str | ExposureMode = _MISSING,  # type: ignore[assignment]
        difficulty: str | DifficultyProfile = _MISSING,  # type: ignore[assignment]
        n_accounts: int | None = None,
        n_contacts: int | None = None,
        n_leads: int | None = None,
        horizon_days: int | None = None,
        primary_task: str | None = None,
        label_window_days: int | None = None,
        snapshot_day: int | None = None,
        output_path: str = _MISSING,  # type: ignore[assignment]
        override: dict[str, Any] | None = None,
    ) -> Generator:
        """Create a :class:`Generator` from a recipe ID, applying config precedence.

        Args:
            recipe_id: Identifier of a registered recipe (e.g.
                ``"b2b_saas_procurement_v1"``).
            seed: Master RNG seed. Defaults to the package default (42).
            exposure_mode: ``"student_public"`` or ``"research_instructor"``.
                Defaults to the package default (``student_public``).
            difficulty: ``"intro"``, ``"intermediate"``, or ``"advanced"``.
                Defaults to the package default (``intermediate``).
            n_accounts: Override recipe default account count.
            n_contacts: Override recipe default contact count.
            n_leads: Override recipe default lead count.
            horizon_days: Override recipe default simulation horizon.
            primary_task: Override recipe default task identifier (e.g.
                ``"converted_within_60_days"``).  Controls the task
                directory name and manifest key.
            label_window_days: Override recipe default label observation
                window in days.
            snapshot_day: Override recipe default snapshot day for windowed
                feature aggregation.  ``None`` means full-horizon (legacy)
                aggregation; an integer ``N`` means features aggregate only
                events with ``timestamp <= lead_created_at + N days``.
            output_path: Directory where the bundle will be saved.
            override: Optional dict of overrides (mirrors a ``--override`` file).
                Applied after recipe defaults but before explicit kwargs.

        Returns:
            A configured :class:`Generator` with a populated
            :attr:`world_spec` (narrative resolved from the recipe).

        Raises:
            :class:`~leadforge.core.exceptions.InvalidRecipeError`: if the
                recipe does not exist, is malformed, or the requested
                exposure mode / difficulty is not supported.
        """
        from leadforge.api.recipes import Recipe
        from leadforge.narrative.spec import NarrativeSpec
        from leadforge.recipes.registry import load_recipe

        raw = load_recipe(recipe_id)
        recipe = Recipe.from_dict(raw)
        config = recipe.resolve_config(
            seed=seed,
            exposure_mode=exposure_mode,
            difficulty=difficulty,
            n_accounts=n_accounts,
            n_contacts=n_contacts,
            n_leads=n_leads,
            horizon_days=horizon_days,
            primary_task=primary_task,
            label_window_days=label_window_days,
            snapshot_day=snapshot_day,
            output_path=output_path,
            override=override,
        )

        narrative_data = recipe.load_narrative()
        narrative = NarrativeSpec.from_dict(narrative_data) if narrative_data else None
        world_spec = WorldSpec(config=config, narrative=narrative)

        return cls(world_spec)

    def generate(
        self,
        *,
        n_accounts: int | None = None,
        n_contacts: int | None = None,
        n_leads: int | None = None,
        difficulty: str | DifficultyProfile = _MISSING,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> WorldBundle:
        """Run the full world simulation and return an in-memory bundle.

        Overrides in *n_accounts*, *n_contacts*, *n_leads*, and *difficulty*
        take effect for this call only — they do not mutate the Generator.
        When *difficulty* is omitted the Generator's configured difficulty is used.

        Args:
            n_accounts: Override account count.
            n_contacts: Override contact count.
            n_leads: Override lead count.
            difficulty: Difficulty profile name or enum value.  Defaults to
                the difficulty set on the Generator (i.e. from the recipe).
            **kwargs: Reserved for future use.

        Returns:
            A fully populated :class:`~leadforge.core.models.WorldBundle`.
            Call :meth:`~leadforge.core.models.WorldBundle.save` to write it
            to disk.
        """
        import dataclasses

        from leadforge.simulation.engine import simulate_world
        from leadforge.simulation.population import build_population
        from leadforge.structure.sampler import sample_hidden_graph

        config = self._world_spec.config

        # Apply per-call overrides without mutating the shared config.
        overrides: dict[str, Any] = {}
        if n_accounts is not None:
            overrides["n_accounts"] = n_accounts
        if n_contacts is not None:
            overrides["n_contacts"] = n_contacts
        if n_leads is not None:
            overrides["n_leads"] = n_leads
        if difficulty is not _MISSING:
            if not isinstance(difficulty, DifficultyProfile):
                difficulty = DifficultyProfile(difficulty)  # type: ignore[arg-type]
            if difficulty != config.difficulty:
                overrides["difficulty"] = difficulty
        if overrides:
            config = dataclasses.replace(config, **overrides)

        narrative = self._world_spec.narrative
        if narrative is None:
            raise RuntimeError(
                "No narrative loaded.  Initialise the Generator via "
                "Generator.from_recipe() to resolve the narrative."
            )

        rng_root = RNGRoot(config.seed)
        world_graph = sample_hidden_graph(rng_root)

        # Load category-latent correlations from difficulty profile if available.
        from leadforge.api.recipes import Recipe
        from leadforge.recipes.registry import load_recipe

        category_latent_correlations = None
        try:
            raw = load_recipe(config.recipe_id)
            recipe = Recipe.from_dict(raw)
            profiles = recipe.load_difficulty_profiles()
            profile = profiles.get(config.difficulty.value, {})
            category_latent_correlations = profile.get("category_latent_correlations")

            # Construct DifficultyParams from profile and attach to config.
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
            config = dataclasses.replace(config, difficulty_params=difficulty_params)
        except (FileNotFoundError, KeyError):
            category_latent_correlations = None

        population = build_population(
            config,
            narrative,
            world_graph,
            category_latent_correlations=category_latent_correlations,
        )
        latent_touch_intensity = kwargs.pop("latent_touch_intensity", False)
        result = simulate_world(
            config, population, world_graph, latent_touch_intensity=latent_touch_intensity
        )

        spec = WorldSpec(config=config, narrative=narrative)
        return WorldBundle(
            spec=spec,
            population=population,
            simulation_result=result,
            world_graph=world_graph,
        )
