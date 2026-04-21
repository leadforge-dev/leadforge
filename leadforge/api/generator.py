"""Public Generator API."""

from __future__ import annotations

from typing import Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.models import GenerationConfig, WorldBundle, WorldSpec
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
        difficulty: str | DifficultyProfile = DifficultyProfile.intermediate,
        **kwargs: Any,
    ) -> WorldBundle:
        """Run the world simulation and return a bundle.

        Not yet implemented — available in v0.3.0+.
        """
        raise NotImplementedError("Generator.generate() is not yet implemented. Coming in v0.3.0.")
