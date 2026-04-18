"""Public Generator API — stub for Milestone 1.

The Generator class is the primary entry point for programmatic dataset
generation. It is fully specified in the architecture doc (§6) and will
be implemented across Milestones 1–9.
"""

from __future__ import annotations

from typing import Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.models import GenerationConfig, WorldBundle


class Generator:
    """High-level entry point for generating a synthetic CRM dataset bundle.

    Usage (once implemented)::

        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1",
            seed=42,
            exposure_mode="student_public",
        )
        bundle = gen.generate(n_leads=5000, difficulty="intermediate")
        bundle.save("./out/demo_bundle")

    Implemented in Milestone 1 (config/recipe) through Milestone 9 (rendering).
    """

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config

    @classmethod
    def from_recipe(
        cls,
        recipe_id: str,
        *,
        seed: int = 42,
        exposure_mode: str | ExposureMode = ExposureMode.student_public,
        **kwargs: Any,
    ) -> Generator:
        """Create a Generator from a recipe ID.

        Not yet implemented — available in v0.2.0.
        """
        raise NotImplementedError(
            "Generator.from_recipe() is not yet implemented. Coming in v0.2.0."
        )

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

        Not yet implemented — available in v0.2.0.
        """
        raise NotImplementedError("Generator.generate() is not yet implemented. Coming in v0.2.0.")
