"""Top-level typed configuration and result models.

WorldSpec and WorldBundle are stubs in M0; they will be populated in M1+.
"""

from dataclasses import dataclass, field

from leadforge.core.enums import DifficultyProfile, ExposureMode


@dataclass
class GenerationConfig:
    """Fully resolved configuration for a single generation run."""

    recipe_id: str = "b2b_saas_procurement_v1"
    seed: int = 42
    exposure_mode: ExposureMode = ExposureMode.student_public
    difficulty: DifficultyProfile = DifficultyProfile.intermediate
    n_accounts: int = 1500
    n_contacts: int = 4200
    n_leads: int = 5000
    horizon_days: int = 90
    output_path: str = "./out"


@dataclass
class WorldSpec:
    """Fully instantiated hidden world specification (post-sampling, pre-simulation).

    Populated in Milestone 1 (config/recipe) through Milestone 6 (mechanisms).
    """

    config: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class WorldBundle:
    """In-memory result of one complete generation run.

    Populated in Milestone 7+ (simulation and rendering).
    """

    spec: WorldSpec = field(default_factory=WorldSpec)
