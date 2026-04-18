"""Top-level typed configuration and result models."""

from __future__ import annotations

from dataclasses import dataclass, field

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.version import __version__


@dataclass
class GenerationConfig:
    """Fully resolved configuration for a single generation run.

    All fields are validated in ``__post_init__``. Instances are produced
    via :meth:`leadforge.api.recipes.Recipe.resolve_config` which applies
    the config precedence rules (CLI flags > override > recipe > package).
    """

    recipe_id: str = "b2b_saas_procurement_v1"
    seed: int = 42
    exposure_mode: ExposureMode = ExposureMode.student_public
    difficulty: DifficultyProfile = DifficultyProfile.intermediate
    n_accounts: int = 1500
    n_contacts: int = 4200
    n_leads: int = 5000
    horizon_days: int = 90
    output_path: str = "./out"
    package_version: str = field(default_factory=lambda: __version__)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise TypeError(f"seed must be int, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.n_accounts <= 0:
            raise ValueError(f"n_accounts must be positive, got {self.n_accounts}")
        if self.n_contacts <= 0:
            raise ValueError(f"n_contacts must be positive, got {self.n_contacts}")
        if self.n_leads <= 0:
            raise ValueError(f"n_leads must be positive, got {self.n_leads}")
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")
        # Coerce string enums supplied as plain strings
        if not isinstance(self.exposure_mode, ExposureMode):
            self.exposure_mode = ExposureMode(self.exposure_mode)
        if not isinstance(self.difficulty, DifficultyProfile):
            self.difficulty = DifficultyProfile(self.difficulty)


@dataclass
class WorldSpec:
    """Fully instantiated hidden world specification (post-sampling, pre-simulation).

    Populated in Milestone 2 (narrative/schema) through Milestone 6 (mechanisms).
    """

    config: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class WorldBundle:
    """In-memory result of one complete generation run.

    Populated in Milestone 7+ (simulation and rendering).
    """

    spec: WorldSpec = field(default_factory=WorldSpec)
