"""Top-level typed configuration and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.exceptions import InvalidConfigError
from leadforge.version import __version__


def _require_positive_int(value: Any, name: str) -> None:
    """Raise ``InvalidConfigError`` unless *value* is a positive plain ``int``.

    ``bool`` is rejected because it is an ``int`` subclass and would otherwise
    silently pass numeric comparisons (``True > 0`` is ``True``).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidConfigError(f"{name} must be a positive int, got {type(value).__name__!r}")
    if value <= 0:
        raise InvalidConfigError(f"{name} must be positive, got {value}")


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
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise InvalidConfigError(f"seed must be an int, got {type(self.seed).__name__!r}")
        if self.seed < 0:
            raise InvalidConfigError(f"seed must be non-negative, got {self.seed}")
        _require_positive_int(self.n_accounts, "n_accounts")
        _require_positive_int(self.n_contacts, "n_contacts")
        _require_positive_int(self.n_leads, "n_leads")
        _require_positive_int(self.horizon_days, "horizon_days")
        # Coerce string enums supplied as plain strings
        if not isinstance(self.exposure_mode, ExposureMode):
            try:
                self.exposure_mode = ExposureMode(self.exposure_mode)
            except ValueError as exc:
                raise InvalidConfigError(
                    f"exposure_mode has invalid value {self.exposure_mode!r}. "
                    f"Valid values: {[m.value for m in ExposureMode]}"
                ) from exc
        if not isinstance(self.difficulty, DifficultyProfile):
            try:
                self.difficulty = DifficultyProfile(self.difficulty)
            except ValueError as exc:
                raise InvalidConfigError(
                    f"difficulty has invalid value {self.difficulty!r}. "
                    f"Valid values: {[d.value for d in DifficultyProfile]}"
                ) from exc


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
