"""Typed Recipe model and config-resolution logic.

A ``Recipe`` is the canonical user-facing generation preset. It binds a
vertical, narrative defaults, difficulty profiles, available tasks, and
supported exposure modes into a single validated object.

Config precedence (highest → lowest):
  1. Explicit kwargs passed to ``from_recipe`` / ``resolve_config``
  2. Override dict (e.g. loaded from a ``--override`` YAML/JSON file)
  3. Recipe defaults (``default_population``, ``horizon_days``)
  4. Package defaults (defined in ``GenerationConfig``)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.exceptions import InvalidRecipeError
from leadforge.core.serialization import load_yaml

# Sentinel for "not provided by caller" — distinct from None
_MISSING = object()

_RECIPES_DIR = Path(__file__).parent.parent / "recipes"


@dataclass(frozen=True)
class Recipe:
    """Fully parsed and validated recipe object."""

    id: str
    title: str
    vertical: str
    description: str
    primary_task: str
    supported_modes: tuple[ExposureMode, ...]
    supported_difficulty: tuple[DifficultyProfile, ...]
    default_population: dict[str, int]
    horizon_days: int

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        """Build a ``Recipe`` from a raw YAML/dict payload.

        Raises:
            InvalidRecipeError: if required keys are missing or values are invalid.
        """
        required = {
            "id",
            "title",
            "vertical",
            "description",
            "primary_task",
            "supported_modes",
            "supported_difficulty",
            "default_population",
            "horizon_days",
        }
        missing = required - data.keys()
        if missing:
            raise InvalidRecipeError(f"Recipe dict is missing required keys: {sorted(missing)}")

        try:
            supported_modes = tuple(ExposureMode(m) for m in data["supported_modes"])
        except ValueError as exc:
            raise InvalidRecipeError(f"Invalid exposure mode in recipe: {exc}") from exc

        try:
            supported_difficulty = tuple(DifficultyProfile(d) for d in data["supported_difficulty"])
        except ValueError as exc:
            raise InvalidRecipeError(f"Invalid difficulty profile in recipe: {exc}") from exc

        pop = data["default_population"]
        if not isinstance(pop, dict) or not all(isinstance(v, int) for v in pop.values()):
            raise InvalidRecipeError(
                f"'default_population' must be a mapping of str→int, got: {pop!r}"
            )

        return cls(
            id=data["id"],
            title=data["title"],
            vertical=data["vertical"],
            description=data["description"],
            primary_task=data["primary_task"],
            supported_modes=supported_modes,
            supported_difficulty=supported_difficulty,
            default_population=dict(pop),
            horizon_days=int(data["horizon_days"]),
        )

    # ------------------------------------------------------------------ #
    # Config resolution
    # ------------------------------------------------------------------ #

    def resolve_config(
        self,
        *,
        seed: int = 42,
        exposure_mode: str | ExposureMode = ExposureMode.student_public,
        difficulty: str | DifficultyProfile = DifficultyProfile.intermediate,
        n_accounts: int | None = None,
        n_contacts: int | None = None,
        n_leads: int | None = None,
        horizon_days: int | None = None,
        output_path: str = "./out",
        override: dict[str, Any] | None = None,
    ) -> GenerationConfig:
        """Resolve a :class:`GenerationConfig` applying config precedence rules.

        Precedence (highest → lowest):
          1. Explicit kwargs (any non-None value passed by the caller)
          2. *override* dict
          3. Recipe defaults (``default_population``, ``horizon_days``)
          4. Package defaults (``GenerationConfig`` field defaults)
        """
        from leadforge.core.models import GenerationConfig  # avoid circular import

        # Layer 3 — recipe defaults
        pop = self.default_population
        resolved: dict[str, Any] = {
            "n_accounts": pop.get("n_accounts", 1500),
            "n_contacts": pop.get("n_contacts", 4200),
            "n_leads": pop.get("n_leads", 5000),
            "horizon_days": self.horizon_days,
        }

        # Layer 2 — override dict
        if override:
            for key in (
                "n_accounts",
                "n_contacts",
                "n_leads",
                "horizon_days",
                "seed",
                "output_path",
            ):
                if key in override:
                    resolved[key] = override[key]
            if "exposure_mode" in override:
                exposure_mode = override["exposure_mode"]
            if "difficulty" in override:
                difficulty = override["difficulty"]

        # Layer 1 — explicit kwargs
        if n_accounts is not None:
            resolved["n_accounts"] = n_accounts
        if n_contacts is not None:
            resolved["n_contacts"] = n_contacts
        if n_leads is not None:
            resolved["n_leads"] = n_leads
        if horizon_days is not None:
            resolved["horizon_days"] = horizon_days

        mode = ExposureMode(exposure_mode)
        diff = DifficultyProfile(difficulty)

        if mode not in self.supported_modes:
            raise InvalidRecipeError(
                f"Exposure mode '{mode}' is not supported by recipe '{self.id}'. "
                f"Supported: {[m.value for m in self.supported_modes]}"
            )
        if diff not in self.supported_difficulty:
            raise InvalidRecipeError(
                f"Difficulty '{diff}' is not supported by recipe '{self.id}'. "
                f"Supported: {[d.value for d in self.supported_difficulty]}"
            )

        return GenerationConfig(
            recipe_id=self.id,
            seed=seed,
            exposure_mode=mode,
            difficulty=diff,
            n_accounts=resolved["n_accounts"],
            n_contacts=resolved["n_contacts"],
            n_leads=resolved["n_leads"],
            horizon_days=resolved["horizon_days"],
            output_path=output_path,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def load_narrative(self) -> dict[str, Any]:
        """Load the ``narrative.yaml`` for this recipe, if present."""
        path = _RECIPES_DIR / self.id / "narrative.yaml"
        if not path.exists():
            return {}
        return load_yaml(path) or {}  # type: ignore[return-value]

    def load_difficulty_profiles(self) -> dict[str, Any]:
        """Load the ``difficulty_profiles.yaml`` for this recipe, if present."""
        path = _RECIPES_DIR / self.id / "difficulty_profiles.yaml"
        if not path.exists():
            return {}
        return load_yaml(path) or {}  # type: ignore[return-value]


# Avoid a circular import — GenerationConfig is defined in core.models
# but uses Recipe indirectly; we reference it via TYPE_CHECKING only.
from leadforge.core.models import GenerationConfig as GenerationConfig  # noqa: E402,F401
