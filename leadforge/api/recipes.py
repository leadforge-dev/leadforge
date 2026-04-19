"""Typed Recipe model and config-resolution logic.

A ``Recipe`` is the canonical user-facing generation preset. It binds a
vertical, narrative defaults, difficulty profiles, available tasks, and
supported exposure modes into a single validated object.

Config precedence (highest → lowest):
  1. Explicit kwargs passed to ``from_recipe`` / ``resolve_config``
  2. Override dict (e.g. loaded from a ``--override`` YAML/JSON file)
  3. Recipe defaults (``default_population``, ``horizon_days``)
  4. Package defaults (defined in ``GenerationConfig`` field defaults —
     the single source of truth; never duplicated here)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.exceptions import InvalidRecipeError
from leadforge.core.sentinels import _MISSING
from leadforge.core.serialization import load_yaml

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
        seed: int = _MISSING,  # type: ignore[assignment]
        exposure_mode: str | ExposureMode = _MISSING,  # type: ignore[assignment]
        difficulty: str | DifficultyProfile = _MISSING,  # type: ignore[assignment]
        n_accounts: int | None = None,
        n_contacts: int | None = None,
        n_leads: int | None = None,
        horizon_days: int | None = None,
        output_path: str = _MISSING,  # type: ignore[assignment]
        override: dict[str, Any] | None = None,
    ) -> GenerationConfig:
        """Resolve a :class:`GenerationConfig` applying config precedence rules.

        Precedence (highest → lowest):
          1. Explicit kwargs — only values *actually passed* by the caller win.
          2. *override* dict — beats recipe and package defaults.
          3. Recipe defaults — ``default_population`` keys and ``horizon_days``.
          4. Package defaults — ``GenerationConfig`` field defaults (authoritative
             source; never duplicated in this file).
        """
        from leadforge.core.models import GenerationConfig  # avoid circular import

        # Layer 4 — package defaults: read directly from GenerationConfig fields.
        pkg: dict[str, Any] = {
            f.name: f.default
            for f in dataclasses.fields(GenerationConfig)
            if f.default is not dataclasses.MISSING
        }
        resolved: dict[str, Any] = {
            "seed": pkg["seed"],
            "exposure_mode": pkg["exposure_mode"],
            "difficulty": pkg["difficulty"],
            "output_path": pkg["output_path"],
            "n_accounts": pkg["n_accounts"],
            "n_contacts": pkg["n_contacts"],
            "n_leads": pkg["n_leads"],
            "horizon_days": pkg["horizon_days"],
        }

        # Layer 3 — recipe defaults
        pop = self.default_population
        for key in ("n_accounts", "n_contacts", "n_leads"):
            if key in pop:
                resolved[key] = pop[key]
        resolved["horizon_days"] = self.horizon_days

        # Layer 2 — override dict (beats recipe/package defaults)
        if override:
            for key in (
                "n_accounts",
                "n_contacts",
                "n_leads",
                "horizon_days",
                "seed",
                "output_path",
                "exposure_mode",
                "difficulty",
            ):
                if key in override:
                    resolved[key] = override[key]

        # Layer 1 — explicit kwargs: only apply when the caller actually passed
        # the argument (sentinel guards all params that have package defaults).
        if seed is not _MISSING:
            resolved["seed"] = seed
        if exposure_mode is not _MISSING:
            resolved["exposure_mode"] = exposure_mode
        if difficulty is not _MISSING:
            resolved["difficulty"] = difficulty
        if output_path is not _MISSING:
            resolved["output_path"] = output_path
        if n_accounts is not None:
            resolved["n_accounts"] = n_accounts
        if n_contacts is not None:
            resolved["n_contacts"] = n_contacts
        if n_leads is not None:
            resolved["n_leads"] = n_leads
        if horizon_days is not None:
            resolved["horizon_days"] = horizon_days

        mode = ExposureMode(resolved["exposure_mode"])
        diff = DifficultyProfile(resolved["difficulty"])

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
            seed=resolved["seed"],
            exposure_mode=mode,
            difficulty=diff,
            n_accounts=resolved["n_accounts"],
            n_contacts=resolved["n_contacts"],
            n_leads=resolved["n_leads"],
            horizon_days=resolved["horizon_days"],
            output_path=resolved["output_path"],
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def load_narrative(self) -> dict[str, Any]:
        """Load the ``narrative.yaml`` for this recipe, if present."""
        path = _RECIPES_DIR / self.id / "narrative.yaml"
        if not path.exists():
            return {}
        data = load_yaml(path)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise InvalidRecipeError(
                f"narrative.yaml for recipe '{self.id}' must be a YAML mapping, "
                f"got {type(data).__name__!r}"
            )
        return data  # type: ignore[return-value]

    def load_difficulty_profiles(self) -> dict[str, Any]:
        """Load the ``difficulty_profiles.yaml`` for this recipe, if present."""
        path = _RECIPES_DIR / self.id / "difficulty_profiles.yaml"
        if not path.exists():
            return {}
        data = load_yaml(path)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise InvalidRecipeError(
                f"difficulty_profiles.yaml for recipe '{self.id}' must be a YAML mapping, "
                f"got {type(data).__name__!r}"
            )
        return data  # type: ignore[return-value]


# Forward reference resolution — GenerationConfig is used as a return type above.
from leadforge.core.models import GenerationConfig as GenerationConfig  # noqa: E402,F401
