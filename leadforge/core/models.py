"""Top-level typed configuration and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.exceptions import InvalidConfigError
from leadforge.version import __version__

if TYPE_CHECKING:
    from leadforge.narrative.spec import NarrativeSpec
    from leadforge.simulation.engine import SimulationResult
    from leadforge.simulation.population import PopulationResult
    from leadforge.structure.graph import WorldGraph


# Default generation scheme when a recipe/world does not declare one.  Kept here
# (the shared core layer) because ``leadforge.core`` must not import
# ``leadforge.schemes`` (the scheme package depends on core, not the reverse).
# ``LeadScoringScheme.name`` must equal this value; a test guards the match.
DEFAULT_SCHEME = "lead_scoring"


@dataclass(frozen=True)
class DifficultyParams:
    """Numeric parameters from a difficulty profile.

    Carried on :class:`GenerationConfig` to thread difficulty-dependent
    behaviour through the simulation engine and snapshot builder.
    """

    signal_strength: float
    noise_scale: float
    missing_rate: float
    outlier_rate: float
    conversion_rate_lo: float
    conversion_rate_hi: float
    committee_friction: float


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
    primary_task: str = "converted_within_90_days"
    label_window_days: int = 90
    snapshot_day: int | None = None
    output_path: str = "./out"
    package_version: str = field(default_factory=lambda: __version__)
    difficulty_params: DifficultyParams | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise InvalidConfigError(f"seed must be an int, got {type(self.seed).__name__!r}")
        if self.seed < 0:
            raise InvalidConfigError(f"seed must be non-negative, got {self.seed}")
        _require_positive_int(self.n_accounts, "n_accounts")
        _require_positive_int(self.n_contacts, "n_contacts")
        _require_positive_int(self.n_leads, "n_leads")
        _require_positive_int(self.horizon_days, "horizon_days")
        _require_positive_int(self.label_window_days, "label_window_days")
        if not isinstance(self.primary_task, str) or not self.primary_task:
            raise InvalidConfigError(
                f"primary_task must be a non-empty string, got {self.primary_task!r}"
            )
        if self.label_window_days > self.horizon_days:
            raise InvalidConfigError(
                f"label_window_days ({self.label_window_days}) must not exceed "
                f"horizon_days ({self.horizon_days})"
            )
        if self.snapshot_day is not None:
            if isinstance(self.snapshot_day, bool) or not isinstance(self.snapshot_day, int):
                raise InvalidConfigError(
                    f"snapshot_day must be a positive int or None, "
                    f"got {type(self.snapshot_day).__name__!r}"
                )
            if self.snapshot_day <= 0:
                raise InvalidConfigError(
                    f"snapshot_day must be a positive int or None, got {self.snapshot_day}"
                )
            if self.snapshot_day > self.horizon_days:
                raise InvalidConfigError(
                    f"snapshot_day ({self.snapshot_day}) must not exceed "
                    f"horizon_days ({self.horizon_days})"
                )
            # A snapshot anchored after the label closes would let features
            # observe events that occur beyond the label-scoring window —
            # exactly the structural leakage the windowed snapshot is here
            # to prevent.  Reject at config time.
            if self.snapshot_day > self.label_window_days:
                raise InvalidConfigError(
                    f"snapshot_day ({self.snapshot_day}) must not exceed "
                    f"label_window_days ({self.label_window_days}); a snapshot "
                    f"anchored after the label closes would re-introduce "
                    f"structural leakage."
                )
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

    Populated incrementally across milestones:
    - M2: config + narrative
    - M3–M6: schema, structure, mechanisms
    """

    config: GenerationConfig = field(default_factory=GenerationConfig)
    narrative: NarrativeSpec | None = None
    # Generation scheme this world runs (see leadforge.schemes).  Defaults to
    # the lead-scoring pipeline so direct WorldSpec construction is unchanged.
    scheme: str = DEFAULT_SCHEME


@dataclass
class WorldBundle:
    """In-memory result of one complete generation run.

    Holds all generated artefacts and provides :meth:`save` to write the
    full output bundle to disk.

    Attributes:
        spec: Fully resolved world specification (config + narrative).
        population: Generated accounts, contacts, leads, and latent state.
        simulation_result: Simulated event tables and final lead outcomes.
        world_graph: Sampled hidden world graph used during simulation.
    """

    spec: WorldSpec = field(default_factory=WorldSpec)
    population: PopulationResult | None = None
    simulation_result: SimulationResult | None = None
    world_graph: WorldGraph | None = None

    def save(self, path: str, generation_timestamp: str | None = None) -> None:
        """Write the full output bundle to *path*.

        Creates the directory if it does not exist.  The bundle layout
        matches the canonical structure defined in ``CLAUDE.md``::

            path/
              manifest.json
              dataset_card.md
              feature_dictionary.csv
              tables/          # one .parquet per relational table
              tasks/converted_within_90_days/{train,valid,test}.parquet
              tasks/converted_within_90_days/task_manifest.json

        Args:
            path: Destination directory (created if absent).
            generation_timestamp: ISO-8601 UTC timestamp.  Defaults to now.
                Pass a fixed value to produce byte-identical manifests.

        Raises:
            RuntimeError: if :attr:`simulation_result`, :attr:`population`,
                or :attr:`world_graph` have not been populated (i.e. if
                :meth:`~leadforge.api.generator.Generator.generate` was not
                called).
        """
        from leadforge.api.bundle import write_bundle

        write_bundle(self, path, generation_timestamp=generation_timestamp)
