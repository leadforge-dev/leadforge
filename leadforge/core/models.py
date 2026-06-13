"""Top-level typed configuration and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.exceptions import InvalidConfigError
from leadforge.version import __version__

if TYPE_CHECKING:
    from leadforge.narrative.spec import NarrativeSpec


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

    # --- lifecycle scheme (b2b_saas_ltv_v1) config -------------------------
    # Consumed only by the lifecycle generation scheme; the lead-scoring scheme
    # ignores these.  They live on the shared config (like ``n_leads`` /
    # ``snapshot_day`` do for lead-scoring) so recipe/CLI resolution stays
    # uniform across schemes.  A nested per-scheme config is a possible future
    # refactor; kept flat here to match the existing precedent.
    #
    # NOTE: these are not threaded into the lifecycle pipeline yet — that wiring
    # is LTV-Pn.4, at which point this config becomes the source of truth and
    # overrides the scheme's module-level defaults.  Until then the scheme's own
    # constants are authoritative.  ``forward_windows_days`` / ``early_tenure_weeks``
    # intentionally duplicate ``schemes.lifecycle.snapshots.FORWARD_WINDOWS_DAYS``
    # / ``DEFAULT_EARLY_TENURE_WEEKS`` (core must not import a scheme — see the
    # LTV-Pn.2 layering cleanup), so a cross-layer test
    # (tests/schemes/lifecycle/test_config_consistency.py) pins the defaults
    # equal to guard against drift.
    n_customers: int = 1500
    # pLTV forward-window targets, in days (D6): ltv_revenue_{90,365,730}d.
    forward_windows_days: tuple[int, ...] = (90, 365, 730)
    # Tenure anchor (whole weeks) for the early-pLTV regime (D8).
    early_tenure_weeks: int = 4
    # Absolute calendar observation anchor (ISO date) for the calendar regime
    # (D4).  ``None`` lets the population builder derive it from the world
    # calendar.
    observation_date: str | None = None

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
        self._validate_lifecycle_fields()

    def _validate_lifecycle_fields(self) -> None:
        """Validate the lifecycle-scheme config fields.

        Kept separate from the main body for readability; these constrain only
        the lifecycle fields and never touch the lead-scoring path.
        """
        _require_positive_int(self.n_customers, "n_customers")
        _require_positive_int(self.early_tenure_weeks, "early_tenure_weeks")

        windows = self.forward_windows_days
        if not isinstance(windows, tuple) or not windows:
            raise InvalidConfigError(
                f"forward_windows_days must be a non-empty tuple, got {windows!r}"
            )
        for w in windows:
            _require_positive_int(w, "forward_windows_days entry")
        if list(windows) != sorted(set(windows)):
            raise InvalidConfigError(
                f"forward_windows_days must be strictly increasing and unique, got {windows!r}"
            )

        if self.observation_date is not None:
            if not isinstance(self.observation_date, str):
                raise InvalidConfigError(
                    f"observation_date must be an ISO date string or None, "
                    f"got {type(self.observation_date).__name__!r}"
                )
            from datetime import date

            try:
                date.fromisoformat(self.observation_date)
            except ValueError as exc:
                raise InvalidConfigError(
                    f"observation_date must be an ISO date (YYYY-MM-DD), "
                    f"got {self.observation_date!r}"
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
        artifacts: The producing scheme's in-memory result (e.g.
            :class:`~leadforge.schemes.lead_scoring.artifacts.LeadScoringArtifacts`).
            Opaque to the shared core layer — typed ``Any`` so ``core`` never
            references a scheme.  Each scheme stores and unwraps its own
            container; ``None`` until :meth:`~leadforge.api.generator.Generator.generate`
            populates it.
    """

    spec: WorldSpec = field(default_factory=WorldSpec)
    artifacts: Any = None

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
            RuntimeError: if :attr:`artifacts` has not been populated (i.e. if
                :meth:`~leadforge.api.generator.Generator.generate` was not
                called).
        """
        from leadforge.api.bundle import write_bundle

        write_bundle(self, path, generation_timestamp=generation_timestamp)
