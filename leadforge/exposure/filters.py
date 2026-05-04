"""Per-mode bundle filter rules.

:data:`FILTERS` maps every :class:`~leadforge.core.enums.ExposureMode` to a
:class:`BundleFilter` that governs which artefacts are written when
:func:`~leadforge.api.bundle.write_bundle` produces an output bundle.

Adding a new mode: define its ``BundleFilter`` entry in ``FILTERS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from leadforge.core.enums import ExposureMode
from leadforge.schema.features import STUDENT_PUBLIC_REDACTED_COLUMNS


@dataclass(frozen=True)
class BundleFilter:
    """Rules that govern bundle publication for one :class:`ExposureMode`.

    Attributes:
        write_metadata: Whether to create ``metadata/`` with hidden-truth
            files (``graph.json``, ``graph.graphml``, ``world_spec.json``,
            ``latent_registry.json``, ``mechanism_summary.json``).
        redacted_columns: Snapshot/task-split columns that must be stripped
            from the published bundle.  Empty for ``research_instructor``.
            For ``student_public`` this is the set of features flagged
            ``leakage_risk=True`` and ``is_leakage_trap=False`` — i.e. true
            label leaks (``current_stage``), but not pedagogical traps
            (``total_touches_all``).
    """

    write_metadata: bool
    redacted_columns: frozenset[str] = field(default_factory=frozenset)


#: Canonical filter rules for every supported exposure mode.
FILTERS: dict[ExposureMode, BundleFilter] = {
    ExposureMode.student_public: BundleFilter(
        write_metadata=False,
        redacted_columns=STUDENT_PUBLIC_REDACTED_COLUMNS,
    ),
    ExposureMode.research_instructor: BundleFilter(write_metadata=True),
}


def get_filter(mode: str | ExposureMode) -> BundleFilter:
    """Return the :class:`BundleFilter` for *mode*.

    Args:
        mode: An :class:`ExposureMode` or its string value.

    Raises:
        ValueError: if *mode* is a string that is not a valid
            :class:`ExposureMode` value.
        KeyError: if *mode* has no registered filter (should never happen
            with well-typed callers, but guards against future enum additions
            that forget to update ``FILTERS``).
    """
    if isinstance(mode, str):
        mode = ExposureMode(mode)
    return FILTERS[mode]
