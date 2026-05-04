"""Per-mode bundle filter rules.

:data:`FILTERS` maps every :class:`~leadforge.core.enums.ExposureMode` to a
:class:`BundleFilter` that governs which artefacts are written when
:func:`~leadforge.api.bundle.write_bundle` produces an output bundle.

The per-feature redaction policy lives separately on
:attr:`leadforge.schema.features.FeatureSpec.redact_in_modes` and is queried
via :func:`leadforge.schema.features.redacted_columns_for`.  ``BundleFilter``
deliberately does *not* duplicate that information so that the writer and
the validator both consult the same source of truth.

Adding a new mode: define its ``BundleFilter`` entry in ``FILTERS``.
"""

from __future__ import annotations

from dataclasses import dataclass

from leadforge.core.enums import ExposureMode


@dataclass(frozen=True)
class BundleFilter:
    """Mode-level publication policy.

    Attributes:
        write_metadata: Whether to create ``metadata/`` with hidden-truth
            files (``graph.json``, ``graph.graphml``, ``world_spec.json``,
            ``latent_registry.json``, ``mechanism_summary.json``).
    """

    write_metadata: bool


#: Canonical filter rules for every supported exposure mode.
FILTERS: dict[ExposureMode, BundleFilter] = {
    ExposureMode.student_public: BundleFilter(write_metadata=False),
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
