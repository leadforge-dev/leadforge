"""Shared feature-spec primitive.

:class:`FeatureSpec` is a scheme-agnostic dataclass used by every scheme's
feature catalog.  The lead-scoring catalog
(:data:`~leadforge.schemes.lead_scoring.features.LEAD_SNAPSHOT_FEATURES`)
lives in :mod:`leadforge.schemes.lead_scoring.features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from leadforge.core.enums import ExposureMode


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one column in a scheme's snapshot table.

    Two concerns are kept deliberately separate:

    - :attr:`leakage_risk` is *descriptive*: the value of this column is
      computed from events that may post-date the snapshot anchor and so
      correlates with the label.  It is informational metadata for
      downstream consumers and is preserved in the published feature
      dictionary.
    - :attr:`redact_in_modes` is *prescriptive*: the bundle writer must
      strip this column from any export whose mode is in this set.

    These can disagree: ``total_touches_all`` is ``leakage_risk=True``
    (it does encode post-snapshot information) but
    ``redact_in_modes=frozenset()`` (it is deliberately retained as a
    pedagogical trap).  Conversely a recipe could redact a column that
    is not itself leakage-risky for unrelated policy reasons.

    Attributes:
        name: Column name as it appears in the Parquet file.
        dtype: Pandas-compatible dtype string (``"string"``, ``"Int64"``,
            ``"Float64"``, ``"boolean"``).
        description: Human-readable explanation of what the column captures.
        category: Logical grouping (``"account"``, ``"contact"``,
            ``"lead_meta"``, ``"engagement"``, ``"sales"``, ``"target"``).
        is_target: True for the label column only.
        leakage_risk: Descriptive — this column is post-snapshot correlated.
        non_negative: True for columns that are physically incapable of being
            negative (counts, durations, monetary values).  Used by the
            snapshot builder to clamp values to ``>= 0`` after noise
            injection, preventing non-physical negatives from leaking into
            published bundles.
        redact_in_modes: Prescriptive — exposure modes in which the
            bundle writer must strip this column from snapshot, task
            splits, and feature dictionary.
    """

    name: str
    dtype: str
    description: str
    category: str
    is_target: bool = False
    leakage_risk: bool = False
    non_negative: bool = False
    redact_in_modes: frozenset[ExposureMode] = field(default_factory=frozenset)
