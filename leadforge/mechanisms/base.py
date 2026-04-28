"""Mechanism base classes and shared contracts.

All mechanism implementations inherit from :class:`Mechanism` and expose a
single ``sample(context, rng)`` method.  :class:`MechanismContext` is the
universal carrier of state passed into every ``sample`` call.
:class:`MechanismAssignment` holds the named mechanism instances that the
simulation engine will invoke on each time step.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class MechanismContext:
    """State snapshot passed to every :meth:`Mechanism.sample` call.

    Attributes:
        latents: Merged latent traits for the relevant entity set
            (account + contact + lead for a full lead context).
        stage: Current funnel stage of the lead, or ``None`` if not
            applicable (e.g. account-level mechanisms).
        t: Day index within the simulation window (0-based).
        extra: Mechanism-specific extra fields (e.g. ``"channel"``,
            ``"rep_id"``).
    """

    latents: dict[str, float] = field(default_factory=dict)
    stage: str | None = None
    t: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Mechanism(ABC):
    """Abstract base class for all leadforge mechanism types.

    Subclasses must implement :meth:`sample` and :meth:`to_dict`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable identifier for this mechanism type."""

    @abstractmethod
    def sample(self, context: MechanismContext, rng: random.Random) -> Any:
        """Draw one sample given *context* using *rng*.

        Args:
            context: Current state snapshot.
            rng: Seeded stdlib :class:`random.Random` instance.

        Returns:
            A value whose type depends on the mechanism family
            (float, int, str, bool, or ``None``).
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this mechanism."""

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Mechanism summary
# ---------------------------------------------------------------------------


@dataclass
class MechanismSummary:
    """Serialisable summary of the mechanism assignment for one world.

    Stored in ``mechanism_summary.json`` in ``research_instructor`` mode.

    Attributes:
        motif_family: Name of the motif family that drove parameter choices.
        conversion_hazard: Summary dict for the conversion hazard mechanism.
        stage_transition: Summary dict for the stage-transition mechanism.
        touch_intensity: Summary dict for the touch-count mechanism.
        measurement: Summary dict for the measurement / proxy mechanism.
    """

    motif_family: str
    conversion_hazard: dict[str, Any]
    stage_transition: dict[str, Any]
    touch_intensity: dict[str, Any]
    measurement: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_family": self.motif_family,
            "conversion_hazard": self.conversion_hazard,
            "stage_transition": self.stage_transition,
            "touch_intensity": self.touch_intensity,
            "measurement": self.measurement,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MechanismSummary:
        return cls(
            motif_family=data["motif_family"],
            conversion_hazard=data["conversion_hazard"],
            stage_transition=data["stage_transition"],
            touch_intensity=data["touch_intensity"],
            measurement=data["measurement"],
        )


# ---------------------------------------------------------------------------
# Mechanism assignment
# ---------------------------------------------------------------------------


@dataclass
class MechanismAssignment:
    """Named mechanism instances consumed by the simulation engine.

    All fields are populated by :func:`~leadforge.mechanisms.policies.assign_mechanisms`.
    """

    motif_family: str
    conversion_hazard: Mechanism
    stage_transition: Mechanism
    touch_intensity: Mechanism
    measurement: Mechanism

    def summary(self) -> MechanismSummary:
        """Return a :class:`MechanismSummary` for serialisation."""
        return MechanismSummary(
            motif_family=self.motif_family,
            conversion_hazard=self.conversion_hazard.to_dict(),
            stage_transition=self.stage_transition.to_dict(),
            touch_intensity=self.touch_intensity.to_dict(),
            measurement=self.measurement.to_dict(),
        )
