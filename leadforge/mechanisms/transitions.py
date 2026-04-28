"""Stage-transition mechanisms — advance leads through the funnel.

:class:`HazardTransition` decides whether a lead advances on a given day.
:class:`StageSequence` defines the ordered funnel stages and resolves the
next stage name.  The simulation engine calls these on each day step per lead.
"""

from __future__ import annotations

import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext
from leadforge.mechanisms.scores import LatentScore

# Default v1 funnel stage ordering (matches narrative.yaml funnel_stages).
_DEFAULT_STAGE_ORDER = (
    "mql",
    "sal",
    "sql",
    "demo_scheduled",
    "demo_completed",
    "proposal_sent",
    "negotiation",
    "closed_won",
    "closed_lost",
)

# Stages from which advancement is no longer possible.
_TERMINAL_STAGES = frozenset({"closed_won", "closed_lost"})


class StageSequence(Mechanism):
    """Ordered funnel stage registry.

    Returns the next stage name given the current one, or ``None`` if the
    current stage is terminal or unknown.

    Args:
        stage_order: Ordered tuple of stage names.  The last stage is
            terminal (no advancement).
        terminal_stages: Set of stage names from which no further
            advancement occurs.
    """

    def __init__(
        self,
        stage_order: tuple[str, ...] = _DEFAULT_STAGE_ORDER,
        terminal_stages: frozenset[str] = _TERMINAL_STAGES,
    ) -> None:
        self._order = stage_order
        self._terminal = terminal_stages
        self._next: dict[str, str] = {}
        for i, stage in enumerate(stage_order[:-1]):
            if stage not in terminal_stages:
                self._next[stage] = stage_order[i + 1]

    @property
    def name(self) -> str:
        return "stage_sequence"

    def next_stage(self, current: str) -> str | None:
        """Return the stage after *current*, or ``None`` if terminal."""
        return self._next.get(current)

    def is_terminal(self, stage: str) -> bool:
        return stage in self._terminal

    def sample(self, context: MechanismContext, rng: random.Random) -> str | None:
        """Return the stage after ``context.stage``, or ``None`` if terminal."""
        if context.stage is None:
            return None
        return self.next_stage(context.stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage_order": list(self._order),
            "terminal_stages": sorted(self._terminal),
        }


class HazardTransition(Mechanism):
    """Discrete-time hazard for stage advancement.

    On each day step, computes a transition probability from the lead's
    latent score and returns ``True`` if the lead should advance.

    Daily probability::

        p_advance = clip(base_rate + scale * score, 0, max_daily_rate)

    A minimum dwell time enforces that leads cannot skip through stages
    unrealistically quickly.

    Args:
        score_mech: :class:`~leadforge.mechanisms.scores.LatentScore`
            mapping merged latents → [0, 1] score.
        base_rate: Minimum daily advancement probability.
        scale: Score multiplier.
        max_daily_rate: Hard cap on daily probability.
        min_dwell_days: Minimum days in the current stage before any
            advancement can occur.
    """

    def __init__(
        self,
        score_mech: LatentScore,
        base_rate: float = 0.03,
        scale: float = 0.10,
        max_daily_rate: float = 0.25,
        min_dwell_days: int = 1,
    ) -> None:
        if not (0.0 <= base_rate <= 1.0):
            raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}")
        if not (0.0 < max_daily_rate <= 1.0):
            raise ValueError(f"max_daily_rate must be in (0, 1], got {max_daily_rate}")
        if min_dwell_days < 0:
            raise ValueError(f"min_dwell_days must be >= 0, got {min_dwell_days}")
        self._score_mech = score_mech
        self._base_rate = base_rate
        self._scale = scale
        self._max_daily_rate = max_daily_rate
        self._min_dwell = min_dwell_days

    @property
    def name(self) -> str:
        return "hazard_transition"

    def daily_probability(self, latents: dict[str, float], dwell: int) -> float:
        """Return the daily advancement probability given *dwell* days in stage."""
        if dwell < self._min_dwell:
            return 0.0
        score = self._score_mech.score(latents)
        p = self._base_rate + self._scale * score
        return max(0.0, min(self._max_daily_rate, p))

    def sample(self, context: MechanismContext, rng: random.Random) -> bool:
        """Return ``True`` if the lead advances today.

        ``context.extra["dwell_days"]`` should carry the number of days the
        lead has spent in the current stage.  Defaults to 0 if absent.
        """
        dwell = int(context.extra.get("dwell_days", 0))
        p = self.daily_probability(context.latents, dwell)
        return rng.random() < p

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_mech": self._score_mech.to_dict(),
            "base_rate": self._base_rate,
            "scale": self._scale,
            "max_daily_rate": self._max_daily_rate,
            "min_dwell_days": self._min_dwell,
        }
