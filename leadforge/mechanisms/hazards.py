"""Conversion hazard mechanism — daily probability of lead conversion.

:class:`ConversionHazard` is the primary mechanism called by the simulation
engine on each day step for each active lead.  It maps the merged latent state
to a daily conversion probability via a :class:`~leadforge.mechanisms.scores.LatentScore`.
"""

from __future__ import annotations

import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext
from leadforge.mechanisms.scores import LatentScore


class ConversionHazard(Mechanism):
    """Daily conversion probability driven by latent score.

    Daily probability::

        p_convert = clip(base_rate + scale * score, 0, max_daily_rate)

    Args:
        score_mech: A :class:`~leadforge.mechanisms.scores.LatentScore`
            instance that maps latents → [0, 1] score.
        base_rate: Minimum daily conversion probability (intercept).
        scale: Multiplier on the latent score.
        max_daily_rate: Hard cap on the daily probability.
    """

    def __init__(
        self,
        score_mech: LatentScore,
        base_rate: float = 0.005,
        scale: float = 0.05,
        max_daily_rate: float = 0.20,
    ) -> None:
        if not (0.0 <= base_rate <= 1.0):
            raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}")
        if not (0.0 < max_daily_rate <= 1.0):
            raise ValueError(f"max_daily_rate must be in (0, 1], got {max_daily_rate}")
        self._score_mech = score_mech
        self._base_rate = base_rate
        self._scale = scale
        self._max_daily_rate = max_daily_rate

    @property
    def name(self) -> str:
        return "conversion_hazard"

    def daily_probability(self, latents: dict[str, float]) -> float:
        """Return the daily conversion probability for the given latent state."""
        score = self._score_mech.score(latents)
        p = self._base_rate + self._scale * score
        return max(0.0, min(self._max_daily_rate, p))

    def sample(self, context: MechanismContext, rng: random.Random) -> bool:
        """Return ``True`` if the lead converts on this day step."""
        p = self.daily_probability(context.latents)
        return rng.random() < p

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_mech": self._score_mech.to_dict(),
            "base_rate": self._base_rate,
            "scale": self._scale,
            "max_daily_rate": self._max_daily_rate,
        }
