"""Latent scoring — maps merged latent state to a scalar score in [0, 1].

:class:`LatentScore` is the core building block used by
:class:`~leadforge.mechanisms.hazards.ConversionHazard` and
:class:`~leadforge.mechanisms.transitions.HazardTransition` to collapse
multiple latent traits into a single predictive signal.
"""

from __future__ import annotations

import math
import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext


class LatentScore(Mechanism):
    """Logistic score from a weighted combination of latent keys.

    Computes::

        score = sigmoid(bias + sum(weight_i * latents[key_i]))

    Keys absent from ``context.latents`` contribute 0.

    Args:
        weights: Mapping of latent-key → weight.  Positive weights increase
            the score; negative weights decrease it.
        bias: Additive intercept (controls the base conversion propensity
            before any latent influence).
    """

    def __init__(self, weights: dict[str, float], bias: float = 0.0) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        self._weights = dict(weights)
        self._bias = bias

    @property
    def name(self) -> str:
        return "latent_score"

    def score(self, latents: dict[str, float]) -> float:
        """Return the [0, 1] score without sampling noise."""
        linear = self._bias + sum(
            self._weights.get(k, 0.0) * latents.get(k, 0.0) for k in self._weights
        )
        return 1.0 / (1.0 + math.exp(-linear))

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        return self.score(context.latents)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": self._weights, "bias": self._bias}
