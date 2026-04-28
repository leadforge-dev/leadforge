"""Count / event-intensity mechanisms — generate touch and session counts.

These mechanisms answer "how many events of type X happen today for this lead?"
The simulation engine uses them to populate the ``touches`` and ``sessions``
tables.
"""

from __future__ import annotations

import math
import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext


class PoissonIntensity(Mechanism):
    """Poisson-distributed event count driven by latent traits.

    Expected count per day::

        lambda = base_rate * exp(sum(weight_i * latents[key_i]))

    This is a log-linear intensity model: latent weights are on the log scale
    so they multiplicatively modulate the base rate.

    Args:
        base_rate: Expected daily event count when all latent keys are 0.
        weights: Mapping of latent-key → log-scale weight.
    """

    def __init__(self, base_rate: float, weights: dict[str, float] | None = None) -> None:
        if base_rate <= 0:
            raise ValueError(f"base_rate must be positive, got {base_rate}")
        self._base_rate = base_rate
        self._weights: dict[str, float] = dict(weights) if weights else {}

    @property
    def name(self) -> str:
        return "poisson_intensity"

    def expected_count(self, latents: dict[str, float]) -> float:
        """Return the expected daily event count for the given latent state."""
        log_rate = math.log(self._base_rate) + sum(
            self._weights.get(k, 0.0) * latents.get(k, 0.0) for k in self._weights
        )
        return math.exp(log_rate)

    def sample(self, context: MechanismContext, rng: random.Random) -> int:
        """Draw a Poisson count for today."""
        lam = self.expected_count(context.latents)
        # Simulate Poisson via waiting times (exact for moderate lambda).
        count = 0
        p = math.exp(-lam)
        cum = p
        u = rng.random()
        while u > cum:
            count += 1
            p *= lam / count
            cum += p
            if count > 1000:  # safety cap
                break
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_rate": self._base_rate,
            "weights": self._weights,
        }


class RecencyDecayIntensity(Mechanism):
    """Poisson intensity that decays exponentially with time since lead creation.

    Models the observation that CRM activity is front-loaded: most touches
    happen early in the sales cycle.

    Args:
        base_rate: Expected daily count at ``t=0``.
        decay_factor: Per-day multiplicative decay in (0, 1].  At day *t*,
            the effective rate is ``base_rate * decay_factor ** t``.
        floor_rate: Minimum daily rate (floor applied after decay).
    """

    def __init__(
        self,
        base_rate: float,
        decay_factor: float = 0.97,
        floor_rate: float = 0.01,
    ) -> None:
        if base_rate <= 0:
            raise ValueError(f"base_rate must be positive, got {base_rate}")
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")
        if floor_rate < 0:
            raise ValueError(f"floor_rate must be non-negative, got {floor_rate}")
        self._base_rate = base_rate
        self._decay = decay_factor
        self._floor = floor_rate

    @property
    def name(self) -> str:
        return "recency_decay_intensity"

    def expected_count(self, t: int) -> float:
        """Return the expected daily count at day *t*."""
        return max(self._floor, self._base_rate * (self._decay**t))

    def sample(self, context: MechanismContext, rng: random.Random) -> int:
        lam = self.expected_count(context.t)
        count = 0
        p = math.exp(-lam)
        cum = p
        u = rng.random()
        while u > cum:
            count += 1
            p *= lam / count
            cum += p
            if count > 1000:
                break
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_rate": self._base_rate,
            "decay_factor": self._decay,
            "floor_rate": self._floor,
        }
