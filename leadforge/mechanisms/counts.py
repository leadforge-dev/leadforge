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


class LatentDecayIntensity(Mechanism):
    """Poisson intensity that decays with time AND is modulated by latent traits.

    Combines the recency-decay model with a latent-score multiplier so that
    leads with higher intent/fit have higher touch intensity throughout the
    simulation.  This creates a causal pathway: latent → touches AND
    latent → conversion, making post-snapshot touch counts a meaningful but
    imperfect proxy for conversion propensity.

    Expected count per day::

        lambda = max(floor, base_rate * decay^t * (1 + effective_boost * latent_multiplier))

    where ``latent_multiplier = sum(weight_i * latents[key_i])``.

    After ``followup_boost_after_day``, the effective boost ramps linearly from
    ``boost`` to ``boost * followup_boost_factor`` over ``followup_ramp_days``.
    This models sales teams increasing follow-up intensity for leads that show
    strong latent signals (engagement, fit, intent) — a causally legitimate
    amplification of the latent → touch pathway.

    Args:
        base_rate: Expected daily count at ``t=0`` for a lead with zero
            latent scores.
        decay_factor: Per-day multiplicative decay in (0, 1].
        floor_rate: Minimum daily rate (applied after decay and boost).
        latent_weights: Mapping of latent-key → weight for the multiplier.
        boost: Scaling factor for the latent multiplier (controls how much
            latent traits amplify touch intensity).
        followup_boost_after_day: Day after which latent modulation ramps up.
            Set to ``None`` (default) to disable the ramp.
        followup_boost_factor: Multiplier applied to ``boost`` at the end of
            the ramp period.  E.g. ``3.0`` means the effective boost is
            ``boost * 3.0`` once the ramp completes.
        followup_ramp_days: Number of days over which the ramp transitions
            linearly from ``boost`` to ``boost * followup_boost_factor``.
        followup_latent_weights: Optional separate latent weights used after
            the followup day.  Models sales teams responding to *different*
            latent signals during the follow-up period (e.g. prioritizing
            authority and budget over raw engagement).  Blended with the
            base weights during the ramp period.
    """

    def __init__(
        self,
        base_rate: float,
        decay_factor: float = 0.97,
        floor_rate: float = 0.01,
        latent_weights: dict[str, float] | None = None,
        boost: float = 0.8,
        followup_boost_after_day: int | None = None,
        followup_boost_factor: float = 1.0,
        followup_ramp_days: int = 10,
        followup_latent_weights: dict[str, float] | None = None,
    ) -> None:
        if base_rate <= 0:
            raise ValueError(f"base_rate must be positive, got {base_rate}")
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")
        if floor_rate < 0:
            raise ValueError(f"floor_rate must be non-negative, got {floor_rate}")
        if followup_boost_after_day is not None and followup_boost_after_day < 0:
            raise ValueError(
                f"followup_boost_after_day must be non-negative, got {followup_boost_after_day}"
            )
        if followup_boost_factor < 1.0:
            raise ValueError(f"followup_boost_factor must be >= 1.0, got {followup_boost_factor}")
        if followup_ramp_days < 1:
            raise ValueError(f"followup_ramp_days must be >= 1, got {followup_ramp_days}")
        self._base_rate = base_rate
        self._decay = decay_factor
        self._floor = floor_rate
        self._latent_weights: dict[str, float] = dict(latent_weights) if latent_weights else {}
        self._boost = boost
        self._followup_after: int | None = followup_boost_after_day
        self._followup_factor = followup_boost_factor
        self._followup_ramp = followup_ramp_days
        self._followup_latent_weights: dict[str, float] | None = (
            dict(followup_latent_weights) if followup_latent_weights else None
        )

    @property
    def name(self) -> str:
        return "latent_decay_intensity"

    def _effective_boost(self, t: int) -> float:
        """Return the effective boost at day *t*, accounting for follow-up ramp."""
        if self._followup_after is None or t <= self._followup_after:
            return self._boost
        elapsed = t - self._followup_after
        progress = min(1.0, elapsed / max(1, self._followup_ramp))
        return self._boost * (1.0 + progress * (self._followup_factor - 1.0))

    def _latent_multiplier(self, t: int, latents: dict[str, float] | None) -> float:
        """Compute the weighted latent multiplier, blending follow-up weights if active."""
        if not latents:
            return 0.0

        # Base weights
        base_mult = 0.0
        if self._latent_weights:
            base_mult = sum(
                self._latent_weights.get(k, 0.0) * latents.get(k, 0.0) for k in self._latent_weights
            )

        # If no followup weights or before followup day, use base only
        if (
            self._followup_latent_weights is None
            or self._followup_after is None
            or t <= self._followup_after
        ):
            return base_mult

        # Blend base and followup weights during ramp
        followup_mult = sum(
            self._followup_latent_weights.get(k, 0.0) * latents.get(k, 0.0)
            for k in self._followup_latent_weights
        )
        elapsed = t - self._followup_after
        progress = min(1.0, elapsed / max(1, self._followup_ramp))
        return base_mult * (1.0 - progress) + followup_mult * progress

    def expected_count(self, t: int, latents: dict[str, float] | None = None) -> float:
        """Return the expected daily count at day *t* given *latents*."""
        latent_mult = self._latent_multiplier(t, latents)
        effective_boost = self._effective_boost(t)
        rate = self._base_rate * (self._decay**t) * (1.0 + effective_boost * latent_mult)
        return max(self._floor, rate)

    def sample(self, context: MechanismContext, rng: random.Random) -> int:
        lam = self.expected_count(context.t, context.latents)
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
            "latent_weights": self._latent_weights,
            "boost": self._boost,
            "followup_boost_after_day": self._followup_after,
            "followup_boost_factor": self._followup_factor,
            "followup_ramp_days": self._followup_ramp,
            "followup_latent_weights": self._followup_latent_weights,
        }
