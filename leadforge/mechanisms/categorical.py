"""Categorical influence mechanisms — channel and segment effects.

These mechanisms map categorical context values (e.g. lead source channel,
industry segment) to a numeric influence score, allowing categorical features
to modulate latent dynamics without embedding them as continuous latents.
"""

from __future__ import annotations

import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext


class CategoricalInfluence(Mechanism):
    """Map a categorical context key to a numeric score via a lookup table.

    Looks up ``context.extra[context_key]`` in *lookup* and returns the
    corresponding float.  Falls back to *default* if the value is absent or
    unknown.

    Args:
        context_key: The key to look up in ``context.extra``.
        lookup: Mapping of category label → score in [0, 1].
        default: Score to return when the lookup key or value is absent.
    """

    def __init__(
        self,
        context_key: str,
        lookup: dict[str, float],
        default: float = 0.5,
    ) -> None:
        if not lookup:
            raise ValueError("lookup must not be empty")
        self._context_key = context_key
        self._lookup = dict(lookup)
        self._default = default

    @property
    def name(self) -> str:
        return "categorical_influence"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        value = context.extra.get(self._context_key)
        return self._lookup.get(str(value), self._default) if value is not None else self._default

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context_key": self._context_key,
            "lookup": self._lookup,
            "default": self._default,
        }


# ---------------------------------------------------------------------------
# V1 channel influence scores (used by policies.py)
# ---------------------------------------------------------------------------

#: Default quality multipliers by lead source channel.
#: Partner referrals tend to arrive pre-qualified; outbound is colder.
CHANNEL_QUALITY_SCORES: dict[str, float] = {
    "inbound_marketing": 0.55,
    "sdr_outbound": 0.40,
    "partner_referral": 0.70,
}
