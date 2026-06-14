"""Lifecycle hidden-truth serialisation for ``research_instructor`` metadata.

Helpers used by :meth:`leadforge.schemes.lifecycle.LifecycleScheme.write_metadata`
to serialise the scheme's latent truth.  The lifecycle scheme has no hidden
*graph* (unlike lead scoring); its hidden truth is the per-entity latent
registry and the motif-derived mechanism parameters.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from leadforge.schemes.lifecycle.mechanisms import assign_lifecycle_mechanisms

if TYPE_CHECKING:
    from leadforge.schemes.lifecycle.population import CustomerLatentState

__all__ = ["latent_registry_dict", "mechanism_summary_dict"]


def latent_registry_dict(latent_state: CustomerLatentState) -> dict[str, Any]:
    """Return the per-entity latent registry as a JSON-serialisable dict."""
    return {
        "account_latents": latent_state.account_latents,
        "customer_latents": latent_state.customer_latents,
    }


def _params_to_dict(params: Any) -> dict[str, Any]:
    """Convert a mechanism params dataclass to plain JSON-serialisable types.

    Unwraps the ``MappingProxyType`` latent-weight fields to plain dicts so the
    result is ``json.dumps``-able.
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(params):
        value = getattr(params, f.name)
        if isinstance(value, MappingProxyType):
            value = dict(value)
        out[f.name] = value
    return out


def mechanism_summary_dict(motif_family: str) -> dict[str, Any]:
    """Return the motif's mechanism parameters as a JSON-serialisable dict.

    Reconstructs the assignment deterministically from *motif_family* (the same
    call the engine makes), so the summary always matches the simulated world.
    """
    assignment = assign_lifecycle_mechanisms(motif_family)
    return {
        "motif_family": assignment.motif_family,
        "churn_hazard": _params_to_dict(assignment.churn_hazard),
        "expansion_propensity": _params_to_dict(assignment.expansion_propensity),
        "payment_failure": _params_to_dict(assignment.payment_failure),
    }
