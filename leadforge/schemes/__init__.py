"""Generation-scheme registry.

Importing this package registers the built-in schemes as a side effect, so
``from leadforge.schemes import get_scheme`` is always sufficient to resolve any
shipped scheme.  See ``leadforge.schemes.base`` and ``docs/ltv/design.md`` §2.5.
"""

from __future__ import annotations

# Import built-in scheme modules for their registration side effects.
from leadforge.schemes import lead_scoring as _lead_scoring  # noqa: F401
from leadforge.schemes import lifecycle as _lifecycle  # noqa: F401
from leadforge.schemes.base import (
    SCHEME_REGISTRY,
    GenerationScheme,
    UnknownSchemeError,
    available_schemes,
    get_scheme,
    register_scheme,
)

__all__ = [
    "SCHEME_REGISTRY",
    "GenerationScheme",
    "UnknownSchemeError",
    "available_schemes",
    "get_scheme",
    "register_scheme",
]
