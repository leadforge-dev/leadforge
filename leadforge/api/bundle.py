"""Bundle writer — dispatches serialisation to the producing generation scheme.

:func:`write_bundle` is called by :meth:`WorldBundle.save`.  It resolves the
bundle's generation scheme (``bundle.spec.scheme``) and delegates to that
scheme's :meth:`~leadforge.schemes.base.GenerationScheme.write_bundle`, which
owns the bundle's on-disk shape end to end (relational tables, task splits,
dataset card, feature dictionary, exposure metadata, manifest).

Scope note: each scheme currently orchestrates its *own* write sequence; only
the scheme-agnostic relational-table write is shared today
(:func:`leadforge.render.relational.write_relational_tables`).  A shared bundle
orchestrator with scheme render hooks is deferred to ``LTV-M6`` — it depends on
generalising ``build_manifest`` and ``apply_exposure``, which are still
lead-scoring-coupled (see ``docs/ltv/roadmap.md``).

This thin module preserves the ``write_bundle(bundle, path)`` entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from leadforge.schemes import get_scheme

if TYPE_CHECKING:
    from leadforge.core.models import WorldBundle


def write_bundle(
    bundle: WorldBundle,
    path: str,
    generation_timestamp: str | None = None,
) -> None:
    """Write *bundle* to disk at *path* via its generation scheme.

    Args:
        bundle: Fully populated :class:`~leadforge.core.models.WorldBundle`.
        path: Destination directory (created if absent).
        generation_timestamp: ISO-8601 UTC timestamp.  Defaults to now.
            Pass a fixed value to produce byte-identical manifests.

    Raises:
        UnknownSchemeError: if ``bundle.spec.scheme`` is not registered.
        RuntimeError: if the bundle is not fully populated (raised by the
            scheme's ``write_bundle``).
    """
    scheme = get_scheme(bundle.spec.scheme)
    scheme.write_bundle(bundle, path, generation_timestamp=generation_timestamp)
