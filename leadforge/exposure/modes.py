"""Exposure-mode dispatch for bundle publication.

:func:`apply_exposure` is the single entry point called by each scheme's
``write_bundle``.  It reads the resolved
:class:`~leadforge.exposure.filters.BundleFilter` for the requested mode and,
when hidden truth should be published, writes the scheme-agnostic
``world_spec.json`` and delegates the scheme-specific hidden-truth files to the
producing scheme's :meth:`~leadforge.schemes.base.GenerationScheme.write_metadata`
hook.  This keeps the exposure layer free of any single scheme's types.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from leadforge.exposure.filters import get_filter
from leadforge.exposure.metadata import write_world_spec_json

if TYPE_CHECKING:
    from pathlib import Path

    from leadforge.core.enums import ExposureMode
    from leadforge.core.models import WorldBundle


def apply_exposure(bundle: WorldBundle, bundle_root: Path, mode: ExposureMode) -> None:
    """Apply exposure filtering for *mode* to the bundle at *bundle_root*.

    For modes whose filter sets ``write_metadata`` (e.g. ``research_instructor``)
    this creates ``metadata/``, writes the scheme-agnostic ``world_spec.json``,
    and calls the producing scheme's ``write_metadata`` hook for its
    hidden-truth files.  For modes that must not publish hidden truth (e.g.
    ``student_public``) any pre-existing ``metadata/`` directory is removed so
    truth is never accidentally republished when reusing an output path.

    Args:
        bundle: Fully populated :class:`~leadforge.core.models.WorldBundle`.
        bundle_root: Root directory of the written bundle (must already exist).
        mode: Exposure mode that controls which artefacts are published.
    """
    from leadforge.schemes import get_scheme

    filt = get_filter(mode)
    meta_dir = bundle_root / "metadata"
    if filt.write_metadata:
        meta_dir.mkdir(exist_ok=True)
        write_world_spec_json(bundle.spec, meta_dir)
        get_scheme(bundle.spec.scheme).write_metadata(bundle, meta_dir)
    elif meta_dir.exists():
        shutil.rmtree(meta_dir)
