"""Contains 3ds examples."""

from __future__ import annotations

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.examples import downloads

if pv.version_info >= (0, 52):
    msg = (
        '`pyvista.examples.download_3ds` should be removed. This module and all of its '
        'functions were deprecated in v0.49 and were scheduled for removal in v0.52.'
    )
    raise RuntimeError(msg)

warn_external(
    '`pyvista.examples.download_3ds` is deprecated and will be removed in a future '
    'version. Use `pyvista.examples.downloads` instead.',
    pv.PyVistaDeprecationWarning,
)


def download_iflamigm():
    """Download a iflamigm image.

    .. versionadded:: 0.44.0

    Returns
    -------
    str
        Filename of the 3DS file.

    """
    warn_external(
        '`examples.download_3ds.download_iflamigm` is deprecated. Use '
        '`examples.download_flamingo` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_flamingo(load=False)
