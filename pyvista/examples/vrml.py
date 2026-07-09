"""Contains vrml examples.

.. deprecated:: 0.49
    This module is deprecated and will be removed in a future version.
    Use :mod:`pyvista.examples.downloads` instead.

"""

from __future__ import annotations

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.examples import downloads

if pv.version_info >= (0, 52):
    msg = (
        '`pyvista.examples.vrml` should be removed. This module and all of its '
        'functions were deprecated in v0.49 and were scheduled for removal in v0.52.'
    )
    raise RuntimeError(msg)

warn_external(
    '`pyvista.examples.vrml` is deprecated and will be removed in a future '
    'version. Use `pyvista.examples.downloads` instead.',
    pv.PyVistaDeprecationWarning,
)


def download_teapot():
    """Download a 2-manifold solid version of the famous teapot example.

    The `Utah Teapot <https://en.wikipedia.org/wiki/Utah_teapot>`_,
    originally modeled by Martin Newell at the University of Utah in
    1975. No formal license has ever been issued for the original Newell
    dataset; the model has been freely distributed in computer graphics
    software for 50 years and is conventionally treated as public domain.

    .. deprecated:: 0.49
        Use :func:`pyvista.examples.downloads.download_teapot_vrml` instead.

    Returns
    -------
    str
        Filename of the VRML file.

    """
    warn_external(
        '`examples.vrml.download_teapot` is deprecated. Use '
        '`examples.download_teapot_vrml` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_teapot_vrml(load=False)


def download_sextant():
    """Download the sextant example.

    .. deprecated:: 0.49
        Use :func:`pyvista.examples.downloads.download_sextant` instead.

    Returns
    -------
    str
        Filename of the VRML file.

    """
    warn_external(
        '`examples.vrml.download_sextant` is deprecated. Use `examples.download_sextant` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_sextant(load=False)


def download_grasshopper():
    """Download the grasshoper example.

    .. versionadded:: 0.45

    .. deprecated:: 0.49
        Use :func:`pyvista.examples.downloads.download_grasshopper` instead.

    Returns
    -------
    str
        Filename of the VRML file.

    """
    warn_external(
        '`examples.vrml.download_grasshopper` is deprecated. Use '
        '`examples.download_grasshopper` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_grasshopper(load=False)
