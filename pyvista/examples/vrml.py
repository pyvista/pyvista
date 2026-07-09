"""Contains vrml examples.

.. deprecated:: 0.49
    This module is deprecated and will be removed in a future version.
    Use :mod:`pyvista.examples.downloads` instead.

"""

from __future__ import annotations

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning

from . import downloads

if pv.version_info >= (0, 52):
    msg = (
        '`pyvista.examples.vrml` should be removed. This module and all of its '
        'functions were deprecated in v0.49 and were scheduled for removal in v0.52.'
    )
    raise RuntimeError(msg)

warn_external(
    '`pyvista.examples.vrml` is deprecated and will be removed in a future '
    'version. Use `pyvista.examples.downloads` instead.',
    PyVistaDeprecationWarning,
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

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_teapot()
    >>> pl = pv.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    """
    warn_external(
        '`examples.vrml.download_teapot` is deprecated. Use '
        '`examples.download_teapot_vrml` instead.',
        PyVistaDeprecationWarning,
    )
    if pv.version_info >= (0, 52):  # pragma: no cover
        msg_0 = 'Remove this deprecated function'
        raise RuntimeError(msg_0)
    return downloads.download_teapot_vrml(load=False)


def download_sextant():
    """Download the sextant example.

    .. deprecated:: 0.49
        Use :func:`pyvista.examples.downloads.download_sextant` instead.

    Returns
    -------
    str
        Filename of the VRML file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_sextant()
    >>> pl = pv.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    .. seealso::

        :ref:`Sextant Dataset <sextant_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    warn_external(
        '`examples.vrml.download_sextant` is deprecated. Use `examples.download_sextant` instead.',
        PyVistaDeprecationWarning,
    )
    if pv.version_info >= (0, 52):  # pragma: no cover
        msg_0 = 'Remove this deprecated function'
        raise RuntimeError(msg_0)
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

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_grasshopper()
    >>> pl = pv.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.camera_position = pv.CameraPosition(
    ...     position=(25.0, 32.0, 44.0),
    ...     focal_point=(0.0, 0.931, -6.68),
    ...     viewup=(-0.20, 0.90, -0.44),
    ... )
    >>> pl.show()

    .. seealso::

        :ref:`Grasshopper Dataset <grasshopper_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    warn_external(
        '`examples.vrml.download_grasshopper` is deprecated. Use '
        '`examples.download_grasshopper` instead.',
        PyVistaDeprecationWarning,
    )
    if pv.version_info >= (0, 52):  # pragma: no cover
        msg_0 = 'Remove this deprecated function'
        raise RuntimeError(msg_0)
    return downloads.download_grasshopper(load=False)
