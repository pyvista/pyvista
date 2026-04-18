"""Contains vrml examples."""

from __future__ import annotations

from pyvista.examples.downloads import download_file


def download_teapot():  # pragma: no cover
    """Download the a 2-manifold solid version of the famous teapot example.

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
    return download_file('vrml/teapot.wrl')


def download_sextant():  # pragma: no cover
    """Download the sextant example.

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

    """
    return download_file('vrml/sextant.wrl')


def download_grasshopper():  # pragma: no cover
    """Download the grasshoper example.

    .. versionadded:: 0.45

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

    """
    return download_file('grasshopper/grasshop.wrl')
