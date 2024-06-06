"""Contains vrml examples."""

from __future__ import annotations

from .downloads import download_file


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
    return download_file("vrml/teapot.wrl")


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
    return download_file("vrml/sextant.wrl")
