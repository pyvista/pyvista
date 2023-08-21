"""Contains vrml examples."""

from .downloads import download_file


def download_teapot():  # pragma: no cover
    """Download the a 2-manifold solid version of the famous teapot example.

    Returns
    -------
    str
        Filename of the VRML file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_teapot()
    >>> pl = pyvista.Plotter()
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
    >>> import pyvista
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_sextant()
    >>> pl = pyvista.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    """
    return download_file("vrml/sextant.wrl")
