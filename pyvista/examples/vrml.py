"""vrml examples."""

from .downloads import FETCHER


def download_teapot():
    """Download the a 2-manifold solid version of the famous teapot example.

    Returns
    -------
    str
        Filename of the VRML file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_teapot_vrml()
    >>> pl = pyvista.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    """
    return FETCHER.fetch("vrml/teapot.wrl")


def download_sextant():
    """Download the sextant example.

    Returns
    -------
    str
        Filename of the VRML file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_sextant_vrml()
    >>> pl = pyvista.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    """
    return FETCHER.fetch("vrml/sextant.wrl")
