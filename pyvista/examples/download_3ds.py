"""Contains 3ds examples."""

from .downloads import download_file


def download_iflamigm():  # pragma: no cover
    """Download a iflamigm image.

    .. versionadded:: 0.44.0

    Returns
    -------
    str
        Filename of the 3DS file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> _3ds_file = examples._3ds.download_iflamigm()
    >>> pl = pv.Plotter()
    >>> pl.import_3ds(_3ds_file)
    >>> pl.show()

    """
    return download_file("iflamigm.3ds")
