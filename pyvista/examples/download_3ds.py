"""Contains 3ds examples."""

from __future__ import annotations

from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader


def download_iflamigm():
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
    >>> download_3ds_file = examples.download_3ds.download_iflamigm()
    >>> pl = pv.Plotter()
    >>> pl.import_3ds(download_3ds_file)
    >>> pl.show()

    .. seealso::

        :ref:`Iflamigm Dataset <iflamigm_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_iflamigm.download()


_dataset_iflamigm = _SingleFileDownloadableDatasetLoader('iflamigm.3ds')
