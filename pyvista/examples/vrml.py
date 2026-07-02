"""Contains vrml examples."""

from __future__ import annotations

from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader


def download_teapot():
    """Download a 2-manifold solid version of the famous teapot example.

    The `Utah Teapot <https://en.wikipedia.org/wiki/Utah_teapot>`_,
    originally modeled by Martin Newell at the University of Utah in
    1975. No formal license has ever been issued for the original Newell
    dataset; the model has been freely distributed in computer graphics
    software for 50 years and is conventionally treated as public domain.

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
    # _dataset_teapot should be moved outside of this function, but we cannot due to
    # the teapot name being used already, see https://github.com/pyvista/pyvista/issues/8773
    _dataset_teapot = _SingleFileDownloadableDatasetLoader('vrml/teapot.wrl')
    return _dataset_teapot.download()


def download_sextant():
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

    .. seealso::

        :ref:`Sextant Dataset <sextant_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_sextant.download()


_dataset_sextant = _SingleFileDownloadableDatasetLoader('vrml/sextant.wrl')


def download_grasshopper():
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

    .. seealso::

        :ref:`Grasshopper Dataset <grasshopper_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_grasshopper.download()


_dataset_grasshopper = _SingleFileDownloadableDatasetLoader('grasshopper/grasshop.wrl')
