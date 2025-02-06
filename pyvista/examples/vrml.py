"""Contains vrml examples."""

from __future__ import annotations

from typing import Literal

from pyvista.examples._dataset_loader import _download_dataset_scene
from pyvista.examples._dataset_loader import _DownloadableFile


def download_teapot(load_as: Literal['plotter', 'dataset'] | None = None):  # pragma: no cover
    """Download the 2-manifold solid version of the famous teapot example.

    Parameters
    ----------
    load_as : 'plotter' | 'dataset', optional
        Load the file as a plotter or as a dataset. By default, the file is not loaded
        and only the filename is returned.

    Returns
    -------
    str | Plotter | DataSet
        Filename, plotter, or dataset depending on ``load_as``.

        .. versionadded:: 0.45

    Examples
    --------
    Download the file, import it, and plot it.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_teapot()
    >>> pl = pv.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    Alternatively, load the file directly as a plotter.

    >>> pl = examples.vrml.download_teapot(load_as='plotter')
    >>> isinstance(pl, pv.Plotter)
    True

    You can also load the file as a dataset. This will remove any scene-specific
    properties such as lighting, colors and textures.

    >>> pl = examples.vrml.download_teapot(load_as='dataset')
    >>> isinstance(pl, pv.DataSet)
    True

    """
    return _download_dataset_scene(_dataset_teapot, load_as=load_as, file_type='vrml')


_dataset_teapot = _DownloadableFile(
    'vrml/teapot.wrl',
)


def download_sextant(load_as: Literal['plotter', 'dataset'] | None = None):  # pragma: no cover
    """Download the sextant example.

    Parameters
    ----------
    load_as : 'plotter' | 'dataset', optional
        Load the file as a plotter or as a dataset. By default, the file is not loaded
        and only the filename is returned.

    Returns
    -------
    str | Plotter | DataSet
        Filename, plotter, or dataset depending on ``load_as``.

        .. versionadded:: 0.45

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_sextant()
    >>> pl = pv.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    Alternatively, load the file directly as a plotter.

    >>> pl = examples.vrml.download_sextant(load_as='plotter')
    >>> isinstance(pl, pv.Plotter)
    True

    You can also load the file as a dataset. This will remove any scene-specific
    properties such as colors and textures.

    >>> pl = examples.vrml.download_sextant(load_as='dataset')
    >>> isinstance(pl, pv.DataSet)
    True

    """
    return _download_dataset_scene(_dataset_sextant, load_as=load_as, file_type='vrml')


_dataset_sextant = _DownloadableFile(
    'vrml/sextant.wrl',
)


def download_grasshopper(load_as: Literal['plotter', 'dataset'] | None = None):  # pragma: no cover
    """Download the grasshopper example.

    .. versionadded:: 0.45

    Parameters
    ----------
    load_as : 'plotter' | 'dataset', optional
        Load the file as a plotter or as a dataset. By default, the file is not loaded
        and only the filename is returned.

    Returns
    -------
    str | Plotter | MultiBlock
        Filename, plotter, or dataset depending on ``load_as``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_grasshopper()
    >>> pl = pv.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.camera_position = [
    ...     (25.0, 32.0, 44.0),
    ...     (0.0, 0.931, -6.68),
    ...     (-0.20, 0.90, -0.44),
    ... ]
    >>> pl.show()

    Alternatively, load the file directly as a plotter.

    >>> pl = examples.vrml.download_grasshopper(load_as='plotter')
    >>> isinstance(pl, pv.Plotter)
    True

    You can also load the file as a dataset. This will remove any scene-specific
    properties such as colors and textures.

    >>> pl = examples.vrml.download_grasshopper(load_as='dataset')
    >>> isinstance(pl, pv.MultiBlock)
    True

    """
    return _download_dataset_scene(_dataset_grasshopper, load_as=load_as, file_type='vrml')


_dataset_grasshopper = _DownloadableFile(
    'grasshopper/grasshop.wrl',
)
