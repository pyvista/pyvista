"""Contains glTF examples."""

from __future__ import annotations

import pooch

from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader
from pyvista.examples.downloads import USER_DATA_PATH

_GLTF_PATHS: dict[str, str] = {
    'damaged_helmet': 'DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf',
    'sheen_chair': 'SheenChair/glTF-Binary/SheenChair.glb',
    'gearbox': 'GearboxAssy/glTF-Binary/GearboxAssy.glb',
    'avocado': 'Avocado/glTF-Binary/Avocado.glb',
    'milk_truck': 'CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb',
}

_GLTF_BASE_URL = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/'


GLTF_FETCHER = pooch.create(  # type: ignore[attr-defined]
    path=USER_DATA_PATH,
    base_url=_GLTF_BASE_URL,
    registry=dict.fromkeys(_GLTF_PATHS.values()),
    retry_if_failed=3,
)


def _download_gltf(path: str):
    return GLTF_FETCHER.fetch(path)


def _gltf_loader(name):
    return _SingleFileDownloadableDatasetLoader(
        _GLTF_PATHS[name],
        base_url=_GLTF_BASE_URL,
        download_func=_download_gltf,
    )


def download_damaged_helmet():  # pragma: no cover
    """Download the damaged helmet example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> gltf_file = examples.gltf.download_damaged_helmet()
    >>> cubemap = examples.download_sky_box_cube_map()
    >>> pl = pv.Plotter()
    >>> pl.import_gltf(gltf_file)
    >>> pl.set_environment_texture(cubemap)
    >>> pl.show()

    .. seealso::

        :ref:`Damaged Helmet Dataset <damaged_helmet_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_damaged_helmet.download()


_dataset_damaged_helmet = _gltf_loader('damaged_helmet')


def download_sheen_chair():  # pragma: no cover
    """Download the sheen chair example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> gltf_file = examples.gltf.download_sheen_chair()
    >>> cubemap = examples.download_sky_box_cube_map()
    >>> pl = pv.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.set_environment_texture(cubemap)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    .. seealso::

        :ref:`Sheen Chair Dataset <sheen_chair_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_sheen_chair.download()


_dataset_sheen_chair = _gltf_loader('sheen_chair')


def download_gearbox():  # pragma: no cover
    """Download the gearbox example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> gltf_file = examples.gltf.download_gearbox()
    >>> pl = pv.Plotter()
    >>> pl.import_gltf(gltf_file)
    >>> pl.show()

    .. seealso::

        :ref:`Gearbox Dataset <gearbox_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_gearbox.download()


_dataset_gearbox = _gltf_loader('gearbox')


def download_avocado():  # pragma: no cover
    """Download the avocado example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> gltf_file = examples.gltf.download_avocado()
    >>> pl = pv.Plotter()
    >>> pl.import_gltf(gltf_file)
    >>> pl.show()

    .. seealso::

        :ref:`Avocado Dataset <avocado_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_avocado.download()


_dataset_avocado = _gltf_loader('avocado')


def download_milk_truck():  # pragma: no cover
    """Download the milk truck example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> gltf_file = examples.gltf.download_milk_truck()
    >>> pl = pv.Plotter()
    >>> pl.import_gltf(gltf_file)
    >>> pl.show()

    .. seealso::

        :ref:`Milk Truck Dataset <milk_truck_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_milk_truck.download()


_dataset_milk_truck = _gltf_loader('milk_truck')
