"""Contains glTF examples."""

from __future__ import annotations

import pooch

from .downloads import USER_DATA_PATH

GLTF_FETCHER = pooch.create(  # type: ignore[attr-defined]
    path=USER_DATA_PATH,
    base_url='https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/',
    registry={
        'Avocado/glTF-Binary/Avocado.glb': None,
        'CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb': None,
        'DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf': None,
        'GearboxAssy/glTF-Binary/GearboxAssy.glb': None,
        'SheenChair/glTF-Binary/SheenChair.glb': None,
    },
    retry_if_failed=3,
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

    """
    return GLTF_FETCHER.fetch('DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf')


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

    """
    return GLTF_FETCHER.fetch('SheenChair/glTF-Binary/SheenChair.glb')


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

    """
    return GLTF_FETCHER.fetch('GearboxAssy/glTF-Binary/GearboxAssy.glb')


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

    """
    return GLTF_FETCHER.fetch('Avocado/glTF-Binary/Avocado.glb')


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

    """
    return GLTF_FETCHER.fetch('CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb')
