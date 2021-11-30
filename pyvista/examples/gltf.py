"""glTF examples."""

from .downloads import _retrieve_file

GLTF_SAMPLES_ROOT_URL = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/'


def _download_file(end_url):  # pragma: no cover
    """Download a gltf example file."""
    basename = end_url.split('/')[-1]
    filename, _ = _retrieve_file(GLTF_SAMPLES_ROOT_URL + end_url, basename)
    return filename


def download_damaged_helmet():  # pragma: no cover
    """Download the damaged helmet example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples    # doctest:+SKIP
    >>> gltf_file = examples.gltf.download_damaged_helmet()  # doctest:+SKIP
    >>> cubemap = examples.download_sky_box_cube_map()  # doctest:+SKIP
    >>> pl = pyvista.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.set_environment_texture(cubemap)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """
    return _download_file('DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf')


def download_sheen_chair():  # pragma: no cover
    """Download the sheen chair example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples    # doctest:+SKIP
    >>> gltf_file = examples.gltf.download_sheen_chair()  # doctest:+SKIP
    >>> cubemap = examples.download_sky_box_cube_map()  # doctest:+SKIP
    >>> pl = pyvista.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.set_environment_texture(cubemap)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """
    return _download_file('SheenChair/glTF-Binary/SheenChair.glb')


def download_gearbox():  # pragma: no cover
    """Download the gearbox example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples    # doctest:+SKIP
    >>> gltf_file = examples.gltf.download_gearbox()  # doctest:+SKIP
    >>> pl = pyvista.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """
    return _download_file('GearboxAssy/glTF-Binary/GearboxAssy.glb')


def download_avocado():  # pragma: no cover
    """Download the avocado example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples    # doctest:+SKIP
    >>> gltf_file = examples.gltf.download_avocado()  # doctest:+SKIP
    >>> pl = pyvista.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """
    return _download_file('Avocado/glTF-Binary/Avocado.glb')


def download_milk_truck():  # pragma: no cover
    """Download the avocado example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples    # doctest:+SKIP
    >>> gltf_file = examples.gltf.download_milk_truck()  # doctest:+SKIP
    >>> pl = pyvista.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """
    return _download_file('CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb')
