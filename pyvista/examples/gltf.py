"""Contains glTF examples."""

from __future__ import annotations

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.examples import downloads

if pv.version_info >= (0, 52):
    msg = (
        '`pyvista.examples.gltf` should be removed. This module and all of its '
        'functions were deprecated in v0.49 and were scheduled for removal in v0.52.'
    )
    raise RuntimeError(msg)

warn_external(
    '`pyvista.examples.gltf` is deprecated and will be removed in a future '
    'version. Use `pyvista.examples.downloads` instead.',
    pv.PyVistaDeprecationWarning,
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
    warn_external(
        '`examples.gltf.download_damaged_helmet` is deprecated. Use '
        '`examples.download_damaged_helmet` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_damaged_helmet(load=False)


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
    warn_external(
        '`examples.gltf.download_sheen_chair` is deprecated. Use '
        '`examples.download_sheen_chair` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_sheen_chair(load=False)


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
    warn_external(
        '`examples.gltf.download_gearbox` is deprecated. Use `examples.download_gearbox` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_gearbox(load=False)


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
    warn_external(
        '`examples.gltf.download_avocado` is deprecated. Use `examples.download_avocado` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_avocado(load=False)


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
    warn_external(
        '`examples.gltf.download_milk_truck` is deprecated. Use '
        '`examples.download_milk_truck` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_milk_truck(load=False)
