"""Contains glTF examples."""

from __future__ import annotations

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.examples import downloads
from pyvista.examples.downloads import _gltf_loader

if pv.version_info >= (0, 52):  # pragma: no cover
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

    """
    warn_external(
        '`examples.gltf.download_damaged_helmet` is deprecated. Use '
        '`examples.download_damaged_helmet` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_damaged_helmet(load=False)


def download_sheen_chair():  # pragma: no cover
    """Download the sheen chair example.

    .. deprecated:: 0.49.0
        This example uses the unsupported glTF extension
        ``KHR_texture_transform`` and will be removed in v0.52.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

    """
    warn_external(
        '`download_sheen_chair` is deprecated and will be removed in v0.52. '
        'It uses the unsupported glTF extension `KHR_texture_transform`.',
        pv.PyVistaDeprecationWarning,
    )
    if pv.version_info >= (0, 52):  # pragma: no cover
        msg = (
            "Remove this deprecated function and remove the 'sheen_chair' "
            'dict mapping from the `_gltf_loader`'
        )
        raise RuntimeError(msg)
    return _gltf_loader('sheen_chair').download()


def download_gearbox():  # pragma: no cover
    """Download the gearbox example.

    Files hosted at https://github.com/KhronosGroup/glTF-Sample-Models

    Returns
    -------
    str
        Filename of the gltf file.

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

    """
    warn_external(
        '`examples.gltf.download_milk_truck` is deprecated. Use '
        '`examples.download_milk_truck` instead.',
        pv.PyVistaDeprecationWarning,
    )
    return downloads.download_milk_truck(load=False)
