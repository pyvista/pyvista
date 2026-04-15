"""Cubemap utilities."""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def cubemap(path='', prefix='', ext='.jpg'):
    """Construct a cubemap from 6 images from a directory.

    Each of the 6 images must be in the following format:

    - <prefix>negx<ext>
    - <prefix>negy<ext>
    - <prefix>negz<ext>
    - <prefix>posx<ext>
    - <prefix>posy<ext>
    - <prefix>posz<ext>

    Prefix may be empty, and extension will default to ``'.jpg'``

    For example, if you have 6 images with the skybox2 prefix:

    - ``'skybox2-negx.jpg'``
    - ``'skybox2-negy.jpg'``
    - ``'skybox2-negz.jpg'``
    - ``'skybox2-posx.jpg'``
    - ``'skybox2-posy.jpg'``
    - ``'skybox2-posz.jpg'``

    Parameters
    ----------
    path : str, default: ""
        Directory containing the cubemap images.

    prefix : str, default: ""
        Prefix to the filename.

    ext : str, default: ".jpg"
        The filename extension.  For example ``'.jpg'``.

    Returns
    -------
    pyvista.Texture
        Texture with cubemap.

    Notes
    -----
    Cubemap will appear flipped relative to the XY plane between VTK v9.1 and
    VTK v9.2.

    Examples
    --------
    Load a skybox given a directory, prefix, and file extension.

    >>> import pyvista as pv
    >>> skybox = pv.cubemap('my_directory', 'skybox', '.jpeg')  # doctest:+SKIP

    """
    sets = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz']
    image_paths = [str(Path(path) / f'{prefix}{suffix}{ext}') for suffix in sets]
    return _cubemap_from_paths(image_paths)


def cubemap_from_filenames(image_paths):
    """Construct a cubemap from 6 images.

    Images must be in the following order:

    - Positive X
    - Negative X
    - Positive Y
    - Negative Y
    - Positive Z
    - Negative Z

    Parameters
    ----------
    image_paths : sequence[str]
        Paths of the individual cubemap images.

    Returns
    -------
    pyvista.Texture
        Texture with cubemap.

    Examples
    --------
    Load a skybox given a list of image paths.

    >>> image_paths = [
    ...     '/home/user/_px.jpg',
    ...     '/home/user/_nx.jpg',
    ...     '/home/user/_py.jpg',
    ...     '/home/user/_ny.jpg',
    ...     '/home/user/_pz.jpg',
    ...     '/home/user/_nz.jpg',
    ... ]
    >>> skybox = pv.cubemap(image_paths=image_paths)  # doctest:+SKIP

    """
    if len(image_paths) != 6:
        msg = 'image_paths must contain 6 paths'
        raise ValueError(msg)

    return _cubemap_from_paths(image_paths)


def _cubemap_from_paths(image_paths):
    """Construct a cubemap from image paths."""
    for image_path in image_paths:
        if not Path(image_path).is_file():
            file_str = '\n'.join(image_paths)
            msg = (
                f'Unable to locate {image_path}\nExpected to find the following files:\n{file_str}'
            )
            raise FileNotFoundError(msg)

    texture = pv.Texture()  # type: ignore[abstract]
    texture.mipmap = True
    texture.interpolate = True
    texture.color_mode = 'direct'
    texture.cube_map = True  # Must be set prior to setting images

    # add each image to the cubemap
    for i, fn in enumerate(image_paths):
        # Read and flip along y-axis
        texture.SetInputDataObject(i, pv.read(fn)._flip_uniform(1))

    return texture
