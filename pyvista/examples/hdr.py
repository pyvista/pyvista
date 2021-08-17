"""HDR texture examples."""

from .downloads import _download_and_read


def download_dikhololo_night():  # pragma: no cover
    """Download and read the dikholo night hdr texture example.

    Files hosted at https://polyhaven.com/

    Returns
    -------
    pyvista.texture
        HDR Texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples    # doctest:+SKIP
    >>> gltf_file = examples.gltf.download_damaged_helmet()  # doctest:+SKIP
    >>> texture = examples.hdr.download_dikhololo_night()  # doctest:+SKIP
    >>> pl = pyvista.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.set_environment_texture(texture)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """
    texture = _download_and_read('dikhololo_night_4k.hdr', texture=True)
    texture.SetColorModeToDirectScalars()
    texture.SetMipmap(True)
    texture.SetInterpolate(True)
    return texture


def download_parched_canal():  # pragma: no cover
    """Download and read the parched canal hdr texture example.

    Files hosted at https://polyhaven.com/

    Returns
    -------
    pyvista.texture
        HDR Texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples    # doctest:+SKIP
    >>> gltf_file = examples.gltf.download_damaged_helmet()  # doctest:+SKIP
    >>> texture = examples.hdr.download_parched_canal()  # doctest:+SKIP
    >>> pl = pyvista.Plotter()  # doctest:+SKIP
    >>> pl.import_gltf(gltf_file)  # doctest:+SKIP
    >>> pl.set_environment_texture(texture)  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """
    texture = _download_and_read('parched_canal_4k.hdr', texture=True)
    texture.SetColorModeToDirectScalars()
    texture.SetMipmap(True)
    texture.SetInterpolate(True)
    return texture
