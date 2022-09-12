"""Downloadable datasets of 3D Celestial Bodies.

Examples
--------
>>> from pyvista import examples
>>> mesh = examples.planets.load_moon()
>>> mesh.plot()

"""

import numpy as np

import pyvista
from pyvista import examples

from .downloads import _download_and_read


def _sphere_with_texture_map(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Sphere with texture coordinates.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    lat_resolution : int , optional
        Set the number of points in the latitude direction.

    Returns
    -------
    pyvista.PolyData
        Sphere mesh with texture coordinates.

    """
    # https://github.com/pyvista/pyvista/pull/2994#issuecomment-1200520035
    theta, phi = np.mgrid[0 : np.pi : lat_resolution * 1j, 0 : 2 * np.pi : lon_resolution * 1j]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    sphere = pyvista.StructuredGrid(x, y, z)
    texture_coords = np.empty((sphere.n_points, 2))
    texture_coords[:, 0] = phi.ravel('F') / phi.max()
    texture_coords[:, 1] = theta.ravel('F') / theta.max()
    sphere.active_t_coords = texture_coords
    return sphere


def load_sun(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Sun source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_sun())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_sun_jpg()
    sphere.textures["atmosphere"] = surface
    return sphere


def load_moon(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Moon source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Moon dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_moon())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_moon_jpg(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def load_mercury(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Mercury source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Mercury dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_mercury())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_mercury_jpg(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def load_venus(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Venus source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Venus dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_venus())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_venus_jpg(atmosphere=False, texture=True)
    sphere.textures["surface"] = surface
    atmosphere = download_venus_jpg(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_mars(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Mars source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Mars dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_mars())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = examples.planets.download_mars_jpg(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def load_jupiter(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Jupiter source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Jupiter dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_jupiter())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_jupiter_jpg(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_saturn(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Saturn source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Saturn dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_saturn())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_saturn_jpg(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_saturn_ring_alpha(inner=0.25, outer=0.5, c_res=6):
    """Load a source for Saturn's rings with opacity.

    Arguments are passed on to :func:`pyvista.Disc`.

    Parameters
    ----------
    inner : float, optional
        The inner radius.

    outer : float, optional
        The outer radius.

    c_res : int, optional
        The number of points in the circumferential direction.

    Returns
    -------
    pyvista.PolyData
        Dataset with texture for Saturn's rings.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_saturn_ring_alpha())
    >>> plotter.show()

    """
    disc = pyvista.Disc(inner=inner, outer=outer, c_res=c_res)
    disc.active_t_coords = np.zeros((disc.points.shape[0], 2))
    radius = np.sqrt(disc.points[:, 0] ** 2 + disc.points[:, 1] ** 2)
    disc.active_t_coords[:, 0] = radius / np.max(radius)
    disc.active_t_coords[:, 1] = 0.0
    atmosphere = download_saturn_ring_alpha_png(texture=True)
    disc.textures["atmosphere"] = atmosphere
    return disc


def load_uranus(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Uranus source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Uranus dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_uranus())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_uranus_jpg(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_neptune(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Neptune source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Neptune dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_neptune())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_neptune_jpg(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_pluto(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a Pluto source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int, optional
        Set the number of points in the latitude direction.

    lon_resolution : int, optional
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Pluto dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> plotter = pyvista.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_pluto())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = examples.planets.download_pluto_jpg(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def download_sun_jpg(load=True, texture=False):  # pragma: no cover
    """Download and return the path of ``'sun.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    str
        Filename of the JPEG.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_sun_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('sun.jpg', texture=texture, load=load)


def download_moon_jpg(load=True, texture=False):  # pragma: no cover
    """Download and return the path of ``'moon.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_moon_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('moon.jpg', texture=texture, load=load)


def download_mercury_jpg(load=True, texture=False):  # pragma: no cover
    """Download and return the path of ``'mercury.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    str
        Filename of the JPEG.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_mercury_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('mercury.jpg', texture=texture, load=load)


def download_venus_jpg(atmosphere=True, load=True, texture=False):  # pragma: no cover
    """Download and return the path of ``'venus.jpg'``.

    Parameters
    ----------
    atmosphere : bool, optional
        Load the atmosphere texture when ``True``.

    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_venus_jpg()
    >>> surface.plot(cpos="xy")

    """
    if atmosphere:
        return _download_and_read('venus_atmosphere.jpg', load=load, texture=texture)
    else:
        return _download_and_read('venus_surface.jpg', load=load, texture=texture)


def download_mars_jpg(load=True, texture=False):  # pragma: no cover
    """Download and return the path of ``'mars.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_mars_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('mars.jpg', load=load, texture=texture)


def download_jupiter_jpg(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'jupiter.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_jupiter_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('jupiter.jpg', texture=texture, load=load)


def download_saturn_jpg(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'saturn.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_saturn_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('saturn.jpg', texture=texture, load=load)


def download_saturn_ring_alpha_png(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'saturn_ring_alpha.png'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    str
        Filename of the JPEG.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_saturn_ring_alpha_png()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('saturn_ring_alpha.png', texture=texture, load=load)


def download_uranus_jpg(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'uranus.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_uranus_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('uranus.jpg', texture=texture, load=load)


def download_neptune_jpg(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'neptune.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_neptune_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('neptune.jpg', texture=texture, load=load)


def download_pluto_jpg(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'pluto.png'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista
    >>> surface = examples.planets.download_pluto_jpg()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('pluto.png', texture=texture, load=load)


def download_stars_jpg(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'stars.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> plotter = pv.Plotter()
    >>> image_path = examples.planets.download_stars_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> plotter.show()

    See :func:`load_mars` for another example using this dataset.

    """
    return _download_and_read('stars.jpg', texture=texture, load=load)


def download_milkyway_jpg(texture=False, load=True):  # pragma: no cover
    """Download and return the path of ``'milkyway.jpg'``.

    Parameters
    ----------
    texture : bool, optional
        ``True`` when file being read is a texture.

    load : bool, optional
        Read the file. Default ``True``, when ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet or str
        Dataset or path to the file depending on the ``load`` parameter.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> plotter = pv.Plotter()
    >>> image_path = examples.planets.download_milkyway_jpg(load=False)
    >>> plotter.add_background_image(image_path)
    >>> plotter.show()

    """
    return _download_and_read('milkyway.jpg', texture=texture, load=load)
