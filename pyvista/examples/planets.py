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
    """Sphere with texture coordinate.

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
    """Load a sun source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    """Load a moon source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    >>> _ = plotter.add_mesh(examples.planets.load_moon())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_moon_jpg()
    sphere.textures["surface"] = surface
    return sphere


def load_mercury(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a mercury source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    >>> _ = plotter.add_mesh(examples.planets.load_mercury())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_mercury_jpg()
    sphere.textures["surface"] = surface
    return sphere


def load_venus(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a venus source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    surface = download_venus_jpg(atmosphere=False)
    sphere.textures["surface"] = surface
    atmosphere = download_venus_jpg()
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_mars(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a mars source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    >>> _ = plotter.add_mesh(examples.planets.load_mars())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = examples.planets.download_mars_jpg()
    sphere.textures["surface"] = surface
    return sphere


def load_jupiter(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a jupiter source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    >>> _ = plotter.add_mesh(examples.planets.load_jupiter())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_jupiter_jpg()
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_saturn(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a saturn source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    >>> _ = plotter.add_mesh(examples.planets.load_saturn())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_saturn_jpg()
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_saturn_ring_alpha(*args, **kwargs):
    """Load a saturn_ring_alpha source.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

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
    >>> _ = plotter.add_mesh(examples.planets.load_saturn_ring_alpha())
    >>> plotter.show()

    """
    disc = pyvista.Disc(*args, **kwargs)
    disc.active_t_coords = np.zeros((disc.points.shape[0], 2))
    radius = np.sqrt(disc.points[:, 0] ** 2 + disc.points[:, 1] ** 2)
    disc.active_t_coords[:, 0] = radius / np.max(radius)
    disc.active_t_coords[:, 1] = 0.0
    atmosphere = download_saturn_ring_alpha_png()
    disc.textures["atmosphere"] = atmosphere
    return disc


def load_uranus(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a uranus source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    >>> _ = plotter.add_mesh(examples.planets.load_uranus())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_uranus_jpg()
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_neptune(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a neptune source.

    Parameters
    ----------
    radius : float, optional
        Sphere radius.

    lat_resolution : int , optional
        Set the number of points in the longitude direction.

    lon_resolution : int, optional
        Set the number of points in the latitude direction.

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
    >>> _ = plotter.add_mesh(examples.planets.load_neptune())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_neptune_jpg()
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def download_sun_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'sun.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('sun.jpg', texture=True, load=load)


def download_moon_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'moon.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('moon.jpg', texture=True, load=load)


def download_mercury_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'mercury.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('mercury.jpg', texture=True, load=load)


def download_venus_jpg(atmosphere=True, load=True):  # pragma: no cover
    """Download and return the path of ``'venus.jpg'``.

    Parameters
    ----------
    atmosphere : bool, optional
        Load the atmosphere texture when ``True``.

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
        return _download_and_read('venus_atmosphere.jpg', texture=True, load=load)
    else:
        return _download_and_read('venus_surface.jpg', texture=True, load=load)


def download_mars_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'mars.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('mars.jpg', texture=True, load=load)


def download_jupiter_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'jupiter.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('jupiter.jpg', texture=True, load=load)


def download_saturn_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'saturn.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('saturn.jpg', texture=True, load=load)


def download_saturn_ring_alpha_png(load=True):  # pragma: no cover
    """Download and return the path of ``'saturn_ring_alpha.png'``.

    Parameters
    ----------
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
    return _download_and_read('saturn_ring_alpha.png', texture=True, load=load)


def download_uranus_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'uranus.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('uranus.jpg', texture=True, load=load)


def download_neptune_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'neptune.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('neptune.jpg', texture=True, load=load)


def download_stars_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'stars.jpg'``.

    Parameters
    ----------
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

    See :func:`download_mars_jpg` for another example using this dataset.

    """
    return _download_and_read('stars.jpg', load=load)


def download_milkyway_jpg(load=True):  # pragma: no cover
    """Download and return the path of ``'milkyway.jpg'``.

    Parameters
    ----------
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
    return _download_and_read('milkyway.jpg', load=load)
