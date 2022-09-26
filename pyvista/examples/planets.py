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
    texture_coords[:, 1] = theta[::-1, :].ravel('F') / theta.max()
    sphere.active_t_coords = texture_coords
    return sphere


def load_sun(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the Sun as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_sun())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_sun_texture(texture=True)
    sphere.textures["atmosphere"] = surface
    return sphere


def load_moon(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the Moon as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_moon())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_moon_texture(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def load_mercury(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Mercury as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_mercury())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_mercury_texture(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def load_venus(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Venus as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_venus())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = download_venus_texture(atmosphere=False, texture=True)
    sphere.textures["surface"] = surface
    atmosphere = download_venus_texture(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_mars(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Mars as a textured Sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_mars())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = examples.planets.download_mars_texture(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def load_jupiter(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Jupiter as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_jupiter())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_jupiter_texture(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_saturn(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Saturn as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_saturn())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_saturn_texture(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_saturn_ring_alpha(inner=0.25, outer=0.5, c_res=6):
    """Load the planet Saturn's rings.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
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
    """Load the planet Uranus as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_uranus())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_uranus_texture(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_neptune(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Neptune as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_neptune())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    atmosphere = download_neptune_texture(texture=True)
    sphere.textures["atmosphere"] = atmosphere
    return sphere


def load_pluto(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the dwarf planet Pluto as a textured sphere.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> _ = plotter.add_mesh(examples.planets.load_pluto())
    >>> plotter.show()

    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    surface = examples.planets.download_pluto_texture(texture=True)
    sphere.textures["surface"] = surface
    return sphere


def download_sun_texture(texture=False, load=True):  # pragma: no cover
    """Download and potentially load the Sun's texture.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_sun_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('sun.jpg', texture=texture, load=load)


def download_moon_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the Earth's Moon.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_moon_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('moon.jpg', texture=texture, load=load)


def download_mercury_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of planet Mercury.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_mercury_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('mercury.jpg', texture=texture, load=load)


def download_venus_texture(atmosphere=True, texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the Planet Venus.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_venus_texture()
    >>> surface.plot(cpos="xy")

    """
    if atmosphere:
        return _download_and_read('venus_atmosphere.jpg', load=load, texture=texture)
    else:
        return _download_and_read('venus_surface.jpg', load=load, texture=texture)


def download_mars_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the planet Mars.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_mars_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('mars.jpg', load=load, texture=texture)


def download_jupiter_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the planet Jupiter.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_jupiter_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('jupiter.jpg', texture=texture, load=load)


def download_saturn_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the planet Saturn.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_saturn_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('saturn.jpg', texture=texture, load=load)


def download_saturn_ring_alpha_png(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the planet Saturn.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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


def download_uranus_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the planet Uranus.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_uranus_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('uranus.jpg', texture=texture, load=load)


def download_neptune_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the planet Neptune.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_neptune_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('neptune.jpg', texture=texture, load=load)


def download_pluto_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the texture of the dwarf planet Pluto.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> surface = examples.planets.download_pluto_texture()
    >>> surface.plot(cpos="xy")

    """
    return _download_and_read('pluto.jpg', texture=texture, load=load)


def download_stars_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load a sky stars texture.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> image_path = examples.planets.download_stars_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> plotter.show()

    See :func:`load_mars` for another example using this dataset.

    """
    return _download_and_read('stars.jpg', texture=texture, load=load)


def download_milkyway_texture(texture=False, load=True):  # pragma: no cover
    """Download and optionally load the sky texture of the Milky Way galaxy.

    Textures obtained from `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

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
    >>> image_path = examples.planets.download_milkyway_texture(load=False)
    >>> plotter.add_background_image(image_path)
    >>> plotter.show()

    """
    return _download_and_read('milkyway.jpg', texture=texture, load=load)
