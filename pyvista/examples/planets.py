"""Downloadable datasets of 3D Celestial Bodies."""

from __future__ import annotations

import numpy as np

import pyvista

from .downloads import _download_and_read


def _sphere_with_texture_map(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Sphere with texture coordinates.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 100
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

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
    sphere.active_texture_coordinates = texture_coords
    return sphere.extract_surface(pass_pointid=False, pass_cellid=False)


def load_sun(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the Sun as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Sun dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_sun()
    >>> texture = examples.planets.download_sun_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_moon(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the Moon as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Moon dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_moon()
    >>> texture = examples.planets.download_moon_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_mercury(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the planet Mercury as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Mercury dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_mercury()
    >>> texture = examples.planets.download_mercury_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_venus(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the planet Venus as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Venus dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_venus()
    >>> texture = examples.planets.download_venus_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_earth(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Earth as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Earth dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_earth()
    >>> texture = examples.load_globe_texture()
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_mars(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the planet Mars as a textured Sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Mars dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_mars()
    >>> texture = examples.planets.download_mars_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_jupiter(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the planet Jupiter as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Jupiter dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_jupiter()
    >>> texture = examples.planets.download_jupiter_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_saturn(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the planet Saturn as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Saturn dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_saturn()
    >>> texture = examples.planets.download_saturn_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_saturn_rings(inner=0.25, outer=0.5, c_res=6):  # pragma: no cover
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
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_saturn_rings()
    >>> texture = examples.planets.download_saturn_rings(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    disc = pyvista.Disc(inner=inner, outer=outer, c_res=c_res)
    disc.active_texture_coordinates = np.zeros((disc.points.shape[0], 2))
    radius = np.sqrt(disc.points[:, 0] ** 2 + disc.points[:, 1] ** 2)
    disc.active_texture_coordinates[:, 0] = radius / np.max(radius)
    disc.active_texture_coordinates[:, 1] = 0.0
    return disc


def load_uranus(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the planet Uranus as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Uranus dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_uranus()
    >>> texture = examples.planets.download_uranus_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_neptune(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the planet Neptune as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Neptune dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_neptune()
    >>> texture = examples.planets.download_neptune_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def load_pluto(radius=1.0, lat_resolution=50, lon_resolution=100):  # pragma: no cover
    """Load the dwarf planet Pluto as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Pluto dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_pluto()
    >>> texture = examples.planets.download_pluto_surface(texture=True)
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> _ = pl.add_mesh(mesh, texture=texture)
    >>> pl.show()

    """
    return _sphere_with_texture_map(
        radius=radius,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
    )


def download_sun_surface(texture=False, load=True):  # pragma: no cover
    """Download the surface of the Sun.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_sun_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/sun.jpg', texture=texture, load=load)


def download_moon_surface(texture=False, load=True):  # pragma: no cover
    """Download the surface of the Earth's Moon.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_moon_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/moon.jpg', texture=texture, load=load)


def download_mercury_surface(texture=False, load=True):  # pragma: no cover
    """Download the surface of planet Mercury.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_mercury_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/mercury.jpg', texture=texture, load=load)


def download_venus_surface(atmosphere=True, texture=False, load=True):  # pragma: no cover
    """Download the surface or atmosphere of Planet Venus.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    atmosphere : bool, optional
        Load the atmosphere texture when ``True``.

    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_venus_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    if atmosphere:
        return _download_and_read('solar_textures/venus_atmosphere.jpg', load=load, texture=texture)
    else:
        return _download_and_read('solar_textures/venus_surface.jpg', load=load, texture=texture)


def download_mars_surface(texture=False, load=True):  # pragma: no cover
    """Download the surface of the planet Mars.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_mars_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/mars.jpg', load=load, texture=texture)


def download_jupiter_surface(texture=False, load=True):  # pragma: no cover
    """Download the surface of the planet Jupiter.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_jupiter_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/jupiter.jpg', texture=texture, load=load)


def download_saturn_surface(texture=False, load=True):  # pragma: no cover
    """Download the surface of the planet Saturn.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_saturn_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/saturn.jpg', texture=texture, load=load)


def download_saturn_rings(texture=False, load=True):  # pragma: no cover
    """Download the texture of Saturn's rings.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.ImageData, pyvista.Texture, or str
        Dataset, texture, or filename of the Saturn's rings.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_saturn_rings(texture=True)
    >>> texture.plot(cpos='xy')

    """
    return _download_and_read('solar_textures/saturn_ring_alpha.png', texture=texture, load=load)


def download_uranus_surface(texture=False, load=True):  # pragma: no cover
    """Download and the texture of the surface of planet Uranus.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_uranus_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/uranus.jpg', texture=texture, load=load)


def download_neptune_surface(texture=False, load=True):  # pragma: no cover
    """Download the texture of the surface of planet Neptune.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_neptune_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/neptune.jpg', texture=texture, load=load)


def download_pluto_surface(texture=False, load=True):  # pragma: no cover
    """Download the texture of the surface of the dwarf planet Pluto.

    Textures obtained from `Solar Textures
    <https://www.solarsystemscope.com/textures/>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_pluto_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    """
    return _download_and_read('solar_textures/pluto.jpg', texture=texture, load=load)


def download_stars_sky_background(texture=False, load=True):  # pragma: no cover
    """Download the night sky stars texture.

    Textures obtained from `tamaskis/planet3D-MATLAB
    <https://github.com/tamaskis/planet3D-MATLAB>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    Load the night sky image as a background image.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> pl.show()

    See :func:`load_mars` for another example using this dataset.

    """
    return _download_and_read('planet3d-matlab/stars.jpg', texture=texture, load=load)


def download_milkyway_sky_background(texture=False, load=True):  # pragma: no cover
    """Download the sky texture of the Milky Way galaxy.

    Textures obtained from `tamaskis/planet3D-MATLAB
    <https://github.com/tamaskis/planet3D-MATLAB>`_.

    Parameters
    ----------
    texture : bool, default: False
        Set to ``True`` when loading the surface as a texture.

    load : bool, default: True
        Load the dataset. When ``False``, return the path to the file.

    Returns
    -------
    pyvista.DataSet, pyvista.Texture, or str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    Load the Milky Way sky image as a background image.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_milkyway_sky_background(
    ...     load=False
    ... )
    >>> pl.add_background_image(image_path)
    >>> pl.show()

    """
    return _download_and_read('planet3d-matlab/milkyway.jpg', texture=texture, load=load)
