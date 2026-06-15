"""Downloadable datasets of 3D Celestial Bodies."""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.examples._dataset_loader import _download_dataset
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader


def _download_dataset_texture(
    loader: _SingleFileDownloadableDatasetLoader, *, load: bool, texture: bool
):
    dataset = _download_dataset(loader, load=load)
    if texture:
        from pyvista.plotting.texture import Texture  # noqa: PLC0415

        return Texture(dataset)  # type: ignore[abstract]
    return dataset


def load_planet(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load a planet or celestial body as a sphere with texture coordinates.

    All planets are geometrically identical spheres. Textures are loaded
    and applied separately; see the ``download_*_surface`` functions.

    .. versionadded:: 0.49

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
        Sphere mesh with texture coordinates.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_planet()
    >>> texture = examples.load_globe_texture()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`planets_example`
            Example plot of the solar system.

    """
    # https://github.com/pyvista/pyvista/pull/2994#issuecomment-1200520035
    theta, phi = np.mgrid[0 : np.pi : lat_resolution * 1j, -np.pi : np.pi : lon_resolution * 1j]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    sphere = pv.StructuredGrid(x, y, z)
    texture_coords = np.empty((sphere.n_points, 2))
    texture_coords[:, 0] = (phi.ravel('F') + np.pi) / (2 * np.pi)
    texture_coords[:, 1] = theta[::-1, :].ravel('F') / theta.max()
    sphere.active_texture_coordinates = texture_coords
    return sphere.extract_surface(algorithm=None, pass_pointid=False, pass_cellid=False)


def _planet_deprecated(name):
    # Deprecated on 0.49.0, estimated removal on v0.52.0
    if pv.version_info >= (0, 52):
        msg = f'Remove deprecated function `load_{name}`.'
        raise RuntimeError(msg)
    warn_external(
        f'`load_{name}` is deprecated and will be removed in v0.52. Use `load_planet` instead.',
        PyVistaDeprecationWarning,
    )


@_deprecate_positional_args
def load_sun(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the Sun as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_sun`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_sun_surface`
            Download the surface of the Sun.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('sun')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_moon(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the Moon as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_moon`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_moon_surface`
            Download the surface of the Moon.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('moon')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_mercury(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Mercury as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_mercury`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_mercury_surface`
            Download the surface of Mercury.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('mercury')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_venus(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Venus as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_venus`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_venus_surface`
            Download the surface of the Venus.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('venus')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_earth(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Earth as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_earth`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.examples.load_globe_texture`
            Download the surface of the Earth.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('earth')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_mars(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Mars as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_mars`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_mars_surface`
            Download the surface of Mars.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('mars')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_jupiter(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Jupiter as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_jupiter`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_jupiter_surface`
            Download the surface of Jupiter.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('jupiter')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_saturn(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Saturn as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_saturn`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_saturn_surface`
            Download the surface of Saturn.

        :func:`~pyvista.examples.planets.load_saturn_rings`
            Load Saturn's rings as a textured disc.

        :func:`~pyvista.examples.planets.download_saturn_rings`
            Download the texture of Saturn's rings.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('saturn')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


def load_planet_rings(*, inner=0.25, outer=0.5, c_res=100):
    """Load planetary as a disc with texture coordinates.

    Arguments are passed on to :func:`pyvista.Disc`.

    Parameters
    ----------
    inner : float, default: 0.25
        The inner radius.

    outer : float, default: 0.5
        The outer radius.

    c_res : int, default: 90
        The number of points in the circumferential direction.

    Returns
    -------
    pyvista.PolyData
        Dataset with texture coordinates for planetary rings.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.planets.load_planet_rings()
    >>> texture = examples.planets.download_saturn_rings(texture=True)
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :func:`~pyvista.examples.planets.download_saturn_rings`
            Download the texture of Saturn's rings.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    disc = pv.Disc(inner=inner, outer=outer, c_res=c_res)
    texture_coordinates = np.zeros((disc.points.shape[0], 2))
    radius = np.sqrt(disc.points[:, 0] ** 2 + disc.points[:, 1] ** 2)
    texture_coordinates[:, 0] = (radius - inner) / (outer - inner)
    texture_coordinates[:, 1] = 0.0
    disc.active_texture_coordinates = texture_coordinates
    return disc


@_deprecate_positional_args
def load_saturn_rings(inner=0.25, outer=0.5, c_res=6):
    """Load the planet Saturn's rings.

    .. deprecated:: 0.49.0
        Use :func:`load_planet_rings` instead. ``load_saturn_rings`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_saturn_rings`
            Download the texture of Saturn's rings.

        :func:`~pyvista.examples.planets.load_planet_rings`
            Load planetary rings as a textured disc.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :func:`~pyvista.examples.planets.download_saturn_surface`
            Download the surface of Saturn.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    # Deprecated on 0.49.0, estimated removal on v0.52.0
    if pv.version_info >= (0, 52):
        msg = 'Remove deprecated `load_saturn_rings`.'
        raise RuntimeError(msg)
    warn_external(
        '`load_saturn_rings` is deprecated and will be removed in v0.52. '
        'Use `load_planet_rings` instead.',
        PyVistaDeprecationWarning,
    )
    return load_planet_rings(inner=inner, outer=outer, c_res=c_res)


@_deprecate_positional_args
def load_uranus(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Uranus as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_uranus`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_uranus_surface`
            Download the surface of Uranus.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('uranus')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_neptune(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the planet Neptune as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_neptune`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_neptune_surface`
            Download the surface of Neptune.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('neptune')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def load_pluto(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Load the dwarf planet Pluto as a textured sphere.

    .. deprecated:: 0.49.0
        Use :func:`load_planet` instead. ``load_pluto`` will be removed in v0.52.

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

    .. seealso::

        :func:`~pyvista.examples.planets.download_pluto_surface`
            Download the surface of Pluto.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('pluto')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


@_deprecate_positional_args
def download_sun_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_sun_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Sun Surface Dataset <sun_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_sun`
            Load the Sun as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_sun_surface, load=load, texture=texture)


_dataset_sun_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/sun.jpg',
)


@_deprecate_positional_args
def download_moon_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_moon_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Moon Surface Dataset <moon_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_moon`
            Load the Moon as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_moon_surface, load=load, texture=texture)


_dataset_moon_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/moon.jpg',
)


@_deprecate_positional_args
def download_mercury_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_mercury_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Mercury Surface Dataset <mercury_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_mercury`
            Load Mercury as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_mercury_surface, load=load, texture=texture)


_dataset_mercury_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/mercury.jpg',
)


@_deprecate_positional_args
def download_venus_surface(
    atmosphere=True,  # noqa: FBT002
    texture=False,  # noqa: FBT002
    load=True,  # noqa: FBT002
):  # pragma: no cover
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_venus_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Venus Surface Dataset <venus_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_venus`
            Load Venus as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    if atmosphere:
        return _download_dataset_texture(_dataset_venus_surface, load=load, texture=texture)
    return _download_dataset_texture(
        __dataset_venus_surface_no_atmosphere, load=load, texture=texture
    )


_dataset_venus_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/venus_atmosphere.jpg',
)
__dataset_venus_surface_no_atmosphere = _SingleFileDownloadableDatasetLoader(
    'solar_textures/venus_surface.jpg',
)


@_deprecate_positional_args
def download_mars_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_mars_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Mars Surface Dataset <mars_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_mars`
            Load Mars as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_mars_surface, load=load, texture=texture)


_dataset_mars_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/mars.jpg',
)


@_deprecate_positional_args
def download_jupiter_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_jupiter_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Jupiter Surface Dataset <jupiter_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_jupiter`
            Load Jupiter as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_jupiter_surface, load=load, texture=texture)


_dataset_jupiter_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/jupiter.jpg',
)


@_deprecate_positional_args
def download_saturn_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_saturn_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Saturn Surface Dataset <saturn_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_saturn`
            Load the planet Saturn as a textured sphere.

        :func:`~pyvista.examples.planets.load_saturn_rings`
            Load Saturn's rings as a textured disc.

        :func:`~pyvista.examples.planets.download_saturn_rings`
            Download the texture of Saturn's rings.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_saturn_surface, load=load, texture=texture)


_dataset_saturn_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/saturn.jpg',
)


@_deprecate_positional_args
def download_saturn_rings(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.ImageData | pyvista.Texture | str
        Dataset, texture, or filename of the Saturn's rings.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_saturn_rings(texture=True)
    >>> texture.plot(cpos='xy')

    .. seealso::

        :ref:`Saturn Rings Dataset <saturn_rings_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_saturn_rings`
            Load Saturn's rings as a textured disc.

        :func:`~pyvista.examples.planets.load_saturn`
            Load the planet Saturn as a textured sphere.

        :func:`~pyvista.examples.planets.download_saturn_surface`
            Download the surface of Saturn.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_saturn_rings, load=load, texture=texture)


_dataset_saturn_rings = _SingleFileDownloadableDatasetLoader(
    'solar_textures/saturn_ring_alpha.png',
)


@_deprecate_positional_args
def download_uranus_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_uranus_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Uranus Surface Dataset <uranus_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_uranus`
            Load Uranus as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_uranus_surface, load=load, texture=texture)


_dataset_uranus_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/uranus.jpg',
)


@_deprecate_positional_args
def download_neptune_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_neptune_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Neptune Surface Dataset <neptune_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_neptune`
            Load Neptune as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_neptune_surface, load=load, texture=texture)


_dataset_neptune_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/neptune.jpg',
)


@_deprecate_positional_args
def download_pluto_surface(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_pluto_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    .. seealso::

        :ref:`Pluto Surface Dataset <pluto_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_pluto`
            Load Pluto as a textured sphere.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_pluto_surface, load=load, texture=texture)


_dataset_pluto_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/pluto.jpg',
)


@_deprecate_positional_args
def download_stars_sky_background(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    Load the night sky image as a background image.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> pl.add_background_image(image_path)
    >>> pl.show()


    .. seealso::

        :ref:`Stars Sky Background Dataset <stars_sky_background_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_mars`
            Load Mars as a textured sphere.

        :ref:`Milkyway Sky Background Dataset <milkyway_sky_background_dataset>`
            Sky texture of the Milky Way galaxy.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_stars_sky_background, load=load, texture=texture)


_dataset_stars_sky_background = _SingleFileDownloadableDatasetLoader(
    'planet3d-matlab/stars.jpg',
)


@_deprecate_positional_args
def download_milkyway_sky_background(texture=False, load=True):  # pragma: no cover  # noqa: FBT002
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
    output : pyvista.DataSet | pyvista.Texture | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    Load the Milky Way sky image as a background image.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> image_path = examples.planets.download_milkyway_sky_background(load=False)
    >>> pl.add_background_image(image_path)
    >>> pl.show()

    .. seealso::

        :ref:`Milkyway Sky Background Dataset <milkyway_sky_background_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Stars Sky Background Dataset <stars_sky_background_dataset>`
            Night sky stars texture.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_milkyway_sky_background, load=load, texture=texture)


_dataset_milkyway_sky_background = _SingleFileDownloadableDatasetLoader(
    'planet3d-matlab/milkyway.jpg',
)
