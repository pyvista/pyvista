"""Downloadable datasets of 3D Celestial Bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _download_dataset
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader

if TYPE_CHECKING:
    from pyvista import ImageData
    from pyvista import PolyData
    from pyvista import Texture


def _download_dataset_texture(
    loader: _SingleFileDownloadableDatasetLoader, *, load: bool, texture: bool
):
    dataset = _download_dataset(loader, load=load)
    if texture:
        from pyvista.plotting.texture import Texture  # noqa: PLC0415

        return Texture(dataset)  # type: ignore[abstract]
    return dataset


def load_planet(
    radius: float = 1.0, lat_resolution: int = 50, lon_resolution: int = 100
) -> PolyData:
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

        :ref:`Planet Dataset <planet_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _dataset_planet.load(  # type: ignore[return-value]
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )


def _planet_load_func(radius=1.0, lat_resolution=50, lon_resolution=100):
    return pv.Sphere(
        radius=radius,
        theta_resolution=lon_resolution,
        phi_resolution=lat_resolution,
        texture_coordinates=True,
        tessellation='phi_theta',
    ).rotate_z(180)  # rotate to align Earth's Prime Meridian to +X axis (0 degrees longitude)


_dataset_planet = _DatasetLoader(_planet_load_func)


def _planet_deprecated(name):
    # Deprecated on 0.49.0, estimated removal on v0.52.0
    if pv.version_info >= (0, 52):  # pragma: no cover
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

        :func:`~pyvista.examples.planets.load_planet_rings`
            Load planetary rings as a disc with texture coordinates.

        :func:`~pyvista.examples.planets.download_saturn_rings`
            Download the texture of Saturn's rings.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    _planet_deprecated('saturn')
    return load_planet(radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution)


def load_planet_rings(*, inner: float = 0.25, outer: float = 0.5, c_res: int = 50) -> PolyData:
    """Load planetary rings as a disc with texture coordinates.

    Arguments are passed on to :func:`pyvista.Disc`.

    .. versionadded:: 0.49

    Parameters
    ----------
    inner : float, default: 0.25
        The inner radius.

    outer : float, default: 0.5
        The outer radius.

    c_res : int, default: 50
        The number of cells in the circumferential direction.

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

        :ref:`Planet Rings Dataset <planet_rings_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.download_saturn_rings`
            Download the texture of Saturn's rings.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :func:`~pyvista.examples.planets.download_saturn_surface`
            Download the surface of Saturn.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _dataset_planet_rings.load(inner=inner, outer=outer, c_res=c_res)  # type: ignore[return-value]


def _planet_rings_load_func(*, inner=0.25, outer=0.5, c_res=50):
    disc = pv.Disc(inner=inner, outer=outer, c_res=c_res)
    texture_coordinates = np.zeros((disc.points.shape[0], 2))
    radius = np.sqrt(disc.points[:, 0] ** 2 + disc.points[:, 1] ** 2)
    texture_coordinates[:, 0] = (radius - inner) / (outer - inner)
    texture_coordinates[:, 1] = 0.0
    disc.active_texture_coordinates = texture_coordinates
    return disc


_dataset_planet_rings = _DatasetLoader(_planet_rings_load_func)


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
    if pv.version_info >= (0, 52):  # pragma: no cover
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


@overload
def download_sun_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_sun_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_sun_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_sun_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_sun_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Sun Surface Dataset <sun_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_sun_surface, load=load, texture=texture)


_dataset_sun_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/sun.jpg',
)


@overload
def download_moon_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_moon_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_moon_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_moon_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_moon_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Moon Surface Dataset <moon_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_moon_surface, load=load, texture=texture)


_dataset_moon_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/moon.jpg',
)


@overload
def download_mercury_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_mercury_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_mercury_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_mercury_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_mercury_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Mercury Surface Dataset <mercury_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_mercury_surface, load=load, texture=texture)


_dataset_mercury_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/mercury.jpg',
)


@overload
def download_venus_surface(
    atmosphere: bool = ...,  # noqa: FBT001
    texture: Literal[False] = ...,
    load: Literal[True] = ...,
) -> ImageData: ...
@overload
def download_venus_surface(
    atmosphere: bool = ...,  # noqa: FBT001
    texture: Literal[False] = ...,
    load: Literal[False] = ...,
) -> str: ...
@overload
def download_venus_surface(
    atmosphere: bool = ...,  # noqa: FBT001
    texture: Literal[True] = ...,
    load: Literal[True] = ...,
) -> Texture: ...
@overload
def download_venus_surface(
    atmosphere: bool = ...,  # noqa: FBT001
    texture: Literal[True] = ...,
    load: Literal[False] = ...,
) -> str: ...
@_deprecate_positional_args
def download_venus_surface(
    atmosphere: bool = True,  # noqa: FBT001, FBT002
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:  # pragma: no cover
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
    output : pyvista.Texture | pyvista.ImageData | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_venus_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Venus Surface Dataset <venus_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

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


@overload
def download_mars_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_mars_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_mars_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_mars_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_mars_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Mars Surface Dataset <mars_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_mars_surface, load=load, texture=texture)


_dataset_mars_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/mars.jpg',
)


@overload
def download_jupiter_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_jupiter_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_jupiter_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_jupiter_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_jupiter_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Jupiter Surface Dataset <jupiter_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_jupiter_surface, load=load, texture=texture)


_dataset_jupiter_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/jupiter.jpg',
)


@overload
def download_saturn_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_saturn_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_saturn_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_saturn_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_saturn_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Saturn Surface Dataset <saturn_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :func:`~pyvista.examples.planets.load_planet_rings`
            Load planetary rings as a disc with texture coordinates.

        :func:`~pyvista.examples.planets.download_saturn_rings`
            Download the texture of Saturn's rings.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_saturn_surface, load=load, texture=texture)


_dataset_saturn_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/saturn.jpg',
)


@overload
def download_saturn_rings(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_saturn_rings(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_saturn_rings(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_saturn_rings(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_saturn_rings(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet_rings()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Saturn Rings Dataset <saturn_rings_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet_rings`
            Load planetary rings as a disc with texture coordinates.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :func:`~pyvista.examples.planets.download_saturn_surface`
            Download the surface of Saturn.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_saturn_rings, load=load, texture=texture)


_dataset_saturn_rings = _SingleFileDownloadableDatasetLoader(
    'solar_textures/saturn_ring_alpha.png',
)


@overload
def download_uranus_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_uranus_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_uranus_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_uranus_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_uranus_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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
    output : pyvista.Texture | pyvista.ImageData | str
        Texture, Dataset, or path to the file depending on the ``load`` and
        ``texture`` parameters.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.planets.download_uranus_surface(texture=True)
    >>> texture.plot(zoom='tight', show_axes=False)

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Uranus Surface Dataset <uranus_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_uranus_surface, load=load, texture=texture)


_dataset_uranus_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/uranus.jpg',
)


@overload
def download_neptune_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_neptune_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_neptune_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_neptune_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_neptune_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Neptune Surface Dataset <neptune_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_neptune_surface, load=load, texture=texture)


_dataset_neptune_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/neptune.jpg',
)


@overload
def download_pluto_surface(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_pluto_surface(texture: Literal[False] = ..., load: Literal[False] = ...) -> str: ...
@overload
def download_pluto_surface(texture: Literal[True], load: Literal[True] = ...) -> Texture: ...
@overload
def download_pluto_surface(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_pluto_surface(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

    >>> mesh = examples.planets.load_planet()
    >>> image_path = examples.planets.download_stars_sky_background(load=False)
    >>> mesh.plot(texture=texture, background=image_path)

    .. seealso::

        :ref:`Pluto Surface Dataset <pluto_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_pluto_surface, load=load, texture=texture)


_dataset_pluto_surface = _SingleFileDownloadableDatasetLoader(
    'solar_textures/pluto.jpg',
)


@overload
def download_stars_sky_background(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_stars_sky_background(
    texture: Literal[False] = ..., load: Literal[False] = ...
) -> str: ...
@overload
def download_stars_sky_background(
    texture: Literal[True], load: Literal[True] = ...
) -> Texture: ...
@overload
def download_stars_sky_background(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_stars_sky_background(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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

        :func:`~pyvista.examples.planets.load_planet`
            Load a planet as a sphere with texture coordinates.

        :ref:`Milkyway Sky Background Dataset <milkyway_sky_background_dataset>`
            Sky texture of the Milky Way galaxy.

        :ref:`planets_example`
            Example plot of the solar system.

    """
    return _download_dataset_texture(_dataset_stars_sky_background, load=load, texture=texture)


_dataset_stars_sky_background = _SingleFileDownloadableDatasetLoader(
    'planet3d-matlab/stars.jpg',
)


@overload
def download_milkyway_sky_background(
    texture: Literal[False] = ..., load: Literal[True] = ...
) -> ImageData: ...
@overload
def download_milkyway_sky_background(
    texture: Literal[False] = ..., load: Literal[False] = ...
) -> str: ...
@overload
def download_milkyway_sky_background(
    texture: Literal[True], load: Literal[True] = ...
) -> Texture: ...
@overload
def download_milkyway_sky_background(texture: Literal[True], load: Literal[False]) -> str: ...
@_deprecate_positional_args
def download_milkyway_sky_background(
    texture: bool = False,  # noqa: FBT001, FBT002
    load: bool = True,  # noqa: FBT001, FBT002
) -> Texture | ImageData | str:
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
