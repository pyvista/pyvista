"""Downloadable datasets of 3D Celestial Bodies.

Examples
--------
>>> from pyvista import examples
>>> mesh = examples.load_moon()
>>> mesh.plot()

"""
import numpy as np

import pyvista
from pyvista import examples


def load_sun(*args, **kwargs):
    """Load a sun source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_sun())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_sun_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["atmosphere"] = surface_tex
    return sphere


def load_moon(*args, **kwargs):
    """Load a moon source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_moon())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_moon_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    return sphere


def load_mercury(*args, **kwargs):
    """Load a mercury source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_mercury())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_mercury_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    return sphere


def load_venus(*args, **kwargs):
    """Load a venus source.

    Returns
    -------
    pyvista.PolyData
        Venus dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_venus())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_venus_jpg(atmosphere=False)
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    atmosphere_jpg = examples.download_venus_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_mars(*args, **kwargs):
    """Load a mars source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_mars())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_mars_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    return sphere


def load_jupiter(*args, **kwargs):
    """Load a jupiter source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_jupiter())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_jupiter_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_saturn(*args, **kwargs):
    """Load a saturn source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_saturn())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_saturn_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_saturn_ring_alpha(*args, **kwargs):
    """Load a saturn_ring_alpha source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_saturn_ring_alpha())
    >>> pl.show()

    """
    disc = pyvista.Disc(*args, **kwargs)
    disc.active_t_coords = np.zeros((disc.points.shape[0], 2))
    radius = np.sqrt(disc.points[:, 0] ** 2 + disc.points[:, 1] ** 2)
    disc.active_t_coords[:, 0] = radius / np.max(radius)
    disc.active_t_coords[:, 1] = 0.0
    atmosphere_png = examples.download_saturn_ring_alpha_png()
    atmosphere_tex = pyvista.read_texture(atmosphere_png)
    disc.textures["atmosphere"] = atmosphere_tex
    return disc


def load_uranus(*args, **kwargs):
    """Load a uranus source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_uranus())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_uranus_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_neptune(*args, **kwargs):
    """Load a neptune source.

    Returns
    -------
    pyvista.PolyData
        Sun dataset with texture.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> pl = pyvista.Plotter()
    >>> pl.add_background_image(examples.download_stars_jpg())
    >>> pl.add_mesh(examples.load_neptune())
    >>> pl.show()

    """
    sphere = pyvista.Sphere(*args, **kwargs)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_neptune_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere
