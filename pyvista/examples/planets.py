import numpy as np
import pyvista
from pyvista import examples


def load_sun():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_sun())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Sun's radius is 696000.0 km
    sphere = pyvista.Sphere(radius = 696000.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_sun_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["atmosphere"] = surface_tex
    return sphere


def load_moon():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_moon())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Moon's radius is 1738.0 km
    sphere = pyvista.Sphere(radius = 1738.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_moon_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    return sphere


def load_mercury():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_mercury())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Mercury's radius is 2439.0 km
    sphere = pyvista.Sphere(radius = 2439.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_mercury_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    return sphere


def load_venus():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_venus())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Venus's radius is 6052.0 km
    sphere = pyvista.Sphere(radius = 6052.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_venus_jpg(atmosphere=False)
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    atmosphere_jpg = examples.download_venus_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_mars():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_mars())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Mars's radius is 3397.2 km
    sphere = pyvista.Sphere(radius = 3397.2, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    surface_jpg = examples.download_mars_jpg()
    surface_tex = pyvista.read_texture(surface_jpg)
    sphere.textures["surface"] = surface_tex
    return sphere


def load_jupiter():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_jupiter())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Jupiter's radius is 71492.0 km
    sphere = pyvista.Sphere(radius = 71492.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_jupiter_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_saturn():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_saturn())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Saturn's radius is 60268.0 km
    sphere = pyvista.Sphere(radius = 60268.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_saturn_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_saturn_ring_alpha():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_saturn_ring_alpha())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Saturn's rings range from 7000.0 km to 80000.0 km from the surface of the planet
    inner = 60268.0 + 7000.0
    outer = 60268.0 + 80000.0
    disc = pyvista.Disc(inner=inner, outer=outer, c_res=50)
    disc.active_t_coords = np.zeros((disc.points.shape[0], 2))
    disc.active_t_coords[:, 0] = np.sqrt(disc.points[:, 0]**2 + disc.points[:, 1]**2) / outer
    disc.active_t_coords[:, 1] = 0.0
    atmosphere_png = examples.download_saturn_ring_alpha_png()
    atmosphere_tex = pyvista.read_texture(atmosphere_png)
    disc.textures["atmosphere"] = atmosphere_tex
    return disc


def load_uranus():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_uranus())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Uranus's radius is 25559.0 km
    sphere = pyvista.Sphere(radius = 25559.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_uranus_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere


def load_neptune():
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
    >>> pl.add_background_image(examples.download_stars_png())
    >>> pl.add_mesh(examples.load_neptune())
    >>> pl.show()

    """
    # https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
    # Neptune's radius is 24764.0 km
    sphere = pyvista.Sphere(radius = 24764.0, theta_resolution=300, phi_resolution=300)
    sphere.texture_map_to_sphere(inplace=True)
    atmosphere_jpg = examples.download_neptune_jpg()
    atmosphere_tex = pyvista.read_texture(atmosphere_jpg)
    sphere.textures["atmosphere"] = atmosphere_tex
    return sphere
