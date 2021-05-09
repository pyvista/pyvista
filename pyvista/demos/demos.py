"""Demos to show off the functionality of pyvista."""

from .logo import text_3d

import pyvista as pv
import numpy as np


def glyphs(grid_sz=3, **kwargs):
    """Create several parametric supertoroids using VTK's glyph table functionality."""
    n = 10
    values = np.arange(n)  # values for scalars to look up glyphs by

    # taken from:
    rng = np.random.default_rng()
    params = rng.uniform(0.5, 2, size=(n, 2))  # (n1, n2) parameters for the toroids

    geoms = [pv.ParametricSuperToroid(n1=n1, n2=n2) for n1, n2 in params]

    # get dataset where to put glyphs
    x, y, z = np.mgrid[:grid_sz, :grid_sz, :grid_sz]
    mesh = pv.StructuredGrid(x, y, z)

    # add random scalars
    rng_int = rng.integers(0, n, size=x.size)
    mesh.point_arrays['scalars'] = rng_int

    # construct the glyphs on top of the mesh; don't scale by scalars now
    return mesh.glyph(geom=geoms, indices=values, scale=False,
                      factor=0.3, rng=(0, n - 1))


def plot_glyphs(grid_sz=3, **kwargs):
    """Plot several parametric supertoroids using VTK's glyph table functionality."""
    # construct the glyphs on top of the mesh; don't scale by scalars now
    mesh = glyphs(grid_sz)

    kwargs.setdefault('specular', 1)
    kwargs.setdefault('specular_power', 15)
    kwargs.setdefault('smooth_shading', True)

    # create plotter and add our glyphs with some nontrivial lighting
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_scalar_bar=False, **kwargs)
    return plotter.show()


def orientation_cube():
    """Return a dictionary containing the meshes composing an orientation cube.

    Examples
    --------
    Load the cube mesh and plot it

    >>> import pyvista
    >>> from pyvista import demos
    >>> ocube = demos.orientation_cube()
    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_mesh(ocube['cube'], show_edges=True)
    >>> _ = pl.add_mesh(ocube['x_p'], color='blue')
    >>> _ = pl.add_mesh(ocube['x_n'], color='blue')
    >>> _ = pl.add_mesh(ocube['y_p'], color='green')
    >>> _ = pl.add_mesh(ocube['y_n'], color='green')
    >>> _ = pl.add_mesh(ocube['z_p'], color='red')
    >>> _ = pl.add_mesh(ocube['z_n'], color='red')
    >>> pl.show_axes()
    >>> pl.show()  # doctest:+SKIP

    """
    cube = pv.Cube()

    x_p = text_3d('X+', depth=0.2)
    x_p.points *= 0.45
    x_p.rotate_y(90)
    x_p.rotate_x(90)
    x_p.translate(-np.array(x_p.center))
    x_p.translate([0.5, 0, 0])
    # x_p.point_arrays['mesh'] = 1

    x_n = text_3d('X-', depth=0.2)
    x_n.points *= 0.45
    x_n.rotate_y(90)
    x_n.rotate_x(90)
    x_n.rotate_z(180)
    x_n.translate(-np.array(x_n.center))
    x_n.translate([-0.5, 0, 0])
    # x_n.point_arrays['mesh'] = 2

    y_p = text_3d('Y+', depth=0.2)
    y_p.points *= 0.45
    y_p.rotate_x(90)
    y_p.rotate_z(180)
    y_p.translate(-np.array(y_p.center))
    y_p.translate([0, 0.5, 0])
    # y_p.point_arrays['mesh'] = 3

    y_n = text_3d('Y-', depth=0.2)
    y_n.points *= 0.45
    y_n.rotate_x(90)
    y_n.translate(-np.array(y_n.center))
    y_n.translate([0, -0.5, 0])
    # y_n.point_arrays['mesh'] = 4

    z_p = text_3d('Z+', depth=0.2)
    z_p.points *= 0.45
    z_p.rotate_z(90)
    z_p.translate(-np.array(z_p.center))
    z_p.translate([0, 0, 0.5])
    # z_p.point_arrays['mesh'] = 5

    z_n = text_3d('Z-', depth=0.2)
    z_n.points *= 0.45
    z_n.rotate_x(180)
    z_n.translate(-np.array(z_n.center))
    z_n.translate([0, 0, -0.5])

    return {'cube': cube,
            'x_p': x_p,
            'x_n': x_n,
            'y_p': y_p,
            'y_n': y_n,
            'z_p': z_p,
            'z_n': z_n,
    }


def orientation_plotter():
    """Return a plotter containing the orientation cube."""
    ocube = orientation_cube()
    pl = pv.Plotter()
    pl.add_mesh(ocube['cube'], show_edges=True)
    pl.add_mesh(ocube['x_p'], color='blue')
    pl.add_mesh(ocube['x_n'], color='blue')
    pl.add_mesh(ocube['y_p'], color='green')
    pl.add_mesh(ocube['y_n'], color='green')
    pl.add_mesh(ocube['z_p'], color='red')
    pl.add_mesh(ocube['z_n'], color='red')
    pl.show_axes()
    return pl
