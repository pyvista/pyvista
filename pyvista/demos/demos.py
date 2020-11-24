"""Demos to show off the functionality of pyvista."""

import pyvista as pv
import numpy as np


def glyphs(grid_sz=3, **kwargs):
    """Plot several parametric supertoroids using VTK's glyph table functionality."""
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
    glyphs = mesh.glyph(geom=geoms, indices=values, scale=False,
                        factor=0.3, rng=(0, n - 1))

    # create plotter and add our glyphs with some nontrivial lighting
    plotter = pv.Plotter()
    plotter.add_mesh(glyphs, specular=1, specular_power=15,
                     smooth_shading=True, show_scalar_bar=False, **kwargs)
    plotter.show()
