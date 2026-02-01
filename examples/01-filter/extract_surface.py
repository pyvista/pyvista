"""
.. _extract_surface_example:

Extract Surface
~~~~~~~~~~~~~~~

You can extract the surface of nearly any object within ``pyvista``
using the :meth:`~pyvista.DataObjectFilters.extract_surface` filter.
"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import CellType

# %%
# Create a quadratic cell and extract its surface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we create a single quadratic hexahedral cell and then extract its surface
# to demonstrate how to extract the surface of an UnstructuredGrid.


lin_pts = np.array(
    [
        [-1, -1, -1],  # point 0
        [1, -1, -1],  # point 1
        [1, 1, -1],  # point 2
        [-1, 1, -1],  # point 3
        [-1, -1, 1],  # point 4
        [1, -1, 1],  # point 5
        [1, 1, 1],  # point 6
        [-1, 1, 1],  # point 7
    ],
    np.double,
)

# these are the "midside" points of a quad cell.  See the definition of a
# :vtk:`vtkQuadraticHexahedron`.

quad_pts = np.array(
    [
        (lin_pts[1] + lin_pts[0]) / 2,  # between point 0 and 1
        (lin_pts[1] + lin_pts[2]) / 2,  # between point 1 and 2
        (lin_pts[2] + lin_pts[3]) / 2,  # and so on...
        (lin_pts[3] + lin_pts[0]) / 2,
        (lin_pts[4] + lin_pts[5]) / 2,
        (lin_pts[5] + lin_pts[6]) / 2,
        (lin_pts[6] + lin_pts[7]) / 2,
        (lin_pts[7] + lin_pts[4]) / 2,
        (lin_pts[0] + lin_pts[4]) / 2,
        (lin_pts[1] + lin_pts[5]) / 2,
        (lin_pts[2] + lin_pts[6]) / 2,
        (lin_pts[3] + lin_pts[7]) / 2,
    ],
)

# introduce a minor variation to the location of the mid-side points
# seed the random numbers for reproducibility
rng = np.random.default_rng(seed=0)
quad_pts += rng.random(quad_pts.shape) * 0.3
pts = np.vstack((lin_pts, quad_pts))

# create the grid
cells = np.hstack((20, np.arange(20))).astype(np.int64, copy=False)
celltypes = np.array([CellType.QUADRATIC_HEXAHEDRON])
grid = pv.UnstructuredGrid(cells, celltypes, pts)

# finally, extract the surface and plot it
surf = grid.extract_surface()
surf.plot(show_scalar_bar=False)


# %%
# Nonlinear Surface Subdivision
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Should your UnstructuredGrid contain quadratic cells, you can
# generate a smooth surface based on the position of the
# "mid-edge" nodes.  This allows the plotting of cells
# containing curvature.  For additional reference, please see:
# https://prod.sandia.gov/techlib-noauth/access-control.cgi/2004/041617.pdf

surf_subdivided = grid.extract_surface(nonlinear_subdivision=5)
surf_subdivided.plot(show_scalar_bar=False)
# %%
# .. tags:: filter
