"""
.. _dimension_line_example:
Dimension Line
~~~~~~~~~~~~~~

Create a dimension line along 2d structured mesh.

"""

import numpy as np

import pyvista as pv

ni, nj = 4, 5
si, sj = 20, 10

xcorn = np.arange(0, (ni + 1) * si, si)
xcorn = np.repeat(xcorn, 2)
xcorn = xcorn[1:-1]
xcorn = np.tile(xcorn, 4 * nj)

ycorn = np.arange(0, (nj + 1) * sj, sj)
ycorn = np.repeat(ycorn, 2)
ycorn = ycorn[1:-1]
ycorn = np.tile(ycorn, (2 * ni, 2))
ycorn = np.transpose(ycorn)
ycorn = ycorn.flatten()

corners = np.stack((xcorn, ycorn))
corners = corners.transpose()

dims = np.asarray((ni, nj)) + 1
grid = pv.ExplicitStructuredGrid(dims, corners)
grid = grid.compute_connectivity()
grid.plot(show_edges=True)
