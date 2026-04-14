"""
.. _create_explicit_structured_grid_example:

Creating an Explicit Structured Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an explicit structured grid from NumPy arrays using
:class:`pyvista.ExplicitStructuredGrid`.

"""

from __future__ import annotations

import numpy as np
import pyvista as pv

ni, nj, nk = 4, 5, 6
si, sj, sk = 20, 10, 1

xcorn = np.arange(0, (ni + 1) * si, si)
xcorn = np.repeat(xcorn, 2)
xcorn = xcorn[1:-1]
xcorn = np.tile(xcorn, 4 * nj * nk)

ycorn = np.arange(0, (nj + 1) * sj, sj)
ycorn = np.repeat(ycorn, 2)
ycorn = ycorn[1:-1]
ycorn = np.tile(ycorn, (2 * ni, 2 * nk))
ycorn = np.transpose(ycorn)
ycorn = ycorn.flatten()

zcorn = np.arange(0, (nk + 1) * sk, sk)
zcorn = np.repeat(zcorn, 2)
zcorn = zcorn[1:-1]
zcorn = np.repeat(zcorn, (4 * ni * nj))

corners = np.stack((xcorn, ycorn, zcorn))
corners = corners.transpose()

dims = np.asarray((ni, nj, nk)) + 1
grid = pv.ExplicitStructuredGrid(dims, corners)
grid = grid.compute_connectivity()
grid.plot(show_edges=True)
# %%
# .. tags:: load
