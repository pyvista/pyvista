"""
.. _gyroid_example:

Plot a Gyroid Surface
---------------------

Contour an implicit gyroid field into a periodic surface.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Sample the implicit field
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The gyroid is the zero level set of
# ``sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)``.

n = 50
x, y, z = np.mgrid[
    -np.pi : np.pi : complex(0, n),
    -np.pi : np.pi : complex(0, n),
    -np.pi : np.pi : complex(0, n),
]
values = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)

grid = pv.ImageData(dimensions=values.shape)
grid.origin = (-np.pi, -np.pi, -np.pi)
grid.spacing = (2 * np.pi / (n - 1),) * 3
grid['gyroid'] = values.ravel(order='F')
grid


# %%
# Extract the zero isosurface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The contour weaves through the periodic sample volume.

surface = grid.contour([0.0], scalars='gyroid')

pl = pv.Plotter()
pl.add_mesh(surface, color='royalblue', smooth_shading=True)
pl.show()
# %%
# .. tags:: advanced
