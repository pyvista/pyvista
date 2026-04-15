"""
.. _lorenz_attractor_example:

Plot a Lorenz Attractor
~~~~~~~~~~~~~~~~~~~~~~~

Integrate the Lorenz system and render the trajectory as a colored tube.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Integrate the Lorenz system
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A forward-Euler scheme is enough to trace the chaotic trajectory.

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
dt = 0.01
n_steps = 8000

points = np.empty((n_steps + 1, 3))
points[0] = (0.0, 1.0, 1.05)

for i in range(n_steps):
    x, y, z = points[i]
    points[i + 1] = (
        x + sigma * (y - x) * dt,
        y + (x * (rho - z) - y) * dt,
        z + (x * y - beta * z) * dt,
    )

trajectory = pv.lines_from_points(points)
trajectory['z'] = points[:, 2]


# %%
# Render the attractor
# ~~~~~~~~~~~~~~~~~~~~
# Tube the polyline so the trajectory has visible thickness in 3D.

pl = pv.Plotter()
pl.add_mesh(trajectory.tube(radius=0.12), scalars='z', cmap='plasma')
pl.show()


# %%
# Inspect the path length
# ~~~~~~~~~~~~~~~~~~~~~~~
# The integrated trajectory accumulates several hundred units of arc length.

trajectory.length
# %%
# .. tags:: plot
