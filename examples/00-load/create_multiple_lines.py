"""
.. _create_multiple_lines_example:

Create Connected Lines from Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a polyline from ordered points with :func:`pyvista.MultipleLines`.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Create a winding polyline
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The points are connected in the order they are given.

t = np.linspace(0, 1, 100)
points = np.column_stack(
    (
        6 * t - 3,
        np.sin(2 * np.pi * t),
        0.5 * np.cos(3 * np.pi * t),
    ),
)

line = pv.MultipleLines(points)
line['height'] = points[:, 1]
line


# %%
# Plot the line as a tube
# ~~~~~~~~~~~~~~~~~~~~~~~
# Tubing makes it easier to follow the path through 3D space.

pl = pv.Plotter()
pl.add_mesh(line.tube(radius=0.08), scalars='height', cmap='viridis')
pl.show()
# %%
# .. tags:: load
