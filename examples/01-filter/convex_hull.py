"""
.. _convex_hull_example:

Wrap a Point Cloud in a Convex Hull
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a convex hull from a point cloud using tetrahedralization and surface
extraction.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Generate a scattered point cloud
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use an anisotropic cloud so the resulting hull is easy to inspect.

rng = np.random.default_rng(seed=2)
points = rng.normal(size=(120, 3)) * (1.8, 0.7, 0.4)
points += rng.normal(scale=0.08, size=points.shape)

cloud = pv.PolyData(points)
cloud


# %%
# Extract the outer hull
# ~~~~~~~~~~~~~~~~~~~~~~
# A Delaunay tetrahedralization followed by surface extraction gives the outer
# wrap of the point cloud.

hull = cloud.delaunay_3d(alpha=1000).extract_surface(algorithm=None)

pl = pv.Plotter()
pl.add_points(
    cloud,
    color='black',
    point_size=12,
    render_points_as_spheres=True,
)
pl.add_mesh(hull, color='royalblue', opacity=0.35, show_edges=True)
pl.show()


# %%
# Inspect the wrapped surface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The result is a closed surface that encloses the full point set.

hull
# %%
# .. tags:: filter
