"""
.. _point_cloud_neighbors_example:

Highlight Nearest Neighbors in a Point Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`pyvista.DataSet.find_closest_point` to inspect local neighborhoods in
a point cloud.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Generate a sample cloud and a query point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The query point does not need to be part of the cloud.

rng = np.random.default_rng(seed=5)
cloud = pv.PolyData(rng.normal(size=(350, 3)) * (1.0, 0.7, 0.4))
query = np.array([0.25, -0.1, 0.0])

neighbor_ids = cloud.find_closest_point(query, n=20)
neighbors = pv.PolyData(cloud.points[neighbor_ids])

segments = np.vstack([np.vstack((query, point)) for point in neighbors.points])
connections = pv.line_segments_from_points(segments)

pl = pv.Plotter()
pl.add_points(
    cloud,
    color='lightgray',
    point_size=10,
    render_points_as_spheres=True,
    opacity=0.35,
)
pl.add_points(
    neighbors,
    color='tomato',
    point_size=14,
    render_points_as_spheres=True,
)
pl.add_points(
    np.array([query]),
    color='gold',
    point_size=18,
    render_points_as_spheres=True,
)
pl.add_mesh(connections, color='black', opacity=0.35)
pl.show()


# %%
# Inspect the neighbor indices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The returned indices can be used to extract or analyze the local subset.

neighbor_ids
# %%
# .. tags:: plot
