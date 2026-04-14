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
from pyvista import examples

# %%
# Load a cosmological point cloud
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :func:`~pyvista.examples.downloads.download_cloud_dark_matter` returns a
# sampled N-body simulation. Any point in the dataset makes a good query seed.

cloud = examples.download_cloud_dark_matter()
query = cloud.points[cloud.n_points // 2]

neighbor_ids = cloud.find_closest_point(query, n=40)
neighbors = pv.PolyData(cloud.points[neighbor_ids])

segments = np.vstack([np.vstack((query, point)) for point in neighbors.points])
connections = pv.line_segments_from_points(segments)

pl = pv.Plotter()
pl.add_points(
    cloud,
    color='lightgray',
    point_size=2,
    render_points_as_spheres=True,
    opacity=0.25,
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
