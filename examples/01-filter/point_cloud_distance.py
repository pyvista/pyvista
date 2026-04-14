"""
.. _point_cloud_distance_example:

Measure Distance Between Point Clouds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Color a source point cloud by the distance to its nearest neighbors in a target
cloud.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Create two related point clouds
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Start from one cloud and perturb it so that every source point has a nearby
# counterpart in the target cloud.

rng = np.random.default_rng(seed=4)
source = pv.PolyData(rng.normal(size=(120, 3)) * (1.2, 0.8, 0.4))

target_points = source.points * (1.0, 1.15, 0.85)
target_points += (0.6, -0.2, 0.1)
target_points += rng.normal(scale=0.05, size=target_points.shape)
target = pv.PolyData(target_points)


# %%
# Compute nearest-neighbor distances
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use the target cloud as a nearest-neighbor lookup and color the source by the
# resulting distances.

closest_ids = np.array([target.find_closest_point(point) for point in source.points])
closest_points = target.points[closest_ids]
source['distance'] = np.linalg.norm(source.points - closest_points, axis=1)

sample_ids = np.linspace(0, source.n_points - 1, 12, dtype=int)
segments = np.vstack(
    [np.vstack((source.points[i], closest_points[i])) for i in sample_ids],
)
connectors = pv.line_segments_from_points(segments)

pl = pv.Plotter()
pl.add_points(
    target,
    color='lightgray',
    point_size=10,
    render_points_as_spheres=True,
    opacity=0.6,
)
pl.add_points(
    source,
    scalars='distance',
    cmap='viridis',
    point_size=12,
    render_points_as_spheres=True,
)
pl.add_mesh(connectors, color='black', opacity=0.35)
pl.show()


# %%
# Inspect the distance range
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# The cloud is close to the target, but not identical to it.

source['distance'].min(), source['distance'].max()
# %%
# .. tags:: filter
