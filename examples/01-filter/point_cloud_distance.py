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
from pyvista import examples

# %%
# Create two related point clouds
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Start from a scanned horse point cloud, subsample it, then displace a
# perturbed copy so each source point has a nearby counterpart in the target.

full_cloud = examples.download_horse_points()
rng = np.random.default_rng(seed=4)
sample_ids = rng.choice(full_cloud.n_points, size=1500, replace=False)
source = pv.PolyData(full_cloud.points[sample_ids])

offset = np.array([source.length * 0.05, 0.0, 0.0])
warp = rng.normal(scale=source.length * 0.01, size=source.points.shape)
target = pv.PolyData(source.points + offset + warp)


# %%
# Compute nearest-neighbor distances
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Query the target cloud for each source point and color the source by the
# resulting distances.

closest_ids = np.array([target.find_closest_point(point) for point in source.points])
closest_points = target.points[closest_ids]
source['distance'] = np.linalg.norm(source.points - closest_points, axis=1)

connector_ids = np.linspace(0, source.n_points - 1, 20, dtype=int)
segments = np.vstack(
    [np.vstack((source.points[i], closest_points[i])) for i in connector_ids],
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
# The clouds are close but not identical.

source['distance'].min(), source['distance'].max()
# %%
# .. tags:: filter
