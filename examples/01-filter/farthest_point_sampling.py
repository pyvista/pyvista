"""
.. _farthest_point_sampling_example:

Farthest Point Sampling
~~~~~~~~~~~~~~~~~~~~~~~

Subsample a point cloud so the kept points stay spaced apart instead of
clumping in dense regions.

Farthest point sampling (FPS) starts from a random seed and repeatedly picks
the point that is farthest from the current sample set. Compared to a
uniform random draw, it gives a much more even covering of the input cloud.

References
----------
Y. Eldar et al., "The farthest point strategy for progressive image sampling,"
*Proc. 12th IAPR Int. Conf. on Pattern Recognition*, Vol. 2, 1994, pp. 93-97,
:doi:`10.1109/ICPR.1994.577129`.

"""

# sphinx_gallery_thumbnail_number = 1
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

# %%
# Load a point cloud
# ~~~~~~~~~~~~~~~~~~
# :func:`~pyvista.examples.downloads.download_horse_points` returns a scanned
# horse with uneven point density.

cloud = examples.download_horse_points()
cloud


# %%
# Implement farthest point sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each iteration tracks the distance from every input point to its closest
# already-sampled neighbor and picks the point with the largest such
# distance as the next sample.

rng = np.random.default_rng(seed=0)


def farthest_point_sampling(points, n_samples):
    """Return indices of ``n_samples`` farthest-spaced points."""
    sampled = np.empty(n_samples, dtype=int)
    sampled[0] = rng.integers(points.shape[0])
    distances = np.full(points.shape[0], np.inf)
    for i in range(1, n_samples):
        last = points[sampled[i - 1]]
        distances = np.minimum(distances, np.linalg.norm(points - last, axis=1))
        sampled[i] = np.argmax(distances)
    return sampled


n_samples = 400
fps_ids = farthest_point_sampling(cloud.points, n_samples)
random_ids = rng.choice(cloud.n_points, size=n_samples, replace=False)

fps_cloud = pv.PolyData(cloud.points[fps_ids])
random_cloud = pv.PolyData(cloud.points[random_ids])


# %%
# Compare with a uniform random draw
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The random subsample (left, red) leaves visible gaps and clumps. The
# farthest-point subsample (right, blue) lays the points down in a more
# uniform pattern over the surface.

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_points(
    cloud,
    color='lightgray',
    point_size=2,
    render_points_as_spheres=True,
    opacity=0.4,
)
pl.add_points(
    random_cloud,
    color='tomato',
    point_size=10,
    render_points_as_spheres=True,
)

pl.subplot(0, 1)
pl.add_points(
    cloud,
    color='lightgray',
    point_size=2,
    render_points_as_spheres=True,
    opacity=0.4,
)
pl.add_points(
    fps_cloud,
    color='royalblue',
    point_size=10,
    render_points_as_spheres=True,
)
pl.link_views()
pl.show()


# %%
# Quantify the coverage gap
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# For each input point, find the distance to its closest sample and report
# the worst case. Lower is better.

random_max_gap = max(
    np.linalg.norm(p - random_cloud.points[random_cloud.find_closest_point(p)])
    for p in cloud.points
)
fps_max_gap = max(
    np.linalg.norm(p - fps_cloud.points[fps_cloud.find_closest_point(p)])
    for p in cloud.points
)
random_max_gap, fps_max_gap
# %%
# .. tags:: filter
