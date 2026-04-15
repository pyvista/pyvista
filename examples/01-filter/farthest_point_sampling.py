"""
.. _farthest_point_sampling_example:

Farthest Point Sampling
~~~~~~~~~~~~~~~~~~~~~~~

Subsample a point cloud so that the kept points are spread as evenly as
possible across the input geometry.

Farthest point sampling (FPS) starts from a seed point and repeatedly picks
the point that is farthest from the current sample set. The result is a
near-uniform covering of the cloud — much more representative than a uniform
random draw, which over-samples dense regions and under-samples sparse ones.

References
----------
Y. Eldar et al., "The farthest point strategy for progressive image sampling,"
*Proc. 12th IAPR Int. Conf. on Pattern Recognition*, Vol. 2, 1994, pp. 93-97,
:doi:`10.1109/ICPR.1994.577129`.

"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

# %%
# Load a point cloud
# ~~~~~~~~~~~~~~~~~~
# :func:`~pyvista.examples.downloads.download_horse_points` returns a scanned
# horse with non-uniform point density — a good stress test for any
# subsampling strategy.

cloud = examples.download_horse_points()
cloud

# %%
# Implement farthest point sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each iteration tracks the distance from every input point to its closest
# already-sampled neighbor and picks the point with the largest such distance
# as the next sample.

rng = np.random.default_rng(seed=0)


def farthest_point_sampling(points, k):
    sampled = np.empty(k, dtype=int)
    sampled[0] = rng.integers(points.shape[0])
    distances = np.full(points.shape[0], np.inf)
    for i in range(1, k):
        last = points[sampled[i - 1]]
        distances = np.minimum(distances, np.linalg.norm(points - last, axis=1))
        sampled[i] = np.argmax(distances)
    return sampled


k = 400
fps_ids = farthest_point_sampling(cloud.points, k)
fps_cloud = pv.PolyData(cloud.points[fps_ids])

# %%
# Compare with a uniform random draw
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Drawing the same number of points uniformly at random produces visibly
# clumpy coverage, while the FPS sample lays the points down on an
# approximately uniform lattice over the surface.

random_ids = rng.choice(cloud.n_points, size=k, replace=False)
random_cloud = pv.PolyData(cloud.points[random_ids])

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_text('Random subsample', font_size=12)
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
pl.add_text('Farthest point sampling', font_size=12)
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
# A simple proxy for "how evenly does each subsample cover the original
# cloud" is the maximum nearest-neighbor distance from any input point to its
# closest sampled neighbor. Lower is better — and FPS wins by construction.


def max_coverage_distance(samples):
    sampled_polydata = pv.PolyData(samples)
    closest = np.array(
        [sampled_polydata.find_closest_point(p) for p in cloud.points],
    )
    return float(np.linalg.norm(cloud.points - samples[closest], axis=1).max())


(
    max_coverage_distance(cloud.points[random_ids]),
    max_coverage_distance(
        cloud.points[fps_ids],
    ),
)
# %%
# .. tags:: filter
