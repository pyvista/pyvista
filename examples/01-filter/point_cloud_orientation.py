"""
.. _point_cloud_orientation_example:

Analyze the Orientation of a Point Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit geometric primitives and an oriented bounding box to a noisy point cloud
using PyVista's principal-axis tools.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

# sphinx_gallery_thumbnail_number = 2

# %%
# Load and tilt a real point cloud
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Start from the :func:`~pyvista.examples.downloads.download_horse_points` scan
# and rotate a subsampled copy away from the world axes so the orientation
# analysis has something to recover.

full_cloud = examples.download_horse_points()
rng = np.random.default_rng(seed=4)
sample_ids = rng.choice(full_cloud.n_points, size=3000, replace=False)
transform = pv.Transform().rotate_vector((1, 1, 0), 33).rotate_y(18)
cloud = pv.PolyData(full_cloud.points[sample_ids]).transform(transform, inplace=False)
cloud


# %%
# Fit a line and a plane
# ~~~~~~~~~~~~~~~~~~~~~~
# The line follows the dominant axis of the cloud while the plane spans the two
# strongest principal directions.

line, length, direction = pv.fit_line_to_points(
    cloud.points,
    init_direction='x',
    return_meta=True,
)
plane = pv.fit_plane_to_points(cloud.points, init_normal='z')

arrow = pv.Arrow(
    start=line.points[0],
    direction=direction,
    scale=length,
    tip_length=0.12,
    tip_radius=0.04,
    shaft_radius=0.015,
)

pl = pv.Plotter()
pl.add_points(
    cloud,
    color='black',
    point_size=4,
    render_points_as_spheres=True,
    opacity=0.4,
)
pl.add_mesh(plane, color='orange', opacity=0.25)
pl.add_mesh(arrow, color='tomato')
pl.show()


# %%
# Compare axis-aligned and oriented boxes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The axis-aligned box ignores the tilt of the data, while the oriented box
# follows the cloud's principal directions.

axis_aligned_box = cloud.bounding_box('outline', as_composite=False)
oriented_box = cloud.oriented_bounding_box(
    'outline',
    axis_2_direction='z',
    as_composite=False,
)

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_points(
    cloud,
    color='black',
    point_size=4,
    render_points_as_spheres=True,
    opacity=0.4,
)
pl.add_mesh(axis_aligned_box, color='tomato', line_width=4)
pl.subplot(0, 1)
pl.add_points(
    cloud,
    color='black',
    point_size=4,
    render_points_as_spheres=True,
    opacity=0.4,
)
pl.add_mesh(oriented_box, color='seagreen', line_width=4)
pl.link_views()
pl.show()


# %%
# Quantify the dominant directions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The principal-axis standard deviations show how strongly the cloud is stretched
# along each fitted axis.

_, std = pv.principal_axes(cloud.points, return_std=True)
std / std.sum()
# %%
# .. tags:: filter
