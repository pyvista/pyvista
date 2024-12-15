"""
.. _maximum_intensity_projection_example:

Maximum Intensity Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to use the maximum intensity projection
to reset the z screen coordinates so that vertices with higher scalar
values are closer to the screen.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# Download Knee dataset
vol = examples.download_knee_full()

###############################################################################
# Get a Sample of vertices from the dataset
sample_n = 100000
rng = np.random.default_rng()
randix = rng.integers(0, vol.n_points, sample_n)
pts = vol.extract_points(randix, adjacent_cells=False, include_cells=False)

###############################################################################
pl = pv.Plotter(shape=(1, 3))
pl.enable_parallel_projection()

# Define actors parameters
display = dict(
    cmap='jet',
    clim=[0, 100],
    opacity=1,
)

###############################################################################
# Plot normally for comparison
pl.subplot(0, 0)
standard_actor = pl.add_mesh(pts, **display)
pl.add_title('Normal Projection\n(point cloud)', font_size=12)

###############################################################################
# Use maximum intensity projection
pl.subplot(0, 1)
mip_actor = pl.add_mesh(pts, **display)
mip_actor.enable_maximum_intensity_projection()
pl.add_title('Maximum Intensity Projection\n(point cloud)', font_size=12)


###############################################################################
# Use maximum intensity projection with spherical point rendering with
# ``render_points_as_spheres``.
pl.subplot(0, 2)
mip_sphere_actor = pl.add_mesh(pts, render_points_as_spheres=True, **display)
mip_sphere_actor.enable_maximum_intensity_projection()
pl.add_title('Maximum Intensity Projection\n(spherical point cloud)', font_size=12)

###############################################################################
pl.link_views()
pl.show()
