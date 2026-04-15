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
from pyvista import examples

# %%
# Load a point cloud
# ~~~~~~~~~~~~~~~~~~
# The :func:`~pyvista.examples.downloads.download_horse_points` dataset is a
# dense scan of a horse statue. Subsample it to keep the hull geometry light.

full_cloud = examples.download_horse_points()
rng = np.random.default_rng(seed=2)
sample_ids = rng.choice(full_cloud.n_points, size=4000, replace=False)
cloud = pv.PolyData(full_cloud.points[sample_ids])
cloud


# %%
# Extract the outer hull
# ~~~~~~~~~~~~~~~~~~~~~~
# A Delaunay tetrahedralization followed by surface extraction returns the
# outer surface of the cloud.

hull = cloud.delaunay_3d(alpha=cloud.length).extract_surface(algorithm=None)

pl = pv.Plotter()
pl.add_points(
    cloud,
    color='black',
    point_size=6,
    render_points_as_spheres=True,
)
pl.add_mesh(hull, color='royalblue', opacity=0.4, show_edges=True)
pl.show()


# %%
# Inspect the wrapped surface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The result is a closed surface enclosing every input point.

hull
# %%
# .. tags:: filter
