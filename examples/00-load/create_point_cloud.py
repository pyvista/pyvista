"""
.. _create_point_cloud_example:

Create Point Cloud
~~~~~~~~~~~~~~~~~~

Create a :class:`pyvista.PolyData` object from a point cloud of vertices and
scalar arrays for those points.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples

# %%
# Point clouds are generally constructed in the :class:`pyvista.PolyData` class
# and can easily have scalar/vector data arrays associated with the point
# cloud. In this example, we'll work a bit backwards using a point cloud that
# that is available from our ``examples`` module. This however is no different
# than creating a PyVista mesh with your own NumPy arrays of vertice locations.


# Seed the random number generator for generating data
rng = np.random.default_rng(seed=0)


# Define some helpers - ignore these and use your own data.
def generate_points(subset=0.02):
    """Make a 3D NumPy array of points (n_points by 3)."""
    dataset = examples.download_lidar()
    ids = rng.integers(
        low=0,
        high=dataset.n_points - 1,
        size=int(dataset.n_points * subset),
    )
    return dataset.points[ids]


points = generate_points()
# Print first 5 rows to prove its a numpy array (n_points by 3)
# Columns are (X Y Z)
points[0:5, :]

# %%
# Now that you have a NumPy array of points/vertices either from our sample
# data or your own project, creating a PyVista mesh of those points is simply:
point_cloud = pv.PolyData(points)
point_cloud

# %%
# And we can even do a sanity check
np.allclose(points, point_cloud.points)

# %%
# And now that we have a PyVista mesh, we can plot it. Note that we add an
# option to use eye dome lighting - this is a shading technique to improve
# depth perception with point clouds (learn more in :ref:`edl_example`).

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore
point_cloud.plot(eye_dome_lighting=True)

# %%
# Now what if you have data attributes (scalar/vector arrays) that you'd like
# to associate with every node of your mesh? You can easily add NumPy data
# arrays that have a length equal to the number of points in the mesh along the
# first axis. For example, lets add a few arrays to this new ``point_cloud``
# mesh.
#
# Make an array of scalar values with the same length as the points array.
# Each element in this array will correspond to points at the same index:

# Make data array using z-component of points array
data = points[:, -1]

# Add that data to the mesh with the name "uniform dist"
point_cloud['elevation'] = data

# %%
# And now we can plot the point cloud with that random data. PyVista is smart
# enough to plot the scalar array you added by default. Note that this time,
# we specify to render every point as its own sphere.
point_cloud.plot(render_points_as_spheres=True)

# %%
# That data is kind of boring, right? You can also add data arrays with
# more than one scalar value - perhaps a vector with three elements? Let's
# make a little function that will compute vectors for every node in the point
# cloud and add those vectors to the mesh.
#
# This time, we're going to create a totally new, random point cloud.

# Create random XYZ points
points = rng.random((100, 3))
# Make PolyData
point_cloud = pv.PolyData(points)


def compute_vectors(mesh):
    origin = mesh.center
    vectors = mesh.points - origin
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


vectors = compute_vectors(point_cloud)
vectors[0:5, :]

# %%

point_cloud['vectors'] = vectors

# %%
# Now we can make arrows using those vectors using the glyph filter
# (see :ref:`glyph_example` for more details).

arrows = point_cloud.glyph(
    orient='vectors',
    scale=False,
    factor=0.15,
)

# Display the arrows
pl = pv.Plotter()
pl.add_mesh(point_cloud, color='maroon', point_size=10.0, render_points_as_spheres=True)
pl.add_mesh(arrows, color='lightblue')
# pl.add_point_labels([point_cloud.center,], ['Center',],
#                          point_color='yellow', point_size=20)
pl.show_grid()
pl.show()
# %%
# .. tags:: load
