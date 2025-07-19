"""
.. _create_polydata_strips_example:

Triangle Strips
~~~~~~~~~~~~~~~

This example shows how to build a simple :class:`pyvista.PolyData` using
triangle strips.

Triangle strips are a more efficient way of storing the connectivity of
adjacent triangles.

"""

# sphinx_gallery_start_ignore
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "pyvista",
# ]
# ///

from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import numpy as np

import pyvista as pv

# Create an array of points
points = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 3.0, 0.0],
        [0.0, 3.0, 0.0],
    ],
)

# %%
# Build the connectivity of the strips
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The first element is the number of points in the strip next three elements is the
# initial triangle the rest of the points is where the strip extends to.
strips = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])


# build the mesh
mesh = pv.PolyData(points, strips=strips)
mesh


# %%
# Plot the triangle strips
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the ``PolyData`` and include the point labels using
# :func:`add_point_labels() <pyvista.Plotter.add_point_labels>` so we can see how
# the PolyData is constructed using triangle strips.

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True)
pl.add_point_labels(mesh.points, range(mesh.n_points))
pl.camera_position = 'yx'
pl.camera.zoom(1.2)
pl.show()


# %%
# Convert strips to triangles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can convert strips to triangle faces using :func:`triangulate
# <pyvista.DataObjectFilters.triangulate>`.

trimesh = mesh.triangulate()
trimesh

# %%
# We can use this new :class:`pyvista.PolyData` to see how VTK represents
# triangle strips as individual faces.
#
# See how the faces array is much larger (~3x more) even for this basic example
# even despite representing the same data.
#
# .. note::
#    The faces array from VTK always contains padding (the number of points in
#    the face) for each face in the face array. This is the initial ``3`` in
#    the following face array.

faces = trimesh.faces.reshape(-1, 4)
faces


# %%
# Convert triangles to strips
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convert faces from a :class:`pyvista.PolyData` to strips using :func:`strip()
# <pyvista.PolyDataFilters.strip>`. Here, for demonstration purposes we convert the
# triangulated mesh back to a stripped mesh.

restripped = trimesh.strip()
restripped


# %%
# The output from the ``strip`` filter is, as expected, identical to the
# original ``mesh``.
restripped == mesh
# %%
# .. tags:: load
