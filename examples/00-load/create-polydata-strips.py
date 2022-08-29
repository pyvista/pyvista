"""
.. _strips_example:

Triangle Strips
~~~~~~~~~~~~~~~

This example shows how to build a simple :class:`pyvista.PolyData` using
triangle strips.

Triangle strips are a more efficient way of storing the connectivity of
adjacent triangles.

"""
import numpy as np

import pyvista as pv

# set an array of vertices
vertices = np.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
    dtype=float,
)

###############################################################################
# Build the connectivity of the strips
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The first element is the number of points in the strip next three elements is the
# initial triangle the rest of the points is where the strip extends to.
strips = np.hstack([10, 0, 1, 2, 3, 6, 7, 4, 5, 0, 1])

# build the mesh
mesh = pv.PolyData(vertices, strips=strips)
mesh


###############################################################################
# Plot the triangle strips
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the ``PolyData`` and include the point labels using
# :func:`add_point_labels <pyvista.Plotter.add_point_labels>` so we can see how
# the PolyData is constructed using triangle strips.

# good viewing angle obtained from pl.show(return_cpos=True)
cpos = [(-1.45, -0.67, 2.95), (0.5, 0.5, 0.5), (0.692, 0.257, 0.674)]

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True)
pl.add_point_labels(mesh.points, range(mesh.n_points))
pl.camera_position = cpos
pl.show()


###############################################################################
# Convert strips to triangles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can convert strips to triangle faces using :func:`triangulate
# <pyvista.Dataset.triangulate>`.

trimesh = mesh.triangulate()
trimesh

###############################################################################
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
