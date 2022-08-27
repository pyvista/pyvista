"""
.. _strips_example:

Triangle Strips
~~~~~~~~~~~~~~~

This example shows how to build a simple :class:`pyvista.PolyData` using triangle strips.

Triangle strips are a more efficient way of storing the connectivity of adjacent triangles.
"""
import numpy as np

import pyvista

# set an array of vertices
vertices = np.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
)

# build the connectivity of the strips:
# first element is the number of points in the strip
# next three elements is the initial triangle
# the rest of the points is where the strip extends to
strips = np.hstack([10, 0, 1, 2, 3, 6, 7, 4, 5, 0, 1])

# build the mesh
mesh = pyvista.PolyData(vertices, strips=strips)

mesh.plot(show_edges=True)
