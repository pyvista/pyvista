"""
.. _ref_create_poly:

Create PolyData
~~~~~~~~~~~~~~~

Creating a PolyData (triangulated surface) object from NumPy arrays of the
vertices and faces.

"""

import numpy as np
import pyvista as pv

###############################################################################
# A PolyData object can be created quickly from numpy arrays.  The vertex array
# contains the locations of the points in the mesh and the face array contains
# the number of points of each face and the indices of the vertices which comprise that face.

# mesh points
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0],
                     [0.5, 0.5, -1]])

# mesh faces
faces = np.hstack([[4, 0, 1, 2, 3],  # square
                   [3, 0, 1, 4],     # triangle
                   [3, 1, 2, 4]])    # triangle

surf = pv.PolyData(vertices, faces)

# plot each face with a different color
surf.plot(scalars=np.arange(3), cpos=[-1, 1, 0.5])
