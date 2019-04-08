"""
.. _ref_create_poly:

Create PolyData
~~~~~~~~~~~~~~~

Creating a PolyData object from NumPy arrays

"""

import numpy as np
import vtki

################################################################################
# A PolyData object can be created quickly from numpy arrays.  The vertex array
# contains the locations of the points of the mesh and the face array contains the
# number of points for each face and the indices of each of those faces.

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

surf = vtki.PolyData(vertices, faces)

# plot each face with a different color
surf.plot(scalars=np.arange(3), cpos=[-1,1,0.5])
