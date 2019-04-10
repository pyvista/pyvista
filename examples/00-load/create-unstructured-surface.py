"""
.. _ref_create_unstructured:

Creating an Unstructured Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an irregular, unstructured grid from NumPy arrays
"""

import vtki
import vtk
import numpy as np

################################################################################
# An unstructured grid can be created directly from numpy arrays.
# This is useful when creating a grid from scratch or copying it from another
# format.  See `vtkUnstructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html>`_
# for available cell types and their descriptions.

# offset array.  Identifies the start of each cell in the cells array
offset = np.array([0, 9])

# Contains information on the points composing each cell.
# Each cell begins with the number of points in the cell and then the points
# composing the cell
cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])

# cell type array. Contains the cell type of each cell
cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON])

cell1 = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]])

cell2 = np.array([[0, 0, 2],
                  [1, 0, 2],
                  [1, 1, 2],
                  [0, 1, 2],
                  [0, 0, 3],
                  [1, 0, 3],
                  [1, 1, 3],
                  [0, 1, 3]])

# points of the cell array
points = np.vstack((cell1, cell2))

# create the unstructured grid directly from the numpy arrays
grid = vtki.UnstructuredGrid(offset, cells, cell_type, points)

# plot the grid
grid.plot(show_edges=True)
