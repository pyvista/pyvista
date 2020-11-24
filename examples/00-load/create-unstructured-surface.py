"""
.. _ref_create_unstructured:

Creating an Unstructured Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an irregular, unstructured grid from NumPy arrays.
"""

import pyvista as pv
import vtk
import numpy as np

###############################################################################
# An unstructured grid can be created directly from NumPy arrays.
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

# in this example, each cell uses separate points
cell1 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

cell2 = np.array(
    [
        [0, 0, 2],
        [1, 0, 2],
        [1, 1, 2],
        [0, 1, 2],
        [0, 0, 3],
        [1, 0, 3],
        [1, 1, 3],
        [0, 1, 3],
    ]
)

# points of the cell array
points = np.vstack((cell1, cell2))

# create the unstructured grid directly from the numpy arrays
# The offset is optional and will be either calculated if not given (VTK version < 9),
# or is not necessary anymore (VTK version >= 9)
grid = pv.UnstructuredGrid(offset, cells, cell_type, points)

# For cells of fixed sizes (like the mentioned Hexahedra), it is also possible to use the
# simplified dictionary interface. This automatically calculates the cell array with types
# and offsets. Note that for mixing with additional cell types, just the appropriate key needs to be
# added to the dictionary.
cells_hex = np.arange(16).reshape([2, 8]) # = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
grid = pv.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells_hex}, points)

# plot the grid (and suppress the camera position output)
_ = grid.plot(show_edges=True)

###############################################################################
# UnstructuredGrid with Shared Points
# -----------------------------------
#
# The next example again creates an unstructured grid containing
# hexahedral cells, but using common points between the cells.

# these points will all be shared between the cells
points = np.array([[0. , 0. , 0. ],
                   [1. , 0. , 0. ],
                   [0.5, 0. , 0. ],
                   [1. , 1. , 0. ],
                   [1. , 0.5, 0. ],
                   [0. , 1. , 0. ],
                   [0.5, 1. , 0. ],
                   [0. , 0.5, 0. ],
                   [0.5, 0.5, 0. ],
                   [1. , 0. , 0.5],
                   [1. , 0. , 1. ],
                   [0. , 0. , 0.5],
                   [0. , 0. , 1. ],
                   [0.5, 0. , 0.5],
                   [0.5, 0. , 1. ],
                   [1. , 1. , 0.5],
                   [1. , 1. , 1. ],
                   [1. , 0.5, 0.5],
                   [1. , 0.5, 1. ],
                   [0. , 1. , 0.5],
                   [0. , 1. , 1. ],
                   [0.5, 1. , 0.5],
                   [0.5, 1. , 1. ],
                   [0. , 0.5, 0.5],
                   [0. , 0.5, 1. ],
                   [0.5, 0.5, 0.5],
                   [0.5, 0.5, 1. ]])


# Each cell in the cell array needs to include the size of the cell
# and the points belonging to the cell.  In this example, there are 8
# hexahedral cells that have common points between them.
cells = np.array([[ 8,  0,  2,  8,  7, 11, 13, 25, 23],
                  [ 8,  2,  1,  4,  8, 13,  9, 17, 25],
                  [ 8,  7,  8,  6,  5, 23, 25, 21, 19],
                  [ 8,  8,  4,  3,  6, 25, 17, 15, 21],
                  [ 8, 11, 13, 25, 23, 12, 14, 26, 24],
                  [ 8, 13,  9, 17, 25, 14, 10, 18, 26],
                  [ 8, 23, 25, 21, 19, 24, 26, 22, 20],
                  [ 8, 25, 17, 15, 21, 26, 18, 16, 22]]).ravel()

# each cell is a VTK_HEXAHEDRON
celltypes = np.empty(8, dtype=np.uint8)
celltypes[:] = vtk.VTK_HEXAHEDRON

# the offset array points to the start of each cell (via flat indexing)
offset = np.array([ 0, 9, 18, 27, 36, 45, 54, 63])

# Effectively, when visualizing a VTK unstructured grid, it will
# sequentially access the cell array by first looking at each index of
# cell array (based on the offset array), and then read the number of
# points based on the first value of the cell.  In this case, the
# VTK_HEXAHEDRON is described by 8 points.

# for example, the 5th cell would be accessed by vtk with:
start_of_cell = offset[4]
n_points_in_cell = cells[start_of_cell]
indices_in_cell = cells[start_of_cell + 1: start_of_cell + n_points_in_cell + 1]
print(indices_in_cell)


###############################################################################
# Finally, create the unstructured grid and plot it

# if you are using VTK 9.0 or newer, you do not need to input the offset array:
# grid = pv.UnstructuredGrid(cells, celltypes, points)

# if you are not using VTK 9.0 or newer, you must use the offset array
grid = pv.UnstructuredGrid(offset, cells, celltypes, points)

# Alternate versions:
grid = pv.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells.reshape([-1, 9])[:, 1:]}, points)
grid = pv.UnstructuredGrid({vtk.VTK_HEXAHEDRON: np.delete(cells, np.arange(0, cells.size, 9))}, points)

# plot the grid (and suppress the camera position output)
_ = grid.plot(show_edges=True)
