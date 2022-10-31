"""
.. _polyhedron_example:

Combining a polyhedron with other cells.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to build a simple :class:`pyvista.UnstructuredGrid` using
polyhedra, which have a concrete way of being built. We will be using VTK
types to determine which type of cells we are building.

"""

import numpy as np

import pyvista as pv

###############################################################################
# Node arrays of the cells
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will mix several cells in one grid for this example, so we determine several
# points for each cell.

quad_points = [
    [0.0, 0.0, 0.0],  # 0
    [0.0, 0.01, 0.0],  # 1
    [0.01, 0.01, 0.0],  # 2
    [0.01, 0.0, 0.0],  # 3
]
polygon_points = [
    [0.02, 0.0, 0.0],  # 4
    [0.02, 0.01, 0.0],  # 5
    [0.03, 0.01, 0.0],  # 6
    [0.035, 0.005, 0.0],  # 7
    [0.03, 0.0, 0.0],  # 8
]
hexa_points = [
    [0.0, 0.0, 0.02],  # 9
    [0.0, 0.01, 0.02],  # 10
    [0.01, 0.01, 0.02],  # 11
    [0.01, 0.0, 0.02],  # 12
    [0.0, 0.0, 0.03],  # 13
    [0.0, 0.01, 0.03],  # 14
    [0.01, 0.01, 0.03],  # 15
    [0.01, 0.0, 0.03],  # 16
]
polyhedron_points = [
    [0.02, 0.0, 0.02],  # 17
    [0.02, 0.01, 0.02],  # 18
    [0.03, 0.01, 0.02],  # 19
    [0.035, 0.005, 0.02],  # 20
    [0.03, 0.0, 0.02],  # 21
    [0.02, 0.0, 0.03],  # 22
    [0.02, 0.01, 0.03],  # 23
    [0.03, 0.01, 0.03],  # 24
    [0.035, 0.005, 0.03],  # 25
    [0.03, 0.0, 0.03],  # 26
]
points = np.array(quad_points + polygon_points + hexa_points + polyhedron_points)


###############################################################################
# Connectivity arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We set which points each of the cells needs. The first element in each array 
# is the number of elements the cell will have. E.g., quad array is composed of 
# 0, 1, 2 and 3 points, so it has 4 elements. This is needed because the connectivity
# array that will contain all cells needs to have only one dimension, so to be able to
# know which points belong to which cell, we set that the following N points in the array
# belong to a single cell, the next N points belong to another cell, etc.
# Note that for polygons, 
# the order of the points is important.

quad = np.array([4, 0, 1, 2, 3])
polygon = np.array([5, 4, 5, 6, 7, 8])
hexa = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16])

###############################################################################
# Polyhedron connectivity array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For polyhedrons, we need to set the faces with the following format:
# [NElements, NFaces, Face1NPoints, Face1Point1, Face1Point2..., Face1PointN, Face2NPoints,...]
# - NElements refers to the total number of elements in the array needed to describe the polyhedron.
# - NFaces is the number of faces the polyhedron will have.
# - Face1Npoints is the number of points the first face will have
# - Face1Point1..Face1PointN are each of the points that describe face1
# In `polyhedron_connectivity`, the first element is `NFaces`. `NElements` is added in `polyhedron`.

polyhedron_connectivity = [
    7,
    5,
    17,
    18,
    19,
    20,
    21,
    4,
    17,
    18,
    23,
    22,
    4,
    17,
    21,
    26,
    22,
    4,
    21,
    26,
    25,
    20,
    4,
    20,
    25,
    24,
    19,
    4,
    19,
    24,
    23,
    18,
    5,
    22,
    23,
    24,
    25,
    26,
]
polyhedron = np.array([len(polyhedron_connectivity)] + polyhedron_connectivity)


###############################################################################
# Cells array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we build the input cells array for `UnstructuredGrid`. We join all cells in 
# a one dimensional array. Internally, the `NElements` previously described is used
# to know which nodes belong to which cells.


cells = np.hstack((quad, polygon, hexa, polyhedron))


###############################################################################
# Cell types
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We need to determine the cell types for each of the cells we define in the cells array.
# The number of elements in this array must coincide with the number of cells in the 
# connectivity array.

cell_type = np.array(
    [pv.CellType.QUAD, pv.CellType.POLYGON, pv.CellType.HEXAHEDRON, pv.CellType.POLYHEDRON]
)


###############################################################################
# Create the grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To create the grid, we use the cells array we built, the cell types, as well
# as the points that describe the faces.

grid = pv.UnstructuredGrid(cells, cell_type, points)
print(grid.cell_type(0))

###############################################################################
# Plot the mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, we can plot the grid we've created

plt = pv.Plotter()
plt.show_axes()
plt.add_mesh(grid)
plt.show()
