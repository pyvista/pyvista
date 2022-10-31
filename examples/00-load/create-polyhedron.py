"""
.. _polyhedron_example:

Combining a polyhedron with other figures.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to build a simple :class:`pyvista.UnstructuredGrid` using
polyhedra, which have a concrete way of being built. We will be using VTK
types to determine which type of figures we are building.

"""

import numpy as np

import pyvista as pv

###############################################################################
# Node arrays of the figures
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will mix several figures in one grid for this example, so we determine several
# nodes for each figure.

quad_nodes = [
    [0.0, 0.0, 0.0],  # 0
    [0.0, 0.01, 0.0],  # 1
    [0.01, 0.01, 0.0],  # 2
    [0.01, 0.0, 0.0],  # 3
]
polygon_nodes = [
    [0.02, 0.0, 0.0],  # 4
    [0.02, 0.01, 0.0],  # 5
    [0.03, 0.01, 0.0],  # 6
    [0.035, 0.005, 0.0],  # 7
    [0.03, 0.0, 0.0],  # 8
]
hexa_nodes = [
    [0.0, 0.0, 0.02],  # 9
    [0.0, 0.01, 0.02],  # 10
    [0.01, 0.01, 0.02],  # 11
    [0.01, 0.0, 0.02],  # 12
    [0.0, 0.0, 0.03],  # 13
    [0.0, 0.01, 0.03],  # 14
    [0.01, 0.01, 0.03],  # 15
    [0.01, 0.0, 0.03],  # 16
]
polyhedron_nodes = [
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
nodes = np.asarray(quad_nodes + polygon_nodes + hexa_nodes + polyhedron_nodes)


###############################################################################
# Connectivity arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We set which nodes each of the figures needs. Note that for polygons,
# the order of the nodes is important.

quad = np.asarray([4, 0, 1, 2, 3])
polygon = np.asarray([5, 4, 5, 6, 7, 8])
hexa = np.asarray([8, 9, 10, 11, 12, 13, 14, 15, 16])

###############################################################################
# Polyhedron connectivity array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For polyhedrons, we need to set the faces with the following format:
# [NElements, NFaces, Face1NPoints, Face1Point1, Face1Point2..., Face1PointN, FaceNNPoints,...]
# - NElements refers to the total number of elements in the array needed to describe the polyhedron.
# - NFaces is the number of faces the figure will have.
# - Face1Npoints is the number of points the first face will have
# - Face1Point1..Face1PointN are each of the points that describe face1

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
polyhedron = np.asarray([len(polyhedron_connectivity)] + polyhedron_connectivity)


###############################################################################
# Cells array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# All of the cells are joined in a 1 dimensional numpy array. Separation between cells
# is determined by the number of elements of each figure (the first element of each
# connectivity array)

cells = np.hstack((quad, polygon, hexa, polyhedron))


###############################################################################
# Cell types
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We need to determine the cell types for each of the cells we define in the cells array.
# The number of elements in this array must coincide with the number of cells in the

cell_type = np.asarray(
    [pv.CellType.QUAD, pv.CellType.POLYGON, pv.CellType.HEXAHEDRON, pv.CellType.POLYHEDRON]
)


###############################################################################
# Create the grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To create the grid, we use the cells array we built, the cells types, as well
# as the nodes that describe the faces.

grid = pv.UnstructuredGrid(cells, cell_type, nodes)
print(grid.cell_type(0))

###############################################################################
# Plot the mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, we can plot the grid we've created

plt = pv.Plotter()
plt.show_axes()
plt.add_mesh(grid)
plt.show()
