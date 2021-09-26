"""
.. _create_truss:

Plot Truss-like FEA Solution with Cylinders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot connections between points in 3D as cylinders, colored by scalars.


"""

import numpy as np
import pyvista

###############################################################################
# Define the points and elements of the truss.  Call them ``nodes``
# here as it comes from finite element analysis.

nodes = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [4.0, 3.0, 0.0],
    [4.0, 0.0, 0.0],
    [0.0, 1.0, 2.0],
    [4.0, 1.0, 2.0],
    [4.0, 3.0, 2.0]
    ]


edges = np.array([
    [0, 4],
    [1, 4],
    [3, 4],
    [5, 4],
    [6, 4],
    [3, 5],
    [2, 5],
    [5, 6],
    [2, 6],
    ])

# We must "pad" the edges to indicate to vtk how many points per edge
padding = np.empty(edges.shape[0], int)*2
padding[:] = 2
edges_w_padding = np.vstack((padding, edges.T)).T
edges_w_padding

###############################################################################
# Plot the truss while rendering the lines as tubes.

mesh = pyvista.PolyData(nodes, edges_w_padding)

colors = range(edges.shape[0])
mesh.plot(scalars=colors,
          render_lines_as_tubes=True,
          style='wireframe',
          line_width=10,
          cmap='jet',
          show_scalar_bar=False,
          background='w')
