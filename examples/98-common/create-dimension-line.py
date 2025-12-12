"""
Create Dimension Line
---------------------

Create a 2d dimension line along 2d structured mesh.

"""

import numpy as np

import pyvista as pv

# Define x and y ranges for the structured mesh
xrng = np.arange(-10, 10, 2)
yrng = np.arange(-10, 10, 5)

# Create the structured mesh
grid = pv.RectilinearGrid(xrng, yrng)


###############################################################################
# Plot mesh with dimension line
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create a plotter object
plotter = pv.Plotter()

# Add the mesh to the plotter object
plotter.add_mesh(grid, show_edges=True, color='tan')

# Enable parallel projection for the plot
plotter.enable_parallel_projection()

# The dimension between ``pointa`` and ``pointb`` is drawn.

plotter.add_dimension_line(
    pointa=np.array([xrng[0], yrng[-1], 0.0]),
    pointb=np.array([xrng[3], yrng[-1], 0.0]),
    offset=np.array([0.0, 1.0, 0.0]),
    shape_color="white",
)
plotter.add_dimension_line(
    pointa=np.array([xrng[3], yrng[-1], 0.0]),
    pointb=np.array([xrng[6], yrng[-1], 0.0]),
    offset=np.array([0.0, 1.0, 0.0]),
    shape_color="white",
)
plotter.add_dimension_line(
    pointa=np.array([xrng[6], yrng[-1], 0.0]),
    pointb=np.array([xrng[9], yrng[-1], 0.0]),
    offset=np.array([0.0, 1.0, 0.0]),
    shape_color="white",
)
plotter.add_dimension_line(
    pointa=np.array([xrng[0], yrng[0], 0.0]),
    pointb=np.array([xrng[0], yrng[1], 0.0]),
    offset=np.array([-1.0, 0.0, 0.0]),
    shape_color="white",
)
plotter.add_dimension_line(
    pointa=np.array([xrng[0], yrng[1], 0.0]),
    pointb=np.array([xrng[0], yrng[2], 0.0]),
    offset=np.array([-1.0, 0.0, 0.0]),
    shape_color="white",
)
plotter.add_dimension_line(
    pointa=np.array([xrng[0], yrng[2], 0.0]),
    pointb=np.array([xrng[0], yrng[3], 0.0]),
    offset=np.array([-1.0, 0.0, 0.0]),
    shape_color="white",
)

plotter.add_dimension_line(
    pointa=np.array([xrng[0], yrng[0], 0.0]),
    pointb=np.array([xrng[9], yrng[0], 0.0]),
    offset=np.array([0.0, -1.0, 0.0]),
    shape_color="white",
)
plotter.add_dimension_line(
    pointa=np.array([xrng[-1], yrng[0], 0.0]),
    pointb=np.array([xrng[-1], yrng[3], 0.0]),
    offset=np.array([1.0, 0.0, 0.0]),
    shape_color="white",
)

plotter.show(cpos="xy")
