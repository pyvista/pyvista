"""
.. _create_dimension_line_example:
Create Dimension Line
---------------------

Create a 2d dimension line along 2d structured mesh.

"""
import numpy as np

import pyvista as pv

# Set plot theme to "document"
pv.set_plot_theme("document")

# Define x and y ranges for the structured mesh
xrng = np.arange(-10, 10, 2)
yrng = np.arange(-10, 10, 5)

# Create the structured mesh
grid = pv.RectilinearGrid(xrng, yrng)

# Create a plotter object
plotter = pv.Plotter()

# Add the mesh to the plotter object
plotter.add_mesh(grid, show_edges=True, color='tan')

# Enable parallel projection for the plot
plotter.enable_parallel_projection()

###############################################################################
# Define a function to create a dimension line.


def create_dimension_line(pointa, pointb, normal):
    """Create a dimension line with labels and arrows for the given points and normal vector"""
    # Define the lines
    pointa += normal
    pointb += normal
    lines = np.array([pointa, pointb])

    # Create multiple lines and set the normal vector
    mlines = pv.MultipleLines(lines)
    mlines["Normal"] = np.array([normal, normal])

    # Create arrows for the lines
    arrows = mlines.glyph(geom=pv.Line(), scale="Normal", factor=1.0, orient="Normal")

    # Define the midpoints between pointa and pointb
    pointc = (pointa + pointb) / 2.0

    # Define the label for the line
    labels = np.array([str(np.linalg.norm(pointb - pointa))])

    # Add the label and line to the plot
    plotter.add_point_labels(points=pointc, labels=labels, shape_color="white")
    plotter.add_lines(lines, color="black", width=2)

    # Add the arrow to the plot
    plotter.add_mesh(arrows, color="black")


###############################################################################
# Define the points and normal vectors for the dimension lines

create_dimension_line(
    pointa=np.array([xrng[0], yrng[-1], 0.0]),
    pointb=np.array([xrng[3], yrng[-1], 0.0]),
    normal=np.array([0.0, 1.0, 0.0]),
)
create_dimension_line(
    pointa=np.array([xrng[3], yrng[-1], 0.0]),
    pointb=np.array([xrng[6], yrng[-1], 0.0]),
    normal=np.array([0.0, 1.0, 0.0]),
)
create_dimension_line(
    pointa=np.array([xrng[6], yrng[-1], 0.0]),
    pointb=np.array([xrng[9], yrng[-1], 0.0]),
    normal=np.array([0.0, 1.0, 0.0]),
)
create_dimension_line(
    pointa=np.array([xrng[0], yrng[0], 0.0]),
    pointb=np.array([xrng[0], yrng[1], 0.0]),
    normal=np.array([-1.0, 0.0, 0.0]),
)
create_dimension_line(
    pointa=np.array([xrng[0], yrng[1], 0.0]),
    pointb=np.array([xrng[0], yrng[2], 0.0]),
    normal=np.array([-1.0, 0.0, 0.0]),
)
create_dimension_line(
    pointa=np.array([xrng[0], yrng[2], 0.0]),
    pointb=np.array([xrng[0], yrng[3], 0.0]),
    normal=np.array([-1.0, 0.0, 0.0]),
)

create_dimension_line(
    pointa=np.array([xrng[0], yrng[0], 0.0]),
    pointb=np.array([xrng[9], yrng[0], 0.0]),
    normal=np.array([0.0, -1.0, 0.0]),
)
create_dimension_line(
    pointa=np.array([xrng[-1], yrng[0], 0.0]),
    pointb=np.array([xrng[-1], yrng[3], 0.0]),
    normal=np.array([1.0, 0.0, 0.0]),
)

###############################################################################
# Plot mesh with dimension line

plotter.show(cpos="xy")
