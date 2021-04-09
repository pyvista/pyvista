"""
Extrude Rotation
~~~~~~~~~~~~~~~~
Sweep polygonal data creating "skirt" from free edges and lines, and
lines from vertices.

This takes polygonal data as input and generates polygonal data on
output. The input dataset is swept around the z-axis to create
new polygonal primitives. These primitives form a "skirt" or
swept surface. For example, sweeping a line results in a
cylindrical shell, and sweeping a circle creates a torus.

"""
import pyvista

# create a line and rotate it about the Z-axis
resolution = 10
line = pyvista.Line(pointa=(0, 0, 0), pointb=(1, 0, 0), resolution=2)
poly = line.extrude_rotate(resolution=resolution)
poly

###############################################################################
# Plot the extruded line
# ~~~~~~~~~~~~~~~~~~~~~~

plotter = pyvista.Plotter(shape=(2, 1))
plotter.subplot(0, 0)
plotter.add_text("Line", font_size=24)
plotter.add_mesh(line, color="tan", show_edges=True)
plotter.add_mesh(
    pyvista.PolyData(line.points),
    color="red",
    point_size=10,
    render_points_as_spheres=True,
)
plotter.subplot(1, 0)
plotter.add_text("Extrude Rotated Line", font_size=24)
plotter.add_mesh(poly, color="tan", show_edges=True)
plotter.add_mesh(
    pyvista.PolyData(poly.points),
    color="red",
    point_size=10,
    render_points_as_spheres=True,
)

plotter.show(cpos="xy")
