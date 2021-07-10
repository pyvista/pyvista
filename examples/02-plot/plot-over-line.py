"""
.. _plot_over_line_example:

Plot Over Line
~~~~~~~~~~~~~~

Plot the values of a dataset over a line through that dataset
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

###############################################################################
# Volumetric Mesh
# +++++++++++++++
#
# First a 3D mesh example to demonstrate
mesh = examples.download_kitchen()

# Make two points to construct the line between
a = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
b = [mesh.bounds[1], mesh.bounds[3], mesh.bounds[5]]

# Preview how this line intersects this mesh
line = pv.Line(a, b)

p = pv.Plotter()
p.add_mesh(mesh, style="wireframe", color="w")
p.add_mesh(line, color="b")
p.show()

###############################################################################
# Run the filter and produce a line plot
mesh.plot_over_line(a, b, resolution=100)


###############################################################################
# Flat Surface
# ++++++++++++
#
# We could also plot the values of a mesh that lies on a flat surface
mesh = examples.download_st_helens()

# Make two points to construct the line between
a = [mesh.center[0], mesh.bounds[2], mesh.bounds[5]]
b = [mesh.center[0], mesh.bounds[3], mesh.bounds[5]]

# Preview how this line intersects this mesh
line = pv.Line(a, b)

p = pv.Plotter()
p.add_mesh(mesh)
p.add_mesh(line, color="white", line_width=10)
p.add_point_labels(
    [a, b], ["A", "B"], font_size=48, point_color="red", text_color="red"
)
p.show()

###############################################################################
# Run the filter and produce a line plot
mesh.plot_over_line(
    a,
    b,
    resolution=10000,
    title="Elevation Profile",
    ylabel="Height above sea level",
    figsize=(10, 5),
)
