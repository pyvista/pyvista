"""
Plot Over Circular Arc
~~~~~~~~~~~~~~~~~~~~~~

Plot the height of a dataset over a circular arc through that dataset
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

###############################################################################
# Volumetric Mesh
# +++++++++++++++
#
# First a 3D mesh example to demonstrate
mesh = examples.load_uniform()
mesh['height'] = mesh.points[:, 2]

# Make two points and center to construct the circular arc between
a = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[5]]
b = [mesh.bounds[1], mesh.bounds[2], mesh.bounds[4]]
center = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]

# Preview how this circular arc intersects this mesh
arc = pv.CircularArc(a, b, center)

p = pv.Plotter()
p.add_mesh(mesh, style="wireframe", color="w")
p.add_mesh(arc, color="b")
p.add_point_labels(
    [a, b], ["A", "B"], font_size=48, point_color="red", text_color="red"
)
p.show()

###############################################################################
# Run the filter and produce a line plot
mesh.plot_over_circular_arc(a, b, center, resolution=100, scalars='height')

