"""
.. _ref_geometric_example:

Geometric Objects
~~~~~~~~~~~~~~~~~

The "Hello, world!" of VTK
"""
import pyvista

################################################################################
# This runs through several of the available geomoetric objects available in VTK
# which ``pyvista`` provides simple conveinance methods for generating.
#
# Let's run through creating a few geometric objects!

cyl = pyvista.Cylinder()
arrow = pyvista.Arrow()
sphere = pyvista.Sphere()
plane = pyvista.Plane()
line = pyvista.Line()
box = pyvista.Box()
cone = pyvista.Cone()
poly = pyvista.Polygon()
disc = pyvista.Disc()

################################################################################
# Now let's plot them all in one window

p = pyvista.Plotter(shape=(3, 3))
# Top row
p.subplot(0,0)
p.add_mesh(cyl, color='tan', show_edges=True)
p.subplot(0,1)
p.add_mesh(arrow, color='tan', show_edges=True)
p.subplot(0,2)
p.add_mesh(sphere, color='tan', show_edges=True)
# Middle row
p.subplot(1,0)
p.add_mesh(plane, color='tan', show_edges=True)
p.subplot(1,1)
p.add_mesh(line, color='tan', line_width=3)
p.subplot(1,2)
p.add_mesh(box, color='tan', show_edges=True)
# Bottom row
p.subplot(2,0)
p.add_mesh(cone, color='tan', show_edges=True)
p.subplot(2,1)
p.add_mesh(poly, color='tan', show_edges=True)
p.subplot(2,2)
p.add_mesh(disc, color='tan', show_edges=True)
# Render all of them
p.show()
