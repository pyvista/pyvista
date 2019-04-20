"""
.. _ref_geometric_example:

Geometric Objects
~~~~~~~~~~~~~~~~~

The "Hello, world!" of VTK
"""
import vtki

################################################################################
# This runs through several of the available geomoetric objects available in VTK
# which ``vtki`` provides simple conveinance methods for generating.
#
# Let's run through creating a few geometric objects!

cyl = vtki.Cylinder()
arrow = vtki.Arrow()
sphere = vtki.Sphere()
plane = vtki.Plane()
line = vtki.Line()
box = vtki.Box()
cone = vtki.Cone()
poly = vtki.Polygon()
disc = vtki.Disc()

################################################################################
# Now let's plot them all in one window

p = vtki.Plotter(shape=(3, 3))
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
