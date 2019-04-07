"""
.. _ref_geometric_example:

Geometric Objects
~~~~~~~~~~~~~~~~~

The "Hello, world!" of VTK
"""
################################################################################
#
# This runs through several of the available geomoetric objects available in VTK
# which ``vtki`` provides simple conveinance methods for generating.

# Let's run through a few geometric objects!

# sphinx_gallery_thumbnail_number = 3
import vtki

sphere = vtki.Sphere()
sphere.plot(color='orange', show_edges=True)


################################################################################
cyl = vtki.Cylinder()
cyl.plot(color='orange', show_edges=True)


################################################################################
arrow = vtki.Arrow()
arrow.plot(color='orange', show_edges=True)


################################################################################
box = vtki.Box()
box.plot(color='orange', show_edges=True)

################################################################################
cone = vtki.Cone()
cone.plot(color='orange', show_edges=True)


################################################################################
poly = vtki.Polygon()
poly.plot(color='orange', show_edges=True)

################################################################################
disc = vtki.Disc()
disc.plot(color='orange', show_edges=True)
