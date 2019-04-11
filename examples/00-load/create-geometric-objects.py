"""
.. _ref_geometric_example:

Geometric Objects
~~~~~~~~~~~~~~~~~

The "Hello, world!" of VTK
"""

# sphinx_gallery_thumbnail_number = 3
import vtki

################################################################################
#
# This runs through several of the available geomoetric objects available in VTK
# which ``vtki`` provides simple conveinance methods for generating.

# Let's run through a few geometric objects!

sphere = vtki.Sphere()
sphere.plot(show_edges=True, color='tan')


################################################################################
cyl = vtki.Cylinder()
cyl.plot(show_edges=True, color='tan')


################################################################################
arrow = vtki.Arrow()
arrow.plot(show_edges=True)


################################################################################
box = vtki.Box()
box.plot(show_edges=True, color='tan')

################################################################################
cone = vtki.Cone()
cone.plot(show_edges=True)


################################################################################
poly = vtki.Polygon()
poly.plot(show_edges=True)

################################################################################
disc = vtki.Disc()
disc.plot(show_edges=True)
