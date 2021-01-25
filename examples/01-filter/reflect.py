"""
.. _ref_reflect_example:

Reflect Meshes
~~~~~~~~~~~~~~

This example reflects a mesh across a plane.

"""

from pyvista import examples

###############################################################################
# This example demonstrates how to reflect a mesh across a plane.
#
# Load an example mesh:
airplane = examples.load_airplane()

###############################################################################
# Reflect the mesh across a plane parallel to Z plane and centered in -100
# (the geometry input is copied to the output):
airplane = airplane.reflect('z', copy=True, center=-100)

###############################################################################
# Plot the reflected mesh:
airplane.plot(show_edges=True)
