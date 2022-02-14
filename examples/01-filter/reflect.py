"""
.. _ref_reflect_example:

Reflect Meshes
~~~~~~~~~~~~~~

This example reflects a mesh across a plane.

"""

import pyvista
from pyvista import examples

###############################################################################
# This example demonstrates how to reflect a mesh across a plane.
#
# Load an example mesh:
airplane = examples.load_airplane()

###############################################################################
# Reflect the mesh across a plane parallel to Z plane and coincident with
# (0, 0, -100)
airplane_reflected = airplane.reflect((0, 0, 1), point=(0, 0, -100))

###############################################################################
# Plot the reflected mesh:
p = pyvista.Plotter()
p.add_mesh(airplane, show_edges=True)
p.add_mesh(airplane_reflected, show_edges=True)
p.show()
