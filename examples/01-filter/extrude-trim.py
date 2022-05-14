"""
.. _extrude_trim_example:

Extrude Trim
~~~~~~~~~~~~

Extrude volume from one surface to palne and trim.
"""

import pyvista as pv

# Create surface and plane
mesh = pv.ParametricRandomHills(randomseed=2)
plane = pv.Plane(center=(0, 0, -5), direction=(0, 0, -1), i_size=100, j_size=100)

# Perform extrude trim
poly = mesh.extrude_trim((0, 0, -1.0), plane)

# Render the result
p = pv.Plotter()
p.add_mesh(plane)
p.add_mesh(mesh)
p.add_mesh(poly, color='red')
p.show()
