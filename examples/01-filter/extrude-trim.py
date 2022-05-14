"""
.. _extrude_trim_example:

Extrude Trim
~~~~~~~~~~~~

Extrude volume from one surface to palne and trim.
"""

import pyvista as pv
import vtk

# Create surface and plane
mesh = pv.ParametricRandomHills(randomseed=2)
plane = pv.Plane(center=(0, 0, -5), direction=(0, 0, -1), i_size=100, j_size=100)

# Perform extrude trim
alg = vtk.vtkTrimmedExtrusionFilter()
alg.SetInputData(mesh)
alg.SetTrimSurfaceData(plane)
alg.SetCappingStrategy(0) # <-- ensure that the cap is defined by the intersection
alg.SetExtrusionDirection(0, 0, -1.0) # <-- set this with the plane normal
alg.Update()
poly = pv.core.filters._get_output(alg)

# Render the result
p = pv.Plotter()
p.add_mesh(plane)
p.add_mesh(mesh)
p.add_mesh(poly, color='red')
p.show()
