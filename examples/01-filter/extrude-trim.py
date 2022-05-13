import pyvista as pv
import vtk

mesh = pv.ParametricRandomHills(randomseed=2)
plane = pv.Plane(center=(0, 0, -5), direction=(0, 0, -1), i_size=100, j_size=100)

alg = vtk.vtkTrimmedExtrusionFilter()
alg.SetInputData(0, mesh)
alg.SetInputData(1, plane)
alg.SetCappingStrategy(0) # <-- ensure that the cap is defined by the intersection
alg.SetExtrusionDirection(0, 0, -1.0) # <-- set this with the plane normal
alg.Update()
output = pv.core.filters._get_output(alg)

p = pv.Plotter()
p.add_mesh(plane)
p.add_mesh(mesh)
p.add_mesh(output, color='red')
p.show()
