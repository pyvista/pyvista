import vtk

import pyvista as pv

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("Data/can.vtu")
reader.Update()
# TODO @tkoyama010: check the warning reason of read function.
# mesh = pv.read("Data/can.vtu")
mesh = pv.wrap(reader.GetOutput())
cqual = mesh.compute_boundary_mesh_quality()

plotter = pv.Plotter(shape=(2, 2))
plotter.add_mesh(mesh)
plotter.subplot(1, 0)
plotter.add_mesh(cqual.copy(), scalars="DistanceFromCellCenterToFaceCenter")
plotter.subplot(0, 1)
plotter.add_mesh(cqual.copy(), scalars="DistanceFromCellCenterToFacePlane")
plotter.subplot(1, 1)
plotter.add_mesh(cqual.copy(), scalars="AngleFaceNormalAndCellCenterToFaceCenterVector")
plotter.show()
