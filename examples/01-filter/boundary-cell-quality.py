import vtk

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("Data/can.vtu")
reader.Update()
print(reader.GetOutput())
# TODO @tkoyama010: check the warning reason of read function.
# mesh = pv.read("Data/can.vtu")

# quality = vtk.vtkBoundaryMeshQuality()
# quality.SetInputConnection(reader.GetOutputPort())
# quality.Update()
# output = quality.GetOutput()
#
# distance1 = output.GetCellData().GetArray("DistanceFromCellCenterToFaceCenter")
# distance2 = output.GetCellData().GetArray("DistanceFromCellCenterToFacePlane")
# angle = output.GetCellData().GetArray("AngleFaceNormalAndCellCenterToFaceCenterVector")
#
# range1 = distance1.GetRange()
# range2 = distance2.GetRange()
# range3 = angle.GetRange()
#
# cqual = mesh.compute_boundary_mesh_quality()
#
# plotter = pv.Plotter(shape=(2, 2))
# plotter.add_mesh(mesh)
# plotter.subplot(1, 0)
# plotter.add_mesh(cqual.copy(), scalars="DistanceFromCellCenterToFaceCenter")
# plotter.subplot(0, 1)
# plotter.add_mesh(cqual.copy(), scalars="DistanceFromCellCenterToFacePlane")
# plotter.subplot(1, 1)
# plotter.add_mesh(cqual.copy(), scalars="AngleFaceNormalAndCellCenterToFaceCenterVector")
# plotter.show()
