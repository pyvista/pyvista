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
