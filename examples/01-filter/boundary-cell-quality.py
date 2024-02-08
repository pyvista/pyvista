import vtk

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("Data/can.vtu")
reader.Update()

boundaryMeshQuality = vtk.vtkBoundaryMeshQuality()
boundaryMeshQuality.SetInputConnection(reader.GetOutputPort())
boundaryMeshQuality.Update()
output = boundaryMeshQuality.GetOutput()

distance1 = output.GetCellData().GetArray("DistanceFromCellCenterToFaceCenter")
distance2 = output.GetCellData().GetArray("DistanceFromCellCenterToFacePlane")
angle = output.GetCellData().GetArray("AngleFaceNormalAndCellCenterToFaceCenterVector")

distanceEpsilon = 1e-6
range1 = distance1.GetRange()

range2 = distance2.GetRange()

angleEpsilon = 1e-4
range3 = angle.GetRange()
