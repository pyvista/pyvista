import pyvista as pv

mesh = pv.read("Data/can.vtu")

output = mesh.compute_boundary_mesh_quality()

distance1 = output["DistanceFromCellCenterToFaceCenter"]
distance2 = output["DistanceFromCellCenterToFacePlane"]
angle = output["AngleFaceNormalAndCellCenterToFaceCenterVector"]
