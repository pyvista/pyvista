import pyvista as pv

mesh = pv.read("Data/can.vtu")
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
