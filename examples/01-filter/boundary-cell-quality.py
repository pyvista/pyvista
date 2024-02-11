import pyvista as pv
from pyvista import examples

mesh = examples.download_can_crushed_vtu()
cqual = mesh.compute_boundary_mesh_quality()

plotter = pv.Plotter(shape=(2, 2))
plotter.add_mesh(mesh, show_edges=True)
plotter.subplot(1, 0)
plotter.add_mesh(cqual, scalars="DistanceFromCellCenterToFaceCenter")
plotter.subplot(0, 1)
plotter.add_mesh(cqual, scalars="DistanceFromCellCenterToFacePlane")
plotter.subplot(1, 1)
plotter.add_mesh(cqual, scalars="AngleFaceNormalAndCellCenterToFaceCenterVector")
plotter.show()
