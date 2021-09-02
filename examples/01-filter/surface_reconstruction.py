"""
.. _surface_reconstruction_example:

Surface Reconstruction
~~~~~~~~~~~~~~~~~~~~~~

Surface reconstruction has a dedicated filter in PyVista and is
handled by :func:`pyvista.DataSetFilters.reconstruct_surface`.  This
tends to perform much better than :func:`DataSetFilters.delaunay_3d`.

"""
import pyvista as pv

###############################################################################
# Create a point cloud from a sphere and then reconstruct a surface from it.

points = pv.wrap(pv.Sphere().points)
surf = points.reconstruct_surface()
surf

###############################################################################
# Plot the point cloud and the reconstructed sphere.

pl = pv.Plotter(shape=(1,2))
pl.add_mesh(points)
pl.add_title('Point Cloud of 3D Surface')
pl.subplot(0,1)
pl.add_mesh(surf, color=True, show_edges=True)
pl.add_title('Reconstructed Surface')
pl.show()
