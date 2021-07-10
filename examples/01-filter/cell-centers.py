"""
.. _cell_centers_example:

Extract Cell Centers
~~~~~~~~~~~~~~~~~~~~

Extract the coordinates of the centers of all cells/faces in a mesh.

Here we use :func:`pyvista.DataSetFilters.cell_centers`
"""
# sphinx_gallery_thumbnail_number = 3
from pyvista import examples
import pyvista as pv

###############################################################################
# First let's fetch the centers of a mesh with 2D geometries (a surface)
mesh = examples.download_teapot()

cpos = [(6.192871661244108, 5.687542355343226, -4.95345468836544),
 (0.48853358141600634, 1.2019347531215714, 0.1656178278582367),
 (-0.40642070472687936, 0.8621356761976646, 0.30256286387543047)]

centers = mesh.cell_centers()

p = pv.Plotter()
p.add_mesh(mesh, show_edges=True, line_width=1)
p.add_mesh(centers, color="r", point_size=8.0, render_points_as_spheres=True)
p.show(cpos=cpos)


###############################################################################
# We can also do this for full 3D meshes.

grid = examples.download_letter_a()

cpos = [(2.704583323659036, 0.7822568412034183, 1.7251126717482546),
 (3.543391913452799, 0.31117673768140197, 0.16407006760146028),
 (0.1481171795711516, 0.96599698246102, -0.2119224645762945)]


centers = grid.cell_centers()

p = pv.Plotter()
p.add_mesh(grid, show_edges=True, opacity=0.5, line_width=1)
p.add_mesh(centers, color="r", point_size=8.0, render_points_as_spheres=True)
p.show(cpos=cpos)

###############################################################################

p = pv.Plotter()
p.add_mesh(grid.extract_all_edges(), color="k", line_width=1)
p.add_mesh(centers, color="r", point_size=8.0, render_points_as_spheres=True)
p.show(cpos=cpos)
