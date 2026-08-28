"""
.. _cell_centers_example:

Extract Cell Centers
~~~~~~~~~~~~~~~~~~~~

Extract the coordinates of the centers of all cells or faces in a mesh.

Here we use :func:`cell_centers <pyvista.DataObjectFilters.cell_centers>`.

"""

import pyvista as pv

# sphinx_gallery_thumbnail_number = 3
from pyvista import examples

# %%
# First let's fetch the centers of a mesh with 2D geometries (a surface)
mesh = examples.download_teapot()

cpos = pv.CameraPosition(
    position=(6.193, 5.688, -4.953),
    focal_point=(0.4885, 1.202, 0.1656),
    viewup=(-0.4064, 0.8621, 0.3026),
)

centers = mesh.cell_centers()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, line_width=1)
pl.add_mesh(centers, color='r', point_size=8.0, render_points_as_spheres=True)
pl.show(cpos=cpos)


# %%
# We can also do this for full 3D meshes.

grid = examples.download_letter_a()

cpos = pv.CameraPosition(
    position=(2.705, 0.7823, 1.725),
    focal_point=(3.543, 0.3112, 0.1641),
    viewup=(0.1481, 0.966, -0.2119),
)


centers = grid.cell_centers()

pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, opacity=0.5, line_width=1)
pl.add_mesh(centers, color='r', point_size=8.0, render_points_as_spheres=True)
pl.show(cpos=cpos)

# %%

pl = pv.Plotter()
pl.add_mesh(grid.extract_all_edges(), color='k', line_width=1)
pl.add_mesh(centers, color='r', point_size=8.0, render_points_as_spheres=True)
pl.show(cpos=cpos)


# %%
# Edge Centers
# ~~~~~~~~~~~~
# You can use :func:`cell_centers <pyvista.DataObjectFilters.cell_centers>` in
# combination with :func:`extract_all_edges
# <pyvista.DataObjectFilters.extract_all_edges>` to get the center of all edges of
# a mesh.

# create a simple mesh and extract all the edges and then centers of the mesh.
mesh = pv.Cube()
edge_centers = mesh.extract_all_edges().cell_centers().points

# Plot the edge centers
pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, line_width=5)
pl.add_points(
    edge_centers,
    render_points_as_spheres=True,
    color='r',
    point_size=20,
)
pl.show()


# %%
# Add Labels to Cells
# ~~~~~~~~~~~~~~~~~~~
# There is not a method to add labels to cells.
# If you want to label it, you need to extract the position to label it.

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore
grid = pv.ImageData(dimensions=(10, 10, 1))
points = grid.cell_centers().points

pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True)
pl.add_point_labels(points, labels=[f'{i}' for i in range(points.shape[0])])
pl.show(cpos='xy')
# %%
# .. tags:: filter
