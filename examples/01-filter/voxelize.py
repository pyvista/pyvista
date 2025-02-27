"""
.. _voxelize_surface_mesh_example:

Voxelize a Surface Mesh
~~~~~~~~~~~~~~~~~~~~~~~

Demonstrate various ways to generate a voxelized mesh using the
:meth:`~pyvista.DataSetFilters.voxelize` filter.

"""

from __future__ import annotations

# sphinx_gallery_thumbnail_number = 2
import numpy as np

import pyvista as pv
from pyvista import examples

# %%
# Voxelize as ImageData
# ---------------------
# Load a coarse mesh of a bunny.
poly = examples.download_bunny_coarse()

# Voxelize the surface mesh as a binary image mask.
# Represent the voxels as points, which is the standard format for 2D/3D images.
mask = poly.voxelize_as('image', 'points')

# The mask is stored as pyvista.ImageData with point data scalars
# (zeros for background, ones for foreground).
mask


# Visualize the mask and polydata for comparison.
def mask_and_polydata_plotter(mask, poly):
    voxel_cells = mask.points_to_cells().threshold(0.5)

    plot = pv.Plotter()
    plot.add_mesh(voxel_cells, color='blue')
    plot.add_mesh(poly, color='lime')
    plot.camera_position = 'xy'
    return plot


plot = mask_and_polydata_plotter(mask, poly)
plot.show()

# Repeat the previous example with a finer mesh.
poly = examples.download_bunny()
mask = poly.voxelize_as('image', 'points')
plot = mask_and_polydata_plotter(mask, poly)
plot.show()

# Control the spacing manually.
mask = poly.voxelize_as('image', 'points', spacing=(0.01, 0.04, 0.02))
plot = mask_and_polydata_plotter(mask, poly)
plot.show()

# The spacing is approximate. Check the mask's actual spacing.
mask.spacing

# Use rounding_func=np.floor to force all spacing values to be greater.
mask = poly.voxelize_as('image', 'points', spacing=(0.01, 0.04, 0.02), rounding_func=np.floor)
mask.spacing

# Set the dimensions instead of the spacing.
mask = poly.voxelize_as('image', 'points', dimensions=(10, 20, 30))
plot = mask_and_polydata_plotter(mask, poly)
plot.show()

# Create a mask using a reference volume.
# First generate polydata from an existing mask.
volume = examples.load_frog_tissues()
poly = volume.contour_labels()

# Now create the mask from the polydata using the volume as a reference.
mask = poly.voxelize_as('image', 'points', reference_volume=volume)
plot = mask_and_polydata_plotter(mask, poly)
plot.show()

# Voxelize as RectilinearGrid
# ---------------------------
# Create a voxelized :class:`pyvista.RectilinearGrid` of a nut with voxels
# represented as cells. By default, the spacing is automatically estimated.

mesh = pv.examples.load_nut()
vox = mesh.voxelize_as('rectilinear', 'cells')

# Plot the mesh together with its volume.
pl = pv.Plotter()
pl.add_mesh(mesh=vox, show_edges=True)
pl.add_mesh(mesh=mesh, show_edges=True, opacity=1)
pl.show()

# Load a mesh of a cow.
mesh = examples.download_cow()

# Create an equal density voxel volume and plot the result.
vox = mesh.voxelize_as('rectilinear', 'cells', spacing=0.15)
cpos = [(15, 3, 15), (0, 0, 0), (0, 0, 0)]
vox.plot(scalars='mask', show_edges=True, cpos=cpos)

# Slice the voxel volume to view the mask scalars.
slices = vox.slice_orthogonal()
slices.plot(scalars='mask', show_edges=True)

# Create a voxel volume from unequal density dimensions and plot the result.
vox = mesh.voxelize_as('rectilinear', 'cells', spacing=(0.15, 0.15, 0.5))
vox.plot(scalars='mask', show_edges=True, cpos=cpos)

# Slice the unequal density voxel volume to view the mask scalars.
slices = vox.slice_orthogonal()
slices.plot(scalars='mask', show_edges=True, cpos=cpos)

# Voxelize as PolyData
# --------------------
# Load an ant mesh.
ant = examples.load_ant()

# Voxelize it as :class:`~pyvista.PolyData` cells. This generates a surface with
# :attr:`~pyvista.CellType.QUAD` cells that is hollow inside.
cpos = [(11.6, 57.3, 28.0), (0.65, -0.78, 1.19), (-0.55, 0.43, -0.71)]
vox = ant.voxelize_as('poly', 'cells', spacing=0.5)
vox.plot(cpos=cpos)

# Voxelize it as a dense point cloud instead.
vox = ant.voxelize_as('poly', 'points', spacing=0.5)

# Plot the point cloud and show the original mesh for context.
pl = pv.Plotter()
pl.add_mesh(ant, opacity=0.1)
pl.add_mesh(vox, color='black', render_points_as_spheres=True)
pl.camera_position = cpos
pl.show()


# Voxelize as UnstructuredGrid
# ----------------------------

# This example also demonstrates how to compute an implicit distance from a
# bounding :class:`pyvista.PolyData` surface.

# Load a surface to voxelize
surface = examples.download_foot_bones()
surface

# %%
cpos = [
    (7.656346967151718, -9.802071079151158, -11.021236183314311),
    (0.2224512272564101, -0.4594554282112895, 0.5549738359311297),
    (-0.6279216753504941, -0.7513057097368635, 0.20311105371647392),
]

surface.plot(cpos=cpos, opacity=0.75)


# %%
# Create a voxel model of the bounding surface
voxels = surface.voxelize()

p = pv.Plotter()
p.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)
p.add_mesh(surface, color='lightblue', opacity=0.5)
p.show(cpos=cpos)


# %%
# We could even add a scalar field to that new voxel model in case we
# wanted to create grids for modelling. In this case, let's add a scalar field
# for bone density noting:
voxels['density'] = np.full(voxels.n_cells, 3.65)  # g/cc
voxels

# %%
voxels.plot(scalars='density', cpos=cpos)


# %%
# A constant scalar field is kind of boring, so let's get a little fancier by
# added a scalar field that varies by the distance from the bounding surface.
voxels.compute_implicit_distance(surface, inplace=True)
voxels

# %%
contours = voxels.contour(6, scalars='implicit_distance')

p = pv.Plotter()
p.add_mesh(voxels, opacity=0.25, scalars='implicit_distance')
p.add_mesh(contours, opacity=0.5, scalars='implicit_distance')
p.show(cpos=cpos)

# %%
# Effects of Internal Surfaces
# ----------------------------

# Visualize the effect of internal surfaces.
mesh = pv.Cylinder() + pv.Cylinder((0, 0.75, 0))
binary_mask = mesh.voxelize_as('image', 'points', dimensions=(1, 100, 50)).points_to_cells()
plot = pv.Plotter()
plot.add_mesh(binary_mask)
plot.add_mesh(mesh.slice(), color='red')
plot.show(cpos='yz')

# Process intersecting parts of the mesh sequentially.
cylinder_1 = pv.Cylinder()
cylinder_2 = pv.Cylinder((0, 0.75, 0))

reference_volume = pv.ImageData(
    dimensions=(1, 100, 50),
    spacing=(1, 0.0175, 0.02),
    origin=(0, -0.5 + 0.0175 / 2, -0.5 + 0.02 / 2),
)

binary_mask_1 = cylinder_1.voxelize_as(
    'image', 'points', reference_volume=reference_volume
).points_to_cells()
binary_mask_2 = cylinder_2.voxelize_as(
    'image', 'points', reference_volume=reference_volume
).points_to_cells()

binary_mask_1['mask'] = binary_mask_1['mask'] | binary_mask_2['mask']

plot = pv.Plotter()
plot.add_mesh(binary_mask_1)
plot.add_mesh(cylinder_1.slice(), color='red')
plot.add_mesh(cylinder_2.slice(), color='red')
plot.show(cpos='yz')

# Visualize nested internal surfaces.
mesh = pv.Tube(radius=2) + pv.Tube(radius=3) + pv.Tube(radius=4)
binary_mask = mesh.voxelize_as('image', 'points', dimensions=(1, 50, 50)).points_to_cells()
plot = pv.Plotter()
plot.add_mesh(binary_mask)
plot.add_mesh(mesh.slice(), color='red')
plot.show(cpos='yz')


# %%
# .. tags:: filter
