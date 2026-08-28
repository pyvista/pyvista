"""
.. _voxelize_example:

Voxelize a Surface Mesh
~~~~~~~~~~~~~~~~~~~~~~~

Create a voxel model (like LEGOs) of a closed surface or volumetric mesh.

This example also demonstrates how to compute an implicit distance from a
bounding :class:`pyvista.PolyData` surface.

"""

import numpy as np
import pyvista as pv

# sphinx_gallery_thumbnail_number = 2
from pyvista import examples

# Load a surface to voxelize
surface = examples.download_foot_bones()
surface

# %%
cpos = pv.CameraPosition(
    position=(7.656, -9.802, -11.02),
    focal_point=(0.2225, -0.4595, 0.555),
    viewup=(-0.6279, -0.7513, 0.2031),
)

surface.plot(cpos=cpos, opacity=0.75)


# %%
# Create a voxel model of the bounding surface
voxels = surface.voxelize()

pl = pv.Plotter()
pl.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)
pl.add_mesh(surface, color='lightblue', opacity=0.5)
pl.show(cpos=cpos)


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

pl = pv.Plotter()
pl.add_mesh(voxels, opacity=0.25, scalars='implicit_distance')
pl.add_mesh(contours, opacity=0.5, scalars='implicit_distance')
pl.show(cpos=cpos)
# %%
# .. tags:: filter
