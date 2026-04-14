"""
.. _compute_normals_example:

Computing Surface Normals
~~~~~~~~~~~~~~~~~~~~~~~~~
Compute normals on a surface.

"""

from __future__ import annotations

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import numpy as np
import pyvista as pv

# sphinx_gallery_thumbnail_number = 2
from pyvista import examples

# %%
# Computing the normals of a surface is quite easy using
# :class:`pyvista.PolyData`'s :func:`pyvista.PolyDataFilters.compute_normals`
# method.

mesh = examples.download_topo_global()
mesh.plot(cmap='gist_earth', show_scalar_bar=False)

# %%
# Now we have a surface dataset of the globe loaded - unfortunately, the
# dataset shows the globe with a uniform radius which hides topographic relief.
# Using :func:`pyvista.PolyDataFilters.compute_normals`, we can compute the normal
# vectors on the globe at all points in the dataset, then use the values given
# in the dataset to warp the surface in the normals direction to create some
# exaggerated topographic relief.

# Compute the normals in-place and use them to warp the globe
mesh.compute_normals(inplace=True)  # this activates the normals as well

# Now use those normals to warp the surface
warp = mesh.warp_by_scalar(factor=0.5e-5)

# And let's see it
warp.plot(cmap='gist_earth', show_scalar_bar=False)


# %%
# We could also use face/cell normals to extract all the faces of a mesh
# facing a general direction. In the following snippet, we take a mesh, compute
# the normals along its cell faces, and extract the faces that face upward.

mesh = examples.download_nefertiti()
# Compute normals
mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)

# Get list of cell IDs that meet condition
ids = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] > 0.0]

# Extract those cells
top = mesh.extract_cells(ids)

cpos = pv.CameraPosition(
    position=(-834.3184529757553, -918.4677714398535, 236.5468795300025),
    focal_point=(11.03829376004883, -13.642289291587957, -35.91218884207208),
    viewup=(0.19212361465657216, 0.11401076390090074, 0.9747256344254143),
)

top.plot(cpos=cpos, color=True)
# %%
# .. tags:: filter
