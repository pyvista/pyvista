"""
.. _fill_holes_example:

Repair a Surface with fill_holes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an open boundary with a planar clip and seal it with
:func:`pyvista.PolyDataFilters.fill_holes`.
"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# sphinx_gallery_thumbnail_number = 2

# %%
# Start from an open surface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# The :func:`~pyvista.examples.downloads.download_face` scan is an open mesh
# with naturally occurring holes along its boundary.

open_mesh = examples.download_face().triangulate()
open_mesh


# %%
# Highlight the boundary loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Boundary edges make it easy to see where the surface is no longer watertight.

boundary = open_mesh.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False,
)

pl = pv.Plotter()
pl.add_mesh(open_mesh, color='wheat', smooth_shading=True)
pl.add_mesh(boundary, color='tomato', line_width=6)
pl.show()


# %%
# Fill the holes
# ~~~~~~~~~~~~~~
# The hole size should be larger than the opening we want to patch.

repaired = open_mesh.fill_holes(open_mesh.length)

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(open_mesh, color='wheat', smooth_shading=True)
pl.add_mesh(boundary, color='tomato', line_width=6)
pl.subplot(0, 1)
pl.add_mesh(repaired, color='wheat', smooth_shading=True)
pl.link_views()
pl.show()


# %%
# Confirm the repair
# ~~~~~~~~~~~~~~~~~~
# The filled surface has no remaining open edges.

open_mesh.n_open_edges, repaired.n_open_edges
# %%
# .. tags:: filter
