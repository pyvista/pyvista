"""
.. _clip_closed_surface_example:

Clip and Cap a Closed Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare a standard planar clip, which leaves an open boundary, with
:func:`pyvista.PolyDataFilters.clip_closed_surface`, which seals the cut face.
"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# sphinx_gallery_thumbnail_number = 2

# %%
# Load a closed surface to inspect
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The :func:`~pyvista.examples.downloads.download_lucy` scan is a watertight,
# manifold PolyData surface that is well suited to closed-surface clipping.

surface = examples.download_lucy().triangulate()
surface


# %%
# Define an oblique cutting plane
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The same plane will be used for both the open clip and the closed-surface
# clip.

plane_origin = surface.center
plane_normal = (1, 1, 0.3)
plane_size = surface.length * 1.2
plane = pv.Plane(
    center=plane_origin, direction=plane_normal, i_size=plane_size, j_size=plane_size
)

pl = pv.Plotter()
pl.add_mesh(surface, color='wheat', smooth_shading=True)
pl.add_mesh(plane, style='wireframe', color='black', line_width=2)
pl.show()


# %%
# Compare an open cut to a capped cut
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A standard clip leaves a boundary loop behind. The closed-surface clip creates
# new faces to keep the result watertight.

open_clip = surface.clip(normal=plane_normal, origin=plane_origin)
closed_clip = surface.clip_closed_surface(normal=plane_normal, origin=plane_origin)

open_boundary = open_clip.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False,
)

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(open_clip, color='wheat', smooth_shading=True)
pl.add_mesh(open_boundary, color='tomato', line_width=6)
pl.add_mesh(plane, style='wireframe', color='black', line_width=2)
pl.subplot(0, 1)
pl.add_mesh(closed_clip, color='wheat', smooth_shading=True)
pl.add_mesh(plane, style='wireframe', color='black', line_width=2)
pl.link_views()
pl.show()


# %%
# Confirm that the capped result is watertight
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The open clip keeps hundreds of boundary edges, while the capped result has
# none.

open_boundary.n_cells, closed_clip.n_open_edges
# %%
# .. tags:: filter
