"""
.. _clip_closed_surface_example:

Clip and Cap a Closed Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare a standard planar clip, which leaves an open boundary, with
:func:`pyvista.PolyDataFilters.clip_closed_surface`, which seals the cut face.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# sphinx_gallery_thumbnail_number = 2

# %%
# Create a closed surface to inspect
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Start from a sphere and displace its points with Perlin noise to create a more
# organic watertight surface.

surface = pv.Sphere(theta_resolution=90, phi_resolution=90).triangulate()
noise = pv.perlin_noise(1.0, (2, 1, 3), (0, 0, 0))
surface['noise'] = np.array([noise.EvaluateFunction(point) for point in surface.points])
surface.points = (
    surface.points + surface.point_normals * (surface['noise'] * 0.18)[:, None]
)
surface


# %%
# Define an oblique cutting plane
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The same plane will be used for both the open clip and the closed-surface
# clip.

plane_origin = (0.1, 0.0, 0.0)
plane_normal = (1, 1, 0.3)
plane = pv.Plane(center=plane_origin, direction=plane_normal, i_size=3.5, j_size=3.5)

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
closed_clip = surface.clip_closed_surface(plane=plane)

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
