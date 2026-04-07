"""
.. _shader_customization_example:

Custom Shader Effects
~~~~~~~~~~~~~~~~~~~~~

Demonstrates shader-based rendering customization in PyVista, including
custom point sprite shapes and Maximum Intensity Projection (MIP).

Both features use VTK's GLSL shader replacement API under the hood but
target different shader stages (fragment and vertex, respectively), so
they compose cleanly on the same actor.
"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Point Sprite Shapes
# ~~~~~~~~~~~~~~~~~~~
# By default, VTK renders points as squares. PyVista provides several
# built-in point sprite shapes that replace the square with a custom
# shape via a GLSL fragment shader.
#
# This only works with ``style='points'`` and
# ``render_points_as_spheres=False``.

rng = np.random.default_rng(42)
points = rng.random((1000, 3))
cloud = pv.PolyData(points)
cloud['elevation'] = cloud.points[:, 2]

shapes = ['circle', 'triangle', 'hexagon', 'diamond', 'asterisk', 'star']
pl = pv.Plotter(shape=(2, 3))

for i, shape in enumerate(shapes):
    pl.subplot(i // 3, i % 3)
    actor = pl.add_mesh(
        cloud,
        scalars='elevation',
        style='points',
        render_points_as_spheres=False,
        point_size=25,
        show_scalar_bar=False,
    )
    actor.set_point_sprite_shape(shape)
    pl.add_text(shape, font_size=12)

pl.link_views()
pl.show()


# %%
# Maximum Intensity Projection (MIP)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MIP reorders vertex depth so that points with higher scalar values
# are rendered in front of points with lower values, regardless of
# their actual distance from the camera. This is useful for visualizing
# dense point clouds where high-value features would otherwise be
# hidden behind lower-value points closer to the viewer.
#
# The technique works by replacing the z-coordinate in clip space with
# the negated, normalized scalar value in a custom vertex shader.
#
# .. note::
#    MIP does not work correctly with ``opacity < 1`` unless depth
#    peeling is enabled via :func:`pyvista.Plotter.enable_depth_peeling`.

from pyvista import examples

vol = examples.download_knee_full()

sample_n = 50000
rng = np.random.default_rng(0)
indices = rng.integers(0, vol.n_points, sample_n)
pts = vol.extract_points(indices, adjacent_cells=False, include_cells=False)

display = dict(
    cmap='jet', clim=[0, 100], style='points', point_size=5, show_scalar_bar=False
)

pl = pv.Plotter(shape=(1, 2))
pl.enable_parallel_projection()

pl.subplot(0, 0)
pl.add_mesh(pts, **display)
pl.add_text('Normal Projection', font_size=12)

pl.subplot(0, 1)
mip_actor = pl.add_mesh(pts, **display)
mip_actor.enable_maximum_intensity_projection()
pl.add_text('Maximum Intensity Projection', font_size=12)

pl.link_views()
pl.show()


# %%
# Combining MIP with Point Sprites
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Because MIP modifies the vertex shader and point sprites modify the
# fragment shader, both features can be used simultaneously on the
# same actor. Here we combine MIP with circular point sprites for
# a cleaner visualization.

combined_display = dict(
    cmap='jet', clim=[0, 100], style='points', point_size=15, show_scalar_bar=False
)

pl = pv.Plotter(shape=(1, 2))
pl.enable_parallel_projection()

pl.subplot(0, 0)
actor_normal = pl.add_mesh(
    pts,
    render_points_as_spheres=False,
    **combined_display,
)
actor_normal.enable_maximum_intensity_projection()
pl.add_text('MIP (squares)', font_size=12)

pl.subplot(0, 1)
actor_combined = pl.add_mesh(
    pts,
    render_points_as_spheres=False,
    **combined_display,
)
actor_combined.enable_maximum_intensity_projection()
actor_combined.set_point_sprite_shape('circle')
pl.add_text('MIP + circle sprites', font_size=12)

pl.link_views()
pl.show()
