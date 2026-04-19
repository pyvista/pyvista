"""
.. _create_circular_arc_example:

Create Circular Arcs
~~~~~~~~~~~~~~~~~~~~

Generate arc geometry with :func:`pyvista.CircularArc` and
:func:`pyvista.CircularArcFromNormal`.
"""

from __future__ import annotations

import pyvista as pv

# sphinx_gallery_thumbnail_number = 2

# %%
# Create an arc from two endpoints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The first arc is defined directly from its endpoints and center.

arc = pv.CircularArc(
    pointa=(1, 0, 0),
    pointb=(0, 1, 0),
    center=(0, 0, 0),
    resolution=60,
)

pl = pv.Plotter()
pl.add_mesh(arc.tube(radius=0.03), color='royalblue')
pl.add_points(
    arc.points[[0, -1]], color='tomato', point_size=18, render_points_as_spheres=True
)
pl.show()


# %%
# Create an arc from a normal and angle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use this form when you already know the plane the arc lies in.

arc_from_normal = pv.CircularArcFromNormal(
    center=(0, 0, 0),
    polar=(1, 0, 0),
    normal=(0, 0, 1),
    angle=120,
    resolution=60,
)

pl = pv.Plotter()
pl.add_mesh(arc_from_normal.tube(radius=0.03), color='seagreen')
pl.add_mesh(pv.Circle(radius=1.0), color='lightgray', style='wireframe')
pl.show()
# %%
# .. tags:: load
