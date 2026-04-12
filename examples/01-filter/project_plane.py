"""
.. _project_plane_example:

Project to a Plane
~~~~~~~~~~~~~~~~~~

:class:`pyvista.PolyData` surfaces and pointsets can easily be projected to a
plane defined by a normal and origin using
:meth:`~pyvista.PolyDataFilters.project_points_to_plane`.
"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv
from pyvista import examples

poly = examples.load_random_hills()
poly.plot()

# %%
# Project that surface to a plane underneath the surface
origin = list(poly.center)
origin[-1] -= poly.length / 3.0
projected = poly.project_points_to_plane(origin=origin)

# Display the results
pl = pv.Plotter()
pl.add_mesh(poly)
pl.add_mesh(projected)
pl.show()
# %%
# .. tags:: filter
