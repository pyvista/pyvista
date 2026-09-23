"""
.. _project_plane_example:

Project to a Plane
~~~~~~~~~~~~~~~~~~

Project a :class:`~pyvista.PolyData` surface or pointset to a plane.

Uses :meth:`~pyvista.PolyDataFilters.project_points_to_plane` with a normal
and origin.

"""

# sphinx_gallery_thumbnail_number = 2
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
