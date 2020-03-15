"""
Project to a Plane
~~~~~~~~~~~~~~~~~~

:class:`pyvista.PolyData` surfaces and pointsets can easily be projected to a
plane defined by a normal and origin
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples


poly = examples.load_random_hills()
poly.plot()

###############################################################################
# Project that surface to a plane underneath the surface
origin = poly.center
origin[-1] -= poly.length / 3.0
projected = poly.project_points_to_plane(origin=origin)

# Display the results
p = pv.Plotter()
p.add_mesh(poly)
p.add_mesh(projected)
p.show()
