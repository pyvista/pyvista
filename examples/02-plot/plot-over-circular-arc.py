"""
Plot Scalars Over a Circular Arc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpolate the scalars of a dataset over a circular arc.

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv
from pyvista import examples

# sphinx_gallery_start_ignore
# labels are not supported in vtk-js
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Volumetric Mesh
# +++++++++++++++
#
# Add the height scalars to a uniform 3D mesh.
mesh = examples.load_uniform()
mesh['height'] = mesh.points[:, 2]

# Make two points at the bounds of the mesh and one at the center to
# construct a circular arc.
normal = [0, 1, 0]
bnds = mesh.bounds
polar = [bnds.x_min, bnds.y_min, bnds.z_max]
center = [bnds.x_min, bnds.y_min, bnds.z_min]
angle = 90.0

# Preview how this circular arc intersects this mesh
arc = pv.CircularArcFromNormal(center, 100, normal, polar, angle)

p = pv.Plotter()
p.add_mesh(mesh, style='wireframe', color='w')
p.add_mesh(arc, color='b')
a = arc.points[0]
b = arc.points[-1]
p.add_point_labels([a, b], ['A', 'B'], font_size=48, point_color='red', text_color='red')
p.show()

# %%
# Run the filter and produce a line plot.
mesh.plot_over_circular_arc_normal(center, 100, normal, polar, angle, 'height')
# %%
# .. tags:: plot
