"""
.. _plot_over_circular_arc_example:

Plot Scalars Over a Circular Arc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpolate the scalars of a dataset over a circular arc
using :meth:`~pyvista.DataSetFilters.plot_over_circular_arc_normal`.

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
# with :func:`~pyvista.CircularArcFromNormal`.
arc = pv.CircularArcFromNormal(
    center=center, resolution=100, normal=normal, polar=polar, angle=angle
)

pl = pv.Plotter()
pl.add_mesh(mesh, style='wireframe', color='w')
pl.add_mesh(arc, color='b')
a = arc.points[0]
b = arc.points[-1]
pl.add_point_labels([a, b], ['A', 'B'], font_size=48, point_color='red', text_color='red')
pl.show()

# %%
# Run the filter and produce a line plot.
mesh.plot_over_circular_arc_normal(
    center=center, resolution=100, normal=normal, polar=polar, angle=angle, scalars='height'
)
# %%
# .. tags:: plot
