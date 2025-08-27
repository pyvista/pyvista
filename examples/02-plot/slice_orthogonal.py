"""
.. _slice_orthogonal_example:

Orthogonal Slices
~~~~~~~~~~~~~~~~~

View three orthogonal slices from a mesh.

Use the :func:`pyvista.DataObjectFilters.slice_orthogonal` filter to create these
slices simultaneously.
"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv
from pyvista import examples

mesh = examples.download_embryo()
mesh.bounds

# %%
# Create three slices. Easily control their locations with the ``x``, ``y``,
# and ``z`` arguments.
slices = mesh.slice_orthogonal(x=100, z=75)

# %%
cpos = pv.CameraPosition(
    position=(540.9115516905358, -617.1912234499737, 180.5084853429126),
    focal_point=(128.31920055083387, 126.4977720785509, 111.77682599082095),
    viewup=(-0.1065160140819035, 0.032750075477590124, 0.9937714884722322),
)
dargs = dict(cmap='gist_ncar_r')

p = pv.Plotter()
p.add_mesh(slices, **dargs)
p.show_grid()
p.show(cpos=cpos)


# %%

p = pv.Plotter(shape=(2, 2))
# XYZ - show 3D scene first
p.subplot(1, 1)
p.add_mesh(slices, **dargs)
p.show_grid()
p.camera_position = cpos
# XY
p.subplot(0, 0)
p.add_mesh(slices, **dargs)
p.show_grid()
p.camera_position = 'xy'
p.enable_parallel_projection()
# ZY
p.subplot(0, 1)
p.add_mesh(slices, **dargs)
p.show_grid()
p.camera_position = 'zy'
p.enable_parallel_projection()
# XZ
p.subplot(1, 0)
p.add_mesh(slices, **dargs)
p.show_grid()
p.camera_position = 'xz'
p.enable_parallel_projection()

p.show()
# %%
# .. tags:: plot
