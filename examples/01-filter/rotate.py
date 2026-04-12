"""
.. _rotate_example:

Rotations
~~~~~~~~~

Rotations of a mesh about its axes using
:meth:`~pyvista.DataObjectFilters.rotate_x`,
:meth:`~pyvista.DataObjectFilters.rotate_y`, and
:meth:`~pyvista.DataObjectFilters.rotate_z`.
In this model, the x axis is from the left to right;
the y axis is from bottom to top; and the z axis emerges from the
image. The camera location is the same in all four images.

"""

# sphinx_gallery_thumbnail_number = 3
from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
# Define camera position and axes
# +++++++++++++++++++++++++++++++
#
# Define camera position and axes. Setting axes origin to ``(3.0, 3.0, 3.0)``.

mesh = examples.download_cow()
mesh.points /= 1.5  # scale the mesh

cpos = [
    (30.0, 30.0, 30.0),  # position
    (5.0, 5.0, 5.0),  # focal point
    (0.0, 1.0, 0.0),  # view up
]


axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
axes.origin = (3.0, 3.0, 3.0)

# %%
# Original Mesh
# +++++++++++++
#
# Plot original mesh. Add axes actor to Plotter.

pl = pv.Plotter()

pl.add_text('Mesh', font_size=24)
pl.add_actor(axes.actor)
pl.add_mesh(mesh)

pl.show(cpos=cpos)

# %%
# Rotation about the x axis
# +++++++++++++++++++++++++
#
# Plot the mesh rotated about the x axis every 60 degrees.
# Add the axes actor to the Plotter and set the axes origin to the point of rotation.

pl = pv.Plotter()

pl.add_text('X-Axis Rotation', font_size=24)
pl.add_actor(axes.actor)

for i in range(6):
    rot = mesh.rotate_x(60 * i, point=axes.origin, inplace=False)
    pl.add_mesh(rot)

pl.show(cpos=cpos)

# %%
# Rotation about the y axis
# +++++++++++++++++++++++++
#
# Plot the mesh rotated about the y axis every 60 degrees.
# Add the axes actor to the Plotter and set the axes origin to the point of rotation.

pl = pv.Plotter()

pl.add_text('Y-Axis Rotation', font_size=24)
pl.add_actor(axes.actor)

for i in range(6):
    rot = mesh.rotate_y(60 * i, point=axes.origin, inplace=False)
    pl.add_mesh(rot)

pl.show(cpos=cpos)

# %%
# Rotation about the z axis
# +++++++++++++++++++++++++
#
# Plot the mesh rotated about the z axis every 60 degrees.
# Add axes actor to the Plotter and set the axes origin to the point of rotation.

pl = pv.Plotter()

pl.add_text('Z-Axis Rotation', font_size=24)
pl.add_actor(axes.actor)

for i in range(6):
    rot = mesh.rotate_z(60 * i, point=axes.origin, inplace=False)
    pl.add_mesh(rot)

pl.show(cpos=cpos)

# %%
# Rotation about a custom vector
# ++++++++++++++++++++++++++++++
#
# Plot the mesh rotated about a custom vector every 60 degrees.
# Add the axes actor to the Plotter and set axes origin to the point of rotation.

pl = pv.Plotter()

pl.add_text('Custom Vector Rotation', font_size=24)
pl.add_actor(axes.actor)
for i in range(6):
    rot = mesh.copy()
    rot.rotate_vector(vector=(1, 1, 1), angle=60 * i, point=axes.origin)
    pl.add_mesh(rot)

pl.show(cpos=cpos)
# %%
# .. tags:: filter
