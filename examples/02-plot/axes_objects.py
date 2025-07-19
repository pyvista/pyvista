"""
.. _axes_objects_example:

Axes Objects
~~~~~~~~~~~~

PyVista has many axes objects which can be used for plotting.
This example highlights many of these objects and shows how
to use them with related plotting methods.

"""

# sphinx_gallery_thumbnail_number = 7

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
# Cube Axes
# =========
# Show axes bounds as a cube with :class:`~pyvista.CubeAxesActor`.

mesh = examples.download_bunny_coarse()

pl = pv.Plotter()
pl.add_mesh(mesh)
axes = pv.CubeAxesActor(camera=pl.camera)
axes.bounds = mesh.bounds
pl.add_actor(axes)
pl.background_color = pv.Color('paraview')
pl.show()

# %%
# Adding the axes like this can be a bit cumbersome since the camera, bounds, and color
# must be set manually. Instead, use :meth:`~pyvista.Plotter.show_bounds` to add a
# :class:`~pyvista.CubeAxesActor` with pre-configured parameters.

pl = pv.Plotter()
pl.add_mesh(mesh)
pl.show_bounds()
pl.show()

# %%
# Alternatively, use :meth:`~pyvista.Plotter.show_grid`. This also adds a
# :class:`~pyvista.CubeAxesActor` to the plot but with different default options.

pl = pv.Plotter()
pl.add_mesh(mesh)
pl.show_grid()
pl.show()

# %%
# .. seealso::
#
#   :meth:`~pyvista.Plotter.remove_bounds_axes`
#   :meth:`~pyvista.Plotter.update_bounds_axes`
#   :ref:`bounds_example`

# %%
# Arrow Axes
# ==========
# Arrow-style axes include :class:`~pyvista.AxesActor`, :class:`~pyvista.AxesAssembly`,
# and :class:`~pyvista.AxesAssemblySymmetric`.
#
# ``AxesActor`` is primarily intended for use as an orientation widget
# (see next section), but can also be added to a plot as a normal actor.
# Use :meth:`~pyvista.Plotter.add_axes_at_origin` to add ``AxesActor``
# at the origin.

pl = pv.Plotter()
pl.add_mesh(mesh)
axes = pl.add_axes_at_origin()
pl.show()

# %%
# The axes are too large and should be scaled down. Transformations with `AxesActor`
# are possible, but with some caveats:
#
# - The bounds of ``AxesActor`` are hard-coded as ``+/- 1``, which makes it challenging
#   to configure the camera bounds for the plot.
# - The user matrix must be used for transformations (scale and position properties
#   do not work).
#
# Create new axes, disable its bounds, and apply a scaling :class:`~pyvista.Transform`.

trans = pv.Transform().scale(0.25)
axes = pv.AxesActor()
axes.UseBoundsOff()
axes.SetUserMatrix(pv.vtkmatrix_from_array(trans.matrix))

# %%
# Plot the axes with a mesh. Note that since the bounds of the axes are not used,
# the tip of the z-axis appears clipped, which is not ideal.

pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_actor(axes)
pl.show()

# %%
# Instead of using :class:`~pyvista.AxesActor`, :class:`~pyvista.AxesAssembly` is
# recommended for positioning axes in a scene.

axes = pv.AxesAssembly(scale=0.25)
pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_actor(axes)
pl.show()

# %%
# Alternatively, use :class:`~pyvista.AxesAssemblySymmetric` for adding
# symmetric axes to a scene.

axes = pv.AxesAssemblySymmetric(scale=0.25)
pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_actor(axes)
pl.show()

# %%
# Axes Widgets
# ============
# Any actor can also be used as an axes orientation widget.
# Here, we demonstrate using four separate axes widgets:
#
# #. Use :meth:`~pyvista.Plotter.add_axes` to add an arrow-style
#    orientation widget. The widget uses :class:`~pyvista.AxesActor`
#    by default.
# #. Use :meth:`~pyvista.Plotter.add_box_axes` to add a box-style
#    orientation widget.
# #. Use :meth:`~pyvista.Plotter.add_north_arrow_widget` to add a
#    north arrow orientation widget.
# #. Add :class:`~pyvista.AxesAssemblySymmetric` as a custom
#    orientation widget using :meth:`~pyvista.Plotter.add_orientation_widget`.

# Load a dataset
mesh = examples.load_airplane()

# Create a plotter with four linked views.
viewport = (0, 0, 0.5, 0.5)
pl = pv.Plotter(shape=(2, 2))
pl.link_views()

# Add arrow-style axes
pl.subplot(0, 0)
pl.add_mesh(mesh)
pl.add_axes(viewport=viewport)

# Add box-style axes
pl.subplot(0, 1)
pl.add_mesh(mesh)
pl.add_box_axes(viewport=viewport)

# Add north arrow
pl.subplot(1, 0)
pl.add_mesh(mesh)
pl.add_north_arrow_widget(viewport=viewport)

# Add symmetric arrow-style axes
pl.subplot(1, 1)
pl.add_mesh(mesh)
axes = pv.AxesAssemblySymmetric(label_size=25)
pl.add_orientation_widget(axes, viewport=viewport)

pl.show()

# %%
# Camera Orientation Widget
# -------------------------
# There is also a specialized camera widget which can added to a plot with
# :class:`~pyvista.Plotter.add_camera_orientation_widget`.

pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_camera_orientation_widget()
pl.show()

# %%
# .. tags:: plot
