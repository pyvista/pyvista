"""
.. _element_picking_example:

Picking elements of a mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrates how to pick different elements on meshes using
:func:`enable_element_picking() <pyvista.Plotter.enable_element_picking>`.

The different elements of a mesh are:

* Mesh: pick the entire mesh (equivalent to
  :func:`~pyvista.Plotter.enable_mesh_picking`.)
* Cell: pick a cell of the mesh (equivalent to
  :func:`~pyvista.Plotter.enable_cell_picking`.)
* Face: pick a single face of a cell on the mesh
* Edge: pick a single edge of a cell on the mesh
* Point: pick a single point on the mesh

These types are captured in the :class:`pyvista.plotting.opts.ElementType` enum class.

"""

# sphinx_gallery_thumbnail_number = 1
from __future__ import annotations

import pyvista as pv
from pyvista.plotting.opts import ElementType

# %%
# Pick Face on Voxel Cell
# +++++++++++++++++++++++
#
mesh = pv.Wavelet()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, pickable=True)
pl.enable_element_picking(mode=ElementType.FACE)

pl.camera_position = pv.CameraPosition(
    position=(13.523728057554308, 9.910583926360937, 11.827103195167833),
    focal_point=(2.229008884793069, -2.782397236304676, 6.84282248642347),
    viewup=(-0.17641568583704878, -0.21978122178947299, 0.9594653304520027),
)

pl.show(auto_close=False)

# Programmatically pick a face to make example look nice
try:
    width, height = pl.window_size
    pl.iren._mouse_right_button_press(419, 263)
    pl.iren._mouse_right_button_release()
except AttributeError:
    # ignore this section when manually closing the window
    pass

# %%
# Pick an Edge of a Cell
# ++++++++++++++++++++++
#
sphere = pv.Sphere()

pl = pv.Plotter()
pl.add_mesh(sphere, show_edges=True, pickable=True)
pl.enable_element_picking(mode=ElementType.EDGE)

pl.camera_position = pv.CameraPosition(
    position=(0.7896646029990011, 0.7520805261169909, 0.5148524767495051),
    focal_point=(-0.014748048334009667, -0.0257133671899262, 0.07194025085895145),
    viewup=(-0.26016740957025775, -0.2603941863919363, 0.9297891087180916),
)

pl.show(auto_close=False)

# Programmatically pick a face to make example look nice
try:
    width, height = pl.window_size
    pl.iren._mouse_right_button_press(480, 300)
    pl.iren._mouse_right_button_release()
except AttributeError:
    # ignore this section when manually closing the window
    pass
# %%
# .. tags:: plot
