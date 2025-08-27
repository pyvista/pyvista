"""
.. _linked_views_example:

Linked Views in Subplots
~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to create linked views in PyVista subplots
using :func:`~pyvista.Plotter.link_views`, where camera movements
in one view are synchronized with other views. This is particularly useful when comparing
different versions or representations of the same model.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples

pv.set_plot_theme('document')

# download mesh
mesh = examples.download_cow()

decimated = mesh.decimate_boundary(target_reduction=0.75)

p = pv.Plotter(shape=(1, 2), border=False)
p.subplot(0, 0)
p.add_text('Original mesh', font_size=24)
p.add_mesh(mesh, show_edges=True, color=True)
p.subplot(0, 1)
p.add_text('Decimated version', font_size=24)
p.add_mesh(decimated, color=True, show_edges=True)

p.link_views()  # link all the views
# Set a camera position to all linked views
p.camera_position = pv.CameraPosition(position=(15, 5, 0), focal_point=(0, 0, 0), viewup=(0, 1, 0))

p.open_gif('linked.gif')
# Update camera and write a frame for each updated position
nframe = 15
for i in range(nframe):
    p.camera_position = pv.CameraPosition(
        position=(15 * np.cos(i * np.pi / 45.0), 5.0, 15 * np.sin(i * np.pi / 45.0)),
        focal_point=(0, 0, 0),
        viewup=(0, 1, 0),
    )
    p.write_frame()

# Close movie and delete object
p.close()
# %%
# .. tags:: plot
