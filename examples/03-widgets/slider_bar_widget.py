"""
.. _slider_bar_widget_example:

Slider Bar Widget
~~~~~~~~~~~~~~~~~

The slider widget can be enabled and disabled by the
:func:`pyvista.Plotter.add_slider_widget` and
:func:`pyvista.Plotter.clear_slider_widgets` methods respectively.
This is one of the most versatile widgets as it can control a value that can
be used for just about anything.
"""

# sphinx_gallery_start_ignore
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

# widgets do not work in interactive examples
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# sphinx_gallery_thumbnail_number = 1

# %%
# One helper method we've added is the
# :func:`pyvista.Plotter.add_mesh_threshold` method which leverages the
# slider widget to control a thresholding value.

import pyvista as pv
from pyvista import examples

mesh = examples.download_knee_full()

pl = pv.Plotter()
pl.add_mesh_threshold(mesh)
pl.show()

# %%
# After interacting with the scene, the threshold mesh is available as:
pl.threshold_meshes

# %%
# And here is a screen capture of a user interacting with this
#
# .. image:: ../../images/gifs/slider-widget-threshold.gif

# %%
# Custom Callback
# +++++++++++++++
#
# Or you could leverage a custom callback function that takes a single value
# from the slider as its argument to do something like control the resolution
# of a mesh. Again note the use of the ``name`` argument in ``add_mesh``:

pl = pv.Plotter()


def create_mesh(value):
    res = int(value)
    sphere = pv.Sphere(phi_resolution=res, theta_resolution=res)
    pl.add_mesh(sphere, name='sphere', show_edges=True)


pl.add_slider_widget(create_mesh, [5, 100], title='Resolution')
pl.show()

# %%
# And here is a screen capture of a user interacting with this
#
# .. image:: ../../images/gifs/slider-widget-resolution.gif
#
# %%
# .. tags:: widgets
