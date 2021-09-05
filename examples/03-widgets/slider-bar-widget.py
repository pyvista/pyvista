"""
.. _slider_bar_widget_example:

Slider Bar Widget
~~~~~~~~~~~~~~~~~

The slider widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_slider_widget` and
:func:`pyvista.WidgetHelper.clear_slider_widgets` methods respectively.
This is one of the most versatile widgets as it can control a value that can
be used for just about anything.
"""
# sphinx_gallery_thumbnail_number = 1

##############################################################################
# One helper method we've added is the
# :func:`pyvista.WidgetHelper.add_mesh_threshold` method which leverages the
# slider widget to control a thresholding value.

import pyvista as pv
from pyvista import examples

mesh = examples.download_knee_full()

p = pv.Plotter()
p.add_mesh_threshold(mesh)
p.show()

###############################################################################
# After interacting with the scene, the threshold mesh is available as:
p.threshold_meshes

##############################################################################
# And here is a screen capture of a user interacting with this
#
# .. image:: ../../images/gifs/slider-widget-threshold.gif

###############################################################################
# Custom Callback
# +++++++++++++++
#
# Or you could leverage a custom callback function that takes a single value
# from the slider as its argument to do something like control the resolution
# of a mesh. Again note the use of the ``name`` argument in ``add_mesh``:

p = pv.Plotter()

def create_mesh(value):
    res = int(value)
    sphere = pv.Sphere(phi_resolution=res, theta_resolution=res)
    p.add_mesh(sphere, name='sphere', show_edges=True)
    return

p.add_slider_widget(create_mesh, [5, 100], title='Resolution')
p.show()

##############################################################################
# And here is a screen capture of a user interacting with this
#
# .. image:: ../../images/gifs/slider-widget-resolution.gif
