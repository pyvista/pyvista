"""
.. _camera3d_widget_example:

Camera Widget
~~~~~~~~~~~~~

"""

# sphinx_gallery_start_ignore
# widgets do not work in interactive examples
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore


import pyvista as pv

sphere = pv.Sphere()

plotter = pv.Plotter(window_size=[600, 300], shape=(1, 2))
plotter.add_mesh(sphere)
plotter.subplot(0, 1)
plotter.add_mesh(sphere)
plotter.add_camera3d_widget()
plotter.show(cpos=plotter.camera_position)
