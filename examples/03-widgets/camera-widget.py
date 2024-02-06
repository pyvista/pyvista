"""
.. _camera3d_widget_example:

Camera Widget
~~~~~~~~~~~~~

"""

# sphinx_gallery_start_ignore
# widgets do not work in interactive examples
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import vtk

import pyvista as pv

sphere = pv.Sphere()

plotter = pv.Plotter(window_size=[600, 300], shape=(1, 2))
plotter.add_mesh(sphere)
plotter.subplot(0, 1)
plotter.add_mesh(sphere)

camera3d_renderer = plotter.renderer
render_window = plotter.render_window
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(render_window)
plotter.add_camera3d_widget(iren)
iren.Initialize()
render_window.Render()
iren.Start()
