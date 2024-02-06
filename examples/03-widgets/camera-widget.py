"""
.. _camera_widget_example:

Camera Widget
~~~~~~~~~~~~

"""

# sphinx_gallery_start_ignore
# widgets do not work in interactive examples
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import vtk

import pyvista as pv

sphere = pv.Sphere()

# Renderers and one render window
plotter = pv.Plotter(window_size=[600, 300], shape=(1, 2))
plotter.add_mesh(sphere)
main_renderer = plotter.renderer

plotter.subplot(0, 1)
plotter.add_mesh(sphere)
camera_renderer = plotter.renderer
camera_renderer.InteractiveOff()

render_window = plotter.render_window
render_window.SetWindowName("camera_widget")

# An interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Camera widget and its representation
camera_representation = vtk.vtkCamera3DRepresentation()
camera_widget = vtk.vtkCamera3DWidget()
camera_widget.SetInteractor(render_window_interactor)
camera_widget.SetRepresentation(camera_representation)

# If you want to set the camera, do it before placing the widget
camera_representation.SetCamera(camera_renderer.GetActiveCamera())
# Placing widget is optional, if you do, camera will be moved toward bounds
# camera_representation.PlaceWidget(actor.GetBounds())

# Render
render_window_interactor.Initialize()
render_window.Render()
camera_widget.On()

# Begin mouse interaction
render_window_interactor.Start()
