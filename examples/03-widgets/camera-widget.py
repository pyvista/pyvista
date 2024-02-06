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

# Renderers and one render window
plotter = pv.Plotter(window_size=[600, 300], shape=(1, 2))
plotter.add_mesh(sphere)
plotter.subplot(0, 1)
plotter.add_mesh(sphere)
camera3d_renderer = plotter.renderer

render_window = plotter.render_window

# An interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Camera widget and its representation
camera3d_representation = vtk.vtkCamera3DRepresentation()
camera3d_representation.SetCamera(camera3d_renderer.GetActiveCamera())
camera3d_widget = vtk.vtkCamera3DWidget()
camera3d_widget.SetInteractor(interactor)
camera3d_widget.SetRepresentation(camera3d_representation)

# If you want to set the camera, do it before placing the widget
# Placing widget is optional, if you do, camera will be moved toward bounds
# camera3d_representation.PlaceWidget(actor.GetBounds())
# plotter.add_camera3d_widget()

# Render
interactor.Initialize()
render_window.Render()
camera3d_widget.On()

# Begin mouse interaction
interactor.Start()
