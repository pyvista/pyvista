"""
.. _camera_widget_example:

Camera Widget
~~~~~~~~~~~~~

"""

# sphinx_gallery_start_ignore
# widgets do not work in interactive examples
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import vtk

import pyvista as pv

sphere_source = pv.SphereSource()
cube_source = pv.CubeSource()
cube_source.center = (0.0, 0.0, 2.0)
source = vtk.vtkAppendPolyData()
source.AddInputConnection(sphere_source.GetOutputPort())
source.AddInputConnection(cube_source.GetOutputPort())
source.Update()

# Create mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(source.GetOutputPort())
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Renderers and one render window
main_renderer = vtk.vtkRenderer()
main_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
main_renderer.AddActor(actor)
main_renderer.SetBackground(0.7, 0.7, 1.0)

camera_renderer = vtk.vtkRenderer()
camera_renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
camera_renderer.InteractiveOff()
camera_renderer.AddActor(actor)
camera_renderer.SetBackground(0.8, 0.8, 1.0)

render_window = vtk.vtkRenderWindow()
render_window.SetSize(600, 300)
render_window.AddRenderer(main_renderer)
render_window.AddRenderer(camera_renderer)
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
camera_representation.PlaceWidget(actor.GetBounds())

# Render
render_window_interactor.Initialize()
render_window.Render()
camera_widget.On()

# Begin mouse interaction
render_window_interactor.Start()
