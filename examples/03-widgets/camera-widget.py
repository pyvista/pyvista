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

sphereSource = vtk.vtkSphereSource()
cubeSource = vtk.vtkCubeSource()
cubeSource.SetCenter(0.0, 0.0, 2.0)
source = vtk.vtkAppendPolyData()
source.AddInputConnection(sphereSource.GetOutputPort())
source.AddInputConnection(cubeSource.GetOutputPort())
source.Update()

# Create mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(source.GetOutputPort())
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Renderers and one render window
mainRenderer = vtk.vtkRenderer()
mainRenderer.SetViewport(0.0, 0.0, 0.5, 1.0)
mainRenderer.AddActor(actor)
mainRenderer.SetBackground(0.7, 0.7, 1.0)

cameraRenderer = vtk.vtkRenderer()
cameraRenderer.SetViewport(0.5, 0.0, 1.0, 1.0)
cameraRenderer.InteractiveOff()
cameraRenderer.AddActor(actor)
cameraRenderer.SetBackground(0.8, 0.8, 1.0)

renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(600, 300)
renderWindow.AddRenderer(mainRenderer)
renderWindow.AddRenderer(cameraRenderer)
renderWindow.SetWindowName("cameraWidget")

# An interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Camera widget and its representation
cameraRepresentation = vtk.vtkCamera3DRepresentation()
cameraWidget = vtk.vtkCamera3DWidget()
cameraWidget.SetInteractor(renderWindowInteractor)
cameraWidget.SetRepresentation(cameraRepresentation)

# If you want to set the camera, do it before placing the widget
cameraRepresentation.SetCamera(cameraRenderer.GetActiveCamera())
# Placing widget is optional, if you do, camera will be moved toward bounds
cameraRepresentation.PlaceWidget(actor.GetBounds())

# Render
renderWindowInteractor.Initialize()
renderWindow.Render()
cameraWidget.On()

# Begin mouse interaction
renderWindowInteractor.Start()
