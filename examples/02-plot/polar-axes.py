import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCamera, vtkLight, vtkRenderWindow, vtkRenderWindowInteractor, vtkRenderer
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkRenderingAnnotation import vtkPolarAxesActor

# Load the teapot geometry using PyVista's built-in methods
filename = pv.examples.download_teapot(load=False)  # Download only, do not load
mesh = pv.read(filename)

# Apply normals to the geometry
normals = mesh.compute_normals()

# Create a PyVista plotter
plotter = pv.Plotter()

# Add the geometry actor (teapot) to the plotter
actor = plotter.add_mesh(normals, color=(0.5, 0.8, 0.3), show_edges=True)

# Create an outline filter using VTK and add it to the plotter
outline = vtkOutlineFilter()
outline.SetInputData(normals)
outline.Update()
outline_polydata = pv.wrap(outline.GetOutput())
plotter.add_mesh(outline_polydata, color="white")

# Set up the camera using VTK
camera = vtkCamera()
camera.SetClippingRange(1.0, 100.0)
camera.SetFocalPoint(0.0, 0.5, 0.0)
camera.SetPosition(5.0, 6.0, 14.0)

# Set up the light using VTK
light = vtkLight()
light.SetFocalPoint(0.21406, 1.5, 0.0)
light.SetPosition(7.0, 7.0, 4.0)

# Set up the polar axes using VTK and add it to the plotter
polaxes = vtkPolarAxesActor()
polaxes.SetBounds(normals.bounds)
polaxes.SetPole(0.5, 1.0, 3.0)
polaxes.SetMaximumRadius(3.0)
polaxes.SetMinimumAngle(-60.0)
polaxes.SetMaximumAngle(210.0)
polaxes.SetRequestedNumberOfRadialAxes(10)
polaxes.SetCamera(camera)
polaxes.SetPolarLabelFormat("%6.1f")
polaxes.GetLastRadialAxisProperty().SetColor(0.0, 1.0, 0.0)
polaxes.GetSecondaryRadialAxesProperty().SetColor(0.0, 0.0, 1.0)
polaxes.GetPolarArcsProperty().SetColor(1.0, 0.0, 0.0)
polaxes.GetSecondaryPolarArcsProperty().SetColor(1.0, 0.0, 1.0)
polaxes.GetPolarAxisProperty().SetColor(1.0, 0.5, 0.0)
polaxes.GetPolarAxisTitleTextProperty().SetColor(0.0, 0.0, 0.0)
polaxes.GetPolarAxisLabelTextProperty().SetColor(1.0, 1.0, 0.0)
polaxes.GetLastRadialAxisTextProperty().SetColor(0.0, 0.5, 0.0)
polaxes.GetSecondaryRadialAxesTextProperty().SetColor(0.0, 1.0, 1.0)
polaxes.SetScreenSize(9.0)

# Add the polar axes to the plotter
plotter.renderer.AddActor(polaxes)

# Set background color
plotter.set_background(0.8, 0.8, 0.8)

# Set the camera and light
plotter.camera = camera
plotter.add_light(light)

# Show the result
plotter.show(window_size=[600, 600])
