"""Image Magnify."""

# Handle the arguments.
from __future__ import annotations

import sys

import vtk

if len(sys.argv) < 2:
    print('Required arguments: filename.png e.g. Gourds.png')
    sys.exit(1)

filename = sys.argv[1]

# Read the image.
reader = vtk.vtkPNGReader()
reader.SetFileName(filename)

# Increase the dimensions of the image.
magnify_filter = vtk.vtkImageMagnify()
magnify_filter.SetInputConnection(reader.GetOutputPort())
magnify_filter.SetMagnificationFactors(2, 1, 1)
magnify_filter.Update()

# Adjust the spacing of the magnified image. This will stretch the image.
change_filter = vtk.vtkImageChangeInformation()
change_filter.SetInputConnection(magnify_filter.GetOutputPort())
magnification_factors = magnify_filter.GetMagnificationFactors()
change_filter.SetSpacingScale(
    magnification_factors[0], magnification_factors[1], magnification_factors[2]
)

# Create actors for the original and magnified images.
original_actor = vtk.vtkImageActor()
original_actor.GetMapper().SetInputConnection(reader.GetOutputPort())

magnified_actor = vtk.vtkImageActor()
magnified_actor.GetMapper().SetInputConnection(change_filter.GetOutputPort())

# Define viewport ranges.
# (xmin, ymin, xmax, ymax)
original_viewport = [0.0, 0.0, 0.5, 1.0]
magnified_viewport = [0.5, 0.0, 1.0, 1.0]

# Setup renderers.
colors = vtk.vtkNamedColors()

original_renderer = vtk.vtkRenderer()
original_renderer.SetViewport(original_viewport)
original_renderer.AddActor(original_actor)
original_renderer.ResetCamera()
original_renderer.SetBackground(colors.GetColor3d('CornflowerBlue'))

magnified_renderer = vtk.vtkRenderer()
magnified_renderer.SetViewport(magnified_viewport)
magnified_renderer.AddActor(magnified_actor)
magnified_renderer.ResetCamera()
magnified_renderer.SetBackground(colors.GetColor3d('SteelBlue'))

# Setup render window.
render_window = vtk.vtkRenderWindow()
render_window.SetSize(600, 300)
render_window.AddRenderer(original_renderer)
render_window.AddRenderer(magnified_renderer)
render_window.SetWindowName('ImageMagnify')

# Setup render window interactor.
render_window_interactor = vtk.vtkRenderWindowInteractor()
style = vtk.vtkInteractorStyleImage()
render_window_interactor.SetInteractorStyle(style)

render_window_interactor.SetRenderWindow(render_window)

# Start interaction.
render_window.Render()
render_window_interactor.Initialize()
render_window_interactor.Start()
