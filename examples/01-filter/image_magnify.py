"""Image Magnify."""

# Handle the arguments.
from __future__ import annotations

import vtk

import pyvista as pv
from pyvista import examples

filename = examples.download_gourds(load=False)

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


plotter = pv.Plotter(shape=(1, 2))
plotter.add_actor(original_actor)
plotter.camera_position = 'xy'
plotter.subplot(0, 1)
plotter.add_actor(magnified_actor)
plotter.camera_position = 'xy'
plotter.show()
