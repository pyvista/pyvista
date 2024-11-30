"""Image Gradient"""
#!/usr/bin/env python

# noinspection PyUnresolvedReferences
from __future__ import annotations

import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkImagingColor import vtkImageHSVToRGB
from vtkmodules.vtkImagingCore import vtkImageConstantPad
from vtkmodules.vtkImagingCore import vtkImageExtractComponents
from vtkmodules.vtkImagingCore import vtkImageMagnify
from vtkmodules.vtkImagingGeneral import vtkImageEuclideanToPolar
from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
from vtkmodules.vtkImagingGeneral import vtkImageGradient
from vtkmodules.vtkInteractionImage import vtkImageViewer
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor

import pyvista as pv
from pyvista import examples

# noinspection PyUnresolvedReferences


file_name = examples.downloads.download_full_head(load=False)
colors = vtkNamedColors()

# Read the CT data of the human head.
reader = pv.get_reader(file_name)
image_data = reader.read()
image_data['MetaImage'] = image_data['MetaImage'].astype(np.float64)

# Magnify the image.
magnify = vtkImageMagnify()
magnify.SetInputConnection(cast.GetOutputPort())  # noqa: F821
magnify.SetMagnificationFactors(2, 2, 1)
magnify.InterpolateOn()

# Smooth the data.
# Remove high frequency artifacts due to linear interpolation.
smooth = vtkImageGaussianSmooth()
smooth.SetInputConnection(magnify.GetOutputPort())
smooth.SetDimensionality(2)
smooth.SetStandardDeviations(1.5, 1.5, 0.0)
smooth.SetRadiusFactors(2.01, 2.01, 0.0)

# Compute the 2D gradient.
gradient = vtkImageGradient()
gradient.SetInputConnection(smooth.GetOutputPort())
gradient.SetDimensionality(2)

# Convert the data to polar coordinates.
# The image magnitude is mapped into saturation value,
# whilst the gradient direction is mapped into hue value.
polar = vtkImageEuclideanToPolar()
polar.SetInputConnection(gradient.GetOutputPort())
polar.SetThetaMaximum(255.0)

# Add a third component to the data.
# This is needed since the gradient filter only generates two components,
#  and we need three components to represent color.
pad = vtkImageConstantPad()
pad.SetInputConnection(polar.GetOutputPort())
pad.SetOutputNumberOfScalarComponents(3)
pad.SetConstant(200.0)

# At this point we have Hue, Value, Saturation.
# Permute components so saturation will be constant.
# Re-arrange components into HSV order.
permute = vtkImageExtractComponents()
permute.SetInputConnection(pad.GetOutputPort())
permute.SetComponents(0, 2, 1)

# Convert back into RGB values.
rgb = vtkImageHSVToRGB()
rgb.SetInputConnection(permute.GetOutputPort())
rgb.SetMaximum(255.0)

# Set up a viewer for the image.
# Note that vtkImageViewer and vtkImageViewer2 are convenience wrappers around
# vtkActor2D, vtkImageMapper, vtkRenderer, and vtkRenderWindow.
# So all that needs to be supplied is the interactor.
viewer = vtkImageViewer()
viewer.SetInputConnection(rgb.GetOutputPort())
viewer.SetZSlice(22)
viewer.SetColorWindow(255.0)
viewer.SetColorLevel(127.0)
viewer.GetRenderWindow().SetSize(512, 512)
viewer.GetRenderer().SetBackground(colors.GetColor3d('Silver'))
viewer.GetRenderWindow().SetWindowName('ImageGradient')

# Create the RenderWindowInteractor.
iren = vtkRenderWindowInteractor()
viewer.SetupInteractor(iren)
viewer.Render()

plotter = pv.Plotter()
plotter.iren = iren
plotter = pv.Plotter()
plotter.iren = iren
plotter.show()
