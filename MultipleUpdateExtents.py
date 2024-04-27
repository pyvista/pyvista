#!/usr/bin/env python
from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkIOImage import vtkPNGReader
from vtkmodules.vtkImagingCore import vtkImageBlend, vtkImageClip, vtkImageShiftScale
from vtkmodules.vtkInteractionImage import vtkImageViewer

VTK_DATA_ROOT = vtkGetDataRoot()

# Simple example that used to crash when an UpdateExtent request
# from one algorithm is overwritten by a smaller UpdateExtent
# request from another algorithm.  The COMBINED_UPDATE_EXTENT
# key was added to vtkStreamingDemandDrivenPipeline to fix the
# bug that caused the crash.
# read an image that has an extent of 0 255 0 255 0 0
reader = vtkPNGReader()
reader.SetDataSpacing(0.8, 0.8, 1.5)
reader.SetFileName(VTK_DATA_ROOT + "/Data/fullhead15.png")
# Uncomment this to make the crash disappear
# reader Update
# clip the image down to 0 127 0 127 0 0
clip = vtkImageClip()
clip.SetInputConnection(reader.GetOutputPort())
clip.SetOutputWholeExtent(0, 127, 0, 127, 0, 0)
# darken the background
darken = vtkImageShiftScale()
darken.SetInputConnection(reader.GetOutputPort())
darken.SetScale(0.2)
# do an operation on the clipped and unclipped data
blend = vtkImageBlend()
blend.SetInputConnection(darken.GetOutputPort())
blend.AddInputConnection(clip.GetOutputPort())
viewer = vtkImageViewer()
viewer.SetInputConnection(blend.GetOutputPort())
viewer.SetColorWindow(2000)
viewer.SetColorLevel(1000)
viewer.Render()
# --- end of script --
