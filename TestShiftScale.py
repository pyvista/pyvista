#!/usr/bin/env python
from pathlib import Path

from vtkmodules.vtkIOImage import vtkImageReader
from vtkmodules.vtkImagingCore import vtkImageMagnify, vtkImageShiftScale
from vtkmodules.vtkInteractionImage import vtkImageViewer
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingFreeType
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

from pyvista import examples

filename = examples.download_headsq(load=False)

# Shift and scale an image (in that order)
# This filter is useful for converting to a lower precision data type.
reader = vtkImageReader()
reader.GetExecutive().SetReleaseDataFlag(0, 0)
reader.SetDataByteOrderToLittleEndian()
reader.SetDataExtent(0, 63, 0, 63, 1, 93)
reader.SetFilePrefix(str(Path(filename).parent) + "/quarter")
reader.SetDataMask(0x7FFF)
shiftScale = vtkImageShiftScale()
shiftScale.SetInputConnection(reader.GetOutputPort())
shiftScale.SetShift(0)
shiftScale.SetScale(0.5)
shiftScale.SetOutputScalarTypeToDouble()
shiftScale2 = vtkImageShiftScale()
shiftScale2.SetInputConnection(shiftScale.GetOutputPort())
shiftScale2.SetShift(0)
shiftScale2.SetScale(2.0)
mag = vtkImageMagnify()
mag.SetInputConnection(shiftScale2.GetOutputPort())
mag.SetMagnificationFactors(4, 4, 1)
mag.InterpolateOff()
viewer = vtkImageViewer()
viewer.SetInputConnection(mag.GetOutputPort())
viewer.SetColorWindow(1024)
viewer.SetColorLevel(1024)
viewer.Render()

# make interface
# skipping source
# vtkPNMWriter w
# w SetFileName "D:/vtknew/vtk/graphics/examplesTcl/mace2.ppm"
# w SetInputConnection [shiftScale GetOutputPort]
# w Write
# --- end of script --
