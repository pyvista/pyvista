#!/usr/bin/env python
from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkImagingCore import vtkImageShiftScale
from vtkmodules.vtkImagingSources import vtkImageEllipsoidSource

import pyvista as pv

VTK_DATA_ROOT = vtkGetDataRoot()

# First one tests the changing display extent without
# changing the size of the display extent (so it
# reuses a texture, but not a contiguous one)
gsOne = vtkImageEllipsoidSource()
gsOne.SetWholeExtent(0, 999, 0, 999, 0, 0)
gsOne.SetCenter(500, 500, 0)
gsOne.SetRadius(300, 400, 0)
gsOne.SetInValue(0)
gsOne.SetOutValue(255)
gsOne.SetOutputScalarTypeToUnsignedChar()
gsOne.Update()
ssOne = vtkImageShiftScale()
ssOne.SetInputConnection(gsOne.GetOutputPort())
ssOne.SetOutputScalarTypeToUnsignedChar()
ssOne.SetShift(100)
ssOne.SetScale(1)
ssOne.UpdateWholeExtent()
ssOne.Update()
source = pv.wrap(gsOne.GetOutput())
output = pv.wrap(ssOne.GetOutput())
source.plot(cpos="xy")  # type: ignore[union-attr]
output.plot(cpos="xy")  # type: ignore[union-attr]
