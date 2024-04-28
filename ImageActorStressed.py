#!/usr/bin/env python
from vtkmodules.vtkImagingCore import vtkImageShiftScale

import pyvista as pv

gsOne = pv.ImageEllipsoidSource()
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
