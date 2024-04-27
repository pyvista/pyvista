#!/usr/bin/env python
from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkImagingCore import vtkImageShiftScale
from vtkmodules.vtkImagingSources import vtkImageEllipsoidSource
from vtkmodules.vtkRenderingCore import (
    vtkImageActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

VTK_DATA_ROOT = vtkGetDataRoot()

# Create the RenderWindow, Renderer and both Actors
ren1 = vtkRenderer()
renWin = vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
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
ssOne = vtkImageShiftScale()
ssOne.SetInputConnection(gsOne.GetOutputPort())
ssOne.SetOutputScalarTypeToUnsignedChar()
ssOne.SetShift(0)
ssOne.SetScale(1)
ssOne.UpdateWholeExtent()
iaOne = vtkImageActor()
iaOne.GetMapper().SetInputConnection(ssOne.GetOutputPort())
ren1.AddActor(iaOne)
# The second one tests a really large texture
gsTwo = vtkImageEllipsoidSource()
gsTwo.SetWholeExtent(1000, 8999, 1000, 8999, 0, 0)
gsTwo.SetCenter(4000, 4000, 0)
gsTwo.SetRadius(1800, 1800, 0)
gsTwo.SetInValue(250)
gsTwo.SetOutValue(150)
gsTwo.SetOutputScalarTypeToUnsignedChar()
ssTwo = vtkImageShiftScale()
ssTwo.SetInputConnection(gsTwo.GetOutputPort())
ssTwo.SetOutputScalarTypeToUnsignedChar()
ssTwo.SetShift(0)
ssTwo.SetScale(1)
ssTwo.UpdateWholeExtent()
iaTwo = vtkImageActor()
iaTwo.GetMapper().SetInputConnection(ssTwo.GetOutputPort())
iaTwo.SetScale(0.1, 0.1, 1.0)
iaTwo.AddPosition(1000, 1000, 0)
ren1.AddActor(iaTwo)
# The third one will test changing input and a
# power of two texture
gsThree = vtkImageEllipsoidSource()
gsThree.SetWholeExtent(0, 511, 2000, 2511, 0, 0)
gsThree.SetCenter(255, 2255, 0)
gsThree.SetRadius(100, 200, 0)
gsThree.SetInValue(250)
gsThree.SetOutValue(0)
gsThree.SetOutputScalarTypeToUnsignedChar()
ssThree = vtkImageShiftScale()
ssThree.SetInputConnection(gsThree.GetOutputPort())
ssThree.SetOutputScalarTypeToUnsignedChar()
ssThree.SetShift(0)
ssThree.SetScale(1)
ssThree.UpdateWholeExtent()
iaThree = vtkImageActor()
iaThree.GetMapper().SetInputConnection(ssThree.GetOutputPort())
ren1.AddActor(iaThree)
# Same as first one, but the display extents
# represent contiguous section of memory that
# are powers of two
gsFour = vtkImageEllipsoidSource()
gsFour.SetWholeExtent(2000, 2511, 0, 511, 0, 0)
gsFour.SetCenter(2255, 255, 0)
gsFour.SetRadius(130, 130, 0)
gsFour.SetInValue(40)
gsFour.SetOutValue(190)
gsFour.SetOutputScalarTypeToUnsignedChar()
ssFour = vtkImageShiftScale()
ssFour.SetInputConnection(gsFour.GetOutputPort())
ssFour.SetOutputScalarTypeToUnsignedChar()
ssFour.SetShift(0)
ssFour.SetScale(1)
ssFour.UpdateWholeExtent()
iaFour = vtkImageActor()
iaFour.GetMapper().SetInputConnection(ssFour.GetOutputPort())
ren1.AddActor(iaFour)
# Same as previous one, but the display extents
# represent contiguous section of memory that
# are not powers of two
gsFive = vtkImageEllipsoidSource()
gsFive.SetWholeExtent(1200, 1712, 0, 512, 0, 0)
gsFive.SetCenter(1456, 256, 0)
gsFive.SetRadius(130, 180, 0)
gsFive.SetInValue(190)
gsFive.SetOutValue(100)
gsFive.SetOutputScalarTypeToUnsignedChar()
ssFive = vtkImageShiftScale()
ssFive.SetInputConnection(gsFive.GetOutputPort())
ssFive.SetOutputScalarTypeToUnsignedChar()
ssFive.SetShift(0)
ssFive.SetScale(1)
ssFive.UpdateWholeExtent()
iaFive = vtkImageActor()
iaFive.GetMapper().SetInputConnection(ssFive.GetOutputPort())
ren1.AddActor(iaFive)
ren1.SetBackground(0.1, 0.2, 0.4)
renWin.SetSize(400, 400)
# render the image
ren1.ResetCamera()
ren1.GetActiveCamera().Zoom(1.5)
renWin.Render()
# prevent the tk window from showing up then start the event loop
renWin.Render()
iaOne.SetDisplayExtent(200, 999, 200, 999, 0, 0)
iaFour.SetDisplayExtent(2000, 2511, 0, 300, 0, 0)
iaFive.SetDisplayExtent(1200, 1712, 0, 300, 0, 0)
gsThree.SetRadius(120, 120, 0)
renWin.Render()
iaOne.SetDisplayExtent(0, 799, 0, 799, 0, 0)
iaFour.SetDisplayExtent(2000, 2511, 200, 500, 0, 0)
iaFive.SetDisplayExtent(1200, 1712, 200, 500, 0, 0)
gsThree.SetRadius(150, 150, 0)
renWin.Render()
# --- end of script --
