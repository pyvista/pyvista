#!/usr/bin/env python
from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkImagingSources import vtkImageEllipsoidSource
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkImageMapper, vtkTextMapper

import pyvista as pv

VTK_DATA_ROOT = vtkGetDataRoot()

#
# display text over an image
#
pl = pv.Plotter()
ellipse = vtkImageEllipsoidSource()
mapImage = vtkImageMapper()
mapImage.SetInputConnection(ellipse.GetOutputPort())
mapImage.SetColorWindow(255)
mapImage.SetColorLevel(127.5)
img = vtkActor2D()
img.SetMapper(mapImage)
mapText = vtkTextMapper()
mapText.SetInput("Text Overlay")
mapText.GetTextProperty().SetFontSize(15)
mapText.GetTextProperty().SetColor(0, 1, 1)
mapText.GetTextProperty().BoldOn()
mapText.GetTextProperty().ShadowOn()
txt = vtkActor2D()
txt.SetMapper(mapText)
txt.SetPosition(138, 128)
pl.add_actor(img)
pl.add_actor(txt)
pl.show()
# --- end of script --
