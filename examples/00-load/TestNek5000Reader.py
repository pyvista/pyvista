"""Test the Nek5000 reader with a simple example."""

#!/usr/bin/env python
from __future__ import annotations

import vtk
from vtk.util.misc import vtkGetDataRoot

VTK_DATA_ROOT = vtkGetDataRoot()

# Create the RenderWindow, Renderer and both Actors
#
ren1 = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
# create pipeline
#
reader = vtk.vtkNek5000Reader()
reader.SetFileName('' + str(VTK_DATA_ROOT) + '/Data/nek5000/eddy_uv/eddy_uv.nek5000')

# Pressure has 1 components and 16384 tuples
#   comp0 range: (-3.599372625350952, 1.3908110857009888)
var_name = 'Pressure'
pres_range = [-3.599372625350952, 1.3908110857009888]

lut = vtk.vtkLookupTable()
lut.SetHueRange(0.66, 0.0)
lut.SetNumberOfTableValues(256)
lut.Build()

lut.SetTableRange(pres_range[0], pres_range[1])
lut.Build()

geom = vtk.vtkGeometryFilter()
geom.SetInputConnection(reader.GetOutputPort(0))

mapper1 = vtk.vtkPolyDataMapper()
mapper1.SetInputConnection(geom.GetOutputPort(0))
mapper1.ScalarVisibilityOn()
mapper1.SetScalarModeToUsePointFieldData()

mapper1.SelectColorArray(var_name)
mapper1.SetLookupTable(lut)
mapper1.UseLookupTableScalarRangeOn()

actor1 = vtk.vtkActor()
actor1.SetMapper(mapper1)

ren1.AddActor(actor1)

renWin.SetSize(200, 200)
ren1.ResetCamera()
ren1.ResetCameraClippingRange()
ren1.SetBackground(0.2, 0.3, 0.4)
renWin.Render()
# render the image
#
# prevent the tk window from showing up then start the event loop
# --- end of script --
