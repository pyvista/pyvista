#!/usr/bin/env python
from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkFiltersCore import vtkAssignAttribute, vtkThreshold
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkIONetCDF import vtkNetCDFCFReader
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
import vtkmodules.vtkRenderingFreeType
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

VTK_DATA_ROOT = vtkGetDataRoot()

# This test checks netCDF reader.  It uses the COARDS convention.
# Open the file.
reader = vtkNetCDFCFReader()
reader.SetFileName("tos_O1_2001-2002.nc")
# Set the arrays we want to load.
reader.UpdateMetaData()
reader.SetVariableArrayStatus("tos", 1)
reader.SetSphericalCoordinates(0)
reader.Update()

# Test unit field arrays
grid = reader.GetOutput()
tuarr = grid.GetFieldData().GetAbstractArray("time_units")
if not tuarr:
    print("Unable to retrieve time_units field array")
    exit(1)
tosuarr = grid.GetFieldData().GetAbstractArray("tos_units")
if not tosuarr:
    print("Unable to retrieve tos_units field array")
    exit(1)
if tosuarr.GetValue(0) != "K":
    print("tos_units is not K but " + tosuarr.GetValue(0))
    exit(1)

aa = vtkAssignAttribute()
aa.SetInputConnection(reader.GetOutputPort())
aa.Assign("tos", "SCALARS", "POINT_DATA")
thresh = vtkThreshold()
thresh.SetInputConnection(aa.GetOutputPort())
thresh.SetThresholdFunction(vtkThreshold.THRESHOLD_LOWER)
thresh.SetLowerThreshold(10000.0)

surface = vtkDataSetSurfaceFilter()
surface.SetInputConnection(thresh.GetOutputPort())
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(surface.GetOutputPort())
mapper.SetScalarRange(270, 310)
actor = vtkActor()
actor.SetMapper(mapper)
ren = vtkRenderer()
ren.AddActor(actor)
renWin = vtkRenderWindow()
renWin.SetSize(200, 200)
renWin.AddRenderer(ren)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
renWin.Render()
# # Setup a lookup table.
# vtkLookupTable lut
# lut SetTableRange 270 310
# lut SetHueRange 0.66 0.0
# lut SetRampToLinear
# # Make pretty colors
# vtkImageMapToColors map
# map SetInputConnection [asinine GetOutputPort]
# map SetLookupTable lut
# map SetOutputFormatToRGB
# # vtkImageViewer viewer
# # viewer SetInputConnection [map GetOutputPort]
# # viewer SetColorWindow 256
# # viewer SetColorLevel 127.5
# # viewer Render
# vtkImageViewer2 viewer
# viewer SetInputConnection [map GetOutputPort]
# viewer Render
# --- end of script --
