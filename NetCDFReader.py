#!/usr/bin/env python
from vtkmodules.vtkFiltersCore import vtkAssignAttribute, vtkThreshold
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper
import vtkmodules.vtkRenderingFreeType
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

import pyvista as pv
from pyvista import examples

# This test checks netCDF reader.  It uses the COARDS convention.
# Open the file.
filename = examples.download_tos_O1_2001_2002(load=False)
reader = pv.NetCDFCFReader(filename)
# Set the arrays we want to load.
reader.reader.UpdateMetaData()
reader.reader.SetVariableArrayStatus("tos", 1)
reader.reader.SetSphericalCoordinates(0)
reader.reader.Update()

# Test unit field arrays
grid = reader.reader.GetOutput()
tuarr = grid.GetFieldData().GetAbstractArray("time_units")
tosuarr = grid.GetFieldData().GetAbstractArray("tos_units")

aa = vtkAssignAttribute()
aa.SetInputConnection(reader.reader.GetOutputPort())
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
actor = pv.Actor()
actor.SetMapper(mapper)
pl = pv.Plotter()
pl.add_actor(actor)
pl.show(cpos="xy")
