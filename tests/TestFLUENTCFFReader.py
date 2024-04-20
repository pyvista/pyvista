#!/usr/bin/env python
from pathlib import Path

from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOFLUENTCFF import vtkFLUENTCFFReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCompositePolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

VTK_DATA_ROOT = vtkGetDataRoot()

# Read some Fluent CFF data HDF5 form
r = vtkFLUENTCFFReader()
r.SetFileName(str(Path(__file__).parent) + "/room.cas.h5")
r.EnableAllCellArrays()

g = vtkGeometryFilter()
g.SetInputConnection(r.GetOutputPort())

FluentMapper = vtkCompositePolyDataMapper()
FluentMapper.SetInputConnection(g.GetOutputPort())
FluentMapper.SetScalarModeToUseCellFieldData()
FluentMapper.SelectColorArray("SV_P")
FluentMapper.SetScalarRange(-31, 44)
FluentActor = vtkActor()
FluentActor.SetMapper(FluentMapper)

ren1 = vtkRenderer()
renWin = vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
# Add the actors to the renderer, set the background and size
#
ren1.AddActor(FluentActor)
renWin.SetSize(300, 300)
renWin.Render()
ren1.ResetCamera()
ren1.GetActiveCamera().SetPosition(2, 2, 11)
ren1.GetActiveCamera().SetFocalPoint(2, 2, 0)
ren1.GetActiveCamera().SetViewUp(0, 1, 0)
ren1.GetActiveCamera().SetViewAngle(30)
ren1.ResetCameraClippingRange()

iren.Initialize()
# --- end of script --
