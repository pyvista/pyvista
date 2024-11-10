"""Demonstrates how to use the vtkRuledSurfaceFilter to create a ruled surface from lines."""

# noinspection PyUnresolvedReferences
from __future__ import annotations

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.vtkCommonDataModel import vtkLine
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
from vtkmodules.vtkRenderingCore import vtkActor
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper

import pyvista as pv

pv.set_jupyter_backend('static')

# Create a rendering window and renderer
plotter = pv.Plotter()

# Create the points for the lines.
points = vtkPoints()
points.InsertPoint(0, 0, 0, 1)
points.InsertPoint(1, 1, 0, 0)
points.InsertPoint(2, 0, 1, 0)
points.InsertPoint(3, 1, 1, 1)

# Create line1
line1 = vtkLine()
line1.GetPointIds().SetId(0, 0)
line1.GetPointIds().SetId(1, 1)

# Create line2
line2 = vtkLine()
line2.GetPointIds().SetId(0, 2)
line2.GetPointIds().SetId(1, 3)

# Create a cellArray containing the lines
lines = vtkCellArray()
lines.InsertNextCell(line1)
lines.InsertNextCell(line2)

# Create the vtkPolyData to contain the points and cellArray with the lines
polydata = vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

# Create the ruledSurfaceFilter from the polydata containing the lines
ruledSurfaceFilter = vtkRuledSurfaceFilter()
ruledSurfaceFilter.SetInputData(polydata)
ruledSurfaceFilter.SetResolution(21, 21)
ruledSurfaceFilter.SetRuledModeToResample()

# Create the mapper with the ruledSurfaceFilter as input
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(ruledSurfaceFilter.GetOutputPort())

# Create the actor with the mapper
actor = vtkActor()
actor.SetMapper(mapper)

# Add the actor to the display
plotter.add_actor(actor)

plotter.show()
