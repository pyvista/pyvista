"""Demonstrates how to use the vtkRuledSurfaceFilter to create a ruled surface from lines."""

# noinspection PyUnresolvedReferences
from __future__ import annotations

from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter

import pyvista as pv

pv.set_jupyter_backend('static')

plotter = pv.Plotter()
polydata = pv.PolyData(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]], lines=[[2, 0, 1], [2, 2, 3]], force_float=False
)

ruledSurfaceFilter = vtkRuledSurfaceFilter()
ruledSurfaceFilter.SetInputData(polydata)
ruledSurfaceFilter.SetResolution(21, 21)
ruledSurfaceFilter.SetRuledModeToResample()
ruledSurfaceFilter.Update()

plotter.add_mesh(pv.wrap(ruledSurfaceFilter.GetOutput()), show_edges=True)
plotter.show()
