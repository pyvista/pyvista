"""Demonstrates how to use the vtkRuledSurfaceFilter to create a ruled surface from lines."""

# noinspection PyUnresolvedReferences
from __future__ import annotations

from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter

import pyvista as pv

pv.set_jupyter_backend('static')

pl = pv.Plotter()
poly = pv.PolyData(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]], lines=[[2, 0, 1], [2, 2, 3]], force_float=False
)

alg = vtkRuledSurfaceFilter()
alg.SetInputData(poly)
alg.SetResolution(21, 21)
alg.SetRuledModeToResample()
alg.Update()

pl.add_mesh(pv.wrap(alg.GetOutput()), show_edges=True)
pl.show()
