"""Typing cases for :meth:`pyvista.Plotter.add_axes_at_origin`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


assert_types(a_plotter().add_axes_at_origin(), _vtk.vtkAxesActor)
