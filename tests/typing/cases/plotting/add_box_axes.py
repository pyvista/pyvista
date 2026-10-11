"""Typing cases for :meth:`pyvista.Plotter.add_box_axes`."""

from __future__ import annotations

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv
from pyvista import _vtk


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


with skip_runtime(pv.vtk_version_info < (9, 4), reason='enabling a widget renders, and the VTK 9.3 wheel renders only through a display'):
    assert_types(a_plotter().add_box_axes(), _vtk.vtkPropAssembly)
