"""Typing cases for :meth:`pyvista.Plotter.add_axes`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


SKIP_RUNTIME = (
    dict.fromkeys(
        [
            'a_plotter().add_axes()',
        ],
        'enabling a widget renders, the VTK 9.3 wheel renders only through a display, and the core phase has none',
    )
    if pv.vtk_version_info < (9, 4)
    else {}
)


assert_types(a_plotter().add_axes(), _vtk.vtkAxesActor | _vtk.vtkPropAssembly | _vtk.vtkAnnotatedCubeActor)
