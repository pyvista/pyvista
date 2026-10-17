"""Typing cases for :meth:`pyvista.Plotter.enable_depth_peeling`."""

from __future__ import annotations

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


with skip_runtime(pv.vtk_version_info < (9, 4), reason='probing depth peeling renders, and the VTK 9.3 wheel renders only through a display'):
    assert_types(a_plotter().enable_depth_peeling(), bool | None)
    assert_types(a_plotter().enable_depth_peeling(number_of_peels=4), bool | None)
