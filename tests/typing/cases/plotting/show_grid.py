"""Typing cases for :meth:`pyvista.Plotter.show_grid`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


assert_types(a_plotter().show_grid(), pv.CubeAxesActor)
