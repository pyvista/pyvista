"""Typing cases for :meth:`pyvista.Plotter.compute_bounds`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import BoundsTuple


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


assert_types(a_plotter().compute_bounds(), BoundsTuple)
