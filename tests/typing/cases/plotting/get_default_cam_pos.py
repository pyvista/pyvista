"""Typing cases for :meth:`pyvista.Plotter.get_default_cam_pos`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import VectorLike


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


assert_types(a_plotter().get_default_cam_pos(), list[VectorLike[float]])
assert_types(a_plotter().get_default_cam_pos(negative=True), list[VectorLike[float]])
