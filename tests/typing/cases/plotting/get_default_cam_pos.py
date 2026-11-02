"""Typing cases for :meth:`pyvista.Plotter.get_default_cam_pos`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


assert_types(a_plotter().get_default_cam_pos(), list[tuple[float, float, float]])
assert_types(a_plotter().get_default_cam_pos(negative=True), list[tuple[float, float, float]])
