"""Typing cases for :meth:`pyvista.Plotter.set_chart_interaction`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


def a_chart_plotter() -> pv.Plotter:
    """Return a plotter holding one chart."""
    pl = pv.Plotter()
    pl.add_chart(pv.Chart2D())
    return pl


assert_types(a_chart_plotter().set_chart_interaction(True), list[pv.Chart2D | pv.ChartBox | pv.ChartPie | pv.ChartMPL])
assert_types(a_chart_plotter().set_chart_interaction(False), list[pv.Chart2D | pv.ChartBox | pv.ChartPie | pv.ChartMPL])
