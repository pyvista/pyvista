"""Typing cases for :meth:`pyvista.Renderer.set_chart_interaction`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_renderer() -> pv.Renderer:
    """Return the renderer of a plotter holding one chart."""
    pl = pv.Plotter()
    pl.add_chart(pv.Chart2D())
    return pl.renderer


assert_types(a_renderer().set_chart_interaction(True), list[pv.Chart2D | pv.ChartBox | pv.ChartPie | pv.ChartMPL])
assert_types(a_renderer().set_chart_interaction(False, toggle=True), list[pv.Chart2D | pv.ChartBox | pv.ChartPie | pv.ChartMPL])
