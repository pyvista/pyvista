"""Typing cases for :meth:`pyvista.Plotter.add_title`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter to add text to."""
    return pv.Plotter()


assert_types(a_plotter().add_title('a'), pv.CornerAnnotation | pv.Text)
