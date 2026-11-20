"""Typing cases for :meth:`pyvista.plotting.widgets.WidgetComponent.add_plane_widget`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk

if TYPE_CHECKING:
    from pyvista.plotting.widgets import WidgetComponent


def a_component() -> WidgetComponent:
    """Return the widget component of a plotter."""
    return pv.Plotter().widgets


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(a_component().add_plane_widget(None), _vtk.vtkImplicitPlaneWidget)
assert_types(a_component().add_plane_widget(None, implicit=True), _vtk.vtkImplicitPlaneWidget)

assert_types(a_component().add_plane_widget(None, implicit=False), _vtk.vtkPlaneWidget)

# The catch-all, reached only by a flag widened to `bool`
assert_types(a_component().add_plane_widget(None, implicit=a_flag()), _vtk.vtkImplicitPlaneWidget | _vtk.vtkPlaneWidget)
