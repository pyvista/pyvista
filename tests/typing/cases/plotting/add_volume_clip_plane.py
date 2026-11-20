"""Typing cases for :meth:`pyvista.plotting.widgets.WidgetComponent.add_volume_clip_plane`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk

if TYPE_CHECKING:
    from pyvista.plotting.widgets import WidgetComponent


def a_component() -> WidgetComponent:
    """Return the widget component of a plotter."""
    return pv.Plotter().widgets


def a_volume() -> pv.ImageData:
    """Return a grid with scalars to render as a volume."""
    grid = pv.ImageData(dimensions=(10, 10, 10))
    grid.point_data['scalars'] = np.linspace(0, 255, grid.n_points).astype(np.uint8)
    return grid


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(a_component().add_volume_clip_plane(a_volume()), _vtk.vtkImplicitPlaneWidget)
assert_types(a_component().add_volume_clip_plane(a_volume(), implicit=True), _vtk.vtkImplicitPlaneWidget)

assert_types(a_component().add_volume_clip_plane(a_volume(), implicit=False), _vtk.vtkPlaneWidget)

# The catch-all, reached only by a flag widened to `bool`
assert_types(a_component().add_volume_clip_plane(a_volume(), implicit=a_flag()), _vtk.vtkImplicitPlaneWidget | _vtk.vtkPlaneWidget)
