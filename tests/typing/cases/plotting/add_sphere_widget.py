"""Typing cases for :meth:`pyvista.plotting.widgets.WidgetComponent.add_sphere_widget`."""

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


assert_types(a_component().add_sphere_widget(None), _vtk.vtkSphereWidget)
assert_types(a_component().add_sphere_widget(None, center=(1.0, 0.0, 0.0)), _vtk.vtkSphereWidget)
assert_types(a_component().add_sphere_widget(None, center=[1, 0, 0]), _vtk.vtkSphereWidget)

# Several centers, or an array whose shape is not known statically
assert_types(a_component().add_sphere_widget(None, center=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], color=['r', 'b']), _vtk.vtkSphereWidget | list[_vtk.vtkSphereWidget])
assert_types(a_component().add_sphere_widget(None, center=np.zeros(3)), _vtk.vtkSphereWidget | list[_vtk.vtkSphereWidget])
