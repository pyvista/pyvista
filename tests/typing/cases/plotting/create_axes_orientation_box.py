"""Typing cases for :func:`pyvista.create_axes_orientation_box`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.create_axes_orientation_box(), _vtk.vtkAnnotatedCubeActor)
assert_types(pv.create_axes_orientation_box(color_box=False), _vtk.vtkAnnotatedCubeActor)

assert_types(pv.create_axes_orientation_box(color_box=True), _vtk.vtkPropAssembly)

# The catch-all, reached only by a flag widened to `bool`
assert_types(
    pv.create_axes_orientation_box(color_box=a_flag()),
    _vtk.vtkAnnotatedCubeActor | _vtk.vtkPropAssembly,
)
