"""Typing cases for :meth:`pyvista.Plotter.add_actor`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


def an_actor() -> pv.Actor:
    """Return an actor drawing a cube."""
    return pv.Actor(mapper=pv.DataSetMapper(pv.Cube()))


assert_types(a_plotter().add_actor(an_actor()), tuple[_vtk.vtkProp, _vtk.vtkProperty | None])
assert_types(a_plotter().add_actor(an_actor(), name='cube', reset_camera=True), tuple[_vtk.vtkProp, _vtk.vtkProperty | None])
