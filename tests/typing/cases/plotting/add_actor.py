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


_ActorAndProperty = tuple[_vtk.vtkProp, _vtk.vtkProperty | _vtk.vtkVolumeProperty | None]


def an_actor() -> pv.Actor:
    """Return an actor drawing a cube."""
    return pv.Actor(mapper=pv.DataSetMapper(pv.Cube()))


assert_types(a_plotter().add_actor(an_actor()), _ActorAndProperty)
assert_types(a_plotter().add_actor(an_actor(), name='cube', reset_camera=True), _ActorAndProperty)

# A volume carries a volume property
assert_types(a_plotter().add_actor(pv.Volume()), _ActorAndProperty)
