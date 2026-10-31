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


# A mapper is wrapped in a new actor
assert_types(a_plotter().add_actor(pv.DataSetMapper(pv.Cube())), tuple[pv.Actor, pv.Property])

# An actor comes back as given, with its surface property
assert_types(a_plotter().add_actor(an_actor()), tuple[pv.Actor, _vtk.vtkProperty])
assert_types(
    a_plotter().add_actor(an_actor(), name='cube', reset_camera=True),
    tuple[pv.Actor, _vtk.vtkProperty],
)
assert_types(a_plotter().add_actor(_vtk.vtkActor()), tuple[_vtk.vtkActor, _vtk.vtkProperty])

# A volume carries a volume property
assert_types(a_plotter().add_actor(pv.Volume()), tuple[pv.Volume, _vtk.vtkVolumeProperty])

# A 2D actor carries a 2D property
assert_types(a_plotter().add_actor(pv.Text('a')), tuple[pv.Text, _vtk.vtkProperty2D])

# Any other prop may have no property
assert_types(
    a_plotter().add_actor(pv.AxesAssembly()),
    tuple[pv.AxesAssembly, _vtk.vtkProperty | _vtk.vtkVolumeProperty | _vtk.vtkProperty2D | None],
)
