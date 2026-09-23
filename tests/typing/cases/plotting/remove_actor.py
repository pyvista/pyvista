"""Typing cases for :meth:`pyvista.Plotter.remove_actor`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter holding one mesh, not yet rendered."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


def a_plotter_with_actor() -> tuple[pv.Plotter, pv.Actor]:
    """Return a plotter and the actor it holds."""
    pl = pv.Plotter()
    actor = pl.add_mesh(pv.Sphere())
    return pl, actor


def remove_it() -> bool:
    """Remove the plotter's actor and report whether it was found."""
    pl, actor = a_plotter_with_actor()
    return pl.remove_actor(actor)


assert_types(a_plotter().remove_actor('missing'), bool)
assert_types(remove_it(), bool)
