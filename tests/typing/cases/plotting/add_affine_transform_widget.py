"""Typing cases for :meth:`pyvista.plotting.widgets.WidgetComponent.add_affine_transform_widget`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from pyvista.plotting.widgets import WidgetComponent


def on_matrix(matrix: NDArray[np.float64]) -> None:
    """Accept the actor's float64 user matrix."""


def a_component() -> WidgetComponent:
    """Return the widget component of a plotter."""
    return pv.Plotter().widgets


def an_actor() -> pv.Actor:
    """Return an actor to transform."""
    return pv.Plotter().add_mesh(pv.Sphere())


assert_types(a_component().add_affine_transform_widget(an_actor(), release_callback=on_matrix), pv.AffineWidget3D)
assert_types(a_component().add_affine_transform_widget(an_actor(), interact_callback=on_matrix), pv.AffineWidget3D)
