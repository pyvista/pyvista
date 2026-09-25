"""Typing cases for :meth:`pyvista.Plotter.add_mesh`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def a_mask() -> NDArray[np.bool_]:
    """Return a boolean mask volume."""
    return np.zeros((2, 2, 2), dtype=np.bool_)


def a_label_volume() -> NDArray[np.uint8]:
    """Return a label volume."""
    return np.zeros((2, 2, 2), dtype=np.uint8)


assert_types(pv.Plotter().add_mesh(pv.Sphere()), pv.Actor)
assert_types(pv.Plotter().add_mesh(np.zeros((4, 3))), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_mask()), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_label_volume()), pv.Actor)
