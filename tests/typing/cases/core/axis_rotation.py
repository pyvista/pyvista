"""Typing cases for :func:`pyvista.axis_rotation`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def some_points() -> NDArray[float]:
    """Return a fresh array of points, since `inplace=True` mutates it."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.axis_rotation(some_points(), 90.0), NDArray[float])
assert_types(pv.axis_rotation(some_points(), 90.0, inplace=False), NDArray[float])
assert_types(pv.axis_rotation(some_points(), 90.0, axis='x', deg=True), NDArray[float])
assert_types(pv.axis_rotation(some_points(), np.pi / 2, axis='y', deg=False), NDArray[float])

assert_types(pv.axis_rotation(some_points(), 90.0, inplace=True), None)

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.axis_rotation(some_points(), 90.0, inplace=a_flag()), NDArray[float] | None)
