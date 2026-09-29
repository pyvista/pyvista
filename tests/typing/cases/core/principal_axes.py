"""Typing cases for :func:`pyvista.principal_axes`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def some_points() -> NDArray[np.floating]:
    """Return points with a distinct variance along each axis."""
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.principal_axes(some_points()), NDArray[np.floating])
assert_types(pv.principal_axes(some_points(), return_std=False), NDArray[np.floating])
assert_types(pv.principal_axes(some_points(), return_std=True), tuple[NDArray[np.floating], NDArray[np.floating]])

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.principal_axes(some_points(), return_std=a_flag()), NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]])
