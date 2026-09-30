"""Typing cases for :func:`pyvista.axis_rotation`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def some_points() -> NDArray[np.floating]:
    """Return a fresh array of points, since `inplace=True` mutates it."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def float32_points() -> NDArray[np.float32]:
    """Return float32 points, which rotate to float64."""
    return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def int32_points() -> NDArray[np.int32]:
    """Return integer points, which rotate to float64."""
    return np.array([[1, 0, 0]], dtype=np.int32)


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.axis_rotation(some_points(), 90.0), NDArray[np.float64])
assert_types(pv.axis_rotation(some_points(), 90.0, inplace=False), NDArray[np.float64])
assert_types(pv.axis_rotation(some_points(), 90.0, axis='x', deg=True), NDArray[np.float64])
assert_types(pv.axis_rotation(some_points(), np.pi / 2, axis='y', deg=False), NDArray[np.float64])
assert_types(pv.axis_rotation(float32_points(), 90.0), NDArray[np.float64])
assert_types(pv.axis_rotation(int32_points(), 90.0), NDArray[np.float64])

assert_types(pv.axis_rotation(some_points(), 90.0, inplace=True), None)

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.axis_rotation(some_points(), 90.0, inplace=a_flag()), NDArray[np.float64] | None)


# Never called; the ignore is reported as unused if integer points ever type check in place
def int32_points_in_place() -> None:  # pragma: no cover
    """Pass integer points with ``inplace=True``, which would truncate the result."""
    pv.axis_rotation(int32_points(), 45, inplace=True)  # type: ignore[arg-type]
