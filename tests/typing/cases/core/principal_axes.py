"""Typing cases for :func:`pyvista.principal_axes`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv


def some_points() -> npt.NDArray[float]:
    """Return points with a distinct variance along each axis."""
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.principal_axes(some_points()), npt.NDArray[float])
assert_types(pv.principal_axes(some_points(), return_std=False), npt.NDArray[float])
assert_types(pv.principal_axes(some_points(), return_std=True), tuple[npt.NDArray[float], npt.NDArray[float]])

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.principal_axes(some_points(), return_std=a_flag()), npt.NDArray[float] | tuple[npt.NDArray[float], npt.NDArray[float]])
