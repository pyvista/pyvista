"""Typing cases for :func:`pyvista.principal_axes`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray


def some_points() -> NumpyArray[float]:
    """Return points with a distinct variance along each axis."""
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.principal_axes(some_points()), NumpyArray[float])
assert_types(pv.principal_axes(some_points(), return_std=False), NumpyArray[float])
assert_types(pv.principal_axes(some_points(), return_std=True), tuple[NumpyArray[float], NumpyArray[float]])

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.principal_axes(some_points(), return_std=a_flag()), NumpyArray[float] | tuple[NumpyArray[float], NumpyArray[float]])
