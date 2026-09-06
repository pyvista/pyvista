"""Typing cases for :func:`pyvista.fit_line_to_points`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Meta = tuple[pv.PolyData, float, NumpyArray[float]]


def some_points() -> NumpyArray[float]:
    """Return points spread along one direction."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.1], [3.0, 0.1, 0.0]])


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.fit_line_to_points(some_points()), pv.PolyData)
assert_types(pv.fit_line_to_points(some_points(), return_meta=False), pv.PolyData)
assert_types(pv.fit_line_to_points(some_points(), init_direction='x'), pv.PolyData)
assert_types(pv.fit_line_to_points(some_points(), init_direction=(1.0, 0.0, 0.0)), pv.PolyData)

assert_types(pv.fit_line_to_points(some_points(), return_meta=True), _Meta)

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.fit_line_to_points(some_points(), return_meta=a_flag()), pv.PolyData | _Meta)
