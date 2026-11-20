"""Typing cases for :meth:`pyvista.DataSet.find_closest_cell`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

_Cell = int | NDArray[np.int_]
_CellAndPoint = tuple[int | NDArray[np.int_], NDArray[np.float64]]


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0)), int)
assert_types(pv.Sphere().find_closest_cell([0, 0, 0], return_closest_point=False), int)
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True), tuple[int, NDArray[np.float64]])

assert_types(pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]), NDArray[np.int_])
assert_types(pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0)], return_closest_point=True), tuple[NDArray[np.int_], NDArray[np.float64]])

# An array, whose shape is not known statically
assert_types(pv.Sphere().find_closest_cell(np.zeros(3)), _Cell)
assert_types(pv.Sphere().find_closest_cell(np.zeros((2, 3)), return_closest_point=True), _CellAndPoint)

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=a_flag()), _Cell | _CellAndPoint)
