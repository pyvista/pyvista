"""Typing cases for :meth:`pyvista.DataSet.find_closest_cell`."""

from __future__ import annotations

from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

_Cell = int | NDArray[int]
_CellAndPoint = tuple[int | NDArray[int], NDArray[float]]


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


SKIP_RUNTIME = {
    'pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])': 'the runtime checker does not accept an int64 array as `NDArray[int]`',
}

assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0)), _Cell)
assert_types(pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]), _Cell)  # pragma: no cover
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=False), _Cell)

assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True), _CellAndPoint)

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=a_flag()), _Cell | _CellAndPoint)
