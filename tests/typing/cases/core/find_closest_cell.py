"""Typing cases for :meth:`pyvista.DataSet.find_closest_cell`."""

from __future__ import annotations

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Cell = int | NumpyArray[int]
_CellAndPoint = tuple[int | NumpyArray[int], NumpyArray[float]]


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0)), _Cell)
with skip_runtime(reason='the runtime checker does not accept an int64 array as `NumpyArray[int]`'):
    assert_types(pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]), _Cell)
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=False), _Cell)

assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True), _CellAndPoint)

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=a_flag()), _Cell | _CellAndPoint)
