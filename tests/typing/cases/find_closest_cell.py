"""Typing cases for :meth:`pyvista.DataSet.find_closest_cell`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Cell = int | NumpyArray[int]
_CellAndPoint = tuple[int | NumpyArray[int], NumpyArray[int]]


def a_flag() -> bool:  # pragma: no cover
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


SKIP_RUNTIME = dict.fromkeys(
    [
        'pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])',
        'pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True)',
        'pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=a_flag())',
    ],
    'the runtime checker has no `npt_promote`, so an int64 array is not a `NumpyArray[int]`',
)

assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0)), _Cell)
assert_types(pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]), _Cell)  # pragma: no cover
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=False), _Cell)

assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True), _CellAndPoint)  # pragma: no cover

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=a_flag()), _Cell | _CellAndPoint)  # pragma: no cover
