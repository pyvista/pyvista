"""Typing cases for :meth:`pyvista.DataSet.find_closest_cell`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Found = int | NumpyArray[int] | tuple[int | NumpyArray[int], NumpyArray[int]]

SKIP_RUNTIME = dict.fromkeys(
    [
        'pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])',
        'pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True)',
    ],
    'an int64 array is not a `NumpyArray[int]` without the `npt_promote` mypy plugin',
)

# fmt: off

assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0)),                              _Found)
assert_types(pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),           _Found)  # pragma: no cover
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=False),  _Found)
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True),   _Found)  # pragma: no cover
