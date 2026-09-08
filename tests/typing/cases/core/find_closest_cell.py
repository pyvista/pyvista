"""Typing cases for :meth:`pyvista.DataSet.find_closest_cell`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Result = int | NumpyArray[int] | tuple[int | NumpyArray[int], NumpyArray[float]]

SKIP_RUNTIME = {
    'pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])': 'the runtime checker does not accept an int64 array as `NumpyArray[int]`',
}

# The closest point comes back as floats, whichever half of the union is taken
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0)), _Result)
assert_types(pv.Sphere().find_closest_cell([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]), _Result)
assert_types(pv.Sphere().find_closest_cell((0.0, 0.0, 0.0), return_closest_point=True), _Result)
