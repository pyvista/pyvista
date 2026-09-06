"""Typing cases for :meth:`pyvista.DataSet.point_is_inside_cell`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray


def a_grid() -> pv.ImageData:
    """Return a small grid to query."""
    return pv.ImageData(dimensions=(3, 3, 3))


# fmt: off

assert_types(a_grid().point_is_inside_cell(0, (0.5, 0.5, 0.5)),                     bool | NumpyArray[np.bool_])
assert_types(a_grid().point_is_inside_cell(0, [(0.5, 0.5, 0.5), (9.0, 9.0, 9.0)]),  bool | NumpyArray[np.bool_])
