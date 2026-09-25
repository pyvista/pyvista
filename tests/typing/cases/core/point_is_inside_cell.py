"""Typing cases for :meth:`pyvista.DataSet.point_is_inside_cell`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import pointset

SKIP_RUNTIME = {
    'pointset().point_is_inside_cell(0, (0.5, 0.5, 0.5))': ('a `PointSet` has no cells, so the call raises'),
}


def a_grid() -> pv.ImageData:
    """Return a small grid to query."""
    return pv.ImageData(dimensions=(3, 3, 3))


assert_types(a_grid().point_is_inside_cell(0, (0.5, 0.5, 0.5)), bool | npt.NDArray[np.bool_])
assert_types(a_grid().point_is_inside_cell(0, [(0.5, 0.5, 0.5), (9.0, 9.0, 9.0)]), bool | npt.NDArray[np.bool_])

assert_types(pointset().point_is_inside_cell(0, (0.5, 0.5, 0.5)), Never)  # pragma: no cover
