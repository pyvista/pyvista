"""Typing cases for :meth:`pyvista.DataSet.find_cells_within_bounds`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import poly


def float64_bounds() -> NDArray[np.float64]:
    """Return float64 bounds."""
    return np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


assert_types(poly().find_cells_within_bounds(poly().bounds), NDArray[np.intp])
assert_types(poly().find_cells_within_bounds([0.0, 1, 0, 1, 0, 1]), NDArray[np.intp])
assert_types(poly().find_cells_within_bounds(float64_bounds()), NDArray[np.intp])
