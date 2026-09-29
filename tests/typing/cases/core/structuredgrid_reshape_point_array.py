"""Typing cases for ``pyvista.StructuredGrid._reshape_point_array``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import structured

GRID = structured()


def int32_values() -> NDArray[np.int32]:
    """Return one int32 value per point."""
    return np.zeros(GRID.n_points, dtype=np.int32)


def float32_values() -> NDArray[np.float32]:
    """Return one float32 value per point."""
    return np.zeros(GRID.n_points, dtype=np.float32)


assert_types(GRID._reshape_point_array(int32_values()), NDArray[np.int32])
assert_types(GRID._reshape_point_array(float32_values()), NDArray[np.float32])
