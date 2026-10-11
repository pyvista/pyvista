"""Typing cases for :meth:`pyvista.CellArray.from_irregular_cells`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def int64_cell() -> NDArray[np.int64]:
    """Return the point ids of a triangle."""
    return np.array([0, 1, 2], dtype=np.int64)


def uint16_cell() -> NDArray[np.uint16]:
    """Return the point ids of a quad with an unsigned dtype."""
    return np.array([0, 1, 2, 3], dtype=np.uint16)


def int32_regular_cells() -> NDArray[np.int32]:
    """Return a (n_cells, cell_size) array of point ids."""
    return np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)


assert_types(pv.CellArray.from_irregular_cells([[0, 1, 2], [1, 2, 3, 4]]), pv.CellArray)
assert_types(pv.CellArray.from_irregular_cells([int64_cell(), uint16_cell()]), pv.CellArray)
assert_types(pv.CellArray.from_irregular_cells(int32_regular_cells()), pv.CellArray)
