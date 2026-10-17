"""Typing cases for :func:`pyvista.core.utilities.cells.create_mixed_cells`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv
from pyvista.core.utilities.cells import create_mixed_cells
from pyvista.core.utilities.cells import get_mixed_cells

_CellArrays = tuple[NDArray[np.uint8], NDArray[np.integer]]


def int_keyed_cells() -> dict[int, NDArray[np.int64]]:
    """Return triangle cells keyed by a plain ``int``."""
    return {int(pv.CellType.TRIANGLE): np.array([[0, 1, 2]], dtype=np.int64)}


def celltype_keyed_cells() -> dict[pv.CellType, NDArray[np.uint32]]:
    """Return unsigned triangle cells keyed by a ``CellType``."""
    return {pv.CellType.TRIANGLE: np.array([[0, 1, 2]], dtype=np.uint32)}


def ragged_cells() -> dict[pv.CellType, list[NDArray[np.int32]]]:
    """Return polygon cells of differing sizes, one array per cell."""
    return {pv.CellType.POLYGON: [np.array([0, 1, 2], dtype=np.int32), np.array([0, 1, 2, 3], dtype=np.int32)]}


def mixed_cells_dict() -> dict[np.uint8, NDArray[np.signedinteger] | list[NDArray[np.signedinteger]]]:
    """Return the cells dict of a small grid."""
    return get_mixed_cells(pv.UnstructuredGrid({pv.CellType.TRIANGLE: np.array([[0, 1, 2]])}, np.zeros((3, 3))))


assert_types(create_mixed_cells(int_keyed_cells()), _CellArrays)
assert_types(create_mixed_cells(celltype_keyed_cells()), _CellArrays)
assert_types(create_mixed_cells(ragged_cells()), _CellArrays)
assert_types(create_mixed_cells(mixed_cells_dict()), _CellArrays)
assert_types(create_mixed_cells({pv.CellType.POLYGON: [[0, 1, 2], [0, 1, 2, 3]]}), _CellArrays)
