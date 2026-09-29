"""Typing cases for :func:`pyvista.core.utilities.cells.ncells_from_cells`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

from pyvista.core.utilities.cells import ncells_from_cells

if TYPE_CHECKING:
    from numpy.typing import NDArray


def uint8_cells() -> NDArray[np.uint8]:
    """Return a legacy cell array with an unsigned dtype."""
    return np.array([3, 0, 1, 2], dtype=np.uint8)


def int64_cells() -> NDArray[np.int64]:
    """Return a legacy cell array with a signed dtype."""
    return np.array([3, 0, 1, 2], dtype=np.int64)


assert_types(ncells_from_cells(uint8_cells()), int)
assert_types(ncells_from_cells(int64_cells()), int)
