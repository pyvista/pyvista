"""Typing cases for :attr:`pyvista.RectilinearGrid.y`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pyvista_validation._typing._array_like import _Real
from type_assert import assert_types

from tests.typing.meshes import rectilinear

if TYPE_CHECKING:
    import pyvista as pv


def int_rectilinear() -> pv.RectilinearGrid:
    """Return a rectilinear grid with coordinates set from integers."""
    grid = rectilinear()
    grid.x = np.arange(4)
    grid.y = np.arange(4)
    grid.z = np.arange(4)
    return grid


assert_types(rectilinear().y, NDArray[_Real])
assert_types(int_rectilinear().y, NDArray[_Real])
