"""Typing cases for :attr:`pyvista.RectilinearGrid.points`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from pyvista.core._typing_core import _Real
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


assert_types(rectilinear().points, NDArray[_Real])
assert_types(int_rectilinear().points, NDArray[_Real])
