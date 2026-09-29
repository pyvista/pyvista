"""Typing cases for :attr:`pyvista.StructuredGrid.y`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import _Real
from tests.typing.meshes import structured


def int_structured() -> pv.StructuredGrid:
    """Return a structured grid that keeps integer points."""
    axis = np.arange(3)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return pv.StructuredGrid(x, y, z, force_float=False)


assert_types(structured().y, NDArray[_Real])
assert_types(int_structured().y, NDArray[_Real])
