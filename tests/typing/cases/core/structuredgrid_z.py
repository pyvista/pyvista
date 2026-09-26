"""Typing cases for :attr:`pyvista.StructuredGrid.z`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyvista_validation._typing._array_like import _Real
from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import structured


def int_structured() -> pv.StructuredGrid:
    """Return a structured grid that keeps integer points."""
    axis = np.arange(3)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return pv.StructuredGrid(x, y, z, force_float=False)


assert_types(structured().z, NDArray[_Real])
assert_types(int_structured().z, NDArray[_Real])
