"""Typing cases for :func:`pyvista.transformations.decomposition`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv
from pyvista.core.utilities.transformations import decomposition


def float64_matrix() -> NDArray[np.float64]:
    """Return a float64 4x4 matrix."""
    return np.eye(4)


def float32_matrix() -> NDArray[np.float32]:
    """Return a float32 4x4 matrix."""
    return np.eye(4, dtype=np.float32)


def float32_3x3() -> NDArray[np.float32]:
    """Return a float32 3x3 matrix."""
    return np.eye(3, dtype=np.float32)


def int32_matrix() -> NDArray[np.int32]:
    """Return an int32 4x4 matrix."""
    return np.eye(4, dtype=np.int32)


def float32_nested() -> list[list[NDArray[np.float32]]]:
    """Return a 4x4 matrix as nested lists of float32 scalar arrays."""
    return [[np.array(value, dtype=np.float32) for value in row] for row in np.eye(4)]


def a_mesh_array() -> pv.pyvista_ndarray:
    """Return a float32 4x4 matrix held by a dataset-style array."""
    return pv.pyvista_ndarray(np.eye(4, dtype=np.float32))


_Five64 = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
_Five = tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]

assert_types(decomposition(float64_matrix()), _Five64)
assert_types(decomposition(float64_matrix(), homogeneous=True), _Five64)
assert_types(decomposition(int32_matrix()), _Five64)
assert_types(decomposition([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), _Five64)
assert_types(decomposition(pv.Transform().scale(2.0)), _Five64)
assert_types(decomposition(pv.vtkmatrix_from_array(np.eye(4))), _Five64)

# A float32 4x4 matrix stays float32 but a float32 3x3 one does not
assert_types(decomposition(float32_matrix()), _Five)
assert_types(decomposition(float32_3x3()), _Five)
assert_types(decomposition(float32_nested()), _Five)
assert_types(decomposition(a_mesh_array()), _Five)
