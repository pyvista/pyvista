"""Typing cases for :func:`pyvista.core.utilities.arrays.vtkmatrix_from_array`.

The shape of the argument decides which matrix comes back; both are held to the union.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

from pyvista import _vtk
from pyvista.core.utilities.arrays import vtkmatrix_from_array

if TYPE_CHECKING:
    from numpy.typing import NDArray


def nested_list() -> list[list[float]]:
    """Return a 3x3 identity as nested lists."""
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def int64_matrix() -> NDArray[np.int64]:
    """Return a 4x4 int64 identity."""
    return np.eye(4, dtype=np.int64)


assert_types(vtkmatrix_from_array(np.eye(3)), _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4)
assert_types(vtkmatrix_from_array(np.eye(4)), _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4)
assert_types(vtkmatrix_from_array(nested_list()), _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4)
assert_types(vtkmatrix_from_array(int64_matrix()), _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4)
