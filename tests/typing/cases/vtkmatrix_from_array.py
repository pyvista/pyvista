"""Typing cases for :func:`pyvista.core.utilities.arrays.vtkmatrix_from_array`.

The shape of the argument decides which matrix comes back; both are held to the union.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

from pyvista import _vtk
from pyvista.core.utilities.arrays import vtkmatrix_from_array

# fmt: off

assert_types(vtkmatrix_from_array(np.eye(3)),                  _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4)
assert_types(vtkmatrix_from_array(np.eye(4)),                  _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4)
