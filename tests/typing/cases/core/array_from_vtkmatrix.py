"""Typing cases for :func:`pyvista.array_from_vtkmatrix`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk

assert_types(pv.array_from_vtkmatrix(_vtk.vtkMatrix3x3()), npt.NDArray[np.float64])
assert_types(pv.array_from_vtkmatrix(_vtk.vtkMatrix4x4()), npt.NDArray[np.float64])
