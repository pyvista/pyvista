"""Typing cases for :func:`pyvista.core.utilities.arrays.convert_array`."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista import _vtk
from pyvista.core.utilities.arrays import convert_array

# A VTK array comes back as NumPy
assert_types(convert_array(_vtk.vtkFloatArray()), npt.NDArray[Any])
assert_types(convert_array(_vtk.vtkStringArray()), npt.NDArray[Any])
assert_types(convert_array(_vtk.vtkFloatArray(), 'data'), npt.NDArray[Any])
assert_types(convert_array(_vtk.vtkFloatArray(), deep=True), npt.NDArray[Any])

# Anything array-like goes the other way
assert_types(convert_array(np.zeros(3)), _vtk.vtkAbstractArray)
assert_types(convert_array([1, 2, 3]), _vtk.vtkAbstractArray)
assert_types(convert_array((1.0, 2.0)), _vtk.vtkAbstractArray)
assert_types(convert_array('text'), _vtk.vtkAbstractArray)
assert_types(convert_array(np.zeros(3), 'data'), _vtk.vtkAbstractArray)
assert_types(convert_array(np.zeros(3), deep=True), _vtk.vtkAbstractArray)
assert_types(convert_array(np.zeros(3), array_type=_vtk.VTK_UNSIGNED_CHAR), _vtk.vtkAbstractArray)

assert_types(convert_array(None), None)
assert_types(convert_array(None, 'data'), None)
assert_types(convert_array(None, array_type=_vtk.VTK_UNSIGNED_CHAR), None)
