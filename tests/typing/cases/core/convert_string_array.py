"""Typing cases for :func:`pyvista.core.utilities.arrays.convert_string_array`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista import _vtk
from pyvista.core.utilities.arrays import convert_string_array


def a_vtk_string_array() -> _vtk.vtkStringArray:
    """Return a two-value VTK string array."""
    array = _vtk.vtkStringArray()
    array.SetNumberOfValues(2)
    array.SetValue(0, 'a')
    array.SetValue(1, 'b')
    return array


assert_types(convert_string_array(a_vtk_string_array()), npt.NDArray[np.str_])
assert_types(convert_string_array(a_vtk_string_array(), 'data'), npt.NDArray[np.str_])

assert_types(convert_string_array(np.array(['a', 'b'])), _vtk.vtkStringArray)
assert_types(convert_string_array(np.array(['a']), 'data'), _vtk.vtkStringArray)
