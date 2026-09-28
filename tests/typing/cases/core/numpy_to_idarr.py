"""Typing cases for :func:`pyvista.core.utilities.cells.numpy_to_idarr`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types
from type_assert import skip_runtime

from pyvista import _vtk
from pyvista.core._typing_core import NumpyArray
from pyvista.core.utilities.cells import numpy_to_idarr


def a_flag() -> bool:  # pragma: no cover
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(numpy_to_idarr([0, 1]), _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr(np.array([0, 1])), _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr(np.array([True, False])), _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr([0, 1], deep=True), _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr([0, 1], return_ind=False), _vtk.vtkIdTypeArray)

with skip_runtime(reason='the runtime checker does not accept an int64 array as `NumpyArray[int]`'):
    assert_types(numpy_to_idarr([0, 1], return_ind=True), tuple[_vtk.vtkIdTypeArray, NumpyArray[int]])
    assert_types(numpy_to_idarr([0, 1], deep=True, return_ind=True), tuple[_vtk.vtkIdTypeArray, NumpyArray[int]])

    # The catch-all, reached only by a flag widened to `bool`
    assert_types(numpy_to_idarr([0, 1], return_ind=a_flag()), tuple[_vtk.vtkIdTypeArray, NumpyArray[int]] | _vtk.vtkIdTypeArray)
