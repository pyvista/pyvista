"""Typing cases for :func:`pyvista.core.utilities.cells.numpy_to_idarr`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

from pyvista import _vtk
from pyvista.core._typing_core import NumpyArray
from pyvista.core.utilities.cells import numpy_to_idarr


def a_flag() -> bool:  # pragma: no cover
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


SKIP_RUNTIME = dict.fromkeys(
    [
        'numpy_to_idarr([0, 1], return_ind=True)',
        'numpy_to_idarr([0, 1], deep=True, return_ind=True)',
        'numpy_to_idarr([0, 1], return_ind=a_flag())',
    ],
    'the runtime checker has no `npt_promote`, so an int64 array is not a `NumpyArray[int]`',
)

# fmt: off

assert_types(numpy_to_idarr([0, 1]),                             _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr(np.array([0, 1])),                   _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr(np.array([True, False])),            _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr([0, 1], deep=True),                  _vtk.vtkIdTypeArray)
assert_types(numpy_to_idarr([0, 1], return_ind=False),           _vtk.vtkIdTypeArray)

assert_types(numpy_to_idarr([0, 1], return_ind=True),            tuple[_vtk.vtkIdTypeArray, NumpyArray[int]])  # pragma: no cover
assert_types(numpy_to_idarr([0, 1], deep=True, return_ind=True), tuple[_vtk.vtkIdTypeArray, NumpyArray[int]])  # pragma: no cover

# The catch-all, reached only by a flag widened to `bool`
assert_types(numpy_to_idarr([0, 1], return_ind=a_flag()),        tuple[_vtk.vtkIdTypeArray, NumpyArray[int]] | _vtk.vtkIdTypeArray)  # pragma: no cover
