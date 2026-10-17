"""Typing cases for :func:`pyvista.core.utilities.arrays.vtk_id_list_to_array`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from pyvista import _vtk
from pyvista.core.utilities.arrays import vtk_id_list_to_array


def an_id_list() -> _vtk.vtkIdList:
    """Return an id list holding three ids."""
    ids = _vtk.vtkIdList()
    for i in (3, 1, 2):
        ids.InsertNextId(i)
    return ids


assert_types(vtk_id_list_to_array(an_id_list()), NDArray[np.int_])
