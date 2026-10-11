"""Typing cases for :class:`pyvista.pyvista_ndarray`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk

if TYPE_CHECKING:
    from numpy.typing import NDArray


def complex_values() -> NDArray[np.complex128]:
    """Return complex values."""
    return np.zeros(3, dtype=np.complex128)


assert_types(pv.pyvista_ndarray([0.0, 1.0]), pv.pyvista_ndarray)
assert_types(pv.pyvista_ndarray(['a', 'b']), pv.pyvista_ndarray)
assert_types(pv.pyvista_ndarray(complex_values()), pv.pyvista_ndarray)
assert_types(pv.pyvista_ndarray(_vtk.vtkFloatArray()), pv.pyvista_ndarray)
