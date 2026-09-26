"""Typing cases for :attr:`pyvista.MultipleLinesSource.points`."""

from __future__ import annotations

from numpy.typing import NDArray
from pyvista_validation._typing._array_like import _Real
from type_assert import assert_types

import pyvista as pv

assert_types(pv.MultipleLinesSource().points, NDArray[_Real])
assert_types(pv.MultipleLinesSource(points=[[0, 0, 0], [1, 1, 1]]).points, NDArray[_Real])
