"""Typing cases for :attr:`pyvista.MultipleLinesSource.points`."""

from __future__ import annotations

from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import Real as _Real

assert_types(pv.MultipleLinesSource().points, NDArray[_Real])
assert_types(pv.MultipleLinesSource(points=[[0, 0, 0], [1, 1, 1]]).points, NDArray[_Real])
