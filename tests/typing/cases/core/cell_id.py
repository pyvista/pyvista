"""Typing cases for :meth:`pyvista.ExplicitStructuredGrid.cell_id`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import explicit_structured

assert_types(explicit_structured().cell_id((1, 2, 0)), int | None)
assert_types(explicit_structured().cell_id([1, 2, 0]), int | None)
assert_types(explicit_structured().cell_id((9, 9, 9)), int | None)

assert_types(explicit_structured().cell_id([(1, 2, 0), (0, 0, 2)]), NDArray[np.intp] | None)
assert_types(explicit_structured().cell_id([(9, 9, 9)]), NDArray[np.intp] | None)

# An array, whose shape is not known statically
assert_types(explicit_structured().cell_id(np.array([1, 2, 0])), int | NDArray[np.intp] | None)
