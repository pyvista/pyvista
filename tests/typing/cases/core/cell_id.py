"""Typing cases for :meth:`pyvista.ExplicitStructuredGrid.cell_id`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import explicit_structured

_CellId = int | NDArray[np.intp] | None

assert_types(explicit_structured().cell_id((1, 2, 0)), _CellId)
assert_types(explicit_structured().cell_id([(1, 2, 0), (0, 0, 2)]), _CellId)
assert_types(explicit_structured().cell_id(np.array([1, 2, 0])), _CellId)
