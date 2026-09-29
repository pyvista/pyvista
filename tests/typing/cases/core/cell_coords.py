"""Typing cases for :meth:`pyvista.ExplicitStructuredGrid.cell_coords`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import explicit_structured

assert_types(explicit_structured().cell_coords(5), NDArray[np.intp] | None)
assert_types(explicit_structured().cell_coords((5, 7)), NDArray[np.intp] | None)
assert_types(explicit_structured().cell_coords(np.array([5, 7])), NDArray[np.intp] | None)
