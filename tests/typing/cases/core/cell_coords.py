"""Typing cases for :meth:`pyvista.ExplicitStructuredGrid.cell_coords`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from tests.typing.meshes import explicit_structured

assert_types(explicit_structured().cell_coords(5), npt.NDArray[np.intp] | None)
assert_types(explicit_structured().cell_coords((5, 7)), npt.NDArray[np.intp] | None)
assert_types(explicit_structured().cell_coords(np.array([5, 7])), npt.NDArray[np.intp] | None)
