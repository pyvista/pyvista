"""Typing cases for :meth:`pyvista.DataSet.find_cells_within_bounds`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from tests.typing.meshes import poly

assert_types(poly().find_cells_within_bounds(poly().bounds), npt.NDArray[np.intp])
