"""Typing cases for :meth:`pyvista.DataSet.find_cells_intersecting_line`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import poly

assert_types(poly().find_cells_intersecting_line((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), NDArray[np.intp])
