"""Typing cases for :meth:`pyvista.DataSet.find_cells_along_line`."""

from __future__ import annotations

from typing import Never

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import pointset
from tests.typing.meshes import poly

assert_types(poly().find_cells_along_line((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), NDArray[np.int_])

assert_types(pointset().find_cells_along_line((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), Never)
