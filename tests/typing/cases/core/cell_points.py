"""Typing cases for :attr:`pyvista.Cell.points`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import poly

assert_types(poly().get_cell(0).points, NDArray[np.float64])
