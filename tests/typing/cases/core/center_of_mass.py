"""Typing cases for :meth:`pyvista.PolyData.center_of_mass`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import poly

assert_types(poly().center_of_mass(), NDArray[np.float64])
assert_types(poly().center_of_mass(scalars_weight=False), NDArray[np.float64])
