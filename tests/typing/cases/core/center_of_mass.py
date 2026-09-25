"""Typing cases for :meth:`pyvista.PolyData.center_of_mass`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from tests.typing.meshes import poly

assert_types(poly().center_of_mass(), npt.NDArray[np.float64])
assert_types(poly().center_of_mass(scalars_weight=False), npt.NDArray[np.float64])
